import logging
from collections import OrderedDict

import torch
import timm.layers

from pangu_weather.layers import PatchEmbeddingConv1d, DownSample, EarthSpecificLayer, UpSample, PatchRecovery


logger = logging.getLogger(__name__)


def filter_by_prefix(iterable, prefixes):
    return [x for x in iterable if not any(x.startswith(f'{prefix}.') for prefix in prefixes)]


def load_pangu_pretrained_weights(model, path, device='cpu', expected_missing_modules=None,
                                  expected_additional_modules=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)

    unexpected_missing_keys = filter_by_prefix(missing_keys, expected_missing_modules or [])
    unexpected_additional_keys = filter_by_prefix(unexpected_keys, expected_additional_modules or [])

    logger.debug(f'Pretrained weights for {type(model)} loaded from {path}.')
    if unexpected_missing_keys:
        logger.info(f'Unexpected missing keys: {unexpected_missing_keys}')
    if unexpected_additional_keys:
        logger.info(f'Unexpected missing keys: {unexpected_additional_keys}')


class PanguWeatherBackbone(torch.nn.Module):
    def __init__(self, weather_statistics, constant_maps, const_h, dimension=192, drop_path=False, reproduce_mask=False,
                 checkpoint=True):
        super().__init__()
        self.dimension = dimension
        self._checkpoint = checkpoint

        # Patch embedding
        self.patch_size = (2, 4, 4)
        self._input_layer = PatchEmbeddingConv1d(
            self.patch_size, weather_statistics, constant_maps, const_h, self.dimension)

        # Down- and up-sampling
        self.downsample = DownSample(self.dimension)
        self.upsample = UpSample(2 * self.dimension, self.dimension)

        # Note on the drop path rate: pangu-pytorch uses 16 steps, but the official pseudocode seems to use 8
        drop_path_rate = (torch.linspace(0, 0.2, 16) if drop_path else torch.zeros(16)).tolist()

        # Four earth-specific layers
        layer_kwargs = {'reproduce_mask': reproduce_mask, 'checkpoint': self._checkpoint}
        layers = [
            EarthSpecificLayer(2, self.dimension, drop_path_rate[0:2], 6, (8, 181, 360), **layer_kwargs),
            EarthSpecificLayer(6, 2 * self.dimension, drop_path_rate[2:8], 12, (8, 91, 180), **layer_kwargs),
            EarthSpecificLayer(6, 2 * self.dimension, drop_path_rate[8:14], 12, (8, 91, 180), **layer_kwargs),
            EarthSpecificLayer(2, self.dimension, drop_path_rate[14:], 6, (8, 181, 360), **layer_kwargs),
        ]
        self.layers = torch.nn.Sequential(
            OrderedDict([(f'EarthSpecificLayer{i}', layer) for i, layer in enumerate(layers)]))

        self.initialize_weights()

    @property
    def checkpoint(self):
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, checkpoint):
        self._checkpoint = checkpoint
        for layer in self.layers:
            layer.checkpoint = checkpoint

    @property
    def lora_target_modules(self):
        # list of submodules to be tuned by LoRA: all linear layers
        return [name for name, module in self.named_modules() if isinstance(module, torch.nn.Linear)]

    def initialize_weights(self):
        def init_layer_weight(layer):
            if isinstance(layer, torch.nn.Linear):
                timm.layers.trunc_normal_(layer.weight, std=.02)
                if isinstance(layer, torch.nn.Linear) and layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, torch.nn.LayerNorm):
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.weight, 1.0)

        self.apply(init_layer_weight)

    def load_pretrained_weights(self, path, device='cpu'):
        logger.info(f'Loading pretrained Pangu-Weather backbone weights from {path}.')
        expected_missing_modules = []
        expected_additional_modules = ['_output_layer']
        load_pangu_pretrained_weights(self, path, device, expected_missing_modules, expected_additional_modules)

    def forward(self, upper_air_data, surface_data):
        # patch embedding
        x = self._input_layer(upper_air_data, surface_data)

        # encoder
        x = self.layers[0](x)  # output shape (B, 521280, C)
        skip = x  # remember for skip connection
        x = self.downsample(x)  # (8, 360, 181) -> (8, 180, 91)
        x = self.layers[1](x)  # output shape (B, 131040, 2C)

        # decoder
        x = self.layers[2](x)  # output shape (B, 131040, 2C)
        x = self.upsample(x)  # (8, 180, 91) -> (8, 360, 181)
        x = self.layers[3](x)  # output shape (B, 521280, C)

        # skip connection -> (B, 521280, 2C)
        return torch.cat((skip, x), dim=-1)


class PanguWeather(torch.nn.Module):
    def __init__(self, weather_statistics, constant_maps, const_h, dimension=192, checkpoint=True):
        super().__init__()
        self.dimension = dimension

        self.backbone = PanguWeatherBackbone(weather_statistics, constant_maps, const_h, dimension,
                                             checkpoint=checkpoint)
        self._output_layer = PatchRecovery(2 * self.dimension)

    def load_pretrained_weights(self, path, device='cpu'):
        logger.info(f'Loading pretrained Pangu-Weather weights from {path}.')
        expected_missing_modules = ['backbone']
        expected_additional_modules = ['_input_layer', 'downsample', 'upsample', 'layers']
        load_pangu_pretrained_weights(self, path, device, expected_missing_modules, expected_additional_modules)
        self.backbone.load_pretrained_weights(path, device)

    def forward(self, upper_air_data, surface_data):
        # Pass through backbone: patch embedding, encoder, and decoder
        x = self.backbone(upper_air_data, surface_data)
        # Patch recovery to extract output
        x = self._output_layer(x)
        return x  # upper_air_prediction, surface_prediction
