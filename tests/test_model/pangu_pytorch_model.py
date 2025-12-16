from collections import OrderedDict

import timm.layers
import torch


import pangu_pytorch.models.layers as pangu_pytorch_layers
from pangu_weather.pangu_weather import load_pangu_pretrained_weights


class PanguWeatherBackbone(torch.nn.Module):
    """Counterpart to pangu_finetuning.model.PanguWeatherBackbone using pangu_pytorch layers for testing."""
    def __init__(self, weather_statistics, constant_maps, const_h, dimension=192, drop_path=False, device='cpu'):
        super().__init__()
        self.dimension = dimension

        # Patch embedding
        self.patch_size = (2, 4, 4)
        self.weather_statistics, self.constant_maps, self.const_h = weather_statistics, constant_maps, const_h
        self._input_layer = pangu_pytorch_layers.PatchEmbedding_pretrain(self.patch_size, self.dimension)

        # Down- and up-sampling
        self.downsample = pangu_pytorch_layers.DownSample(self.dimension)
        self.upsample = pangu_pytorch_layers.UpSample(2 * self.dimension, self.dimension)

        # Note on the drop path rate: pangu-pytorch uses 16 steps, but the official pseudocode seems to use 8
        drop_path_rate = (torch.linspace(0, 0.2, 16) if drop_path else torch.zeros(16)).tolist()

        # Four earth-specific layers
        layer_kwargs = {'use_checkpoint': self.training, 'device': device}
        layers = [
            pangu_pytorch_layers.EarthSpecificLayer(2, self.dimension, drop_path_rate[0:2], 6, **layer_kwargs),
            pangu_pytorch_layers.EarthSpecificLayer(6, 2 * self.dimension, drop_path_rate[2:8], 12, **layer_kwargs),
            pangu_pytorch_layers.EarthSpecificLayer(6, 2 * self.dimension, drop_path_rate[8:14], 12, **layer_kwargs),
            pangu_pytorch_layers.EarthSpecificLayer(2, self.dimension, drop_path_rate[14:], 6, **layer_kwargs),
        ]
        self.layers = torch.nn.Sequential(
            OrderedDict([(f'EarthSpecificLayer{i}', layer) for i, layer in enumerate(layers)]))

        self.initialize_weights()

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

    def forward(self, upper_air_data, surface_data):
        # patch embedding
        device = upper_air_data.device
        x = self._input_layer(upper_air_data, surface_data, tuple(stat.to(device) for stat in self.weather_statistics),
                              self.constant_maps.to(device), self.const_h.to(device))

        # encoder
        x = self.layers[0](x, 8, 181, 360)  # output shape (B, 521280, C), C = self.dimension
        skip = x  # remember for skip connection
        x = self.downsample(x, 8, 181, 360)  # (8, 360, 181) -> (8, 180, 91)
        x = self.layers[1](x, 8, 91, 180)  # output shape (B, 131040, 2C), C = self.dimension

        # decoder
        x = self.layers[2](x, 8, 91, 180)  # output shape (B, 131040, 2C), C = self.dimension
        x = self.upsample(x)  # (8, 180, 91) -> (8, 360, 181)
        x = self.layers[3](x, 8, 181, 360)  # output shape (B, 521280, C), C = self.dimension

        # skip connection -> (B, 521280, 2C)
        return torch.cat((skip, x), dim=-1)

    def load_pretrained_weights(self, path, device='cpu'):
        load_pangu_pretrained_weights(self, path, device, [], ['_output_layer'])


class PanguWeather(torch.nn.Module):
    """Counterpart to pangu_finetuning.model.PanguWeather using pangu_pytorch layers for testing."""
    def __init__(self, weather_statistics, constant_maps, const_h, dimension=192, device='cpu'):
        super().__init__()
        self.dimension = dimension
        self.backbone = PanguWeatherBackbone(weather_statistics, constant_maps, const_h, dimension, device=device)
        self._output_layer = pangu_pytorch_layers.PatchRecovery_pretrain(2 * self.dimension)

    def load_pretrained_weights(self, path, device='cpu'):
        expected_missing_modules = ['backbone']
        expected_additional_modules = ['_input_layer', 'downsample', 'upsample', 'layers']
        load_pangu_pretrained_weights(self, path, device, expected_missing_modules, expected_additional_modules)
        self.backbone.load_pretrained_weights(path, device)

    def forward(self, upper_air_data, surface_data):
        # Pass through backbone: patch embedding, encoder, and decoder
        x = self.backbone(upper_air_data, surface_data)
        # Patch recovery to extract output
        x = self._output_layer(x, 8, 181, 360)
        return x  # upper_air_prediction, surface_prediction

