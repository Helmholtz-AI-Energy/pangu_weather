import logging
import pathlib
import re
from collections import OrderedDict

import onnx
import pandas
import torch
import timm.layers

from pangu_weather.layers import PatchEmbeddingConv1d, DownSample, EarthSpecificLayer, UpSample, PatchRecovery


logger = logging.getLogger(__name__)


def filter_by_prefix(iterable, prefixes):
    return [x for x in iterable if not any(x.startswith(f"{prefix}.") for prefix in prefixes)]


def load_pangu_pretrained_weights(
    model, path, device="cpu", expected_missing_modules=None, expected_additional_modules=None
):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)

    unexpected_missing_keys = filter_by_prefix(missing_keys, expected_missing_modules or [])
    unexpected_additional_keys = filter_by_prefix(unexpected_keys, expected_additional_modules or [])

    logger.debug(f"Pretrained weights for {type(model)} loaded from {path}.")
    if unexpected_missing_keys:
        logger.info(f"Unexpected missing keys: {unexpected_missing_keys}")
    if unexpected_additional_keys:
        logger.info(f"Unexpected missing keys: {unexpected_additional_keys}")


def onnx_tensor_to_torch(onnx_tensor):
    return torch.from_numpy(onnx.numpy_helper.to_array(onnx_tensor).copy())


def get_onnx_constant_tensor(onnx_model, node_name):
    node = [node for node in onnx_model.graph.node if node.name == node_name][0]
    onnx_tensor = [attribute for attribute in node.attribute if attribute.name == "value"][0].t
    return onnx_tensor_to_torch(onnx_tensor)


def load_pretrained_onnx_weights(pangu_weather_model, path_to_onnx_weights, onnx2torch_map_path=None):
    # Read mapping between ONNX and torch parameter names from file
    if onnx2torch_map_path is None:
        onnx2torch_map_path = pathlib.Path(__file__).parent / "onnx_to_torch_parameter_mapping.csv"
    onnx2torch_parameter_df = pandas.read_csv(onnx2torch_map_path)
    onnx2torch_parameter_map = dict(zip(onnx2torch_parameter_df.onnx_name, onnx2torch_parameter_df.torch_name))

    # Load onnx file of pretrained pangu model
    onnx_model = onnx.load(path_to_onnx_weights)

    # maps torch parameter name -> pre-trained parameter
    pretrained_parameters = {
        onnx2torch_parameter_map[initializer.name]: onnx_tensor_to_torch(initializer)
        for initializer in onnx_model.graph.initializer
        if initializer.name in onnx2torch_parameter_map
    }

    # assign pre-trained ONNX parameters to pytorch parameters
    for name, param in pangu_weather_model.named_parameters():
        if name not in pretrained_parameters:
            logger.warning(f"No pre-trained ONNX weight available for parameter {name}.")
            continue

        pre_trained_param = pretrained_parameters[name]
        is_linear = bool(re.match(r".*linear\d*.weight", name))
        if is_linear:
            assert len(pre_trained_param.shape) == 2
            logger.debug(f"Transposing weight {name}.")
            pre_trained_param = pre_trained_param.T

        assert param.shape == pre_trained_param.shape
        with torch.no_grad():
            param.copy_(pre_trained_param)

    # collect from ONNX and prepare constant map buffers (reshaping, flipping,...)
    onnx_buffers = {
        torch_name: get_onnx_constant_tensor(onnx_model, onnx_name)
        for onnx_name, torch_name in onnx2torch_parameter_map.items()
        if onnx_name.startswith("/")
    }
    buffer_names = [
        f"backbone._input_layer.{name}"
        for name in ["surface_mean", "surface_std", "upper_mean", "upper_std", "constant_maps", "const_h"]
    ]
    assert all(name in onnx_buffers for name in buffer_names)
    prepared_buffers = pangu_weather_model.backbone._input_layer.prepare_constant_buffers(
        *[onnx_buffers[name] for name in buffer_names]
    )
    for name, buffer in zip(buffer_names, prepared_buffers):
        onnx_buffers[name] = buffer

    # assign constant maps from ONNX to PatchEmbedding
    for name, buffer in pangu_weather_model.named_buffers():
        if name not in onnx_buffers:
            logger.warning(f"No pre-trained ONNX weight available for parameter {name}.")
            continue

        if not buffer.shape == onnx_buffers[name].shape:
            message = f"Mismatched shape for buffer {name}: got {onnx_buffers[name].shape} but expected {buffer.shape}."
            if buffer.squeeze().shape == onnx_buffers[name].squeeze().shape:
                logger.info(f"{message} Reshape by adding/removing singleton dimensions.")
                onnx_buffers[name] = onnx_buffers[name].view_as(buffer)
            else:
                raise ValueError(f"{message}.")

        with torch.no_grad():
            buffer.copy_(onnx_buffers[name])


class PanguWeatherBackbone(torch.nn.Module):
    def __init__(
        self,
        weather_statistics,
        constant_maps,
        const_h,
        dimension=192,
        drop_path=False,
        reproduce_mask=False,
        checkpoint=True,
    ):
        super().__init__()
        self.dimension = dimension
        self._checkpoint = checkpoint

        # Patch embedding
        self.patch_size = (2, 4, 4)
        self._input_layer = PatchEmbeddingConv1d(
            self.patch_size, weather_statistics, constant_maps, const_h, self.dimension
        )

        # Down- and up-sampling
        self.downsample = DownSample(self.dimension)
        self.upsample = UpSample(2 * self.dimension, self.dimension)

        # Note on the drop path rate: pangu-pytorch uses 16 steps, but the official pseudocode seems to use 8
        drop_path_rate = (torch.linspace(0, 0.2, 16) if drop_path else torch.zeros(16)).tolist()

        # Four earth-specific layers
        layer_kwargs = {"reproduce_mask": reproduce_mask, "checkpoint": self._checkpoint}
        layers = [
            EarthSpecificLayer(2, self.dimension, drop_path_rate[0:2], 6, (8, 181, 360), **layer_kwargs),
            EarthSpecificLayer(6, 2 * self.dimension, drop_path_rate[2:8], 12, (8, 91, 180), **layer_kwargs),
            EarthSpecificLayer(6, 2 * self.dimension, drop_path_rate[8:14], 12, (8, 91, 180), **layer_kwargs),
            EarthSpecificLayer(2, self.dimension, drop_path_rate[14:], 6, (8, 181, 360), **layer_kwargs),
        ]
        self.layers = torch.nn.Sequential(
            OrderedDict([(f"EarthSpecificLayer{i}", layer) for i, layer in enumerate(layers)])
        )

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
                timm.layers.trunc_normal_(layer.weight, std=0.02)
                if isinstance(layer, torch.nn.Linear) and layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, torch.nn.LayerNorm):
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.weight, 1.0)

        self.apply(init_layer_weight)

    def load_pretrained_weights(self, path, device="cpu"):
        logger.info(f"Loading pretrained Pangu-Weather backbone weights from {path}.")
        expected_missing_modules = []
        expected_additional_modules = ["_output_layer"]
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

        self.backbone = PanguWeatherBackbone(
            weather_statistics, constant_maps, const_h, dimension, checkpoint=checkpoint
        )
        self._output_layer = PatchRecovery(2 * self.dimension)

    def load_pretrained_weights(self, path, device="cpu"):
        logger.info(f"Loading pretrained Pangu-Weather weights from {path}.")
        expected_missing_modules = ["backbone"]
        expected_additional_modules = ["_input_layer", "downsample", "upsample", "layers"]
        load_pangu_pretrained_weights(self, path, device, expected_missing_modules, expected_additional_modules)
        self.backbone.load_pretrained_weights(path, device)

    def load_pretrained_onnx_weights(self, path):
        logger.info(f"Loading pretrained Pangu-Weather weights from {path}.")
        load_pretrained_onnx_weights(self, path)

    def forward(self, upper_air_data, surface_data):
        # Pass through backbone: patch embedding, encoder, and decoder
        x = self.backbone(upper_air_data, surface_data)
        # Patch recovery to extract output
        x = self._output_layer(x)
        return x  # upper_air_prediction, surface_prediction
