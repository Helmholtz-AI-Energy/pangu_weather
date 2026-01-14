import logging

import pytest
import torch

from pangu_weather.pangu_weather import PanguWeather, PanguWeatherBackbone
import pangu_pytorch_model
from tests.utils import batched_repeat

logger = logging.getLogger("pangu_weather." + __name__)


@pytest.mark.parametrize("batch_size", [1, pytest.param(2, marks=pytest.mark.slow)])
def test_pangu_backbone_shapes(
    batch_size, best_device, random_weather_statistics, random_constant_maps, random_const_h
):
    dim = 192
    surface_data = torch.zeros(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.zeros(batch_size, 5, 13, 721, 1440, device=best_device)
    expected_output_shape = (batch_size, 521280, 2 * dim)

    backbone = (
        PanguWeatherBackbone(random_weather_statistics, random_constant_maps, random_const_h, dim)
        .to(best_device)
        .eval()
    )
    with torch.no_grad():
        output = backbone(upper_air_data, surface_data)

    assert output.shape == expected_output_shape


@pytest.mark.parametrize("batch_size", [1, pytest.param(2, marks=pytest.mark.slow)])
def test_pangu_weather_shapes(batch_size, best_device, random_weather_statistics, random_constant_maps, random_const_h):
    dim = 192
    surface_data = torch.zeros(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.zeros(batch_size, 5, 13, 721, 1440, device=best_device)

    pangu_weather = (
        PanguWeather(random_weather_statistics, random_constant_maps, random_const_h, dim).to(best_device).eval()
    )
    with torch.no_grad():
        upper_air_output, surface_output = pangu_weather(upper_air_data, surface_data)

    assert upper_air_output.shape == upper_air_data.shape
    assert surface_output.shape == surface_data.shape


@pytest.mark.parametrize("batch_size", [1, pytest.param(2, marks=pytest.mark.slow)])
def test_pangu_backbone_random_sample(
    batch_size, best_device, random_weather_statistics, random_constant_maps, random_const_h
):
    dim = 192
    surface_data = torch.randn(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.randn(batch_size, 5, 13, 721, 1440, device=best_device)
    expected_output_shape = (batch_size, 521280, 2 * dim)

    torch.manual_seed(0)
    backbone = (
        PanguWeatherBackbone(random_weather_statistics, random_constant_maps, random_const_h, dim)
        .to(best_device)
        .eval()
    )
    with torch.no_grad():
        output = backbone(upper_air_data, surface_data)

    torch.manual_seed(0)
    # to use the same initialization of the earth-specific bias: create on cpu first, then move to device
    backbone_pangu_pytorch = pangu_pytorch_model.PanguWeatherBackbone(
        random_weather_statistics, random_constant_maps, random_const_h, dim, device="cpu"
    ).to(best_device)
    backbone_pangu_pytorch.apply(lambda module: setattr(module, "device", best_device))
    with torch.no_grad():
        output_pangu_pytorch = torch.cat(
            [backbone_pangu_pytorch(upper_air_data[i : i + 1], surface_data[i : i + 1]) for i in range(batch_size)]
        )

    # check output shapes
    assert output.shape == expected_output_shape
    assert output_pangu_pytorch.shape == expected_output_shape

    # check output content
    # increase acceptable absolute error (atol) slightly for multi-sample batches
    assert output.allclose(output_pangu_pytorch, atol=1e-5 if batch_size > 1 else 1e-8)


@pytest.mark.parametrize("batch_size", [1, pytest.param(2, marks=pytest.mark.slow)])
def test_pangu_weather_random_sample(
    batch_size, best_device, random_weather_statistics, random_constant_maps, random_const_h
):
    dim = 192
    surface_data = torch.randn(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.randn(batch_size, 5, 13, 721, 1440, device=best_device)

    torch.manual_seed(0)
    pangu_weather = (
        PanguWeather(random_weather_statistics, random_constant_maps, random_const_h, dim).to(best_device).eval()
    )
    with torch.no_grad():
        upper_air_output, surface_output = pangu_weather(upper_air_data, surface_data)

    torch.manual_seed(0)
    # to use the same initialization of the earth-specific bias: create on cpu first, then move to device
    pangu_weather_pangu_pytorch = pangu_pytorch_model.PanguWeather(
        [stat.to(best_device) for stat in random_weather_statistics],
        random_constant_maps.to(best_device),
        random_const_h.to(best_device),
        dim,
        device="cpu",
    ).to(best_device)
    pangu_weather_pangu_pytorch.apply(lambda module: setattr(module, "device", best_device))
    with torch.no_grad():
        outputs_pangu_pytorch = [
            pangu_weather_pangu_pytorch(upper_air_data[i : i + 1], surface_data[i : i + 1]) for i in range(batch_size)
        ]
    upper_air_output_pangu_pytorch, surface_output_pangu_pytorch = [
        torch.cat(outputs) for outputs in zip(*outputs_pangu_pytorch)
    ]

    # check output shapes
    assert surface_output.shape == surface_data.shape
    assert upper_air_output.shape == upper_air_data.shape

    assert surface_output_pangu_pytorch.shape == surface_data.shape
    assert upper_air_output_pangu_pytorch.shape == upper_air_data.shape

    # check output content
    # increase acceptable absolute error (atol) slightly for multi-sample batches
    assert surface_output.allclose(surface_output_pangu_pytorch, atol=1e-5 if batch_size > 1 else 1e-8)
    assert upper_air_output.allclose(upper_air_output_pangu_pytorch, atol=1e-5 if batch_size > 1 else 1e-8)


@pytest.mark.parametrize(
    "batch_size", [pytest.param(2, marks=pytest.mark.smoke), pytest.param(2, marks=pytest.mark.slow)]
)
def test_pangu_weather_load_pretrained(
    batch_size,
    best_device,
    pretrained_model_path_onnx,
    pretrained_model_path_torch,
    weather_statistics,
    constant_maps,
    const_h,
):
    logger.info("Preparing input and auxiliary data.")
    dim = 192
    surface_data = torch.randn(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.randn(batch_size, 5, 13, 721, 1440, device=best_device)

    # our model using either pangu-pytorch (pp) or onnx pre-trained weights
    logger.info("Loading PanguWeather with Pangu-Pytorch weights.")
    pangu_weather__pp_weights = PanguWeather(weather_statistics, constant_maps, const_h, dim).to(best_device).eval()
    pangu_weather__pp_weights.load_pretrained_weights(pretrained_model_path_torch, best_device)
    logger.info("Forward pass with PanguWeather with Pangu-Pytorch weights.")
    with torch.no_grad():
        upper_air_output__pp_weights, surface_output__pp_weights = pangu_weather__pp_weights(
            upper_air_data, surface_data
        )

    logger.info("Loading PanguWeather with official ONNX weights.")
    pangu_weather__onnx_weights = PanguWeather(weather_statistics, constant_maps, const_h, dim).to(best_device).eval()
    pangu_weather__onnx_weights.load_pretrained_onnx_weights(pretrained_model_path_onnx)
    logger.info("Forward pass with PanguWeather with official ONNX weights.")
    with torch.no_grad():
        upper_air_output__onnx_weights, surface_output__onnx_weights = pangu_weather__onnx_weights(
            upper_air_data, surface_data
        )

    # pangu-pytorch model with pre-trained weights
    logger.info("Loading PanguPytorch model with pre-trained weights.")
    pangu_weather_pangu_pytorch = pangu_pytorch_model.PanguWeather(
        [stat.to(best_device) for stat in weather_statistics],
        constant_maps.to(best_device),
        const_h.to(best_device),
        dim,
        device="cpu",
    ).to(best_device)
    pangu_weather_pangu_pytorch.load_pretrained_weights(pretrained_model_path_torch, best_device)
    # load_pangu_pretrained_weights(pangu_weather_pangu_pytorch, pretrained_model_path_torch, best_device)
    # checkpoint = torch.load(pretrained_model_path_torch, map_location=best_device, weights_only=True)
    # pangu_weather_pangu_pytorch.load_state_dict(checkpoint['model'])
    pangu_weather_pangu_pytorch.apply(lambda module: setattr(module, "device", best_device))
    logger.info("Forward pass with PanguPytorch model with pre-trained weights.")
    with torch.no_grad():
        outputs_pangu_pytorch = [
            pangu_weather_pangu_pytorch(upper_air_data[i : i + 1], surface_data[i : i + 1]) for i in range(batch_size)
        ]
    upper_air_output_pangu_pytorch, surface_output_pangu_pytorch = [
        torch.cat(outputs) for outputs in zip(*outputs_pangu_pytorch)
    ]

    # check output shapes (should be same as input shapes)
    logger.info("Checking output shapes.")
    assert surface_output__pp_weights.shape == surface_data.shape
    assert upper_air_output__pp_weights.shape == upper_air_data.shape

    assert surface_output__onnx_weights.shape == surface_data.shape
    assert upper_air_output__onnx_weights.shape == upper_air_data.shape

    assert surface_output_pangu_pytorch.shape == surface_data.shape
    assert upper_air_output_pangu_pytorch.shape == upper_air_data.shape

    # check output content
    logger.info("Checking output values.")
    atol = 1e-5 if batch_size > 1 else 1e-8  # increase acceptable absolute error slightly for multi-sample batches
    # ours with torch vs onnx pretrained weights
    assert surface_output__pp_weights.allclose(surface_output__onnx_weights, atol=atol)
    assert upper_air_output__pp_weights.allclose(upper_air_output__onnx_weights, atol=atol)
    # ours with torch pretrained weights vs pangu pytorch
    assert surface_output__pp_weights.allclose(surface_output_pangu_pytorch, atol=atol)
    assert upper_air_output__pp_weights.allclose(upper_air_output_pangu_pytorch, atol=atol)
    # ours with onnx pretrained weights vs pangu pytorch
    assert surface_output__onnx_weights.allclose(surface_output_pangu_pytorch, atol=atol)
    assert upper_air_output__onnx_weights.allclose(upper_air_output_pangu_pytorch, atol=atol)


@pytest.mark.parametrize("batch_size", [pytest.param(1), pytest.param(2, marks=pytest.mark.slow)])
def test_pangu_weather_vs_onnx_on_example_input(
    batch_size,
    best_device,
    pretrained_model_path_onnx,
    weather_statistics,
    constant_maps,
    const_h,
    example_input,
    onnx_output_for_example_input,
):
    upper_air_data, surface_data = [batched_repeat(x, batch_size) for x in example_input]
    upper_air_output_onnx, surface_output_onnx = [batched_repeat(x, batch_size) for x in onnx_output_for_example_input]

    logger.info("Prepare weather statistics.")
    # prepare weather statistics for (de)normalization
    surface_mean, surface_std, upper_mean, upper_std = weather_statistics
    surface_mean, surface_std = surface_mean.view((1, 4, 1, 1)), surface_std.view((1, 4, 1, 1))
    upper_mean, upper_std = upper_mean.flip(0).permute(3, 0, 1, 2), upper_std.flip(0).permute(3, 0, 1, 2)

    def denormalize(upper, surface):
        return upper * upper_std + upper_mean, surface * surface_std + surface_mean

    def normalize(upper, surface):
        return (upper - upper_mean) / upper_std, (surface - surface_mean) / surface_std

    logger.info("Loading PanguWeather with official ONNX weights.")
    dim = 192
    pangu_weather = PanguWeather(weather_statistics, constant_maps, const_h, dim).to(best_device).eval()
    pangu_weather.load_pretrained_onnx_weights(pretrained_model_path_onnx)
    logger.info("Forward pass with PanguWeather with official ONNX weights.")
    with torch.no_grad():
        upper_air_output_normalized, surface_output_normalized = pangu_weather(upper_air_data, surface_data)

    # check output shapes
    logger.info("Checking output shapes.")
    assert surface_output_normalized.shape == surface_data.shape
    assert upper_air_output_normalized.shape == upper_air_data.shape

    assert surface_output_onnx.shape == surface_data.shape
    assert upper_air_output_onnx.shape == upper_air_data.shape

    # check normalized output content
    logger.info("Checking normalized results.")
    upper_air_output_onnx_normalized, surface_output_onnx_normalized = normalize(
        upper_air_output_onnx, surface_output_onnx
    )
    assert upper_air_output_normalized.allclose(upper_air_output_onnx_normalized, atol=5e-5)
    assert surface_output_normalized.allclose(surface_output_onnx_normalized, atol=5e-5)

    # check denormalized output content
    logger.info("Checking denormalized results.")
    upper_air_output, surface_output = denormalize(upper_air_output_normalized, surface_output_normalized)
    max_upper_error_by_variable = (
        (upper_air_output - upper_air_output_onnx).abs().view(batch_size, 5, 13, -1).amax((0, -1))
    )
    max_surface_error_by_variable = (surface_output - surface_output_onnx).abs().view(batch_size, 4, -1).amax((0, -1))
    assert (max_surface_error_by_variable / surface_std.view(1, 4).clamp(min=1)).max() < 5e-5
    assert (max_upper_error_by_variable / upper_std.view(1, 5, 13).clamp(min=1)).max() < 5e-5
