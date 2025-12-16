import pytest
import torch

from pangu_weather.pangu_weather import PanguWeather, PanguWeatherBackbone
import pangu_pytorch_model
from tests.utils import pretrained_model_path_onnx, pretrained_model_path_torch


@pytest.mark.parametrize("batch_size", [1, 2])
def test_pangu_backbone_shapes(
        batch_size, best_device, random_weather_statistics, random_constant_maps, random_const_h):
    dim = 192
    surface_data = torch.zeros(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.zeros(batch_size, 5, 13, 721, 1440, device=best_device)
    expected_output_shape = (batch_size, 521280, 2 * dim)

    backbone = PanguWeatherBackbone(
        random_weather_statistics, random_constant_maps, random_const_h, dim).to(best_device)
    with torch.no_grad():
        output = backbone(upper_air_data, surface_data)

    assert output.shape == expected_output_shape


@pytest.mark.parametrize("batch_size", [1, pytest.param(2, marks=pytest.mark.smoke)])
def test_pangu_weather_shapes(batch_size, best_device, random_weather_statistics, random_constant_maps, random_const_h):
    dim = 192
    surface_data = torch.zeros(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.zeros(batch_size, 5, 13, 721, 1440, device=best_device)

    pangu_weather = PanguWeather(random_weather_statistics, random_constant_maps, random_const_h, dim).to(best_device)
    with torch.no_grad():
        upper_air_output, surface_output = pangu_weather(upper_air_data, surface_data)

    assert upper_air_output.shape == upper_air_data.shape
    assert surface_output.shape == surface_data.shape


@pytest.mark.parametrize("batch_size", [1, pytest.param(2, marks=pytest.mark.slow)])
def test_pangu_backbone_random_sample(batch_size, best_device, random_weather_statistics, random_constant_maps,
                                      random_const_h):
    dim = 192
    surface_data = torch.randn(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.randn(batch_size, 5, 13, 721, 1440, device=best_device)
    expected_output_shape = (batch_size, 521280, 2 * dim)

    torch.manual_seed(0)
    backbone = PanguWeatherBackbone(random_weather_statistics, random_constant_maps, random_const_h, dim)
    backbone.to(best_device)
    with torch.no_grad():
        output = backbone(upper_air_data, surface_data)

    torch.manual_seed(0)
    # to use the same initialization of the earth-specific bias: create on cpu first, then move to device
    backbone_pangu_pytorch = pangu_pytorch_model.PanguWeatherBackbone(
        random_weather_statistics, random_constant_maps, random_const_h, dim, device='cpu').to(best_device)
    backbone_pangu_pytorch.apply(lambda module: setattr(module, 'device', best_device))
    with torch.no_grad():
        output_pangu_pytorch = torch.cat(
            [backbone_pangu_pytorch(upper_air_data[i:i+1], surface_data[i:i+1]) for i in range(batch_size)])

    # check output shapes
    assert output.shape == expected_output_shape
    assert output_pangu_pytorch.shape == expected_output_shape

    # check output content
    # increase acceptable absolute error (atol) slightly for multi-sample batches
    assert output.allclose(output_pangu_pytorch, atol=1e-5 if batch_size > 1 else 1e-8)


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.smoke),
                                        pytest.param(2, marks=pytest.mark.slow)])
def test_pangu_weather_random_sample(batch_size, best_device, random_weather_statistics, random_constant_maps,
                                     random_const_h):
    dim = 192
    surface_data = torch.randn(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.randn(batch_size, 5, 13, 721, 1440, device=best_device)

    torch.manual_seed(0)
    pangu_weather = PanguWeather(random_weather_statistics, random_constant_maps, random_const_h, dim).to(best_device)
    with torch.no_grad():
        upper_air_output, surface_output = pangu_weather(upper_air_data, surface_data)

    torch.manual_seed(0)
    # to use the same initialization of the earth-specific bias: create on cpu first, then move to device
    pangu_weather_pangu_pytorch = pangu_pytorch_model.PanguWeather(
        [stat.to(best_device) for stat in random_weather_statistics],
        random_constant_maps.to(best_device), random_const_h.to(best_device), dim, device='cpu').to(best_device)
    pangu_weather_pangu_pytorch.apply(lambda module: setattr(module, 'device', best_device))
    with torch.no_grad():
        outputs_pangu_pytorch = [
            pangu_weather_pangu_pytorch(upper_air_data[i:i+1], surface_data[i:i+1]) for i in range(batch_size)]
    upper_air_output_pangu_pytorch, surface_output_pangu_pytorch = [
        torch.cat(outputs) for outputs in zip(*outputs_pangu_pytorch)]

    # check output shapes
    assert surface_output.shape == surface_data.shape
    assert upper_air_output.shape == upper_air_data.shape

    assert surface_output_pangu_pytorch.shape == surface_data.shape
    assert upper_air_output_pangu_pytorch.shape == upper_air_data.shape

    # check output content
    # increase acceptable absolute error (atol) slightly for multi-sample batches
    assert surface_output.allclose(surface_output_pangu_pytorch, atol=1e-5 if batch_size > 1 else 1e-8)
    assert upper_air_output.allclose(upper_air_output_pangu_pytorch, atol=1e-5 if batch_size > 1 else 1e-8)


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.slow),
                                        pytest.param(2, marks=pytest.mark.slow)])
@pytest.mark.skipif(condition=not pretrained_onnx_model_path.is_file(),
                    reason=f"Pretrained weights not found at {pretrained_onnx_model_path}")
def test_pangu_weather_vs_onnx(batch_size, best_device, random_weather_statistics, random_constant_maps,
                               random_const_h, pretrained_onnx_model):
    dim = 192
    surface_data = torch.randn(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.randn(batch_size, 5, 13, 721, 1440, device=best_device)

    pangu_weather = PanguWeather(random_weather_statistics, random_constant_maps, random_const_h, dim).to(best_device)
    # TODO: to compare output values (not just shapes) we need to use the exact same weights as the ONNX model
    with torch.no_grad():
        upper_air_output, surface_output = pangu_weather(upper_air_data, surface_data)

    outputs_onnx = [pretrained_onnx_model(upper_air_data[i], surface_data[i]) for i in range(batch_size)]
    upper_air_output_onnx, surface_output_onnx = [torch.stack(outputs) for outputs in zip(*outputs_onnx)]

    # check output shapes
    assert surface_output.shape == surface_data.shape
    assert upper_air_output.shape == upper_air_data.shape

    assert surface_output_onnx.shape == surface_data.shape
    assert upper_air_output_onnx.shape == upper_air_data.shape

    # check output content -> skipped for now until we can import the pre-trained ONNX weights
    # assert surface_output.allclose(surface_output_onnx, atol=1e-5 if batch_size > 1 else 1e-8)
    # assert upper_air_output.allclose(upper_air_output_onnx, atol=1e-5 if batch_size > 1 else 1e-8)
