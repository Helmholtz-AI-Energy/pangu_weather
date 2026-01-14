import onnx
import pytest
import torch

from pangu_weather.pangu_weather import get_onnx_constant_tensor


@pytest.fixture(scope="module")
def onnx_model(pretrained_model_path_onnx):
    return onnx.load(pretrained_model_path_onnx)


@pytest.mark.smoke
def test_const_h(const_h, onnx_model):
    onnx_const_h = get_onnx_constant_tensor(onnx_model, "/b1/Constant_17")

    assert const_h.shape == onnx_const_h.shape
    assert const_h.allclose(onnx_const_h)


@pytest.mark.smoke
def test_constant_maps(constant_maps, onnx_model):
    onnx_constant_maps = get_onnx_constant_tensor(onnx_model, "/b1/Constant_44")

    assert constant_maps.shape == onnx_constant_maps.shape
    assert constant_maps.allclose(onnx_constant_maps)


@pytest.mark.smoke
def test_weather_statistics_input(weather_statistics, onnx_model):
    surface_mean, surface_std, upper_mean, upper_std = weather_statistics

    onnx_surface_mean = get_onnx_constant_tensor(onnx_model, "/b1/Constant_11")
    onnx_surface_std = get_onnx_constant_tensor(onnx_model, "/b1/Constant_12")
    onnx_upper_mean = get_onnx_constant_tensor(onnx_model, "/b1/Constant_9")
    onnx_upper_std = get_onnx_constant_tensor(onnx_model, "/b1/Constant_10")

    assert surface_mean.shape == onnx_surface_mean.shape
    assert surface_mean.allclose(onnx_surface_mean)

    assert surface_std.shape == onnx_surface_std.shape
    assert surface_std.allclose(onnx_surface_std)

    assert upper_mean.shape == onnx_upper_mean.shape
    assert upper_mean.allclose(onnx_upper_mean)

    assert upper_std.shape == onnx_upper_std.shape
    assert upper_std.allclose(onnx_upper_std)


@pytest.mark.smoke
def test_weather_statistics_output(weather_statistics, onnx_model):
    surface_mean, surface_std, upper_mean, upper_std = weather_statistics
    surface_mean, surface_std = surface_mean.view((4, 1, 1, 1)), surface_std.view((4, 1, 1, 1))
    upper_mean = upper_mean.flip(0).permute(3, 0, 1, 2).view(5, 1, 13, 1, 1)
    upper_std = upper_std.flip(0).permute(3, 0, 1, 2).view(5, 1, 13, 1, 1)

    onnx_surface_mean = get_onnx_constant_tensor(onnx_model, "/b1/Constant_3294")
    onnx_surface_std = get_onnx_constant_tensor(onnx_model, "/b1/Constant_3293")
    onnx_upper_mean = get_onnx_constant_tensor(onnx_model, "/b1/Constant_3292")
    onnx_upper_std = get_onnx_constant_tensor(onnx_model, "/b1/Constant_3291")

    assert surface_mean.shape == onnx_surface_mean.shape
    assert surface_mean.allclose(onnx_surface_mean)

    assert surface_std.shape == onnx_surface_std.shape
    assert surface_std.allclose(onnx_surface_std)

    assert upper_mean.shape == onnx_upper_mean.shape
    assert upper_mean.allclose(onnx_upper_mean)

    assert upper_std.shape == onnx_upper_std.shape
    assert upper_std.allclose(onnx_upper_std)


@pytest.mark.smoke
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_normalize_denormalize_weather(batch_size, best_device, weather_statistics):
    surface_mean, surface_std, upper_mean, upper_std = weather_statistics
    surface_mean, surface_std = surface_mean.view((1, 4, 1, 1)), surface_std.view((1, 4, 1, 1))
    upper_mean, upper_std = upper_mean.flip(0).permute(3, 0, 1, 2), upper_std.flip(0).permute(3, 0, 1, 2)

    def denormalize(upper, surface):
        return upper * upper_std + upper_mean, surface * surface_std + surface_mean

    def normalize(upper, surface):
        return (upper - upper_mean) / upper_std, (surface - surface_mean) / surface_std

    surface_data = torch.randn(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.randn(batch_size, 5, 13, 721, 1440, device=best_device)

    normalized_denormalized_input = denormalize(*normalize(upper_air_data, surface_data))
    denormalized_normalized_input = normalize(*denormalize(upper_air_data, surface_data))

    for processed_upper, processed_surface in [normalized_denormalized_input, denormalized_normalized_input]:
        # check that the shape stays the same
        assert processed_upper.shape == upper_air_data.shape
        assert processed_surface.shape == surface_data.shape
        # check that the data after normalization + de-normalization (or vice-versa) is within 1e-5 of the original data
        # (either absolute or relative to the variable (& pressure level) std)
        max_surface_error_by_variable = (processed_surface - surface_data).abs().view(batch_size, 4, -1).amax((0, -1))
        assert (max_surface_error_by_variable / surface_std.squeeze().clamp(min=1)).max() < 1e-5
        max_upper_error_by_variable = (processed_upper - upper_air_data).abs().view(batch_size, 5, 13, -1).amax((0, -1))
        assert (max_upper_error_by_variable / upper_std.squeeze().clamp(min=1)).max() < 1e-5
