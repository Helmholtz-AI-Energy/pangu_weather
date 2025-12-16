import pytest
import torch

from pangu_weather.layers import PatchEmbeddingConv1d, PatchEmbeddingConv3d2d
from pangu_pytorch.models.layers import PatchEmbedding_pretrain
from tests.utils import get_available_torch_devices


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.smoke), 2, 4])
@pytest.mark.parametrize("device", get_available_torch_devices())
def test_patch_embedding_shapes(batch_size, random_weather_statistics, random_constant_maps, random_const_h, device):
    patch_size = (2, 4, 4)
    dim = 192
    patch_embedding_conv1d = PatchEmbeddingConv1d(
        patch_size, random_weather_statistics, random_constant_maps, random_const_h, dim).to(device)
    patch_embedding_conv3d2d = PatchEmbeddingConv3d2d(
        patch_size, random_weather_statistics, random_constant_maps, random_const_h, dim).to(device)

    surface_data = torch.zeros(batch_size, 4, 721, 1440, device=device)
    upper_air_data = torch.zeros(batch_size, 5, 13, 721, 1440, device=device)

    expected_shape = (batch_size, 521280, dim)
    embedded_input = patch_embedding_conv1d(upper_air_data, surface_data)
    assert embedded_input.shape == expected_shape
    embedded_input = patch_embedding_conv3d2d(upper_air_data, surface_data)
    assert embedded_input.shape == expected_shape


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.smoke), 2, 4])
def test_patch_embedding_random_sample(batch_size, random_weather_statistics, random_constant_maps, random_const_h,
                                       best_device):
    patch_size = (2, 4, 4)
    dim = 192

    torch.manual_seed(0)
    patch_embedding_conv1d = PatchEmbeddingConv1d(
        patch_size, random_weather_statistics, random_constant_maps, random_const_h, dim).to(best_device)

    torch.manual_seed(0)
    patch_embedding_conv3d2d = PatchEmbeddingConv3d2d(
        patch_size, random_weather_statistics, random_constant_maps, random_const_h, dim).to(best_device)

    torch.manual_seed(0)
    patch_embedding_pangu_pytorch = PatchEmbedding_pretrain(patch_size, dim).to(best_device)

    surface_data = torch.randn(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.randn(batch_size, 5, 13, 721, 1440, device=best_device)

    embedded_input_conv1d = patch_embedding_conv1d(upper_air_data, surface_data)
    embedded_input_conv3d2d = patch_embedding_conv3d2d(upper_air_data, surface_data)
    embedded_input_pangu_pytorch = torch.cat([patch_embedding_pangu_pytorch(
            upper_air_data[i:i+1], surface_data[i:i+1], [stat.to(best_device) for stat in random_weather_statistics],
            random_constant_maps.to(best_device), random_const_h.to(best_device))
        for i in range(batch_size)])

    # check output shapes
    expected_shape = (batch_size, 521280, dim)
    assert embedded_input_conv1d.shape == expected_shape
    assert embedded_input_conv3d2d.shape == expected_shape
    assert embedded_input_pangu_pytorch.shape == expected_shape

    # check output content
    assert embedded_input_conv1d.allclose(embedded_input_pangu_pytorch)
    assert embedded_input_conv3d2d.allclose(embedded_input_pangu_pytorch)
