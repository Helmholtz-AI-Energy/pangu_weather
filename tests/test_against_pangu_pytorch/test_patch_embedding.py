import pytest
import torch

from pangu_weather.layers import PatchEmbedding
from pangu_pytorch.models.layers import PatchEmbedding_pretrain
from tests.conftest import get_available_torch_devices


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("device", get_available_torch_devices())
def test_patch_embedding_shapes(batch_size, random_weather_statistics, random_constant_maps, random_const_h, device):
    patch_size = (2, 4, 4)
    patch_embedding = PatchEmbedding(
        patch_size, random_weather_statistics, random_constant_maps, random_const_h).to(device)
    surface_data = torch.zeros(batch_size, 4, 721, 1440, device=device)
    upper_air_data = torch.zeros(batch_size, 5, 13, 721, 1440, device=device)

    embedded_input = patch_embedding(upper_air_data, surface_data)
    assert embedded_input.shape == (batch_size, 521280, patch_embedding.dim)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_patch_embedding_random_sample(batch_size, random_weather_statistics, random_constant_maps, random_const_h,
                                       best_device):
    patch_size = (2, 4, 4)
    dim = 192

    torch.manual_seed(0)
    patch_embedding = PatchEmbedding(
        patch_size, random_weather_statistics, random_constant_maps, random_const_h, dim).to(best_device)

    torch.manual_seed(0)
    patch_embedding_pangu_pytorch = PatchEmbedding_pretrain(patch_size, dim).to(best_device)

    surface_data = torch.randn(batch_size, 4, 721, 1440, device=best_device)
    upper_air_data = torch.randn(batch_size, 5, 13, 721, 1440, device=best_device)

    embedded_input = patch_embedding(upper_air_data, surface_data)
    embedded_input_pangu_pytorch = torch.cat([patch_embedding_pangu_pytorch(
            upper_air_data[i:i+1], surface_data[i:i+1], [stat.to(best_device) for stat in random_weather_statistics],
            random_constant_maps.to(best_device), random_const_h.to(best_device))
        for i in range(batch_size)])

    # check output shapes
    assert embedded_input.shape == (batch_size, 521280, dim)
    assert embedded_input_pangu_pytorch.shape == (batch_size, 521280, dim)

    # check output content
    assert embedded_input.allclose(embedded_input_pangu_pytorch)
