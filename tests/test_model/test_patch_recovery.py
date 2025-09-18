import pytest
import torch

from pangu_weather.layers import PatchRecovery
from pangu_pytorch.models.layers import PatchRecovery_pretrain
from tests.conftest import get_available_torch_devices


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.smoke), 2, 4])
@pytest.mark.parametrize("device", get_available_torch_devices())
def test_patch_embedding_shapes(batch_size, device):
    patch_recovery = PatchRecovery().to(device)
    patches = torch.zeros(batch_size, 521280, 384, device=device)

    upper_air_data, surface_data = patch_recovery(patches)
    assert upper_air_data.shape == (batch_size, 5, 13, 721, 1440)
    assert surface_data.shape == (batch_size, 4, 721, 1440)


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.smoke), 2, 4])
def test_patch_embedding_random_sample(batch_size, random_weather_statistics, random_constant_maps, random_const_h,
                                       best_device):
    torch.manual_seed(0)
    patch_recovery = PatchRecovery().to(best_device)

    torch.manual_seed(0)
    patch_recovery_pangu_pytorch = PatchRecovery_pretrain(dim=384).to(best_device)

    patches = torch.zeros(batch_size, 521280, 384, device=best_device)

    upper_air_data, surface_data = patch_recovery(patches)
    pangu_pytorch_outputs = [
        patch_recovery_pangu_pytorch(patches[i:i+1], *patch_recovery.zhw) for i in range(batch_size)]
    upper_air_data_pangu_pytorch, surface_data_pangu_pytorch = [
        torch.cat(outputs) for outputs in zip(*pangu_pytorch_outputs)]

    # check output shapes
    assert upper_air_data.shape == (batch_size, 5, 13, 721, 1440)
    assert upper_air_data_pangu_pytorch.shape == (batch_size, 5, 13, 721, 1440)
    assert surface_data.shape == (batch_size, 4, 721, 1440)
    assert surface_data_pangu_pytorch.shape == (batch_size, 4, 721, 1440)

    # check output content
    assert upper_air_data.allclose(upper_air_data_pangu_pytorch)
    assert surface_data.allclose(surface_data_pangu_pytorch)
