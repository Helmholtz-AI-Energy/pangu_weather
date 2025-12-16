import pytest
import torch

from pangu_weather.layers import DownSample, UpSample
from tests.test_model.pangu_pytorch_model import pangu_pytorch_layers
from tests.utils import get_available_torch_devices


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.smoke), 2, 4])
@pytest.mark.parametrize("device", get_available_torch_devices())
def test_downsample_shapes(batch_size, device):
    dim = 192
    input_shape = (batch_size, 8 * 360 * 181, dim)
    expected_output_shape = (batch_size, 8 * 180 * 91, 2 * dim)

    downsample = DownSample().to(device)
    x = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        output = downsample(x)

    assert output.shape == expected_output_shape


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.smoke), 2, 4])
@pytest.mark.parametrize("device", get_available_torch_devices())
def test_upsample_shapes(batch_size, device):
    dim = 192
    input_shape = (batch_size, 8 * 180 * 91, 2 * dim)
    expected_output_shape = (batch_size, 8 * 360 * 181, dim)

    upsample = UpSample().to(device)
    x = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        output = upsample(x)

    assert output.shape == expected_output_shape


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.smoke), 2, 4])
def test_downsample_random_sample(batch_size, best_device):
    dim = 192
    input_shape = (batch_size, 8 * 360 * 181, dim)
    expected_output_shape = (batch_size, 8 * 180 * 91, 2 * dim)

    torch.manual_seed(0)
    downsample = DownSample().to(best_device)
    torch.manual_seed(0)
    downsample_pangu_pytorch = pangu_pytorch_layers.DownSample(dim).to(best_device)

    x = torch.randn(input_shape, device=best_device)

    with torch.no_grad():
        output = downsample(x)
        output_pangu_pytorch = downsample_pangu_pytorch(x, 8, 181, 360)

    assert output.shape == expected_output_shape
    assert output_pangu_pytorch.shape == expected_output_shape
    assert output.allclose(output_pangu_pytorch)


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.smoke), 2, 4])
def test_upsample_random_sample(batch_size, best_device):
    dim = 192
    input_shape = (batch_size, 8 * 180 * 91, 2 * dim)
    expected_output_shape = (batch_size, 8 * 360 * 181, dim)

    torch.manual_seed(0)
    upsample = UpSample().to(best_device)
    torch.manual_seed(0)
    upsample_pangu_pytorch = pangu_pytorch_layers.UpSample(2 * dim, dim).to(best_device)

    x = torch.randn(input_shape, device=best_device)

    with torch.no_grad():
        output = upsample(x)
        output_pangu_pytorch = upsample_pangu_pytorch(x)

    assert output.shape == expected_output_shape
    assert output_pangu_pytorch.shape == expected_output_shape
    assert output.allclose(output_pangu_pytorch)
