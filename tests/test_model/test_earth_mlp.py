import itertools

import pytest
import torch

from pangu_weather.layers import MLP
import pangu_pytorch.models.layers as pangu_pytorch_layers
from tests.utils import get_available_torch_devices

BATCH_SIZES = [1, 2, 4]
DROPOUTS = [0.1, 0]

parameters = "batch_size,dropout"
parameter_combinations = list(itertools.product(BATCH_SIZES, DROPOUTS))
parameter_combinations[0] = pytest.param(*parameter_combinations[0], marks=pytest.mark.smoke)


@pytest.mark.parametrize(parameters, parameter_combinations)
@pytest.mark.parametrize("device", get_available_torch_devices())
def test_mlp_shapes(batch_size, dropout, device):
    dim = 192
    input_shape = (batch_size, 131040, dim)
    expected_output_shape = (batch_size, 131040, dim)

    mlp = MLP(dim, dropout).to(device)
    x = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        output = mlp(x)

    assert output.shape == expected_output_shape


@pytest.mark.parametrize(parameters, parameter_combinations)
def test_mlp_random_sample(batch_size, dropout, best_device):
    dim = 192
    input_shape = (batch_size, 131040, dim)
    expected_output_shape = (batch_size, 131040, dim)

    torch.manual_seed(0)
    mlp = MLP(dim, dropout).to(best_device)
    torch.manual_seed(0)
    mlp_pangu_pytorch = pangu_pytorch_layers.Mlp(dim, dropout).to(best_device)

    x = torch.randn(input_shape, device=best_device)

    with torch.no_grad():
        torch.manual_seed(0)
        output = mlp(x)
        torch.manual_seed(0)
        output_pangu_pytorch = mlp_pangu_pytorch(x)

    assert output.shape == expected_output_shape
    assert output_pangu_pytorch.shape == expected_output_shape
    assert output.allclose(output_pangu_pytorch)
