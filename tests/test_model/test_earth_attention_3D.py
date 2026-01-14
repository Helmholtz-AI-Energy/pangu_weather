import itertools
import warnings

import pytest
import torch

from pangu_weather.layers import EarthAttention3D
from tests.test_model.pangu_pytorch_model import pangu_pytorch_layers
from tests.utils import get_available_torch_devices

BATCH_SIZES = [1, 2, 4]
INPUT_SHAPES = [(30, 124, 144, 192), (15, 64, 144, 384)]
MASKED = [True, False]

parameters = "batch_size,input_shape,masked"
parameter_combinations = list(itertools.product(BATCH_SIZES, INPUT_SHAPES, MASKED))
parameter_combinations[0] = pytest.param(*parameter_combinations[0], marks=pytest.mark.smoke)


@pytest.mark.parametrize(parameters, parameter_combinations)
@pytest.mark.parametrize("device", get_available_torch_devices())
def test_earth_attention_shapes(batch_size, input_shape, masked, device):
    expected_output_shape = (batch_size, *input_shape)

    x = torch.zeros(batch_size, *input_shape, device=device)
    mask = torch.zeros(input_shape[:2] + (144, 144), device=device) if masked else None

    type_of_windows = 124 if input_shape[-1] == 192 else 64

    earth_attention = EarthAttention3D(input_shape[-1], 6, 0, (2, 6, 12), type_of_windows).to(device)
    with torch.no_grad():
        output = earth_attention(x, mask)

    assert output.shape == expected_output_shape


@pytest.mark.parametrize(parameters, parameter_combinations)
def test_earth_attention_random_sample(batch_size, input_shape, masked, best_device):
    dim = input_shape[-1]
    expected_output_shape = (batch_size, *input_shape)

    torch.manual_seed(0)
    type_of_windows = 124 if dim == 192 else 64
    earth_attention = EarthAttention3D(dim, 6, 0, (2, 6, 12), type_of_windows).to(best_device)
    torch.manual_seed(0)
    # to use the same initialization of the earth-specific bias: create on cpu first, then move to device
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        earth_attention_pangu_pytorch = pangu_pytorch_layers.EarthAttention3D(dim, 6, 0, (2, 6, 12), device="cpu").to(
            best_device
        )

    x = torch.randn(batch_size, *input_shape, device=best_device)
    mask = (torch.rand(input_shape[:2] + (144, 144), device=best_device) > 0.5) * -100 if masked else None

    with torch.no_grad():
        torch.manual_seed(0)
        output = earth_attention(x, mask)
        torch.manual_seed(0)
        output_pangu_pytorch = torch.stack([earth_attention_pangu_pytorch(xi, mask) for xi in x])

    assert output.shape == expected_output_shape
    assert output_pangu_pytorch.shape == expected_output_shape
    assert output.allclose(output_pangu_pytorch)
