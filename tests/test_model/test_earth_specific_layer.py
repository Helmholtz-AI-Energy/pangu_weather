import itertools

import numpy as np
import pytest
import torch

from pangu_weather.layers import EarthSpecificLayer
import pangu_pytorch.models.layers as pangu_pytorch_layers
from tests.conftest import get_available_torch_devices


BATCH_SIZES = [1, 2, 4]
ZHW_DIM = [(8, 181, 360, 192), (8, 91, 180, 384)]
DEPTH = [1, 2]

parameters = "batch_size,zhw_dim,depth"
parameter_combinations = list(itertools.product(BATCH_SIZES, ZHW_DIM, DEPTH))
parameter_combinations[0] = pytest.param(*parameter_combinations[0], marks=pytest.mark.smoke)


@pytest.mark.parametrize(parameters, parameter_combinations)
@pytest.mark.parametrize("device", get_available_torch_devices())
def test_earth_specific_layer_shapes(batch_size, device, zhw_dim, depth):
    *zhw, dim = zhw_dim
    input_shape = (batch_size, int(np.prod(zhw)), dim)
    drop_path_ratios = [0.1 for _ in range(depth)]

    x = torch.zeros(input_shape, device=device)

    earth_specific_layer = EarthSpecificLayer(depth, dim, drop_path_ratios, 6, zhw).to(device)
    with torch.no_grad():
        output = earth_specific_layer(x)

    assert output.shape == input_shape


@pytest.mark.parametrize(parameters, parameter_combinations)
def test_earth_specific_layer_random_sample(batch_size, zhw_dim, depth, best_device):
    *zhw, dim = zhw_dim
    # Note: as for the earth attention, the results between our batched implementation and pangu-pytorch's sample-wise
    # implementation differ slightly for drop path > 0 and batch size > 1. This test thus only covers drop path = 0.
    drop_path_ratios = [0 for _ in range(depth)]
    input_shape = (batch_size, int(np.prod(zhw)), dim)

    torch.manual_seed(0)
    earth_specific_layer = EarthSpecificLayer(depth, dim, drop_path_ratios, 6, zhw, reproduce_mask=True).to(best_device)
    torch.manual_seed(0)
    # to use the same initialization of the earth-specific biases: create on cpu first, then move to device
    earth_specific_layer_pangu_pytorch = pangu_pytorch_layers.EarthSpecificLayer(
        depth, dim, drop_path_ratios, 6, False, 'cpu').to(best_device)
    earth_specific_layer_pangu_pytorch.apply(lambda module: setattr(module, 'device', best_device))

    x = torch.randn(input_shape, device=best_device)
    with torch.no_grad():
        torch.manual_seed(0)
        output = earth_specific_layer(x)
        torch.manual_seed(0)
        output_pangu_pytorch = torch.cat([earth_specific_layer_pangu_pytorch(xi.unsqueeze(0), *zhw) for xi in x])

    assert output.shape == input_shape
    assert output_pangu_pytorch.shape == input_shape
    # increase acceptable absolute error (atol) slightly for multi-sample batches
    assert output.allclose(output_pangu_pytorch, atol=2e-5 if batch_size > 1 else 1e-8)
