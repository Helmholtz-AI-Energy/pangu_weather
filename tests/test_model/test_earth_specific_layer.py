import numpy as np
import pytest
import torch

from pangu_weather.layers import EarthSpecificLayer
import pangu_pytorch.models.layers as pangu_pytorch_layers
from tests.conftest import batch_size_device_product


@pytest.mark.parametrize("batch_size,device", batch_size_device_product())
@pytest.mark.parametrize("dim,zhw", [(192, (8, 181, 360)), (384, (8, 91, 180))])
@pytest.mark.parametrize("depth", [1, 2])
def test_earth_specific_layer_shapes(batch_size, device, dim, zhw, depth):
    input_shape = (batch_size, int(np.prod(zhw)), dim)
    drop_path_ratios = [0.1 for _ in range(depth)]

    x = torch.zeros(input_shape, device=device)

    earth_specific_layer = EarthSpecificLayer(depth, dim, drop_path_ratios, 6, zhw).to(device)
    with torch.no_grad():
        output = earth_specific_layer(x)

    assert output.shape == input_shape


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("dim,zhw", [(192, (8, 181, 360)), (384, (8, 91, 180))])
@pytest.mark.parametrize("depth", [1, 2])
def test_earth_specific_layer_random_sample(batch_size, dim, zhw, depth, best_device):
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
