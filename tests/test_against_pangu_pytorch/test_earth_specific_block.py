import numpy as np
import pytest
import torch

from pangu_weather.layers import EarthSpecificBlock
import pangu_pytorch.models.layers as pangu_pytorch_layers
from tests.conftest import batch_size_device_product


@pytest.mark.parametrize("batch_size,device", batch_size_device_product())
@pytest.mark.parametrize("dim,zhw", [(192, (8, 181, 360)), (384, (8, 91, 180))])
@pytest.mark.parametrize("roll", [False, True])
@pytest.mark.parametrize("drop_path_ratio", [0, 0.1])
@pytest.mark.parametrize("reproduce_mask", [False, True])
def test_earth_specific_block_shapes(batch_size, device, dim, zhw, roll, drop_path_ratio, reproduce_mask):
    input_shape = (batch_size, int(np.prod(zhw)), dim)

    x = torch.zeros(input_shape, device=device)

    earth_specific_block = EarthSpecificBlock(dim, drop_path_ratio, roll, zhw, reproduce_mask=reproduce_mask).to(device)
    with torch.no_grad():
        output = earth_specific_block(x)

    assert output.shape == input_shape


@pytest.mark.parametrize("batch_size,drop_path_ratio", [(1, 0.1), (1, 0), (2, 0), (4, 0)])
@pytest.mark.parametrize("dim,zhw", [(192, (8, 181, 360)), (384, (8, 91, 180))])
@pytest.mark.parametrize("roll", [False, True])
def test_earth_specific_block_random_sample(batch_size, dim, zhw, roll, drop_path_ratio, best_device):
    # note: results for larger batch sizes differ slightly if drop path is enabled (since we cannot exactly reproduce
    # the same random state between the two implementations -> drop path is only tested with batch size 1
    input_shape = (batch_size, int(np.prod(zhw)), dim)

    torch.manual_seed(0)
    earth_specific_block = EarthSpecificBlock(dim, drop_path_ratio, roll, zhw, reproduce_mask=True).to(best_device)
    torch.manual_seed(0)
    # to use the same initialization of the earth-specific bias: create on cpu first, then move to device
    earth_specific_block_pangu_pytorch = pangu_pytorch_layers.EarthSpecificBlock(
        dim, drop_path_ratio, 6, 'cpu').to(best_device)
    earth_specific_block_pangu_pytorch.device = best_device

    x = torch.randn(input_shape, device=best_device)
    with torch.no_grad():
        torch.manual_seed(0)
        output = earth_specific_block(x)
        torch.manual_seed(0)
        output_pangu_pytorch = torch.cat([earth_specific_block_pangu_pytorch(xi.unsqueeze(0), *zhw, roll) for xi in x])

    assert output.shape == input_shape
    assert output_pangu_pytorch.shape == input_shape
    # increase acceptable absolute error (atol) slightly for multi-sample batches
    assert output.allclose(output_pangu_pytorch, atol=1e-5 if batch_size > 1 else 1e-8)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("shape", [(8, 186, 360, 192), (8, 96, 180, 384)])
@pytest.mark.parametrize("reproduce_mask", [True, False])
def test_earth_specific_block_mask(batch_size, shape, reproduce_mask):
    zhw = shape[:3]
    input_shape = (batch_size, *shape)
    dim = input_shape[-1]
    expected_mask_shape = (30, 124, 144, 144) if dim == 192 else (15, 64, 144, 144)

    drop_path_ratio = 0
    earth_specific_block = EarthSpecificBlock(dim, drop_path_ratio, True, zhw, reproduce_mask=reproduce_mask)
    earth_specific_block_pangu_pytorch = pangu_pytorch_layers.EarthSpecificBlock(dim, drop_path_ratio, 6, 'cpu')

    mask = earth_specific_block.generate_attention_mask()
    mask_pangu_pytorch = earth_specific_block_pangu_pytorch.gen_mask(torch.zeros(input_shape))

    assert mask.shape == expected_mask_shape
    assert mask_pangu_pytorch.shape == expected_mask_shape
    assert mask.allclose(mask_pangu_pytorch)
