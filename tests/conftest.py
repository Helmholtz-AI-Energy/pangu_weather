import itertools
import os

import pytest
import torch


@pytest.fixture(scope='session', autouse=True)
def cuda_reproducibility():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


@pytest.fixture
def random_weather_statistics():
    generator = torch.Generator().manual_seed(0)
    surface_mean = torch.randn(4, generator=generator)
    surface_std = torch.randn(4, generator=generator)
    upper_mean = torch.randn((13, 1, 1, 5), generator=generator)
    upper_std = torch.randn((13, 1, 1, 5), generator=generator)
    return surface_mean, surface_std, upper_mean, upper_std


@pytest.fixture
def random_constant_maps():
    generator = torch.Generator().manual_seed(0)
    return torch.randn((1, 3, 724, 1440), generator=generator)


@pytest.fixture
def random_const_h():
    generator = torch.Generator().manual_seed(0)
    return torch.randn((1, 1, 1, 13, 721, 1440), generator=generator)


def get_available_torch_devices():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


def get_best_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def best_device():
    return get_best_device()


def batch_size_device_product(all_device_batch_sizes=(1,), best_device_batch_sizes=(2, 4)):
    all_device_pairs = list(itertools.product(all_device_batch_sizes, get_available_torch_devices()))
    best_device_pairs = [(batch_size, get_best_device()) for batch_size in best_device_batch_sizes]
    return all_device_pairs + best_device_pairs
