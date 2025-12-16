import os

import numpy as np
import pytest
import torch

from tests.utils import *


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
def example_input():
    return load_example_input()


@pytest.fixture
def weather_statistics():
    files = ["surface_mean.npy", "surface_std.npy", "upper_mean.npy", "upper_std.npy"]
    return [load_tensor_from_npy(aux_data_path / file) for file in files]


@pytest.fixture
def constant_maps():
    return load_tensor_from_npy(aux_data_path / "constantMaks3.npy")


@pytest.fixture
def const_h():
    return load_tensor_from_npy(aux_data_path / "Constant_17_output_0.npy")


@pytest.fixture
def random_constant_maps():
    generator = torch.Generator().manual_seed(0)
    return torch.randn((1, 3, 724, 1440), generator=generator)


@pytest.fixture
def random_const_h():
    generator = torch.Generator().manual_seed(0)
    return torch.randn((1, 1, 1, 13, 721, 1440), generator=generator)


@pytest.fixture
def best_device():
    return get_best_device()


@pytest.fixture(scope='session')
def pretrained_onnx_model():
    ort_session_24 = setup_onnxruntime_session(pretrained_model_path_onnx)
    return onnx_inference_model(ort_session_24)


@pytest.fixture(scope='session')
def onnx_output_for_example_input():
    ort_session_24 = setup_onnxruntime_session(pretrained_model_path_onnx)
    onnx_model = onnx_inference_model(ort_session_24)
    return onnx_model(*load_example_input())
