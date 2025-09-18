import itertools
import os
import pathlib

import numpy as np
import onnxruntime as ort
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


pretrained_onnx_model_path = pathlib.Path(__file__).parent / 'test_data' / 'pangu_weather_24.onnx'


@pytest.fixture(scope='session')
def pretrained_onnx_model():
    # Setup ONNX runtime as in official repository
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1
    cuda_provider = ('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'})
    cpu_provider = 'CPUExecutionProvider'
    providers = [cpu_provider]

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session_24 = ort.InferenceSession(pretrained_onnx_model_path, sess_options=options, providers=providers)

    def inference_with_onnx_model(input_upper, input_surface):
        # convert input to float32 numpy arrays, not clear if necessary
        input_upper = np.asarray(input_upper, dtype=np.float32)
        input_surface = np.asarray(input_surface, dtype=np.float32)

        # inference via ONNX runtime
        output_upper, output_surface = ort_session_24.run(None, {'input': input_upper, 'input_surface': input_surface})
        return torch.tensor(output_upper), torch.tensor(output_surface)

    return inference_with_onnx_model
