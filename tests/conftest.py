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


aux_data = pathlib.Path(__file__).parent / 'data' / 'aux_data'


@pytest.fixture
def weather_statistics():
    files = ["surface_mean.npy", "surface_std.npy", "upper_mean.npy", "upper_std.npy"]
    return [torch.from_numpy(np.load(aux_data / file)).to(torch.float32) for file in files]


@pytest.fixture
def constant_maps():
    return torch.from_numpy(np.load(aux_data / "constantMaks3.npy")).to(torch.float32)


@pytest.fixture
def const_h():
    return torch.from_numpy(np.load(aux_data / "Constant_17_output_0.npy")).to(torch.float32)


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


def batch_size_device_product(all_device_batch_sizes=(1,), best_device_batch_sizes=(2, 4), smoke_test_batch_sizes=(1,)):
    smoke_tests = []
    if smoke_test_batch_sizes:
        smoke_tests = [pytest.param(batch_size, get_best_device(), marks=pytest.mark.smoke)
                       for batch_size in smoke_test_batch_sizes]
    all_device_pairs = itertools.product(all_device_batch_sizes, get_available_torch_devices())
    all_device_pairs = [(batch_size, device) for batch_size, device in all_device_pairs
                        if batch_size not in smoke_test_batch_sizes and device != get_best_device()]
    best_device_pairs = [(batch_size, get_best_device()) for batch_size in best_device_batch_sizes]
    return smoke_tests + all_device_pairs + best_device_pairs


pretrained_model_path_onnx = pathlib.Path(__file__).parent / 'data' / 'pangu_weather_24.onnx'
pretrained_model_path_torch = pathlib.Path(__file__).parent / 'data' / 'pangu_weather_24_torch.pth'


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
    ort_session_24 = ort.InferenceSession(pretrained_model_path_onnx, sess_options=options, providers=providers)

    def inference_with_onnx_model(input_upper, input_surface):
        # convert input to float32 numpy arrays, not clear if necessary
        input_upper = np.asarray(input_upper, dtype=np.float32)
        input_surface = np.asarray(input_surface, dtype=np.float32)

        # inference via ONNX runtime
        output_upper, output_surface = ort_session_24.run(None, {'input': input_upper, 'input_surface': input_surface})
        return torch.tensor(output_upper), torch.tensor(output_surface)

    return inference_with_onnx_model
