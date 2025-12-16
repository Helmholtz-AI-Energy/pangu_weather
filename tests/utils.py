import itertools
import logging
import pathlib
import time

import numpy as np
import onnxruntime as ort
import pytest
import torch

__all__ = [
    "aux_data_path",
    "example_input_path",
    "pretrained_model_path_onnx",
    "pretrained_model_path_torch",
    "get_available_torch_devices",
    "get_best_device",
    "batch_size_device_product",
    "batched_repeat",
    "random_input",
    "load_tensor_from_npy",
    "load_example_input",
    "setup_onnxruntime_session",
    "onnx_inference_model",
]

logger = logging.getLogger('pangu_weather.' + __name__)

TEST_DATA_DIR = pathlib.Path(__file__).parent / 'data'

aux_data_path = TEST_DATA_DIR / 'aux_data'
example_input_path = TEST_DATA_DIR / 'example_input'
pretrained_model_path_onnx = TEST_DATA_DIR / 'pangu_weather_24.onnx'
pretrained_model_path_torch = TEST_DATA_DIR / 'pangu_weather_24_torch.pth'


def get_available_torch_devices():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


def get_best_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


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


def batched_repeat(tensor, batch_size):
    target_shape = [batch_size] + [1] * (len(tensor.shape) - 1)  # [batch_size, 1, ..., 1]
    return tensor.repeat(target_shape)


def random_input(device, batch_size=1, seed=0, generator=None):
    if generator is None and seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    upper_air_data = torch.randn(batch_size, 5, 13, 721, 1440, device=device, generator=generator)
    surface_data = torch.randn(batch_size, 4, 721, 1440, device=device, generator=generator)
    return upper_air_data, surface_data


def load_tensor_from_npy(path):
    return torch.from_numpy(np.load(path)).to(torch.float32)


def load_example_input():
    upper_air_data = load_tensor_from_npy(example_input_path / "input_upper.npy").unsqueeze(0)
    surface_data = load_tensor_from_npy(example_input_path / "input_surface.npy").unsqueeze(0)
    return upper_air_data, surface_data


def setup_onnxruntime_session(model_path):
    logger.info('Setup ONNX runtime session.')
    # Setup ONNX runtime as in official repository
    options = ort.SessionOptions()
    # options.enable_cpu_mem_arena = False
    # options.enable_mem_pattern = False
    # options.enable_mem_reuse = False
    # options.intra_op_num_threads = 1
    cuda_provider = ('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'})
    cpu_provider = 'CPUExecutionProvider'
    providers = [cpu_provider]

    # Initialize onnxruntime session for Pangu-Weather Models
    return ort.InferenceSession(model_path, sess_options=options, providers=providers)


def onnx_inference_model(onnxruntime_session):
    def inference_with_onnx_model(input_upper, input_surface):
        start = time.time()
        # inference via ONNX runtime
        output_upper, output_surface = onnxruntime_session.run(
            None, {'input': input_upper, 'input_surface': input_surface})
        logger.info(f'ONNX forward pass took {time.time() - start:.3f}s.')
        return torch.tensor(output_upper), torch.tensor(output_surface)

    def batched_inference_with_onnx_model(batched_input_upper, batched_input_surface):
        start = time.time()
        logger.info('Inference with ONNX runtime.')
        # convert input to float32 numpy arrays, not clear if necessary
        batched_input_upper = np.asarray(batched_input_upper, dtype=np.float32)
        batched_input_surface = np.asarray(batched_input_surface, dtype=np.float32)
        # run one forward pass per sample
        outputs = [inference_with_onnx_model(batched_input_upper[sample], batched_input_surface[sample])
                   for sample in range(batched_input_upper.shape[0])]
        # combine outputs back to batched tensor
        batched_output_upper, batched_output_surface = [torch.stack(outputs) for outputs in zip(*outputs)]
        logger.info(f'Total ONNX inference took {time.time() - start:.3f}s.')
        return batched_output_upper, batched_output_surface

    return batched_inference_with_onnx_model
