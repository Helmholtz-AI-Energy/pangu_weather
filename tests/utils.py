import itertools
import pathlib

import pytest
import torch

aux_data_path = pathlib.Path(__file__).parent / 'data' / 'aux_data'
pretrained_model_path_onnx = pathlib.Path(__file__).parent / 'data' / 'pangu_weather_24.onnx'
pretrained_model_path_torch = pathlib.Path(__file__).parent / 'data' / 'pangu_weather_24_torch.pth'


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
