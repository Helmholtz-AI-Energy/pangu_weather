import os

import pytest
import torch

import tests.utils


@pytest.fixture(scope="session", autouse=True)
def cuda_reproducibility():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
    try:
        return tests.utils.load_example_input()
    except FileNotFoundError as e:
        pytest.skip(f"Missing example input file: {e}")


@pytest.fixture
def weather_statistics():
    file_names = ["surface_mean.npy", "surface_std.npy", "upper_mean.npy", "upper_std.npy"]
    paths = [tests.utils.aux_data_path / file for file in file_names]
    if missing_paths := [path for path in paths if not path.exists()]:
        pytest.skip(f"Missing weather statistic files: {missing_paths}.")
    return [tests.utils.load_tensor_from_npy(path) for path in paths]


@pytest.fixture
def constant_maps():
    path = tests.utils.aux_data_path / "constantMaks3.npy"
    if not path.exists():
        pytest.skip(f"Missing constant map file: {path}.")
    return tests.utils.load_tensor_from_npy(path)


@pytest.fixture
def const_h():
    path = tests.utils.aux_data_path / "Constant_17_output_0.npy"
    if not path.exists():
        pytest.skip(f"Missing const_h file: {path}.")
    return tests.utils.load_tensor_from_npy(path)


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
    return tests.utils.get_best_device()


@pytest.fixture(scope="session")
def pretrained_model_path_onnx():
    path = tests.utils.pretrained_model_path_onnx
    if not path.exists():
        pytest.skip(f"Missing ONNX model file: {path}.")
    return path


@pytest.fixture(scope="session")
def pretrained_model_path_torch():
    path = tests.utils.pretrained_model_path_torch
    if not path.exists():
        pytest.skip(f"Missing PyTorch model file: {path}.")
    return path


@pytest.fixture(scope="session")
def pretrained_onnx_model(pretrained_model_path_onnx):
    ort_session_24 = tests.utils.setup_onnxruntime_session(pretrained_model_path_onnx)
    return tests.utils.onnx_inference_model(ort_session_24)


@pytest.fixture(scope="session")
def onnx_output_for_example_input(pretrained_model_path_onnx):
    ort_session_24 = tests.utils.setup_onnxruntime_session(pretrained_model_path_onnx)
    onnx_model = tests.utils.onnx_inference_model(ort_session_24)
    return onnx_model(*tests.utils.load_example_input())
