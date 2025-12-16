import collections
import logging

import numpy
import onnx
import torch

from pangu_weather.pangu_weather import get_onnx_constant_tensor, PanguWeather
from tests.utils import aux_data_path, pretrained_model_path_onnx, pretrained_model_path_torch

logger = logging.getLogger('pangu_weather.' + __name__)


def extract_auxiliary_data_from_onnx(onnx_path, overwrite=False):
    logger.info(f'Loading ONNX model from {onnx_path}.')
    onnx_model = onnx.load(onnx_path)

    def save_aux_data(onnx_name, output_path):
        logger.info(f'Extracting {onnx_name} from ONNX model and saving to {output_path}.')
        aux_tensor = get_onnx_constant_tensor(onnx_model, onnx_name)
        if output_path.exists():
            message = f'{output_path} already exists'
            saved_tensor = numpy.load(output_path)
            if saved_tensor.shape != aux_tensor.shape:
                message += f', but has mismatching shape (old: {saved_tensor.shape}, new: {aux_tensor.shape}).'
            elif not numpy.allclose(saved_tensor, aux_tensor):
                message += f', but values are not allclose to new version.'
            else:
                message += f' and is identical to new version.'

            if not overwrite:
                logger.info(f'{message} Skipping because {overwrite=}.')
                return
            logger.info(f'{message} Overwriting.')
        numpy.save(output_path, aux_tensor.numpy())

    save_aux_data('/b1/Constant_17', aux_data_path / "Constant_17_output_0.npy")  # const_h
    save_aux_data('/b1/Constant_44', aux_data_path / "constantMaks3.npy")  # constant maps
    # weather statistics
    save_aux_data('/b1/Constant_11', aux_data_path / "surface_mean.npy")
    save_aux_data('/b1/Constant_12', aux_data_path / "surface_std.npy")
    save_aux_data('/b1/Constant_9', aux_data_path / "upper_mean.npy")
    save_aux_data('/b1/Constant_10', aux_data_path / "upper_std.npy")


def convert_onnx_to_torch_checkpoint(onnx_path, torch_output_path, overwrite=False, device='cpu'):
    dim = 192
    generator = torch.Generator().manual_seed(0)
    const_h = torch.randn((1, 1, 1, 13, 721, 1440), generator=generator)
    constant_maps = torch.randn((1, 3, 724, 1440), generator=generator)
    surface_mean = torch.randn(4, generator=generator)
    surface_std = torch.randn(4, generator=generator)
    upper_mean = torch.randn((13, 1, 1, 5), generator=generator)
    upper_std = torch.randn((13, 1, 1, 5), generator=generator)
    weather_statistics = surface_mean, surface_std, upper_mean, upper_std

    pangu_weather_onnx = PanguWeather(weather_statistics, constant_maps, const_h, dim)
    pangu_weather_onnx.load_pretrained_onnx_weights(onnx_path)

    pangu_pytorch_style_state_dict = collections.OrderedDict(
        {key.removeprefix('backbone.'): value for key, value in pangu_weather_onnx.state_dict().items()})
    checkpoint = {'model': pangu_pytorch_style_state_dict}

    if torch_output_path.exists():
        previous_checkpoint = torch.load(torch_output_path, map_location=device, weights_only=True)
        message = f'Output path {torch_output_path} already exists'
        if ((previous_checkpoint.keys() != checkpoint.keys()) or
                (previous_checkpoint['model'].keys() != checkpoint['model'].keys())):
            message += f', but has mismatching keys.'
        elif not all(torch.equal(checkpoint['model'][key], previous_checkpoint['model'][key])
                     for key in pangu_pytorch_style_state_dict):
            message += f', but values are not allclose to new version.'
        else:
            message += f' and is identical to new version.'

        if not overwrite:
            logger.info(f'{message} Aborting because {overwrite=}.')
            return
        logger.info(f'{message} Overwriting.')
    torch.save(checkpoint, torch_output_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(name)s:%(lineno)d] - %(message)s")

    aux_data_path.mkdir(exist_ok=True, parents=True)
    extract_auxiliary_data_from_onnx(pretrained_model_path_onnx)
    convert_onnx_to_torch_checkpoint(pretrained_model_path_onnx, pretrained_model_path_torch)
