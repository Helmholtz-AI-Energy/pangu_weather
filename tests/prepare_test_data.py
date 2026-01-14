import argparse
import collections
import logging

import gdown
import numpy
import onnx
import torch

from pangu_weather.pangu_weather import get_onnx_constant_tensor, PanguWeather
from utils import aux_data_path, pretrained_model_path_onnx, pretrained_model_path_torch, example_input_path

logger = logging.getLogger('pangu_weather.' + __name__)


def extract_auxiliary_data_from_onnx(onnx_path, overwrite=False):
    logger.info(f'Extracting auxiliary data from ONNX weights.')
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
    logger.info(f'Converting ONNX weights to torch checkpoint.')
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


def download_onnx_weights_and_example_input(overwrite=False):
    logger.info(f'Downloading ONNX weights and example input from official pangu-weather repository.')
    # url to the pre-trained weights of the 24h pangu weather model and the example inputs
    # from the official pangu weather repository (github.com/198808xc/Pangu-Weather)
    gdrive_urls = {
        pretrained_model_path_onnx:
            'https://drive.google.com/file/d/1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP/view?usp=share_link',
        example_input_path / "input_upper.npy":
            'https://drive.google.com/file/d/1--7xEBJt79E3oixizr8oFmK_haDE77SS/view?usp=share_link',
        example_input_path / "input_surface.npy":
            'https://drive.google.com/file/d/1pj8QEVNpC1FyJfUabDpV4oU3NpSe0BkD/view?usp=share_link',
    }
    example_input_path.mkdir(parents=True, exist_ok=True)

    for path, url in gdrive_urls.items():
        if overwrite or not path.exists():
            logger.info(f'{path.name} not found, from {url}.')
            gdown.download(url, str(path), fuzzy=True)
    else:
        logger.info(f'Done, all files already downloaded.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(name)s:%(lineno)d] - %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true',
                        help='Download missing data (onnx weights and example inputs)')
    parser.add_argument('--extract_aux_data', action='store_true',
                        help='Extract auxiliary data from ONNX weights.')
    parser.add_argument('--to_torch', action='store_true',
                        help='Convert ONNX weights to torch checkpoint.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files.')
    cli_args = parser.parse_args()

    if cli_args.download:
        download_onnx_weights_and_example_input(overwrite=cli_args.overwrite)

    if (cli_args.extract_aux_data or cli_args.to_torch) and not pretrained_model_path_onnx.exists():
        raise FileNotFoundError(f'ONNX model file not found at {pretrained_model_path_onnx}.')

    if cli_args.extract_aux_data:
        aux_data_path.mkdir(exist_ok=True, parents=True)
        extract_auxiliary_data_from_onnx(pretrained_model_path_onnx, overwrite=cli_args.overwrite)

    if cli_args.to_torch:
        convert_onnx_to_torch_checkpoint(pretrained_model_path_onnx, pretrained_model_path_torch,
                                         overwrite=cli_args.overwrite)
