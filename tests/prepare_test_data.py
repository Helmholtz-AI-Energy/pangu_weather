import logging

import numpy
import onnx

from pangu_weather.pangu_weather import get_onnx_constant_tensor
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
            if not (saved_tensor.shape == aux_tensor.shape):
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


def convert_onnx_to_torch_checkpoint(onnx_path, torch_output_path, overwrite=False):
    if torch_output_path.exists():
        if not overwrite:
            logger.info(f'Output path {torch_output_path} already exists, aborting because {overwrite=}.')
            return
        logger.info(f'Output path {torch_output_path} already exists, will be overwritten.')

    # TODO: convert to torch checkpoint


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(name)s:%(lineno)d] - %(message)s")

    aux_data_path.mkdir(exist_ok=True, parents=True)
    extract_auxiliary_data_from_onnx(pretrained_model_path_onnx)
    convert_onnx_to_torch_checkpoint(pretrained_model_path_onnx, pretrained_model_path_torch)
