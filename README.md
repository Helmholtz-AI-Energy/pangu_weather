# A PyTorch Implementation of Pangu-Weather with Multi-Sample Batch Support
[![Python](https://img.shields.io/badge/Python-3.9+-69E2BC)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-69E2BC)](https://github.com/Helmholtz-AI-Energy/pangu_weather/blob/main/LICENSE)
[![Ruff](https://img.shields.io/badge/Code_Style-Ruff-69E2BC)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/Helmholtz-AI-Energy/pangu_weather/graph/badge.svg?token=H6KACMG2GX)](https://codecov.io/gh/Helmholtz-AI-Energy/pangu_weather)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Helmholtz-AI-Energy/pangu_weather/main.svg)](https://results.pre-commit.ci/latest/github/Helmholtz-AI-Energy/pangu_weather/main)

This repository implements the Pangu-Weather model [1] based on the official pseudocode [2] in `pytorch` with support for batch sizes > 1.

## Usage
### Installation
Clone the repository and install:
```bash
git clone <url>
cd pangu-weather
pip install .
```

### Using the Pangu-Weather Model

To use the full Pangu-Weather model:

```python
from pangu_weather import PanguWeather

# create Pangu-Weather model with auxiliary data weather_statistics, constant_maps, and const_h
pangu_weather_model = PanguWeather(weather_statistics, constant_maps, const_h)
# load pretrained weights from path to the given torch device
pangu_weather_model.load_pretrained_weights(path, device)
```

We also include a backbone only model, i.e., without the `PatchRecovery` output layer:
```python
from pangu_weather import PanguWeatherBackbone

# create Pangu-Weather backbone model with auxiliary data weather_statistics, constant_maps, and const_h
pangu_weather_backbone = PanguWeatherBackbone(weather_statistics, constant_maps, const_h)
# load pretrained weights from path to the given torch device
pangu_weather_backbone.load_pretrained_weights(path, device)
```

Additionally, the individual layers can be found in `pangu_weather.layers`.

### Running the tests

The tests use `pytest` and compare the layer-wise outputs to pangu-pytorch.
Clone the repository with submodules to clone the `pangu-pytorch` [3] submodule to test against and install with dev requirements.
```bash
git clone --recurse-submodules <url>
cd pangu-weather
pip install '.[dev]'
```
Run the tests
```bash
python -m pytest tests
```

#### Selecting test subsets
We provide pytest markers to select subsets of the tests:
- smoke: fast tests for the basic functionality, select with `-m smoke`
- slow: slow tests, deselect with `-m "not slow"`

#### Downloading additional test data
In addition to tests on random data and weights, we can also test using actual inputs and pre-trained weights.
The run these tests, you first need to download and prepare the necessary data by running
```
python tests/prepare_test_data.py
```
This downloads the example input and pre-trained weights from links provided in the README to the official repository https://github.com/198808xc/Pangu-Weather and extract the auxiliary data and torch checkpoint from the ONNX weights.
The final structure should look like this:
```
├── pangu-weather
│   ├── tests
│   │   ├── data
│   │   │   ├── aux_data
│   │   │   │   ├── surface_mean.npy
│   │   │   │   ├── surface_std.npy
│   │   │   │   ├── upper_mean.npy
│   │   │   │   ├── upper_std.npy
│   │   │   │   ├── constantMaks3.npy
│   │   │   │   ├── Constant_17_output_0.npy
│   │   │   ├── example_input
│   │   │   │   ├── input_surface.npy
│   │   │   │   ├── input_upper.npy
│   │   │   ├── pangu_weather_24.onnx
│   │   │   ├── pangu_weather_24_torch.pth
```

If the necessary files are not available, the corresponding tests are skipped automatically.

## References
[1] Bi, K., Xie, L., Zhang, H. et al. Accurate medium-range global weather forecasting with 3D neural networks. Nature 619, 533–538 (2023). https://doi.org/10.1038/s41586-023-06185-3
[2] https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py
[3] https://github.com/zhaoshan2/pangu-pytorch
