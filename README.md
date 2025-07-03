# A PyTorch Implementation of Pangu-Weather with Multi-Sample Batch Support

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
Clone the repository with submodules to clone the `pangu-pytorch` submodule to test against and install with dev requirements.
```bash
git clone --recurse-submodules <url>
cd pangu-weather
pip install '.[dev]'
```
Run the tests
```bash
python -m pytest tests
```

## References
[1] Bi, K., Xie, L., Zhang, H. et al. Accurate medium-range global weather forecasting with 3D neural networks. Nature 619, 533–538 (2023). https://doi.org/10.1038/s41586-023-06185-3  
[2] https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py  
[3] https://github.com/zhaoshan2/pangu-pytorch  