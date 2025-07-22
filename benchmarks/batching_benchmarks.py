import argparse
import time

import torch
import tqdm

import pangu_weather
import pangu_weather.layers


def generate_random_auxiliary_data(seed=0):
    generator = torch.Generator().manual_seed(seed)

    surface_mean = torch.randn(4, generator=generator)
    surface_std = torch.randn(4, generator=generator)
    upper_mean = torch.randn((13, 1, 1, 5), generator=generator)
    upper_std = torch.randn((13, 1, 1, 5), generator=generator)
    weather_statistics = surface_mean, surface_std, upper_mean, upper_std

    constant_maps = torch.randn((1, 3, 724, 1440), generator=generator)
    const_h = torch.randn((1, 1, 1, 13, 721, 1440), generator=generator)
    return weather_statistics, constant_maps, const_h


def initialize_pangu_weather(seed=0):
    weather_statistics, constant_maps, const_h = generate_random_auxiliary_data(seed)
    return pangu_weather.PanguWeather(weather_statistics, constant_maps, const_h)


def benchmark_layers_on_dummy_data(batch_size, iterations, device, patch_size=(2, 4, 4), dimension=192):
    layer_kwargs = {'reproduce_mask': False, 'checkpoint': False}
    zeros = torch.zeros(6).tolist()

    layers_and_shapes = {
        'patch_embedding': (
            pangu_weather.layers.PatchEmbedding(patch_size, *generate_random_auxiliary_data(), dimension),
            (batch_size, 5, 13, 721, 1440),
            (batch_size, 4, 721, 1440),
        ),
        'earth_specific_layer0': (
            pangu_weather.layers.EarthSpecificLayer(2, dimension, zeros[:2], 6, (8, 181, 360), **layer_kwargs),
            (batch_size, 521280, dimension),
        ),
        'downsample': (pangu_weather.layers.DownSample(dimension), (batch_size, 521280, dimension)),
        'earth_specific_layer1': (
            pangu_weather.layers.EarthSpecificLayer(6, 2 * dimension, zeros, 12, (8, 91, 180), **layer_kwargs),
            (batch_size, 131040, 2 * dimension),
        ),
        'earth_specific_layer2': (
            pangu_weather.layers.EarthSpecificLayer(6, 2 * dimension, zeros, 12, (8, 91, 180), **layer_kwargs),
            (batch_size, 131040, 2 * dimension),
        ),
        'upsample': (pangu_weather.layers.UpSample(2 * dimension, dimension), (batch_size, 131040, 2 * dimension)),
        'earth_specific_layer3': (
            pangu_weather.layers.EarthSpecificLayer(2, dimension, zeros[:2], 6, (8, 181, 360), **layer_kwargs),
            (batch_size, 521280, dimension),
        ),
        'patch_recovery': (pangu_weather.layers.PatchRecovery(2 * dimension), (batch_size, 521280, 2 * dimension)),
    }

    for key, (layer, *input_shapes) in layers_and_shapes.items():
        name = f"{key}, {batch_size=}"
        benchmark_inference_on_dummy_data(layer, input_shapes, iterations, device, name)


def summarize_results(iteration_times, label):
    results = {
        'mean': iteration_times.mean(),
        'min': iteration_times.min(),
        'max': iteration_times.max(),
        'median': iteration_times.median(),
        'std': iteration_times.std(),
    }
    print(f'Benchmark results for {label}:\n' + '\n'.join(f'{key:>20}: {value}' for key, value in results.items()))


def benchmark_inference_on_dummy_data(model, input_shapes, iterations, device, name="", seed=0):
    model.eval().to(device)
    generator = torch.Generator().manual_seed(seed)
    x = [torch.rand(shape, generator=generator).to(device) for shape in input_shapes]

    iteration_times = torch.zeros(iterations)
    label = f'Inference on Dummy Data – {name} {iterations=}, {device=}'

    with torch.no_grad():
        for i in tqdm.trange(iterations, desc=label, ncols=len(label) + 80):
            start_time = time.perf_counter()
            model(*x)
            end_time = time.perf_counter()
            iteration_times[i] = end_time - start_time

    summarize_results(iteration_times, label)


def benchmark_pangu_inference_on_dummy_data(batch_size, iterations, device):
    model = initialize_pangu_weather()
    input_shapes = [(batch_size, 5, 13, 721, 1440), (batch_size, 4, 721, 1440)]
    benchmark_inference_on_dummy_data(model, input_shapes, iterations, device, name=f"PanguWeather with {batch_size=}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PanguWeather Benchmarks")
    parser.add_argument("--batch_sizes", "-b", type=int, nargs="+", default=[1])
    parser.add_argument("--iterations", "-i", type=int, default=10)
    parser.add_argument("--device", "-d", type=str, default='cpu')
    config = parser.parse_args()

    for batch_size in config.batch_sizes:
        benchmark_pangu_inference_on_dummy_data(batch_size, config.iterations, config.device)
        benchmark_layers_on_dummy_data(batch_size, config.iterations, config.device)

