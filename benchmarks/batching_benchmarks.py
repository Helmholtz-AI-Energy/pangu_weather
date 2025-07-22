import argparse
import time

import torch
import tqdm

from pangu_weather import PanguWeather


def initialize_pangu_weather(seed=0):
    generator = torch.Generator().manual_seed(seed)

    surface_mean = torch.randn(4, generator=generator)
    surface_std = torch.randn(4, generator=generator)
    upper_mean = torch.randn((13, 1, 1, 5), generator=generator)
    upper_std = torch.randn((13, 1, 1, 5), generator=generator)
    weather_statistics = surface_mean, surface_std, upper_mean, upper_std

    constant_maps = torch.randn((1, 3, 724, 1440), generator=generator)
    const_h = torch.randn((1, 1, 1, 13, 721, 1440), generator=generator)
    return PanguWeather(weather_statistics, constant_maps, const_h)


def generate_dummy_input_data(batch_size, seed=0):
    generator = torch.Generator().manual_seed(seed)
    upper_air_input = torch.rand((batch_size, 5, 13, 721, 1440), generator=generator)
    surface_input = torch.rand((batch_size, 4, 721, 1440), generator=generator)
    return upper_air_input, surface_input


def benchmark_inference_on_dummy_data(pangu_weather_model, batch_size, iterations, device):
    pangu_weather_model.eval().to(device)
    upper_air_input, surface_input = generate_dummy_input_data(batch_size)
    upper_air_input, surface_input = upper_air_input.to(device), surface_input.to(device)

    iteration_times = torch.zeros(iterations)
    label = f'Inference on Dummy Data with {batch_size=}, {iterations=}, {device=}'

    with torch.no_grad():
        for i in tqdm.trange(iterations, desc=label, ncols=len(label) + 40):
            start_time = time.perf_counter()
            pangu_weather_model(upper_air_input, surface_input)
            end_time = time.perf_counter()
            iteration_times[i] = end_time - start_time

    results = {
        'mean': iteration_times.mean(),
        'min': iteration_times.min(),
        'max': iteration_times.max(),
        'median': iteration_times.median(),
        'std': iteration_times.std(),
    }
    print(f'Benchmark results for {label}:\n' + '\n'.join(f'{key:>20}: {value}' for key, value in results.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PanguWeather Benchmarks")
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--iterations", "-i", type=int, default=10)
    parser.add_argument("--device", "-d", type=str, default='cpu')
    config = parser.parse_args()
    benchmark_inference_on_dummy_data(initialize_pangu_weather(), config.batch_size, config.iterations, config.device)

