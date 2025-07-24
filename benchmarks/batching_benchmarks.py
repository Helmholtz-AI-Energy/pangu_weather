import argparse
import time

import torch
import tqdm

import pangu_weather
import pangu_weather.layers
from benchmarks.timer import Timer


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


def benchmark_earth_specific_block_on_dummy_data(batch_size, iterations, device, inner=False, base_dim=192,
                                                 drop_path=0., roll=False, reproduce_mask=False, seed=0):
    print(f'Benchmark Earth Specific Block Inference on Dummy Data '
          f'with {inner=}, {base_dim=}, {drop_path=}, {roll=}, {reproduce_mask=}')
    if inner:  # inner (earth specific layers 1 and 2)
        layer_dim = 2 * base_dim
        num_heads = 12
        zhw = (8, 91, 180)
        input_shape = (batch_size, 131040, layer_dim)
    else:  # outer (earth specific layers 0 and 3)
        layer_dim = base_dim
        num_heads = 6
        zhw = (8, 181, 360)
        input_shape = (batch_size, 521280, layer_dim)

    earth_specific_block = pangu_weather.layers.EarthSpecificBlock(
        layer_dim, drop_path, roll, zhw, num_heads, reproduce_mask=reproduce_mask).to(device).eval()
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(input_shape, generator=generator).to(device)
    iteration_times = {}  # key: torch.zeros(iterations) for key in stages
    kwargs = {'x': x}

    def run_measurements(in_kwargs, fn, iteration_times, key):
        iteration_times[key] = {'cpu': torch.zeros(iterations), 'gpu': torch.zeros(iterations)}
        for i in tqdm.trange(iterations, desc=key, ncols=100):
            with Timer(print_on_exit=False) as t:
                with torch.no_grad():
                    out_kwargs = fn(**in_kwargs)
            iteration_times[key]['cpu'][i] = t.elapsed_time_cpu_s
            iteration_times[key]['gpu'][i] = t.elapsed_time_gpu_s
        return out_kwargs

    run_measurements(kwargs, lambda x, **_: earth_specific_block(x), iteration_times, 'earth_specific_block')

    # ---------------------------- Padding ----------------------------
    def padding(x, **_):
        # Example shapes for x = [B, 521280, C] with Z, H, W = 8, 181, 360 and window_size = [2, 6, 12]
        # Input shape [B, Z * H * W, C], save the shortcut for skip-connection
        shortcut = x

        # Reshape input to three dimensions: [B, Z * H * W, C] -> [B, Z, H, W, C] = [B, 8, 181, 360, C]
        x = x.view(x.shape[0], *earth_specific_block.zhw, x.shape[2])
        # Pad to window size: [B, 8, 181, 360, 192] -> [B, 8, 186, 360, 192]
        x = pangu_weather.layers.pad_to_shape(x, (1, *earth_specific_block.window_size, 1))
        original_shape = x.shape  # remember shape for later [B, 8, 186, 360, 192]

        return {'shortcut': shortcut, 'x': x, 'original_shape': original_shape}

    padding_results = run_measurements(kwargs, padding, iteration_times, 'padding')
    kwargs = {**kwargs, **padding_results}

    # ---------------------------- Compute Mask ----------------------------
    def compute_mask(x, **_):
        # 3D SwinTransformer: shift windows every other block (set via self.roll in __init__) to connect patches in
        # between different windows, in contrast to the original SwinTransformer, Pangu uses 3D windows
        if earth_specific_block.roll:
            # Shift by half a window in all 3 dimensions Z, H, W
            x = torch.roll(x, shifts=[-size // 2 for size in earth_specific_block.window_size], dims=(1, 2, 3))
            # mask out non-adjacent pixels
            mask = earth_specific_block.generate_attention_mask(x.shape[1:4], x.device)
        else:  # if not shifting, no mask needed
            mask = None
        return {'mask': mask, 'x': x}

    compute_mask_results = run_measurements(kwargs, compute_mask, iteration_times, 'compute_mask')
    kwargs = {**kwargs, **compute_mask_results}

    # ---------------------------- Window Partition ----------------------------
    def window_partition(x, **_):
        # Reshape to windows: [B, 8, 186, 360, 192] -> [B, 30, 124, 144, 192]
        x = earth_specific_block.window_partition(x)
        return {'x': x}

    window_partition_results = run_measurements(kwargs, window_partition, iteration_times, 'window_partition')
    kwargs = {**kwargs, **window_partition_results}

    # ---------------------------- 3D Window Attention ----------------------------
    def window_attention(x, mask, **_):
        # Apply 3D window attention with earth-specific bias
        x = earth_specific_block.attention(x, mask)
        return {'x': x}

    window_attention_results = run_measurements(kwargs, window_attention, iteration_times, '3d_window_attention')
    kwargs = {**kwargs, **window_attention_results}

    # ---------------------------- Window Reverse ----------------------------
    def window_reverse(x, original_shape, **_):
        # Revert back from windows: [B, 30, 124, 144, 192] -> [B, 8, 186, 360, 192]
        x = earth_specific_block.window_reverse(x, original_shape)

        # Revert shifted windows by shifting in the other direction
        if earth_specific_block.roll:
            x = torch.roll(x, shifts=[size // 2 for size in earth_specific_block.window_size], dims=(1, 2, 3))
        return {'x': x}

    window_reverse_results = run_measurements(kwargs, window_reverse, iteration_times, 'window_reverse')
    kwargs = {**kwargs, **window_reverse_results}

    # ---------------------------- Reverse Padding & Reshape ----------------------------
    def reverse_padding(x, shortcut, **_):
        # Crop to revert zero-padding [B, 8, 186, 360, 192] -> [B, 8, 181, 360, 192] = [B, Z, H, W, C]
        z, h, w = earth_specific_block.zhw
        x = x[:, :z, :h, :w, :]

        # Reshape back to input shape [B, Z, H, W, C] -> [B, Z * H * W, C]
        x = x.reshape(shortcut.shape)
        return {'x': x}

    reverse_padding_results = run_measurements(kwargs, reverse_padding, iteration_times, 'reverse_padding')
    kwargs = {**kwargs, **reverse_padding_results}

    # ---------------------------- Main calculation stages ----------------------------
    def main_calculation(x, shortcut, **_):
        # Main calculation stages
        x = shortcut + earth_specific_block.drop_path(earth_specific_block.norm1(x))
        x = x + earth_specific_block.drop_path(earth_specific_block.norm2(earth_specific_block.linear(x)))
        return {'x': x}

    run_measurements(kwargs, main_calculation, iteration_times, 'main_calculation')

    for key, results in iteration_times.items():
        label = f"{key}, {iterations=}, {batch_size=}, {device=}"
        summarize_results(results['cpu'], f"{label}, CPU Seconds", aggs=['mean'])
        if torch.cuda.is_available():
            summarize_results(results['gpu'], f"{label}, GPU Seconds", aggs=['mean'])


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


def summarize_results(iteration_times, label, aggs=None):
    results = {
        'mean': iteration_times.mean(),
        'min': iteration_times.min(),
        'max': iteration_times.max(),
        'median': iteration_times.median(),
        'std': iteration_times.std(),
    }
    print(f'Benchmark results for {label}:\n' + '\n'.join(
        f'{key:>20}: {value}' for key, value in results.items() if aggs is None or key in aggs))


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
        # benchmark_pangu_inference_on_dummy_data(batch_size, config.iterations, config.device)
        # benchmark_layers_on_dummy_data(batch_size, config.iterations, config.device)
        benchmark_earth_specific_block_on_dummy_data(batch_size, config.iterations, config.device, inner=False)
        benchmark_earth_specific_block_on_dummy_data(batch_size, config.iterations, config.device, inner=True)
        benchmark_earth_specific_block_on_dummy_data(batch_size, config.iterations, config.device, inner=True,
                                                     roll=True, drop_path=0.1)

