import time

import torch

from benchmarks.measurements import ContextTracker, AverageMeter


class GPUEvent:
    """Wrapper to torch.cuda.Event to avoid error when cuda is not available (all ops are noops in that case)"""
    def __init__(self, *args, **kwargs):
        self.event = torch.cuda.Event(*args, **kwargs) if torch.cuda.is_available() else None

    def record(self, *args, **kwargs):
        return self.event.record(*args, **kwargs) if torch.cuda.is_available() else None

    def elapsed_time(self, end_event):
        return self.event.elapsed_time(end_event.event) if torch.cuda.is_available() else 0.0


class Timer(ContextTracker):
    DEFAULT_FORMAT = "Elapsed time {0.name}: CPU {0.elapsed_time_cpu_s:.3f}s, GPU {0.elapsed_time_gpu_s:.3f}s"

    def __init__(self, name="", print_on_exit=True, output_format=DEFAULT_FORMAT, print_fn=print, disable=False):
        super().__init__(name, print_on_exit, output_format, print_fn, disable)
        self.start_time = 0.0
        self.end_time = 0.0

        self.start_event = GPUEvent(enable_timing=True)
        self.end_event = GPUEvent(enable_timing=True)

        self._elapsed_time_cpu = AverageMeter(f'{self.name}_cpu_s')  # in seconds
        self._elapsed_time_gpu = AverageMeter(f'{self.name}_gpu_ms')  # in milliseconds

    @property
    def elapsed_time_cpu_s(self):
        return self._elapsed_time_cpu.average

    @property
    def elapsed_time_gpu_s(self):
        return self._elapsed_time_gpu.average / 1000  # ms to s

    def reset(self):
        self._elapsed_time_cpu.reset()
        self._elapsed_time_gpu.reset()

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # probably not necessary, at least not for gpu measurements
        self.start_time = time.perf_counter()
        self.start_event.record()

    def stop(self):
        self.end_time = time.perf_counter()
        self.end_event.record()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._elapsed_time_cpu.update(self.end_time - self.start_time)
        self._elapsed_time_gpu.update(self.start_event.elapsed_time(self.end_event))
