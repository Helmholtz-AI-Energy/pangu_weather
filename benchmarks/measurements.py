import enum
import logging

import torch


logger = logging.getLogger(__name__)


class AverageMeter:
    STR_FORMAT = '{0.name}: {0.average:{0.value_fmt}} (last {0.last_value:{0.value_fmt}})'

    def __init__(self, name, str_fmt=STR_FORMAT, value_fmt=".4g", device=None):
        self.name = name
        self.str_fmt = str_fmt
        self.value_fmt = value_fmt

        # set or infer device
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.last_value = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self):
        self.last_value = 0.0
        self.sum = 0.0
        self.count = 0

    def all_reduce(self):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return self   # no need to allreduce if not distributed
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        torch.distributed.all_reduce(total, torch.distributed.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        return self

    @property
    def average(self):
        return self.sum / self.count if self.count > 0 else self.last_value

    def update(self, value: float, n: int = 1) -> None:
        self.last_value = value
        self.sum += value * n
        self.count += n

    def __str__(self) -> str:
        return self.str_fmt.format(self)


class ContextTracker:
    DEFAULT_FORMAT = "{0.name}"

    def __init__(self, name="", print_on_exit=True, output_format=DEFAULT_FORMAT, print_fn=print, disable=False):
        self.output_format = output_format
        self.print_on_exit = print_on_exit
        self.name = name
        self.print_fn = print_fn
        self.disable = disable

    def reset(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def __str__(self):
        return self.output_format.format(self)

    def print(self):
        self.print_fn(str(self))

    def __enter__(self):
        if not self.disable:
            self.start()
        return self

    def __exit__(self, *args):
        if not self.disable:
            self.stop()
            if self.print_on_exit:
                self.print()