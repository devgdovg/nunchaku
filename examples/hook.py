import json
import typing

import torch
import torch.nn as nn


class TraceHook:
    def __init__(self, module_name, output_file):
        self.module_name = module_name
        self.output_file = output_file

    def __call__(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, typing.Any],
        output: tuple[torch.Tensor, ...],
    ):

        def _get_stats(tensor: torch.Tensor):
            return {
                "shape": tuple(tensor.shape),
                "max": torch.max(tensor).item(),
                "min": torch.min(tensor).item(),
                "mean": torch.mean(tensor).item(),
                "std": torch.std(tensor).item(),
            }

        input_stats = []
        for input in input_args:
            input_stats.append(_get_stats(input))
        output_stats = []
        for out in output:
            output_stats.append(_get_stats(out))
        info = {
            "module_name": self.module_name,
            "type": str(type(module)),
            "input_stats": input_stats,
            "output_stats": output_stats,
        }
        self.output_file.write(json.dumps(info, indent=4))
        self.output_file.flush()


def add_hook(model: nn.Module, file, cls):
    for name, module in model.named_modules():
        if isinstance(module, cls):
            print(f"adding hook for {name},  output to: {file}")
            module.register_forward_hook(TraceHook(name, file), with_kwargs=True)
