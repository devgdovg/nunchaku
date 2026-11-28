"""Activation helpers for Nunchaku."""

import torch

from .._C import ops

__all__ = ["situ_and_mul_cuda"]


def situ_and_mul_cuda(input: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    """Fused ``situ`` over ``[up, gate]`` into ``[ffn_dim]``.

    ``input`` is ``[..., 2 * D]`` and ``output`` is ``[..., D]``.
    The kernel is a single high-occupancy elementwise pass.
    """
    if input.shape[-1] % 2:
        raise ValueError("input last dimension must be even")
    d = input.shape[-1] // 2
    if output is None:
        output = torch.empty((*input.shape[:-1], d), dtype=input.dtype, device=input.device)
    elif output.shape != (*input.shape[:-1], d):
        raise ValueError(f"output must have shape {(*input.shape[:-1], d)}, got {tuple(output.shape)}")

    input_2d = input.reshape(-1, input.shape[-1])
    output_2d = output.reshape(-1, d)
    if not input_2d.is_contiguous():
        input_2d = input_2d.contiguous()
    if not output_2d.is_contiguous():
        output_2d = output_2d.contiguous()
    ops.situ_and_mul(input_2d, output_2d)
    return output
