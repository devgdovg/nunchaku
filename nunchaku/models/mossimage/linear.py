"""MOSS Image-specific quantized projection modules."""

from __future__ import annotations

import torch

from ...ops.gemm import svdq_gemm_w4a4_gated_residual_cuda
from ..linear import SVDQW4A4Linear

__all__ = ["MOSSImageW4A4OProject"]


class MOSSImageW4A4OProject(SVDQW4A4Linear):
    """SVDQ W4A4 o_proj with fused gated residual for MOSS Image.

    The gate is a per-all vector shared by every token and sample; it is not a
    per-sample or per-token tensor.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        gate: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``residual + o_proj(x) * gate`` with a fused epilogue.

        Parameters
        ----------
        x : torch.Tensor
            Attention output with shape ``(B, S, in_features)``.
        residual : torch.Tensor
            Residual stream with shape ``(B, S, out_features)``.
        gate : torch.Tensor
            Shared gate vector with ``out_features`` elements.
        output : torch.Tensor or None, optional
            Optional 2D output buffer with shape ``(B * S, out_features)``.
        """
        if self.precision != "nvfp4":
            raise ValueError("gated residual fusion currently requires nvfp4")
        if self.bias is not None:
            raise ValueError("gated residual fusion currently expects a bias-free linear")

        batch_size, seq_len, channels = x.shape
        if residual.shape != (batch_size, seq_len, self.out_features):
            raise ValueError(
                f"residual must have shape {(batch_size, seq_len, self.out_features)}, "
                f"got {tuple(residual.shape)}"
            )
        if gate.numel() != self.out_features:
            raise ValueError(f"gate must contain {self.out_features} elements, got {gate.numel()}")

        x = x.reshape(batch_size * seq_len, channels)
        residual = residual.reshape(batch_size * seq_len, self.out_features)
        gate = gate.reshape(-1)
        if not x.is_contiguous():
            x = x.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        if not gate.is_contiguous():
            gate = gate.contiguous()

        if output is None:
            output = torch.empty(
                batch_size * seq_len, self.out_features, dtype=x.dtype, device=x.device
            )

        quantized_x, ascales, lora_act_out = self.quantize(x)
        svdq_gemm_w4a4_gated_residual_cuda(
            act=quantized_x,
            wgt=self.qweight,
            out=output,
            ascales=ascales,
            wscales=self.wscales,
            lora_act_in=lora_act_out,
            lora_up=self.proj_up,
            wcscales=self.wcscales,
            residual=residual,
            gate=gate,
            fp4=True,
            alpha=self.wtscale,
        )
        return output.reshape(batch_size, seq_len, -1)
