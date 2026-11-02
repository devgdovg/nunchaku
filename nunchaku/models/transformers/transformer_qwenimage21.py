"""
Nunchaku inference implementation for Qwen Image 2.1.

Qwen Image 2.1 is a 32-layer single-stream DiT. The quantized checkpoint contains
fused ``attn.to_qkv``, ``attn.to_out.0``, ``img_mlp.net.0.proj`` (gate + up), and
``img_mlp.net.2`` projections. This module patches the diffusers transformer in place:

- the original Qwen SwiGLU FFN is first converted to a standard diffusers
  ``FeedForward(activation_fn="swiglu")`` so its state-dict keys match the checkpoint;
- attention and FFN linears are replaced by :class:`SVDQW4A4Linear`;
- attention uses :func:`fused_qkv_norm_rottary` for qkv + RMSNorm + RoPE;
- KV-cache extract/cached semantics remain in the diffusers transformer forward and
  are handled by the custom attention processor.

The old ``NunchakuQwenImageTransformer2DModel`` targets the legacy Qwen Image model
and is intentionally not reused.
"""

import json
import os
from pathlib import Path
from typing import Any

import torch
from diffusers.models.attention import FeedForward
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Rope,
    QwenImage21SwiGLUFeedForward,
    QwenImage21Transformer2DModel,
)
from huggingface_hub import utils
from torch import nn

from ...utils import get_precision, pad_tensor
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..attention_processors.qwenimage21 import NunchakuQwenImage21AttnProcessor
from ..embeddings import pack_rotemb
from ..linear import SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key

__all__ = [
    "NunchakuQwenImage21Attention",
    "NunchakuQwenImage21RopeHook",
    "NunchakuQwenImage21Transformer2DModel",
]


class NunchakuQwenImage21RopeHook:
    """Convert Qwen 2.1 complex RoPE to nunchaku's packed ``rotemb`` layout.

    Qwen 2.1 passes ``rotary_emb`` as ``[S, D/2]`` complex64 (no batch dimension).
    The fused QKV kernel expects a packed float32 tensor shaped
    ``[1, ceil(B*S / 256) * 256, D]``. The packed tensor is cached for all 32 blocks
    within one transformer forward; the model pre-hook clears the cache at the start
    of each forward to avoid stale entries from allocator address reuse.
    """

    def __init__(self) -> None:
        self.packed_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def clear(self) -> None:
        self.packed_cache.clear()

    def __call__(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        rotary = kwargs.get("rotary_emb", None)
        if rotary is None or not torch.is_complex(rotary):
            return None

        hidden = kwargs.get("hidden_states", None)
        if hidden is None and args and isinstance(args[0], torch.Tensor):
            hidden = args[0]
        batch_size = hidden.shape[0] if isinstance(hidden, torch.Tensor) else 1

        if rotary.ndim == 3:
            # [B, S, D/2] -> [B*S, D/2]
            rotary = rotary.reshape(-1, rotary.shape[-1])
        elif rotary.ndim != 2:
            raise ValueError(f"unexpected Qwen rotary_emb shape: {tuple(rotary.shape)}")

        cache_key = (rotary.data_ptr(), tuple(rotary.shape), batch_size)
        packed = self.packed_cache.get(cache_key)
        if packed is None:
            real = torch.view_as_real(rotary).unsqueeze(0).unsqueeze(3)  # [1, S, D/2, 1, 2]
            if batch_size > 1:
                # The rotary embedding is shared by all samples in the batch; flatten
                # it to [1, B*S, ...] so the packed layout matches the flattened GEMM M.
                real = real.expand(batch_size, -1, -1, -1, -1).reshape(
                    1, batch_size * rotary.shape[0], rotary.shape[-1], 1, 2
                )
            real = torch.flip(real, dims=[-1])
            real = pad_tensor(real, 256, 1)
            packed = pack_rotemb(real)
            self.packed_cache[cache_key] = packed

        new_kwargs = kwargs.copy()
        new_kwargs["rotary_emb"] = packed
        return args, new_kwargs


class NunchakuQwenImage21Attention(NunchakuBaseAttention):
    """Qwen Image 2.1 attention with a quantized fused QKV projection."""

    def __init__(self, orig_attn: Any, **kwargs: Any) -> None:
        super().__init__(processor="qwenimage21")
        self.heads = orig_attn.heads
        self.inner_dim = orig_attn.inner_dim
        self.use_bias = orig_attn.use_bias
        self.norm_q = orig_attn.norm_q
        self.norm_k = orig_attn.norm_k

        with torch.device("meta"):
            to_qkv = fuse_linears([orig_attn.to_q, orig_attn.to_k, orig_attn.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_out = orig_attn.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

    def set_processor(self, processor: str) -> None:
        if processor not in (None, "qwenimage21"):
            raise ValueError(f"Processor {processor!r} is not supported for Qwen Image 2.1")
        self.processor = NunchakuQwenImage21AttnProcessor()

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.processor(self, hidden_states, **kwargs)


def _convert_qwen_ff(ffn: nn.Module) -> FeedForward:
    """Convert Qwen's custom SwiGLU FFN to a standard diffusers ``FeedForward``.

    ``up = cat([proj, gate_layer])`` because diffusers ``SwiGLU`` returns
    ``first * silu(second)``; the first chunk must be the multiplier.
    """
    if isinstance(ffn, FeedForward):
        return ffn
    assert isinstance(ffn, QwenImage21SwiGLUFeedForward), f"unexpected Qwen FFN type: {type(ffn).__name__}"
    assert ffn.proj.in_features == ffn.gate_layer.in_features
    assert ffn.proj.out_features == ffn.gate_layer.out_features
    assert ffn.proj.out_features == ffn.out.in_features

    converted = FeedForward(
        dim=ffn.proj.in_features,
        dim_out=ffn.out.out_features,
        dropout=0.0,
        activation_fn="swiglu",
        inner_dim=ffn.proj.out_features,
        bias=False,
    ).to(dtype=ffn.proj.weight.dtype, device=ffn.proj.weight.device)

    if not ffn.proj.weight.is_meta:
        with torch.no_grad():
            converted.net[0].proj.weight.copy_(torch.cat([ffn.proj.weight, ffn.gate_layer.weight], dim=0))
            converted.net[2].weight.copy_(ffn.out.weight)
    return converted


class NunchakuQwenImage21Transformer2DModel(QwenImage21Transformer2DModel, NunchakuModelLoaderMixin):
    """Nunchaku-optimized Qwen Image 2.1 transformer."""

    def _patch_model(self, precision: str = "nvfp4", rank: int = 128) -> "NunchakuQwenImage21Transformer2DModel":
        self._rope_hook = NunchakuQwenImage21RopeHook()
        self.register_forward_pre_hook(self._clear_rope_cache)

        for block in self.transformer_blocks:
            block.img_mlp = NunchakuFeedForward(
                _convert_qwen_ff(block.img_mlp),
                precision=precision,
                rank=rank,
            )
            block.attn = NunchakuQwenImage21Attention(block.attn, precision=precision, rank=rank)
            block.attn.register_forward_pre_hook(self._rope_hook, with_kwargs=True)
        return self

    def _clear_rope_cache(self, module: nn.Module, args: tuple[Any, ...]) -> None:
        self._rope_hook.clear()

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: Any):
        """Load a quantized Qwen Image 2.1 checkpoint into the nunchaku model."""
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)
        if offload:
            raise NotImplementedError("Offload is not supported for Qwen Image 2.1")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"

        transformer, state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        config = json.loads(metadata["config"])
        rank = int(quantization_config.get("rank", 32))

        precision = get_precision()
        if precision != "fp4":
            raise NotImplementedError(
                f"Qwen Image 2.1 inference currently supports nvfp4/Blackwell only; detected precision={precision!r}"
            )
        precision = "nvfp4"

        transformer = transformer.to(torch_dtype)
        transformer._patch_model(precision=precision, rank=rank)
        transformer = transformer.to_empty(device=device)

        # QwenImage21Rope keeps `freqs` in a plain list, not a buffer; `to_empty`
        # cannot migrate those tensors, so rebuild the module after materialization.
        transformer.pos_embed = QwenImage21Rope(
            theta=int(config.get("rope_theta", 10000)),
            axes_dim=list(config.get("axes_dims_rope", [16, 56, 56])),
        )

        patch_scale_key(transformer, state_dict)
        transformer.load_state_dict(state_dict)
        return transformer
