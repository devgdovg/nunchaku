"""
Attention processor for Qwen Image 2.1 with fused QKV projection.

The cache semantics, block-causal segment handling, and output trimming are copied
from :class:`diffusers.models.transformers.transformer_qwenimage21.QwenImage21AttnProcessor`.
Only the QKV preparation is replaced by the fused SVDQuant kernel, which performs
``to_q/to_k/to_v`` projection, Q/K RMSNorm, and RoPE in one launch.
"""

from typing import Any

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn

__all__ = ["NunchakuQwenImage21AttnProcessor"]


class NunchakuQwenImage21AttnProcessor:
    """Attention processor for Qwen Image 2.1 using the fused QKV/RMSNorm/RoPE kernel.

    The processor deliberately stays on the non-flex path used by the default
    ``QwenImage21AttnProcessor``. ``QwenImage21Transformer2DModel.forward`` detects
    flex processors by type and passes ``segments``/``key_valid`` otherwise; this
    processor consumes exactly those arguments.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        attention_mask: Any | None = None,
        rotary_emb: torch.Tensor | None = None,
        layer_cache: Any | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = attn.fused_module(hidden_states, rotary_emb)

        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        seq_len_q = query.shape[1]

        # Same cache placement as diffusers `_qwenimage21_prepare_qkv`: K/V are
        # stored after RMSNorm + RoPE.
        if layer_cache is not None:
            if kv_cache_mode == "extract" and cache_write_slice is not None:
                layer_cache.store(
                    key[:, cache_write_slice].clone(),
                    value[:, cache_write_slice].clone(),
                )
            elif kv_cache_mode == "cached":
                cached_k, cached_v = layer_cache.get()
                key = torch.cat([cached_k, key], dim=1)
                value = torch.cat([cached_v, value], dim=1)

        if segments is None:
            # Decode / no-cache: full attention over [cached prefix, target].
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:
            # Prefill: each segment attends to keys `[0, end)`; text segments get
            # a causal triangle over their own keys. Padded text keys are dropped.
            prefix_len = segments[-1][1] if segments else 0
            outputs = []
            for start, end, is_text in segments:
                seg_mask = None
                if is_text:
                    seg_len = end - start
                    seg_mask = torch.cat(
                        [
                            torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                            torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
                        ],
                        dim=1,
                    )[None, None]
                if key_valid is not None:
                    seg_key_valid = key_valid[:, None, None, :end]
                    seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
                outputs.append(
                    dispatch_attention_fn(
                        query[:, start:end],
                        key[:, :end],
                        value[:, :end],
                        attn_mask=seg_mask,
                        dropout_p=0.0,
                        backend=None,
                        parallel_config=self._parallel_config,
                    )
                )
            outputs.append(
                dispatch_attention_fn(
                    query[:, prefix_len:],
                    key,
                    value,
                    attn_mask=None if key_valid is None else key_valid[:, None, None, :],
                    dropout_p=0.0,
                    backend=None,
                    parallel_config=self._parallel_config,
                )
            )
            hidden_states = torch.cat(outputs, dim=1)

        hidden_states = hidden_states[:, :seq_len_q]
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)
