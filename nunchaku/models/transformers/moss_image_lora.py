"""MOSS-Image-SFT 8-step DMD LoRA 合并（仅用于 bf16 基线对比）。

量化 ckpt 已包含 LoRA，不需要调用本模块。
"""

from __future__ import annotations

import logging
import os
import re

import torch
from safetensors import safe_open

__all__ = ["merge_moss_image_lora"]

logger = logging.getLogger(__name__)

_PROJ_NAMES = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
_LORA_KEY = re.compile(
    r"^(?:diffusion_model\.)?layers\.(\d+)\.proj\."
    + r"("
    + "|".join(_PROJ_NAMES)
    + r")\.(lora_down|lora_up|alpha)(?:\.weight)?$"
)


def _strip_prefix(key: str) -> str:
    return key.removeprefix("diffusion_model.")


def merge_moss_image_lora(transformer: torch.nn.Module, lora_path: str, strength: float = 0.7) -> dict[str, int]:
    """按 ``W + strength · alpha/rank · up@down`` 逐层合并 MOSS DMD LoRA。"""
    if not lora_path or not os.path.isfile(lora_path):
        raise ValueError(f"LoRA file not found: {lora_path!r}")
    layers = getattr(transformer, "layers", None)
    if layers is None:
        raise ValueError("transformer has no `layers`; is this a MOSSImageTransformer?")

    groups: dict[tuple[int, str], dict[str, str]] = {}
    unmatched: list[str] = []
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 - safetensors safe_open is not iterable
            match = _LORA_KEY.match(_strip_prefix(key))
            if match is None:
                unmatched.append(key)
                continue
            layer_index, proj_name, kind = int(match.group(1)), match.group(2), match.group(3)
            groups.setdefault((layer_index, proj_name), {})[kind] = key
        if unmatched:
            logger.warning("ignoring %d unmatched LoRA key(s), e.g. %s", len(unmatched), unmatched[:5])

        merged = 0
        missing: list[str] = []
        for (layer_index, proj_name), parts in sorted(groups.items()):
            if "lora_down" not in parts or "lora_up" not in parts:
                missing.append(f"layers.{layer_index}.proj.{proj_name}")
                continue
            if layer_index >= len(layers):
                raise ValueError(f"LoRA references layer {layer_index}, but transformer has {len(layers)} layers")
            down = f.get_tensor(parts["lora_down"])
            up = f.get_tensor(parts["lora_up"])
            alpha = f.get_tensor(parts["alpha"]) if "alpha" in parts else None
            module = layers[layer_index].proj[proj_name]
            if module.weight.shape != (up.shape[0], down.shape[1]):
                raise ValueError(
                    f"shape mismatch for layers.{layer_index}.proj.{proj_name}: "
                    f"base={tuple(module.weight.shape)}, lora={(up.shape[0], down.shape[1])}"
                )
            device = module.weight.device
            down = down.to(device=device, dtype=torch.float32)
            up = up.to(device=device, dtype=torch.float32)
            rank = int(down.shape[0])
            scale = strength * (float(alpha.reshape(-1)[0].item()) / rank) if alpha is not None else strength
            delta = (up @ down).mul_(scale).to(dtype=module.weight.dtype)
            with torch.no_grad():
                module.weight.data.add_(delta)
            merged += 1

    if missing:
        raise ValueError(f"incomplete LoRA groups (missing lora_down/lora_up): {missing[:8]}")
    expected = len(layers) * len(_PROJ_NAMES)
    if merged != expected:
        logger.warning(
            "merged %d/%d LoRA projections; the LoRA does not cover every layer/projection", merged, expected
        )
    logger.info("merged %d MOSS LoRA projection(s) with strength=%s", merged, strength)
    return {"merged": merged, "unmatched": len(unmatched)}
