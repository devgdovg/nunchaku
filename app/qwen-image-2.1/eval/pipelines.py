"""Qwen Image 2.1 评估 pipeline 构建与逐样本调用参数。"""

import os
import typing as tp

import torch
from dataset import EvalSample, open_conditional_images

__all__ = ["build_call_kwargs", "build_pipeline", "sample_seed"]


def sample_seed(sample_id: str) -> int:
    """把 ``sample_id`` 稳定地映射为随机种子；bf16 与量化 pipeline 使用相同种子。"""
    modulus = 10**9 + 7
    hash_int = 0
    for char in sample_id:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def _transformer_info(transformer) -> dict[str, tp.Any]:
    """识别实际参与推理的 transformer 类型与量化线性层数量。

    按类名匹配 ``SVDQW4A4Linear``，避免在纯 bf16 评估时触发 nunchaku 编译扩展的导入。
    """
    return {
        "transformer_class": type(transformer).__name__,
        "svdq_linears": sum(type(module).__name__ == "SVDQW4A4Linear" for module in transformer.modules()),
    }


def build_pipeline(
    precision: str,
    model_path: str,
    ckpt_path: str | None = None,
    offload: str = "model",
) -> tuple[tp.Any, dict[str, tp.Any]]:
    """构建原版 bf16 pipeline 或替换 SVDQ transformer 的 fp4 pipeline。

    Returns:
        ``(pipeline, info)``；``info`` 记录实际 transformer 的类名和 SVDQ 线性层数量，
        用于确认量化/原版模型确实被正确替换。
    """
    from diffusers import QwenImage21Pipeline

    kwargs: dict[str, tp.Any] = {"torch_dtype": torch.bfloat16}
    if precision == "fp4":
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise ValueError("--ckpt must point to an existing quantized safetensors file when --precision fp4")
        from nunchaku import NunchakuQwenImage21Transformer2DModel
        from nunchaku.utils import get_precision

        detected = get_precision()
        if detected != "fp4":
            raise RuntimeError(f"fp4 evaluation requires Blackwell/nvfp4; detected {detected!r}")
        kwargs["transformer"] = NunchakuQwenImage21Transformer2DModel.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16, device="cpu"
        )
    elif precision != "bf16":
        raise ValueError(f"unsupported precision {precision!r} (expected 'bf16' or 'fp4')")

    pipeline = QwenImage21Pipeline.from_pretrained(model_path, **kwargs)
    info = _transformer_info(pipeline.transformer)
    expected_class = "NunchakuQwenImage21Transformer2DModel" if precision == "fp4" else "QwenImage21Transformer2DModel"
    if info["transformer_class"] != expected_class:
        raise RuntimeError(
            f"pipeline uses transformer {info['transformer_class']!r}, "
            f"expected {expected_class!r} for precision {precision!r}"
        )
    if precision == "fp4":
        if kwargs["transformer"] is not pipeline.transformer:
            raise RuntimeError("pipeline did not keep the quantized transformer instance")
        if info["svdq_linears"] == 0:
            raise RuntimeError("quantized transformer contains no SVDQW4A4Linear modules")
    elif info["svdq_linears"] != 0:
        raise RuntimeError(f"bf16 pipeline unexpectedly contains {info['svdq_linears']} SVDQW4A4Linear modules")

    if offload == "model":
        pipeline.enable_model_cpu_offload()
    elif offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"unsupported offload mode {offload!r} (expected 'model' or 'sequential')")
    return pipeline, info


def build_call_kwargs(
    sample: EvalSample,
    num_inference_steps: int,
    true_cfg_scale: float,
    use_kv_cache: bool = True,
) -> tuple[dict[str, tp.Any], list, int]:
    """构造单条样本的 pipeline 调用参数。

    Returns:
        ``(kwargs, opened_images, seed)``；``opened_images`` 由调用方在推理结束后关闭。
    """
    seed = sample_seed(sample.sample_id)
    kwargs: dict[str, tp.Any] = {
        "prompt": sample.prompt,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "use_kv_cache": use_kv_cache,
        "generator": torch.Generator().manual_seed(seed),
    }
    opened = []
    if sample.images:
        opened = open_conditional_images(sample)
        kwargs["image"] = opened[0] if len(opened) == 1 else opened
    else:
        if sample.width is not None:
            kwargs["width"] = sample.width
        if sample.height is not None:
            kwargs["height"] = sample.height
    return kwargs, opened, seed
