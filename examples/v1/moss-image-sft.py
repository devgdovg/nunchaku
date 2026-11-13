import os
import sys

# Keep the example CI-safe: both a quantized checkpoint and the 8-step DMD LoRA
# are required for the baseline/quantized comparison.
ckpt_path = os.environ.get("MOSS_IMAGE_CKPT", "")
lora_path = os.environ.get("MOSS_IMAGE_LORA", "")
model_path = os.environ.get("MOSS_IMAGE_MODEL", "")
if not ckpt_path or not os.path.isfile(ckpt_path) or not lora_path or not os.path.isfile(lora_path) or not model_path:
    print(
        "Set MOSS_IMAGE_CKPT, MOSS_IMAGE_LORA and MOSS_IMAGE_MODEL to existing files/directories "
        "to run the MOSS-Image-SFT comparison; skipping."
    )
    sys.exit(0)

import gc
import math

import numpy as np
import torch
from diffusers import DiffusionPipeline

from nunchaku import NunchakuMOSSImageTransformer
from nunchaku.models.transformers.moss_image_lora import merge_moss_image_lora
from nunchaku.utils import get_precision


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name, "1" if default else "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _load_pipeline(transformer=None):
    kwargs = {
        "custom_pipeline": "pipeline_moss_image",
        "trust_remote_code": True,
        "torch_dtype": {"default": torch.bfloat16, "vae": torch.float32},
        "local_files_only": _env_flag("MOSS_IMAGE_LOCAL_FILES_ONLY", True),
    }
    if transformer is not None:
        kwargs["transformer"] = transformer
        try:
            pipe = DiffusionPipeline.from_pretrained(model_path, **kwargs)
        except (TypeError, ValueError) as exc:
            print(f"warning: custom pipeline rejected the transformer override ({exc}); loading then replacing it")
            kwargs.pop("transformer")
            pipe = DiffusionPipeline.from_pretrained(model_path, **kwargs)
            pipe.register_modules(transformer=transformer)
    else:
        pipe = DiffusionPipeline.from_pretrained(model_path, **kwargs)
    if transformer is not None and pipe.transformer is not transformer:
        print("warning: pipeline did not keep the passed transformer; replacing it")
        pipe.register_modules(transformer=transformer)
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe


def _generate(pipe, prompt: str, height: int, width: int, steps: int, guidance_scale: float, seed: int):
    generator = torch.Generator("cpu").manual_seed(seed)
    return pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        uncond_skip_layers=None,
        slg_layers=None,
        slg_scale=0.0,
        generator=generator,
    ).images[0]


def _save_and_compare(bf16_image, fp4_image, bf16_path: str, fp4_path: str) -> None:
    bf16_image.save(bf16_path)
    fp4_image.save(fp4_path)
    a = np.asarray(bf16_image.convert("RGB"), dtype=np.float32)
    b = np.asarray(fp4_image.convert("RGB"), dtype=np.float32)
    diff = np.abs(a - b)
    mse = float((diff**2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0**2) / mse)
    print(
        f"saved bf16={bf16_path}, fp4={fp4_path}; "
        f"pixel_abs_max={diff.max():.1f}, pixel_abs_mean={diff.mean():.4f}, psnr={psnr:.2f} dB"
    )


if __name__ == "__main__":
    precision = get_precision()
    if precision != "fp4":
        print(f"MOSS-Image-SFT quantized inference currently requires nvfp4/Blackwell; detected {precision!r}.")
        sys.exit(0)

    prompt = os.environ.get(
        "MOSS_IMAGE_PROMPT",
        "A wide view of a refrigerator beside a kitchen counter with a fruit bowl, soft daylight, highly detailed.",
    )
    height = int(os.environ.get("MOSS_IMAGE_HEIGHT", "1024"))
    width = int(os.environ.get("MOSS_IMAGE_WIDTH", "1024"))
    steps = int(os.environ.get("MOSS_IMAGE_STEPS", "8"))
    guidance_scale = float(os.environ.get("MOSS_IMAGE_GUIDANCE", "1.0"))
    seed = int(os.environ.get("MOSS_IMAGE_SEED", "42"))
    strength = float(os.environ.get("MOSS_IMAGE_LORA_STRENGTH", "0.7"))
    bf16_output = os.environ.get("MOSS_IMAGE_BF16_OUTPUT", "moss-image-sft-bf16-lora.png")
    fp4_output = os.environ.get("MOSS_IMAGE_FP4_OUTPUT", "moss-image-sft-fp4.png")

    # 1) bf16 baseline: explicitly merge the DMD LoRA into the official transformer.
    pipe = _load_pipeline()
    merge_result = merge_moss_image_lora(pipe.transformer, lora_path, strength)
    print(f"bf16 baseline: merged {merge_result['merged']} LoRA projection(s) at strength={strength}")
    bf16_image = _generate(pipe, prompt, height, width, steps, guidance_scale, seed)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 2) quantized: the checkpoint already contains the merged LoRA weights.
    transformer = NunchakuMOSSImageTransformer.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    pipe = _load_pipeline(transformer=transformer)
    print(f"quantized pipeline: transformer={type(pipe.transformer).__name__}, fused qkv+norm+RoPE")
    fp4_image = _generate(pipe, prompt, height, width, steps, guidance_scale, seed)

    _save_and_compare(bf16_image, fp4_image, os.path.abspath(bf16_output), os.path.abspath(fp4_output))
