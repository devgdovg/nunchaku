"""Benchmark MOSS-Image-SFT eager / fused-simple / fused-packed qkv paths.

The quantized checkpoint already contains the merged 8-step DMD LoRA, so this
script only compares inference paths. Text blocks remain eager in all three
modes; the measurement isolates the main-block qkv + RMSNorm + RoPE path.

Environment variables
---------------------
MOSS_IMAGE_CKPT
    Quantized MOSS-Image-SFT safetensors file (required).
MOSS_IMAGE_MODEL
    Local MOSS-Image diffusers pipeline directory. Required for
    ``MOSS_IMAGE_BENCH_SCOPE=pipeline`` or ``all``.
MOSS_IMAGE_BENCH_SCOPE
    ``transformer`` (default) measures one transformer forward and the rope
    pack call. ``pipeline`` measures the 8-step denoising pipeline.
    ``all`` runs both.
MOSS_IMAGE_BENCH_HEIGHT / MOSS_IMAGE_BENCH_WIDTH
    Image size used to derive the latent size (default 1024).
MOSS_IMAGE_BENCH_TEXT_LEN
    Synthetic text token count for the transformer scope (default 64).
MOSS_IMAGE_BENCH_STEPS
    Pipeline steps (default 8).
MOSS_IMAGE_BENCH_WARMUP / MOSS_IMAGE_BENCH_ITERS
    Warmup and measured iterations (default 2 / 5).
MOSS_IMAGE_FP4_OUTPUT
    Optional path for the last generated image in pipeline scope.
"""

import gc
import os
import statistics
import sys
import time

import torch
from diffusers import DiffusionPipeline

from nunchaku import NunchakuMOSSImageTransformer
from nunchaku.utils import get_precision

ckpt_path = os.environ.get("MOSS_IMAGE_CKPT", "")
model_path = os.environ.get("MOSS_IMAGE_MODEL", "")
if not ckpt_path or not os.path.isfile(ckpt_path):
    print("Set MOSS_IMAGE_CKPT to an existing quantized MOSS-Image-SFT safetensors file; skipping.")
    sys.exit(0)

SCOPE = os.environ.get("MOSS_IMAGE_BENCH_SCOPE", "transformer").strip().lower()
if SCOPE not in ("transformer", "pipeline", "all"):
    raise ValueError("MOSS_IMAGE_BENCH_SCOPE must be transformer, pipeline, or all")
if SCOPE in ("pipeline", "all") and (not model_path or not os.path.isdir(model_path)):
    raise ValueError("set MOSS_IMAGE_MODEL to a local MOSS-Image pipeline directory")

HEIGHT = int(os.environ.get("MOSS_IMAGE_BENCH_HEIGHT", "1024"))
WIDTH = int(os.environ.get("MOSS_IMAGE_BENCH_WIDTH", "1024"))
TEXT_LEN = int(os.environ.get("MOSS_IMAGE_BENCH_TEXT_LEN", "64"))
STEPS = int(os.environ.get("MOSS_IMAGE_BENCH_STEPS", "8"))
WARMUP = int(os.environ.get("MOSS_IMAGE_BENCH_WARMUP", "2"))
ITERS = int(os.environ.get("MOSS_IMAGE_BENCH_ITERS", "5"))
SEED = int(os.environ.get("MOSS_IMAGE_BENCH_SEED", "42"))
PROMPT = os.environ.get(
    "MOSS_IMAGE_PROMPT",
    "A wide view of a refrigerator beside a kitchen counter with a fruit bowl, soft daylight, highly detailed.",
)

MODES = {
    "eager": {"fused_rope": False, "rope_layout": "packed"},
    "fused-simple": {"fused_rope": True, "rope_layout": "simple"},
    "fused-packed": {"fused_rope": True, "rope_layout": "packed"},
}


def _sync() -> None:
    torch.cuda.synchronize()


def _timed(fn, warmup: int = WARMUP, iters: int = ITERS) -> tuple[list[float], float]:
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end))
    return times, torch.cuda.max_memory_allocated() / 2**30


def _summary(name: str, times: list[float], peak_gib: float) -> None:
    print(
        f"{name}: median={statistics.median(times):.3f} ms, mean={statistics.mean(times):.3f} ms, "
        f"min={min(times):.3f} ms, max={max(times):.3f} ms, peak_vram={peak_gib:.2f} GiB"
    )


def _load_transformer(mode: str):
    cfg = MODES[mode]
    return NunchakuMOSSImageTransformer.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
        fused_rope=cfg["fused_rope"],
        rope_layout=cfg["rope_layout"],
    )


def _transformer_inputs(transformer):
    latent_h, latent_w = HEIGHT // 16, WIDTH // 16
    latent_channels = transformer.config.latent_channels
    text_dim = transformer.config.text_dim
    hidden = torch.randn(1, latent_channels, latent_h, latent_w, device="cuda", dtype=torch.bfloat16)
    context = torch.randn(1, TEXT_LEN, text_dim, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, TEXT_LEN, device="cuda", dtype=torch.bool)
    timestep = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    return hidden, timestep, context, mask


def _bench_transformer(mode: str) -> None:
    torch.cuda.empty_cache()
    transformer = _load_transformer(mode)
    hidden, timestep, context, mask = _transformer_inputs(transformer)

    if transformer._rope_cache is not None:
        rope = transformer.rope(HEIGHT // 16, WIDTH // 16, TEXT_LEN, hidden.device)
        pack_times = []
        for _ in range(5):
            transformer._rope_cache.clear()
            _sync()
            start = time.perf_counter()
            transformer._rope_cache.pack(rope)
            _sync()
            pack_times.append((time.perf_counter() - start) * 1e3)
        print(f"[{mode}] first rope pack: median={statistics.median(pack_times):.3f} ms")

    def call():
        with torch.no_grad():
            transformer(hidden, timestep, context, encoder_attention_mask=mask)

    times, peak_gib = _timed(call)
    _summary(f"[{mode}] transformer forward", times, peak_gib)
    del transformer
    gc.collect()
    torch.cuda.empty_cache()


def _load_pipeline(transformer):
    kwargs = {
        "custom_pipeline": "pipeline_moss_image",
        "trust_remote_code": True,
        "torch_dtype": {"default": torch.bfloat16, "vae": torch.float32},
        "local_files_only": True,
        "transformer": transformer,
    }
    try:
        pipe = DiffusionPipeline.from_pretrained(model_path, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("transformer")
        pipe = DiffusionPipeline.from_pretrained(model_path, **kwargs)
        pipe.register_modules(transformer=transformer)
    if pipe.transformer is not transformer:
        pipe.register_modules(transformer=transformer)
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe


def _bench_pipeline(mode: str) -> None:
    torch.cuda.empty_cache()
    transformer = _load_transformer(mode)
    pipe = _load_pipeline(transformer)

    def call():
        pipe(
            PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            guidance_scale=1.0,
            uncond_skip_layers=None,
            slg_layers=None,
            slg_scale=0.0,
            generator=torch.Generator("cpu").manual_seed(SEED),
        ).images[0]

    times, peak_gib = _timed(call)
    _summary(f"[{mode}] pipeline {STEPS}-step", times, peak_gib)
    output = os.environ.get("MOSS_IMAGE_FP4_OUTPUT")
    if output:
        call().save(output)
        print(f"[{mode}] saved {output}")
    del pipe, transformer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if get_precision() != "fp4":
        print(f"MOSS-Image-SFT quantized inference currently requires nvfp4/Blackwell; detected {get_precision()!r}.")
        sys.exit(0)

    print(
        f"scope={SCOPE}, image={HEIGHT}x{WIDTH}, text_len={TEXT_LEN}, steps={STEPS}, "
        f"warmup={WARMUP}, iters={ITERS}"
    )
    for name in MODES:
        if SCOPE in ("transformer", "all"):
            _bench_transformer(name)
        if SCOPE in ("pipeline", "all"):
            _bench_pipeline(name)
