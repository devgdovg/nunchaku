import os
import sys

# The quantized Qwen Image 2.1 checkpoint is not published yet. Keep the example
# CI-safe: exit successfully when it is not configured, instead of downloading or
# failing inside `tests/v1/test_examples.py`.
ckpt_path = os.environ.get("QWEN_IMAGE_21_CKPT", "")
if not ckpt_path or not os.path.isfile(ckpt_path):
    print(
        "QWEN_IMAGE_21_CKPT is not set to an existing safetensors file; "
        "skipping the Qwen Image 2.1 text-to-image example."
    )
    sys.exit(0)

import torch
from diffusers import QwenImage21Pipeline

from nunchaku import NunchakuQwenImage21Transformer2DModel
from nunchaku.utils import get_precision

_BAD_SEEN: set[str] = set()
_STATS_SEEN: set[str] = set()


def _as_tensor(output):
    if isinstance(output, (tuple, list)):
        if not output:
            return None
        output = output[0]
    sample = getattr(output, "sample", None)
    if sample is not None:
        output = sample
    return output if torch.is_tensor(output) else None


def _report(name: str, tensor: torch.Tensor) -> None:
    if tensor is None or not tensor.is_floating_point():
        return
    try:
        finite = torch.isfinite(tensor)
    except RuntimeError:
        finite = torch.isfinite(tensor.float())
    if finite.all():
        return
    if name in _BAD_SEEN:
        return
    _BAD_SEEN.add(name)
    xf = tensor.float()
    x = torch.nan_to_num(xf)
    print(
        f"[BAD] {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"nan={torch.isnan(xf).sum().item()} inf={torch.isinf(xf).sum().item()} "
        f"finite_min={x.min().item():.6g} finite_max={x.max().item():.6g} "
        f"finite_std={x.std().item():.6g}",
        flush=True,
    )
    if os.environ.get("QWEN_IMAGE_21_RAISE_ON_NAN", "0").strip().lower() in ("1", "true", "yes"):
        raise RuntimeError(f"first NaN/Inf at {name}")


def _log_stats(name: str, tensor: torch.Tensor) -> None:
    """Print finite min/max/std once for the most informative tensors."""
    if tensor.is_complex():
        return
    is_key = (
        name in ("transformer.output", "img_in", "txt_in", "modulation", "norm_out", "proj_out", "vae.output")
        or name.startswith(("step[", "pipeline.output.", "transformer.input."))
        or name.endswith((".attn.fused_module.q", ".attn.fused_module.k", ".attn.fused_module.v", ".attn"))
        or (name.startswith("transformer_blocks.") and name.count(".") == 1)
    )
    if not is_key or name in _STATS_SEEN:
        return
    _STATS_SEEN.add(name)
    xf = tensor.float()
    x = torch.nan_to_num(xf)
    print(
        f"[stats] {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"nan={torch.isnan(xf).sum().item()} inf={torch.isinf(xf).sum().item()} "
        f"min={x.min().item():.6g} max={x.max().item():.6g} std={x.std().item():.6g}",
        flush=True,
    )


def _check_output(name: str, output) -> None:
    tensor = _as_tensor(output)
    if tensor is None:
        return
    if name.endswith("attn.fused_module"):
        q, k, v = tensor.chunk(3, dim=-1)
        _log_stats(f"{name}.q", q)
        _log_stats(f"{name}.k", k)
        _log_stats(f"{name}.v", v)
        _report(f"{name}.q", q)
        _report(f"{name}.k", k)
        _report(f"{name}.v", v)
    else:
        _log_stats(name, tensor)
        _report(name, tensor)


def _register_transformer_hooks(pipe) -> None:
    transformer = pipe.transformer

    def _pre_hook(module, args, kwargs):
        for index, value in enumerate(args):
            if torch.is_tensor(value):
                _log_stats(f"transformer.input.args[{index}]", value)
                _report(f"transformer.input.args[{index}]", value)
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                _log_stats(f"transformer.input.{key}", value)
                _report(f"transformer.input.{key}", value)

    transformer.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    for name, module in transformer.named_modules():
        module.register_forward_hook(
            lambda module, inputs, output, name=name: _check_output(name or "transformer.output", output)
        )
    print(f"[debug] registered transformer hooks on {len(list(transformer.named_modules()))} modules", flush=True)


def _register_vae_hooks(pipe) -> None:
    if getattr(pipe, "vae", None) is None:
        return
    pipe.vae.register_forward_hook(lambda module, inputs, output: _check_output("vae.output", output))
    print("[debug] registered VAE output hook", flush=True)


def _step_callback(pipe, step: int, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents")
    if torch.is_tensor(latents):
        _log_stats(f"step[{step}].latents", latents)
        _report(f"step[{step}].latents", latents)
    return callback_kwargs


if __name__ == "__main__":
    precision = get_precision()
    if precision != "fp4":
        print(f"Qwen Image 2.1 nunchaku inference currently requires nvfp4/Blackwell; detected {precision!r}.")
        sys.exit(0)

    model_path = os.environ.get("QWEN_IMAGE_21_MODEL", "Qwen/Qwen-Image-2.1")
    prompt = os.environ.get(
        "QWEN_IMAGE_21_PROMPT",
        "a red apple on a wooden table, soft natural light, highly detailed",
    )
    output_path = os.environ.get("QWEN_IMAGE_21_OUTPUT", "qwen-image-2.1-nunchaku.png")
    num_steps = int(os.environ.get("QWEN_IMAGE_21_STEPS", "25"))
    use_kv_cache = os.environ.get("QWEN_IMAGE_21_USE_KV_CACHE", "1").strip().lower() in ("1", "true", "yes")
    output_type = os.environ.get("QWEN_IMAGE_21_OUTPUT_TYPE", "pil")
    sequential_offload = os.environ.get("QWEN_IMAGE_21_SEQUENTIAL_OFFLOAD", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if num_steps < 2:
        print(
            "[warn] QwenImage21Pipeline with FlowMatchEulerDiscreteScheduler uses "
            "sigmas=linspace(1, 1/num_steps, num_steps); num_steps=1 makes "
            "stretch_shift_to_terminal divide by zero and produces NaN timesteps. "
            "Use QWEN_IMAGE_21_STEPS>=2 for diagnostics.",
            flush=True,
        )

    print(
        f"[debug] model={model_path} ckpt={ckpt_path} steps={num_steps} "
        f"kv_cache={use_kv_cache} output_type={output_type} sequential_offload={sequential_offload}",
        flush=True,
    )

    transformer = NunchakuQwenImage21Transformer2DModel.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device="cpu" if sequential_offload else "cuda",
    )
    pipe = QwenImage21Pipeline.from_pretrained(
        model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    if sequential_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    _register_transformer_hooks(pipe)
    if output_type != "latent":
        _register_vae_hooks(pipe)

    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=num_steps,
        use_kv_cache=use_kv_cache,
        output_type=output_type,
        generator=torch.Generator().manual_seed(12345),
        callback_on_step_end=_step_callback,
    )

    output = result.images
    if output_type == "latent":
        if isinstance(output, (list, tuple)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError(f"expected latent tensor, got {type(output).__name__}")
        _log_stats("pipeline.output.latents", output)
        _report("pipeline.output.latents", output)
        latent_path = os.environ.get(
            "QWEN_IMAGE_21_LATENT_OUTPUT",
            output_path.rsplit(".", 1)[0] + ".pt",
        )
        torch.save(output, latent_path)
        print(f"saved latents {latent_path}")
    else:
        image = output[0] if isinstance(output, (list, tuple)) else output
        image.save(output_path)
        print(f"saved {output_path}")
