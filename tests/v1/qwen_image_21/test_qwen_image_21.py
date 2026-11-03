"""Qwen Image 2.1 nunchaku inference tests (Blackwell/nvfp4 only)."""

import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import QwenImage21Pipeline
from diffusers.utils import load_image
from PIL import Image

from nunchaku import NunchakuQwenImage21Transformer2DModel
from nunchaku.utils import get_precision

from ...utils import already_generate, compute_lpips
from ..utils import run_pipeline

precision = get_precision()
torch_dtype = torch.bfloat16
dtype_str = "bf16"

rank = int(os.environ.get("QWEN_IMAGE_21_RANK", "128"))
num_inference_steps = int(os.environ.get("QWEN_IMAGE_21_STEPS", "25"))
max_lpips = float(os.environ.get("QWEN_IMAGE_21_MAX_LPIPS", "0.5"))

ckpt_path = os.environ.get("QWEN_IMAGE_21_CKPT", "")
model_id = os.environ.get("QWEN_IMAGE_21_MODEL", "Qwen/Qwen-Image-2.1")

pytestmark = [
    pytest.mark.skipif(precision != "fp4", reason="Qwen Image 2.1 tests currently require nvfp4/Blackwell"),
    pytest.mark.skipif(
        not ckpt_path or not Path(ckpt_path).is_file(),
        reason="set QWEN_IMAGE_21_CKPT to a quantized Qwen Image 2.1 safetensors file",
    ),
]

t2i_dataset = [
    {
        "prompt": "a red apple on a wooden table, soft natural light, highly detailed",
        "filename": "apple",
    },
    {
        "prompt": "a lighthouse on a rocky coast at sunset, dramatic clouds, cinematic",
        "filename": "lighthouse",
    },
]


def _edit_dataset() -> list[dict]:
    image_path = os.environ.get("QWEN_IMAGE_21_EDIT_IMAGE", "")
    if image_path and Path(image_path).is_file():
        image = Image.open(image_path).convert("RGB")
    else:
        image = load_image(
            "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/neon_sign.png"
        ).convert("RGB")
    return [
        {
            "prompt": "change the text on the sign to read 'Qwen Image 2.1'",
            "filename": "neon_sign",
            "image": image,
        }
    ]


ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))


def _ref_dir(name: str) -> Path:
    return Path(ref_root) / f"qwen-image-2.1-{name}" / dtype_str / f"t{num_inference_steps}"


def _nunchaku_dir(name: str) -> Path:
    return (
        Path("test_results")
        / "nunchaku"
        / f"qwen-image-2.1-{name}"
        / f"{precision}_r{rank}-{dtype_str}"
        / f"t{num_inference_steps}-bs1"
    )


def _generate_reference(name: str, dataset: list[dict], forward_kwargs: dict) -> Path:
    save_dir = _ref_dir(name)
    if already_generate(save_dir, len(dataset)):
        return save_dir

    pipe = QwenImage21Pipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.enable_model_cpu_offload()
    run_pipeline(
        dataset=dataset,
        batch_size=1,
        pipeline=pipe,
        save_dir=save_dir,
        forward_kwargs=forward_kwargs,
    )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return save_dir


def _output_tensor(output):
    if isinstance(output, (tuple, list)):
        if not output:
            return None
        output = output[0]
    sample = getattr(output, "sample", None)
    if sample is not None:
        output = sample
    return output if torch.is_tensor(output) else None


class _FiniteChecker:
    """Record the first non-finite transformer/VAE output without noisy logging."""

    def __init__(self, pipe) -> None:
        self.bad = None
        self.handles = []
        targets = [("transformer", pipe.transformer), ("vae", pipe.vae)]
        for index, block in enumerate(pipe.transformer.transformer_blocks):
            targets.append((f"block[{index}]", block))
            if hasattr(block.attn, "fused_module"):
                targets.append((f"block[{index}].fused_qkv", block.attn.fused_module))
        for name, module in targets:
            if module is not None:
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(module, inputs, output) -> None:
            if self.bad is not None:
                return
            tensor = _output_tensor(output)
            if tensor is None or not tensor.is_floating_point():
                return
            finite = torch.isfinite(tensor)
            if not finite.all():
                self.bad = (
                    name,
                    int(torch.isnan(tensor).sum().item()),
                    int(torch.isinf(tensor).sum().item()),
                    tuple(tensor.shape),
                )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _run_nunchaku(name: str, dataset: list[dict], forward_kwargs: dict) -> Path:
    save_dir = _nunchaku_dir(name)
    sequential_offload = os.environ.get("QWEN_IMAGE_21_SEQUENTIAL_OFFLOAD", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    transformer = NunchakuQwenImage21Transformer2DModel.from_pretrained(
        ckpt_path,
        torch_dtype=torch_dtype,
        device="cpu" if sequential_offload else "cuda",
    )
    assert torch.isfinite(transformer.time_text_embed.time_proj.freqs).all(), "time_proj.freqs is not initialized"
    assert all(
        torch.isfinite(freq.real).all() and torch.isfinite(freq.imag).all() for freq in transformer.pos_embed.freqs
    ), "pos_embed.freqs is not initialized"

    pipe = QwenImage21Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )
    if sequential_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    checker = _FiniteChecker(pipe)
    try:
        run_pipeline(
            dataset=dataset,
            batch_size=1,
            pipeline=pipe,
            save_dir=save_dir,
            forward_kwargs=forward_kwargs,
        )
    finally:
        checker.close()
    assert checker.bad is None, f"non-finite output at {checker.bad}"

    del transformer
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return save_dir


def test_qwen_image_21_text_to_image():
    forward_kwargs = {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": num_inference_steps,
        "use_kv_cache": True,
    }
    ref_dir = _generate_reference("t2i", t2i_dataset, forward_kwargs)
    gen_dir = _run_nunchaku("t2i", t2i_dataset, forward_kwargs)

    lpips = compute_lpips(ref_dir, gen_dir)
    print(f"qwen-image-2.1 t2i lpips={lpips}")
    assert lpips < max_lpips


def test_qwen_image_21_edit():
    forward_kwargs = {
        "num_inference_steps": num_inference_steps,
        "use_kv_cache": True,
    }
    dataset = _edit_dataset()
    ref_dir = _generate_reference("edit", dataset, forward_kwargs)
    gen_dir = _run_nunchaku("edit", dataset, forward_kwargs)

    lpips = compute_lpips(ref_dir, gen_dir)
    print(f"qwen-image-2.1 edit lpips={lpips}")
    assert lpips < max_lpips
