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

    transformer = NunchakuQwenImage21Transformer2DModel.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    pipe = QwenImage21Pipeline.from_pretrained(
        model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=25,
        use_kv_cache=True,
        generator=torch.Generator().manual_seed(12345),
    ).images[0]
    image.save(output_path)
    print(f"saved {output_path}")
