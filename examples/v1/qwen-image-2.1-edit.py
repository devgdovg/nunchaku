import os
import sys

# See qwen-image-2.1.py: keep the example runnable by CI only when a local
# quantized checkpoint and edit image are explicitly provided.
ckpt_path = os.environ.get("QWEN_IMAGE_21_CKPT", "")
edit_image_path = os.environ.get("QWEN_IMAGE_21_EDIT_IMAGE", "")
if not ckpt_path or not os.path.isfile(ckpt_path) or not edit_image_path or not os.path.isfile(edit_image_path):
    print(
        "Set QWEN_IMAGE_21_CKPT and QWEN_IMAGE_21_EDIT_IMAGE to existing files to run "
        "the Qwen Image 2.1 editing example; skipping."
    )
    sys.exit(0)

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

from nunchaku import NunchakuQwenImage21Transformer2DModel
from nunchaku.utils import get_precision

if __name__ == "__main__":
    precision = get_precision()
    if precision != "fp4":
        print(f"Qwen Image 2.1 nunchaku inference currently requires nvfp4/Blackwell; detected {precision!r}.")
        sys.exit(0)

    model_path = os.environ.get("QWEN_IMAGE_21_MODEL", "Qwen/Qwen-Image-2.1")
    prompt = os.environ.get(
        "QWEN_IMAGE_21_EDIT_PROMPT",
        "change the text on the sign to read 'Qwen Image 2.1'",
    )
    output_path = os.environ.get("QWEN_IMAGE_21_EDIT_OUTPUT", "qwen-image-2.1-edit-nunchaku.png")
    image = Image.open(edit_image_path).convert("RGB")

    sequential_offload = os.environ.get("QWEN_IMAGE_21_SEQUENTIAL_OFFLOAD", "0").strip().lower() in (
        "1",
        "true",
        "yes",
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

    result = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=25,
        use_kv_cache=True,
        generator=torch.Generator().manual_seed(12345),
    ).images[0]
    result.save(output_path)
    print(f"saved {output_path}")
