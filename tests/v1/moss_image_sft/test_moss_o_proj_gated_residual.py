"""Compare the MOSS o_proj gated-residual epilogue with the eager path."""

import os
from pathlib import Path

import pytest
import torch

from nunchaku import NunchakuMOSSImageTransformer
from nunchaku.utils import get_precision

ckpt_path = os.environ.get("MOSS_IMAGE_CKPT", "")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the W4A4 fused kernel"),
    pytest.mark.skipif(
        not ckpt_path or not Path(ckpt_path).is_file(),
        reason="set MOSS_IMAGE_CKPT to a quantized MOSS-Image-SFT safetensors file",
    ),
]


def test_o_proj_gated_residual_matches_eager():
    if get_precision() != "fp4":
        pytest.skip("MOSS gated residual fusion currently requires nvfp4/Blackwell")

    seq_len = int(os.environ.get("MOSS_GATED_RESIDUAL_SEQ_LEN", "256"))
    bf16_eps = torch.finfo(torch.bfloat16).eps
    atol = float(os.environ.get("MOSS_GATED_RESIDUAL_ATOL", str(4 * bf16_eps)))
    rtol = float(os.environ.get("MOSS_GATED_RESIDUAL_RTOL", str(2 * bf16_eps)))

    torch.manual_seed(0)
    transformer = NunchakuMOSSImageTransformer.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    block = transformer.layers[0]
    dim = transformer.config.dim

    attn = torch.randn(1, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(1, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, 1, dim, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        eager = residual + block.proj["o"](attn) * gate
        fused = block.proj["o"].forward_gated_residual(attn, residual=residual, gate=gate)

    diff = (fused.float() - eager.float()).abs()
    print(
        f"max_abs_diff={diff.max().item():.6f}, mean_abs_diff={diff.mean().item():.6f}, "
        f"max_abs_value={eager.float().abs().max().item():.6f}"
    )
    torch.testing.assert_close(fused.float(), eager.float(), atol=atol, rtol=rtol)
