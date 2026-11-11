"""Compare the MOSS fused qkv kernel with the eager qkv + norm + RoPE path.

This is a block-level GPU diagnostic: it loads the quantized transformer and
compares q/k/v outputs for one main block. It does not need the text encoder,
VAE, or a full pipeline.
"""

import os
from pathlib import Path

import pytest
import torch

from nunchaku import NunchakuMOSSImageTransformer
from nunchaku.models.transformers.transformer_mossimage import _rotate
from nunchaku.utils import get_precision

ckpt_path = os.environ.get("MOSS_IMAGE_CKPT", "")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the W4A4 fused kernel"),
    pytest.mark.skipif(
        not ckpt_path or not Path(ckpt_path).is_file(),
        reason="set MOSS_IMAGE_CKPT to a quantized MOSS-Image-SFT safetensors file",
    ),
]


def test_qkv_fused_matches_eager():
    if get_precision() != "fp4":
        pytest.skip("MOSS fused qkv currently requires nvfp4/Blackwell")

    seq_len = int(os.environ.get("MOSS_QKV_FUSED_SEQ_LEN", "256"))
    atol = float(os.environ.get("MOSS_QKV_FUSED_ATOL", "5e-2"))
    rtol = float(os.environ.get("MOSS_QKV_FUSED_RTOL", "5e-2"))

    torch.manual_seed(0)
    transformer = NunchakuMOSSImageTransformer.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
        fused_rope=True,
    )
    block = transformer.layers[0]
    heads = transformer.config.num_heads
    dim = transformer.config.dim
    head_dim = dim // heads

    y = torch.randn(1, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    # Simple stage-1 layout: first half sin, second half cos, already padded to 256.
    sin = torch.rand(1, seq_len, head_dim // 2, device="cuda", dtype=torch.float32)
    cos = torch.rand(1, seq_len, head_dim // 2, device="cuda", dtype=torch.float32)
    rope = torch.cat((sin, cos), dim=-1)

    with torch.no_grad():
        qkv_eager = block._qkv_eager(y)
        qkv_fused = block._qkv_fused(y, rope)

    qe, ke, ve = qkv_eager.chunk(3, dim=-1)
    qf, kf, vf = qkv_fused.chunk(3, dim=-1)

    # _qkv_eager applies q/k RMSNorm but not RoPE; _qkv_fused applies both.
    rope_eager = rope.to(dtype=y.dtype)
    qe = _rotate(qe.reshape(1, seq_len, heads, head_dim), rope_eager).flatten(2)
    ke = _rotate(ke.reshape(1, seq_len, heads, head_dim), rope_eager).flatten(2)

    def _diff(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        diff = (actual.float() - expected.float()).abs()
        print(f"{name}: max_abs_diff={diff.max().item():.6f}, mean_abs_diff={diff.mean().item():.6f}")

    _diff("q", qf, qe)
    _diff("k", kf, ke)
    _diff("v", vf, ve)

    torch.testing.assert_close(qf.float(), qe.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(kf.float(), ke.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(vf.float(), ve.float(), atol=atol, rtol=rtol)
