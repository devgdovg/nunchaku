"""Compare the MOSS text-block fused qkv kernel with the eager path.

Text blocks have a bias in qkv (unlike the main blocks), so this test also
covers the fused kernel's bias path and the sliced text RoPE.
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


def test_text_qkv_fused_matches_eager():
    if get_precision() != "fp4":
        pytest.skip("MOSS fused qkv currently requires nvfp4/Blackwell")

    seq_len = int(os.environ.get("MOSS_TEXT_QKV_FUSED_SEQ_LEN", "64"))
    bf16_eps = torch.finfo(torch.bfloat16).eps
    qk_atol = float(os.environ.get("MOSS_TEXT_QKV_FUSED_QK_ATOL", str(4 * bf16_eps)))
    qk_rtol = float(os.environ.get("MOSS_TEXT_QKV_FUSED_QK_RTOL", str(2 * bf16_eps)))
    v_atol = float(os.environ.get("MOSS_TEXT_QKV_FUSED_V_ATOL", str(2 * bf16_eps)))
    v_rtol = float(os.environ.get("MOSS_TEXT_QKV_FUSED_V_RTOL", str(bf16_eps)))

    torch.manual_seed(0)
    transformer = NunchakuMOSSImageTransformer.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    block = transformer.text_embedder.blocks[0]
    heads = transformer.config.num_heads
    dim = transformer.config.dim
    head_dim = dim // heads

    x = torch.randn(1, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.rand(1, seq_len, head_dim // 2, device="cuda", dtype=torch.float32)
    cos = torch.rand(1, seq_len, head_dim // 2, device="cuda", dtype=torch.float32)
    random_rope = torch.cat((sin, cos), dim=-1)
    identity_rope = torch.cat(
        (
            torch.zeros(1, seq_len, head_dim // 2, device="cuda", dtype=torch.float32),
            torch.ones(1, seq_len, head_dim // 2, device="cuda", dtype=torch.float32),
        ),
        dim=-1,
    )

    def _qkv_eager(y: torch.Tensor) -> torch.Tensor:
        q, k, v = block.qkv(y).chunk(3, -1)
        q = block.q_norm(q.unflatten(-1, (heads, -1))).flatten(2)
        k = block.k_norm(k.unflatten(-1, (heads, -1))).flatten(2)
        return torch.cat((q, k, v), dim=-1)

    def _diff(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        actual_f, expected_f = actual.float(), expected.float()
        diff = (actual_f - expected_f).abs()
        abs_value = expected_f.abs()
        rel = diff / abs_value.clamp_min(1e-12)
        print(
            f"{name}: max_abs_diff={diff.max().item():.6f}, mean_abs_diff={diff.mean().item():.6f}, "
            f"max_abs_value={abs_value.max().item():.6f}, max_rel_diff={rel.max().item():.6f}"
        )

    def _compare(label: str, rope: torch.Tensor) -> None:
        with torch.no_grad():
            y = block.norm1(x)
            qkv_eager = _qkv_eager(y)
            qkv_fused = block._qkv_fused(y, rope)

        qe, ke, ve = qkv_eager.chunk(3, dim=-1)
        qf, kf, vf = qkv_fused.chunk(3, dim=-1)

        rope_eager = rope.to(dtype=x.dtype)
        qe = _rotate(qe.reshape(1, seq_len, heads, head_dim), rope_eager).flatten(2)
        ke = _rotate(ke.reshape(1, seq_len, heads, head_dim), rope_eager).flatten(2)

        print(f"[{label}]")
        _diff("q", qf, qe)
        _diff("k", kf, ke)
        _diff("v", vf, ve)

        torch.testing.assert_close(qf.float(), qe.float(), atol=qk_atol, rtol=qk_rtol)
        torch.testing.assert_close(kf.float(), ke.float(), atol=qk_atol, rtol=qk_rtol)
        torch.testing.assert_close(vf.float(), ve.float(), atol=v_atol, rtol=v_rtol)

    _compare("random-rope", random_rope)
    _compare("identity-rope", identity_rope)
