"""GPU tests for the fused RMSNorm + RoPE epilogue (both rope layouts)."""

import pytest
import torch
from nunchaku._C import ops

from nunchaku.models.embeddings import pack_rotemb

HEAD_DIM = 128
M = 256
N = 3 * HEAD_DIM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the W4A4 epilogue test")


def _reference(
    input_t: torch.Tensor, norm_q: torch.Tensor, norm_k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, mode: int
):
    x = input_t.float()
    q, k, v = x[:, :HEAD_DIM], x[:, HEAD_DIM : 2 * HEAD_DIM], x[:, 2 * HEAD_DIM :]

    def rmsnorm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        coef = torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        return value * coef * weight.float()

    q = rmsnorm(q, norm_q)
    k = rmsnorm(k, norm_k)

    if mode == 0:
        q0, q1 = q[:, 0::2], q[:, 1::2]
        k0, k1 = k[:, 0::2], k[:, 1::2]
        q = torch.stack((q0 * cos - q1 * sin, q1 * cos + q0 * sin), dim=-1).flatten(-2)
        k = torch.stack((k0 * cos - k1 * sin, k1 * cos + k0 * sin), dim=-1).flatten(-2)
    else:
        qa, qb = q[:, : HEAD_DIM // 2], q[:, HEAD_DIM // 2 :]
        ka, kb = k[:, : HEAD_DIM // 2], k[:, HEAD_DIM // 2 :]
        q = torch.cat((qa * cos - qb * sin, qb * cos + qa * sin), dim=-1)
        k = torch.cat((ka * cos - kb * sin, kb * cos + ka * sin), dim=-1)
    return torch.cat((q, k, v), dim=-1)


@pytest.mark.parametrize("mode", [0, 1])
def test_rmsnorm_rope_epilogue(mode):
    torch.manual_seed(0)
    input_t = torch.randn(M, N, device="cuda", dtype=torch.float16)
    output = torch.empty_like(input_t)
    norm_q = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float16)
    norm_k = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float16)

    sin = torch.rand(M, HEAD_DIM // 2, device="cuda", dtype=torch.float32)
    cos = torch.rand(M, HEAD_DIM // 2, device="cuda", dtype=torch.float32)
    if mode == 0:
        logical = torch.stack((sin, cos), dim=-1).unsqueeze(-2).unsqueeze(0)
        rotemb = pack_rotemb(logical)
    else:
        rotemb = torch.cat((sin, cos), dim=-1).unsqueeze(0).float()

    ops.test_rmsnorm_rope(input_t, output, norm_q, norm_k, rotemb, mode)

    expected = _reference(input_t.cpu(), norm_q.cpu(), norm_k.cpu(), sin.cpu(), cos.cpu(), mode)
    torch.testing.assert_close(output.cpu().float(), expected, atol=2e-2, rtol=2e-2)
