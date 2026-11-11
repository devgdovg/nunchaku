"""CPU-only tests for the half-split packed rotemb layout.

``nunchaku.models.embeddings`` has no CUDA dependency, but importing the
``nunchaku`` package itself builds/loads the compiled extension. Load the module
directly so these layout tests can run in a plain CPU environment.
"""

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_embeddings():
    path = Path(__file__).resolve().parents[3] / "nunchaku" / "models" / "embeddings.py"
    spec = importlib.util.spec_from_file_location("nunchaku_embeddings_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embeddings = _load_embeddings()


def _make_rotemb(seq_len: int) -> torch.Tensor:
    token = torch.arange(seq_len)
    angle = torch.arange(64)
    sin = (token[:, None] * 64 + angle[None, :]).float()
    cos = -sin
    return torch.stack((sin, cos), dim=-1).unsqueeze(-2).unsqueeze(0)  # [1, S, 64, 1, 2]


def _reference_float4(rotemb: torch.Tensor, bm: int, warp: int, m_tile: int, lane: int, n: int, k: int):
    sin = rotemb[0, :, :, 0, 0]
    cos = rotemb[0, :, :, 0, 1]
    row = lane // 4
    col = (lane % 4) * 2
    angle = n * 16 + col + (0, 1, 8, 9)[k]
    token = bm * 256 + warp * 32 + m_tile * 16 + row
    token_next = token + 8
    return torch.stack((sin[token, angle], cos[token, angle], sin[token_next, angle], cos[token_next, angle]))


@pytest.mark.parametrize("seq_len", [256, 512])
def test_half_split_pack_matches_reference(seq_len):
    rotemb = _make_rotemb(seq_len)
    packed = embeddings.pack_rotemb_for_half_split(rotemb)
    assert packed.shape == (1, seq_len, 128)
    assert packed.dtype == torch.float32
    flat = packed.reshape(-1).view(-1, 4)

    for bm in range(seq_len // 256):
        for warp in range(8):
            for m_tile in range(2):
                for lane in range(32):
                    for n in range(4):
                        for k in range(4):
                            index = (((bm * 8 + warp) * 2 + m_tile) * 16 + (n * 4 + k)) * 32 + lane
                            expected = _reference_float4(rotemb, bm, warp, m_tile, lane, n, k)
                            assert torch.equal(flat[index], expected), (
                                f"mismatch at bm={bm} warp={warp} m_tile={m_tile} lane={lane} n={n} k={k}"
                            )


def test_half_split_pack_zero_padded_tail():
    seq_len = 100
    rotemb = _make_rotemb(seq_len)
    rotemb = torch.cat([rotemb, torch.zeros(1, 256 - seq_len, 64, 1, 2)], dim=1)
    packed = embeddings.pack_rotemb_for_half_split(rotemb)
    flat = packed.reshape(-1).view(-1, 4)
    for warp in range(8):
        for m_tile in range(2):
            for lane in range(32):
                row = lane // 4
                for n in range(4):
                    for k in range(4):
                        index = ((warp * 2 + m_tile) * 16 + (n * 4 + k)) * 32 + lane
                        token = warp * 32 + m_tile * 16 + row
                        if token < seq_len:
                            expected = _reference_float4(rotemb, 0, warp, m_tile, lane, n, k)
                        else:
                            expected = torch.zeros(4)
                        assert torch.equal(flat[index], expected)


def test_half_split_pack_rejects_unsupported_shapes():
    rotemb = _make_rotemb(256)
    with pytest.raises(AssertionError):
        embeddings.pack_rotemb_for_half_split(rotemb.repeat(2, 1, 1, 1, 1))
    with pytest.raises(AssertionError):
        embeddings.pack_rotemb_for_half_split(_make_rotemb(128))
    with pytest.raises(AssertionError):
        embeddings.pack_rotemb_for_half_split(rotemb[..., :32, :, :])
    with pytest.raises(AssertionError):
        embeddings.pack_rotemb_for_half_split(rotemb.double())
