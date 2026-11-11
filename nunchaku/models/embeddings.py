"""
Embedding layers for Nunchaku.
"""

import diffusers
import torch
from packaging.version import Version
from torch import nn


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """
    Rotary positional embedding function.
    Copied from https://github.com/huggingface/diffusers/blob/c9ff360966327ace3faad3807dc871a4e5447501/src/diffusers/models/transformers/transformer_flux.py#L38

    Parameters
    ----------
    pos : torch.Tensor, shape (..., n), dtype int
        Position indices.
    dim : int
        Embedding dimension (must be even).
    theta : int
        Rotary base.

    Returns
    -------
    out : torch.Tensor, shape (B, M, D//2, 1, 2), dtype float32
        Rotary embedding tensor.

    Notes
    -----
    - B: batch size
    - M: sequence length
    - D: embedding dimension
    """
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)

    # Sin/cos representation for rotary embedding
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    stacked_out = torch.stack([sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 1, 2)

    return out.float()


class NunchakuFluxPosEmbed(nn.Module):
    """
    Nunchaku multi-dimensional rotary embedding module for FLUX.
    Adapted from https://github.com/huggingface/diffusers/blob/c9ff360966327ace3faad3807dc871a4e5447501/src/diffusers/models/transformers/transformer_flux.py#L55

    Parameters
    ----------
    dim : int
        Embedding dimension.
    theta : int
        Rotary base.
    axes_dim : list of int
        Dimension for each spatial axis.
    """

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super(NunchakuFluxPosEmbed, self).__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary embeddings for multi-dimensional positions.

        Parameters
        ----------
        ids : torch.Tensor, shape (..., n_axes), dtype int
            Position indices.

        Returns
        -------
        out : torch.Tensor, shape (B, 1, ...), dtype float32
            Rotary embedding tensor.

        Notes
        -----
        - B: batch size
        - n_axes: number of spatial axes
        """
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def pack_rotemb_for_half_split(rotemb: torch.Tensor) -> torch.Tensor:
    """Pack rotary embeddings for half-split RoPE with an explicit index map.

    This is the half-split counterpart of :func:`pack_rotemb`. It accepts the same
    logical input layout ``[B, M, D/2, 1, 2]`` (``sin, cos``), but arranges the
    values for a kernel that rotates ``(j, j+64)`` within a 128-wide head. The
    caller must pad ``M`` to a multiple of 256 first (for example with
    ``pad_tensor(rotemb, 256, dim=1)``).

    The output is ``[B, M, D]`` float32. For each lane/tile ``n = 0..3`` the
    four ``float4`` slots ``4*n + {0,1,2,3}`` contain, respectively::

        A0: (sin_j,     cos_j,     sin_j(r+8),     cos_j(r+8))
        A1: (sin_{j+1}, cos_{j+1}, sin_{j+1}(r+8), cos_{j+1}(r+8))
        B0: (sin_{j+8}, cos_{j+8}, sin_{j+8}(r+8), cos_{j+8}(r+8))
        B1: (sin_{j+9}, cos_{j+9}, sin_{j+9}(r+8), cos_{j+9}(r+8))

    where ``j = n*16 + 2*(lane % 4)``. Tile ``n+4`` reuses the same four slots
    in the half-split epilogue, because ``j+64`` has the same angle as ``j``.

    This function currently supports ``B == 1`` and ``D == 128``; ``M`` must
    already be aligned to 256.
    """
    assert rotemb.dtype == torch.float32
    assert rotemb.ndim == 5 and rotemb.shape[3] == 1 and rotemb.shape[4] == 2
    B, M, half_dim, _, _ = rotemb.shape
    D = half_dim * 2
    assert B == 1, "pack_rotemb_for_half_split currently supports B == 1"
    assert D == 128, f"half-split pack expects D == 128, got {D}"
    assert M % 256 == 0, f"M must be aligned to 256 before packing, got {M}"

    sin = rotemb[0, :, :, 0, 0]  # [M, 64]
    cos = rotemb[0, :, :, 0, 1]  # [M, 64]
    out = torch.empty((1, M, D), dtype=rotemb.dtype, device=rotemb.device)
    flat = out.reshape(-1)

    # Constants from GEMMConfig_W4A4 / EpilogueRMSNormRope.
    num_warps = 8
    warp_m_tiles = 2
    rotemb_tiles = 16
    warp_size = 32
    block_m = 256

    device = rotemb.device
    bm = torch.arange(M // block_m, device=device).view(-1, 1, 1, 1, 1, 1)
    warp = torch.arange(num_warps, device=device).view(1, -1, 1, 1, 1, 1)
    m_tile = torch.arange(warp_m_tiles, device=device).view(1, 1, -1, 1, 1, 1)
    lane = torch.arange(warp_size, device=device).view(1, 1, 1, -1, 1, 1)
    n = torch.arange(4, device=device).view(1, 1, 1, 1, -1, 1)
    k = torch.arange(4, device=device).view(1, 1, 1, 1, 1, -1)

    row = lane // 4  # 0..7
    col = (lane % 4) * 2  # 0,2,4,6
    angle_offsets = torch.tensor([0, 1, 8, 9], device=device, dtype=torch.long).view(1, 1, 1, 1, 1, -1)
    angle = n * 16 + col + angle_offsets  # [1,1,1,32,4,4]

    token = bm * block_m + warp * 32 + m_tile * 16 + row
    token_next = token + 8
    slot = n * 4 + k  # 0..15, matching WARP_N_ROTEMB_TILES
    float4_index = (((bm * num_warps + warp) * warp_m_tiles + m_tile) * rotemb_tiles + slot) * warp_size + lane
    base = float4_index * 4

    flat[base + 0] = sin[token, angle]
    flat[base + 1] = cos[token, angle]
    flat[base + 2] = sin[token_next, angle]
    flat[base + 3] = cos[token_next, angle]
    return out


def pack_rotemb(rotemb: torch.Tensor) -> torch.Tensor:
    """
    Pack rotary embeddings for efficient CUDA computation.

    Parameters
    ----------
    rotemb : torch.Tensor, shape (B, M, D//2, 1, 2), dtype float32
        Rotary embedding tensor.

    Returns
    -------
    packed : torch.Tensor, shape (B, M, D), dtype float32
        Packed rotary embedding tensor.

    Notes
    -----
    - B: batch size
    - M: sequence length (must be divisible by 16)
    - D: embedding dimension (must be divisible by 8)
    """
    assert rotemb.dtype == torch.float32
    B = rotemb.shape[0]
    M = rotemb.shape[1]
    D = rotemb.shape[2] * 2
    assert rotemb.shape == (B, M, D // 2, 1, 2)
    assert M % 16 == 0
    assert D % 8 == 0
    rotemb = rotemb.reshape(B, M // 16, 16, D // 8, 8)
    rotemb = rotemb.permute(0, 1, 3, 2, 4)
    # 16*8 pack, FP32 accumulator (C) format
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-c
    ##########################################|--M--|--D--|
    ##########################################|-3--4--5--6|
    ##########################################  :  :  :  :
    rotemb = rotemb.reshape(*rotemb.shape[0:3], 2, 8, 4, 2)
    rotemb = rotemb.permute(0, 1, 2, 4, 5, 3, 6)
    rotemb = rotemb.contiguous()
    rotemb = rotemb.view(B, M, D)
    return rotemb
