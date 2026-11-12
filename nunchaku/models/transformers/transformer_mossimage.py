"""MOSS-Image-SFT 量化推理实现。

量化 ckpt 中主 block 的 qkv/o/ffn_up_gate/ffn_down 以及 text block 的
qkv/proj/ffn_up_gate/ffn_down 已融合并量化。默认只把线性层替换为
:class:`SVDQW4A4Linear`，norm_q/norm_k、RoPE、SDPA 保持 PyTorch 实现；
``fused_rope=True`` 时主 block 和 text block 都使用 half-split fused kernel
完成 qkv + q/k RMSNorm + RoPE；``rope_layout`` 可选 ``"packed"``（默认，
MMA lane 对齐布局）或 ``"simple"``（stage-1 sin|cos 布局，用于对比）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from huggingface_hub import utils
from torch import nn
from torch.nn import functional as F

from ...ops.gemm import svdq_gemm_w4a4_cuda
from ...ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from ...utils import get_precision, pad_tensor
from ..embeddings import pack_rotemb_for_half_split
from ..linear import SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key

__all__ = ["NunchakuMOSSImageTransformer"]


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _validate_skip_layers(layers, depth, name="skip_layers"):
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f"{name} must contain integer block indices in [0, {depth}).")
    return frozenset(layers)


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


def _attention_pre_roped(q, k, v, heads):
    """Attention for q/k already normalized and RoPE-rotated by the fused kernel."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = q.reshape(shape).transpose(1, 2)
    k = k.reshape(shape).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class NunchakuMOSSImageRopeCache:
    """Half-split rotemb cache for the fused qkv kernel.

    ``layout="simple"`` keeps the stage-1 ``[1, padded_S, HEAD_DIM]`` layout
    (first half sin, second half cos). ``layout="packed"`` first pads the
    logical ``[1, S, HEAD_DIM/2, 1, 2]`` tensor to 256 tokens and then applies
    :func:`pack_rotemb_for_half_split`. The cache is cleared at the start of
    every transformer forward.
    """

    def __init__(self, layout: str = "packed") -> None:
        if layout not in ("simple", "packed"):
            raise ValueError(f"rope layout must be 'simple' or 'packed', got {layout!r}")
        self.layout = layout
        self.cache: dict[tuple[object, ...], torch.Tensor] = {}

    def clear(self) -> None:
        self.cache.clear()

    def pack(self, rope: torch.Tensor) -> torch.Tensor:
        key = (rope.data_ptr(), tuple(rope.shape))
        packed = self.cache.get(key)
        if packed is None:
            rope = rope.float().contiguous()
            if self.layout == "packed":
                sin, cos = rope.chunk(2, dim=-1)
                rotemb = torch.stack((sin, cos), dim=-1).unsqueeze(-2)  # [1, S, HEAD_DIM/2, 1, 2]
                packed = pack_rotemb_for_half_split(pad_tensor(rotemb, 256, dim=1))
            else:
                packed = pad_tensor(rope, 256, dim=1)
            self.cache[key] = packed
        return packed


class _TextBlock(nn.Module):
    """官方 text block 结构（量化前，仅用于骨架构建）。"""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    """官方主 block 结构（量化前，仅用于骨架构建）。"""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device):
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(ModelMixin, ConfigMixin):
    """官方 MOSS-Image transformer 结构（未量化）。"""

    @register_to_config
    def __init__(
        self,
        dim=3072,
        ffn_dim=8192,
        num_heads=24,
        num_layers=42,
        text_dim=4096,
        freq_dim=256,
        latent_channels=128,
        rope_base=10000.0,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        return_dict=True,
        skip_layers=None,
    ):
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width :])
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index : index + 1])
            x = self.final_norm_video(x[:, : height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


def _run_fused_qkv(
    y: torch.Tensor,
    rope: torch.Tensor,
    linear: SVDQW4A4Linear,
    q_norm: nn.Module,
    k_norm: nn.Module,
    rope_cache: NunchakuMOSSImageRopeCache,
    rope_mode: int,
) -> torch.Tensor:
    """Run qkv projection + q/k RMSNorm + half-split RoPE in one kernel."""
    batch, seq_len, channels = y.shape
    assert batch == 1, "MOSS fused RoPE currently expects one sample per block call"
    flat = y.reshape(batch * seq_len, channels)
    quantized, ascales, lora_act = svdq_quantize_w4a4_act_fuse_lora_cuda(
        flat,
        lora_down=linear.proj_down,
        smooth=linear.smooth_factor,
        fp4=linear.precision == "nvfp4",
        pad_size=256,
    )
    out = torch.empty(flat.shape[0], linear.out_features, dtype=flat.dtype, device=flat.device)
    svdq_gemm_w4a4_cuda(
        act=quantized,
        wgt=linear.qweight,
        out=out,
        ascales=ascales,
        wscales=linear.wscales,
        lora_act_in=lora_act,
        lora_up=linear.proj_up,
        bias=linear.bias,
        fp4=linear.precision == "nvfp4",
        alpha=1.0 if linear.precision == "nvfp4" else None,
        wcscales=linear.wcscales,
        norm_q=q_norm.weight,
        norm_k=k_norm.weight,
        rotary_emb=rope_cache.pack(rope),
        rope_mode=rope_mode,
    )
    return out.view(batch, seq_len, -1)


class NunchakuMOSSImageBlock(nn.Module):
    """量化后的主 block：qkv/o/ffn_up_gate/ffn_down 使用 SVDQW4A4Linear。"""

    def __init__(
        self,
        orig: _Block,
        *,
        precision: str,
        rank: int,
        use_fused_rope: bool = False,
        rope_cache: NunchakuMOSSImageRopeCache | None = None,
        rope_mode: int = 2,
    ) -> None:
        super().__init__()
        self.heads = orig.heads
        self.norm1, self.norm2 = orig.norm1, orig.norm2
        self.q_norm, self.k_norm = orig.q_norm, orig.k_norm
        self.use_fused_rope = use_fused_rope
        self.rope_cache = rope_cache
        self.rope_mode = rope_mode
        with torch.device("meta"):
            qkv = fuse_linears([orig.proj["q"], orig.proj["k"], orig.proj["v"]])
        self.proj = nn.ModuleDict(
            {
                "qkv": SVDQW4A4Linear.from_linear(qkv, precision=precision, rank=rank),
                "o": SVDQW4A4Linear.from_linear(orig.proj["o"], precision=precision, rank=rank),
                "ffn_up_gate": SVDQW4A4Linear.from_linear(orig.proj["ffn_up_gate"], precision=precision, rank=rank),
                "ffn_down": SVDQW4A4Linear.from_linear(orig.proj["ffn_down"], precision=precision, rank=rank),
            }
        )

    def _qkv_fused(self, y: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        assert self.rope_cache is not None, "fused RoPE requires a rope cache"
        return _run_fused_qkv(
            y, rope, self.proj["qkv"], self.q_norm, self.k_norm, self.rope_cache, self.rope_mode
        )

    def _qkv_eager(self, y: torch.Tensor) -> torch.Tensor:
        qkv = self.proj["qkv"](y).chunk(3, -1)
        q, k, v = qkv
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        return torch.cat((q, k, v), dim=-1)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["qkv"].proj_up.dtype
        y = y.to(dtype)
        qkv = self._qkv_fused(y, rope) if self.use_fused_rope else self._qkv_eager(y)
        q, k, v = qkv.chunk(3, -1)
        attn = (
            _attention_pre_roped(q, k, v, self.heads) if self.use_fused_rope else _attention(q, k, v, rope, self.heads)
        )
        x = x + self.proj["o"](attn) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


class NunchakuMOSSImageTextBlock(nn.Module):
    """量化后的 text block；qkv 可选 fused norm+RoPE，ffn_up/gate 已融合。"""

    def __init__(
        self,
        orig: _TextBlock,
        *,
        precision: str,
        rank: int,
        use_fused_rope: bool = False,
        rope_cache: NunchakuMOSSImageRopeCache | None = None,
        rope_mode: int = 2,
    ) -> None:
        super().__init__()
        self.heads = orig.heads
        self.norm1, self.norm2 = orig.norm1, orig.norm2
        self.q_norm, self.k_norm = orig.q_norm, orig.k_norm
        self.use_fused_rope = use_fused_rope
        self.rope_cache = rope_cache
        self.rope_mode = rope_mode
        self.qkv = SVDQW4A4Linear.from_linear(orig.qkv, precision=precision, rank=rank)
        self.proj = SVDQW4A4Linear.from_linear(orig.proj, precision=precision, rank=rank)
        with torch.device("meta"):
            ffn_up_gate = fuse_linears([orig.ffn_up, orig.ffn_gate])
        self.ffn_up_gate = SVDQW4A4Linear.from_linear(ffn_up_gate, precision=precision, rank=rank)
        self.ffn_down = SVDQW4A4Linear.from_linear(orig.ffn_down, precision=precision, rank=rank)

    def _qkv_fused(self, y: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        assert self.rope_cache is not None, "fused RoPE requires a rope cache"
        return _run_fused_qkv(y, rope, self.qkv, self.q_norm, self.k_norm, self.rope_cache, self.rope_mode)

    def _qkv_eager(self, y: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(y).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        return torch.cat((q, k, v), dim=-1)

    def forward(self, x, rope):
        y = self.norm1(x)
        qkv = self._qkv_fused(y, rope) if self.use_fused_rope else self._qkv_eager(y)
        q, k, v = qkv.chunk(3, -1)
        attn = (
            _attention_pre_roped(q, k, v, self.heads) if self.use_fused_rope else _attention(q, k, v, rope, self.heads)
        )
        x = x + self.proj(attn)
        y = self.norm2(x)
        hidden, gate = self.ffn_up_gate(y).chunk(2, -1)
        return x + self.ffn_down(situ(hidden, gate))


class NunchakuMOSSImageTransformer(MOSSImageTransformer, NunchakuModelLoaderMixin):
    """Nunchaku 量化版 MOSS-Image transformer。"""

    def _patch_model(
        self, precision: str = "nvfp4", rank: int = 128, fused_rope: bool = False, rope_layout: str = "packed"
    ):
        if rope_layout not in ("simple", "packed"):
            raise ValueError(f"rope_layout must be 'simple' or 'packed', got {rope_layout!r}")
        self._rope_cache = NunchakuMOSSImageRopeCache(layout=rope_layout) if fused_rope else None
        rope_mode = 1 if rope_layout == "simple" else 2
        if fused_rope:
            self.register_forward_pre_hook(self._clear_rope_cache)
        for index, block in enumerate(self.layers):
            self.layers[index] = NunchakuMOSSImageBlock(
                block,
                precision=precision,
                rank=rank,
                use_fused_rope=fused_rope,
                rope_cache=self._rope_cache,
                rope_mode=rope_mode,
            )
        for index, block in enumerate(self.text_embedder.blocks):
            self.text_embedder.blocks[index] = NunchakuMOSSImageTextBlock(
                block,
                precision=precision,
                rank=rank,
                use_fused_rope=fused_rope,
                rope_cache=self._rope_cache,
                rope_mode=rope_mode,
            )
        return self

    def _clear_rope_cache(self, module: nn.Module, args: tuple[Any, ...]) -> None:
        if self._rope_cache is not None:
            self._rope_cache.clear()

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: Any):
        device = kwargs.get("device", "cpu")
        fused_rope = bool(kwargs.pop("fused_rope", False))
        rope_layout = str(kwargs.pop("rope_layout", "packed")).lower()
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"

        transformer, state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        rank = int(quantization_config.get("rank", 32))

        precision = get_precision()
        if precision != "fp4":
            raise NotImplementedError(
                f"MOSS-Image-SFT inference currently supports nvfp4/Blackwell only; detected precision={precision!r}"
            )
        precision = "nvfp4"

        transformer = transformer.to(torch_dtype)
        transformer._patch_model(
            precision=precision, rank=rank, fused_rope=fused_rope, rope_layout=rope_layout
        )
        transformer = transformer.to_empty(device=device)
        if "rope.bands" not in state_dict:
            state_dict["rope.bands"] = transformer.rope._default_bands(transformer.rope.bands.device)
        patch_scale_key(transformer, state_dict)
        transformer.load_state_dict(state_dict)
        return transformer.eval()
