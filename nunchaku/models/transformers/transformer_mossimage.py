"""MOSS-Image-SFT 量化推理实现。

量化 ckpt 中主 block 的 qkv/o/ffn_up_gate/ffn_down 以及 text block 的
qkv/proj/ffn_up_gate/ffn_down 已融合并量化；这里只把对应线性层替换为
:class:`SVDQW4A4Linear`，norm_q/norm_k、RoPE、SDPA 保持 PyTorch 实现。
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from huggingface_hub import utils
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from ...utils import get_precision
from ..linear import SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key

__all__ = ["NunchakuMOSSImageTransformer", "load_bf16_text_embedder"]


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


class NunchakuMOSSImageBlock(nn.Module):
    """量化后的主 block：qkv/o/ffn_up_gate/ffn_down 使用 SVDQW4A4Linear。"""

    def __init__(self, orig: _Block, *, precision: str, rank: int) -> None:
        super().__init__()
        self.heads = orig.heads
        self.norm1, self.norm2 = orig.norm1, orig.norm2
        self.q_norm, self.k_norm = orig.q_norm, orig.k_norm
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

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["qkv"].proj_up.dtype
        y = y.to(dtype)
        q, k, v = self.proj["qkv"](y).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, v, rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


class NunchakuMOSSImageTextBlock(nn.Module):
    """量化后的 text block；ffn_up/gate 已融合为 ffn_up_gate。"""

    def __init__(self, orig: _TextBlock, *, precision: str, rank: int) -> None:
        super().__init__()
        self.heads = orig.heads
        self.norm1, self.norm2 = orig.norm1, orig.norm2
        self.q_norm, self.k_norm = orig.q_norm, orig.k_norm
        self.qkv = SVDQW4A4Linear.from_linear(orig.qkv, precision=precision, rank=rank)
        self.proj = SVDQW4A4Linear.from_linear(orig.proj, precision=precision, rank=rank)
        with torch.device("meta"):
            ffn_up_gate = fuse_linears([orig.ffn_up, orig.ffn_gate])
        self.ffn_up_gate = SVDQW4A4Linear.from_linear(ffn_up_gate, precision=precision, rank=rank)
        self.ffn_down = SVDQW4A4Linear.from_linear(orig.ffn_down, precision=precision, rank=rank)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        hidden, gate = self.ffn_up_gate(y).chunk(2, -1)
        return x + self.ffn_down(situ(hidden, gate))


class NunchakuMOSSImageTransformer(MOSSImageTransformer, NunchakuModelLoaderMixin):
    """Nunchaku 量化版 MOSS-Image transformer。"""

    def _patch_model(self, precision: str = "nvfp4", rank: int = 128):
        for index, block in enumerate(self.layers):
            self.layers[index] = NunchakuMOSSImageBlock(block, precision=precision, rank=rank)
        for index, block in enumerate(self.text_embedder.blocks):
            self.text_embedder.blocks[index] = NunchakuMOSSImageTextBlock(block, precision=precision, rank=rank)
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: Any):
        device = kwargs.get("device", "cpu")
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
        transformer._patch_model(precision=precision, rank=rank)
        transformer = transformer.to_empty(device=device)
        if "rope.bands" not in state_dict:
            state_dict["rope.bands"] = transformer.rope._default_bands(transformer.rope.bands.device)
        patch_scale_key(transformer, state_dict)
        transformer.load_state_dict(state_dict)
        return transformer.eval()


# ---------------------------------------------------------------------------
# DIAGNOSTIC ONLY:
# 用于定位 text block 量化（含 bias）对生成结果的影响。仅读取原版 bf16
# transformer 的 text_embedder.* 权重，不加载 5B 主干；确认问题后可移除本函数
# 以及 example 中对应的 MOSS_IMAGE_TEXT_BLOCKS=bf16 分支。
# ---------------------------------------------------------------------------
_TEXT_EMBEDDER_PREFIX = "text_embedder."


def _read_text_embedder_state(transformer_dir: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    index_candidates = (
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    )
    index_path = next(
        (
            os.path.join(transformer_dir, n)
            for n in index_candidates
            if os.path.isfile(os.path.join(transformer_dir, n))
        ),
        None,
    )
    if index_path is None:
        matches = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors.index.json")))
        index_path = matches[0] if len(matches) == 1 else None

    state: dict[str, torch.Tensor] = {}
    if index_path is not None:
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        keys_by_shard: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            if key.startswith(_TEXT_EMBEDDER_PREFIX):
                keys_by_shard.setdefault(shard, []).append(key)
        for shard, keys in keys_by_shard.items():
            with safe_open(os.path.join(transformer_dir, shard), framework="pt", device="cpu") as f:
                for key in keys:
                    state[key[len(_TEXT_EMBEDDER_PREFIX) :]] = f.get_tensor(key).to(dtype=dtype)
    else:
        files = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no safetensors found under {transformer_dir!r}")
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118 - safetensors safe_open is not iterable
                    if key.startswith(_TEXT_EMBEDDER_PREFIX):
                        state[key[len(_TEXT_EMBEDDER_PREFIX) :]] = f.get_tensor(key).to(dtype=dtype)

    if not state:
        raise ValueError(f"no {_TEXT_EMBEDDER_PREFIX}* weights found under {transformer_dir!r}")
    return state


def load_bf16_text_embedder(model_root: str, config: Any, dtype: torch.dtype = torch.bfloat16) -> _TextProjection:
    """DIAGNOSTIC ONLY: 从原版模型的 transformer/ 读取 bf16 text_embedder。

    主 transformer 层的量化权重仍来自 svdq ckpt；这里仅用于对比 text block
    量化（含 bias）对输出的影响，不参与正常量化推理路径。text block 使用本文件中
    与官方逐操作同构的 ``_TextBlock.forward``。
    """
    transformer_dir = os.path.join(model_root, "transformer")
    if not os.path.isdir(transformer_dir):
        raise ValueError(f"MOSS_IMAGE_TEXT_BLOCKS=bf16 需要本地模型目录，找不到 {transformer_dir!r}")
    state = _read_text_embedder_state(transformer_dir, dtype)
    text_embedder = _TextProjection(
        int(config.text_dim),
        int(config.dim),
        int(config.ffn_dim),
        int(config.num_heads),
    )
    text_embedder.load_state_dict(state, strict=True)
    return text_embedder.to(dtype=dtype).eval()
