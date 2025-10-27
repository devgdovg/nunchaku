import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D

from nunchaku.models.linear import SVDQW4A4Linear


def _least_multiple_of(orig_value: int, multiple: int):
    return (orig_value + multiple - 1) // multiple * multiple


# for conv1
class UnfoldAndLinear(nn.Module):
    def __init__(self, orig_conv: nn.Conv2d, **kwargs):
        super().__init__()
        self.in_channels = orig_conv.in_channels
        self.out_channels = orig_conv.out_channels
        self.kernel_size = orig_conv.kernel_size
        self.stride = orig_conv.stride
        self.padding = orig_conv.padding
        self.bias_flag = orig_conv.bias is not None

        self.default_warp_n = 128
        self.default_num_k_unrolls = 2
        self.default_comp_k = 256 // 4
        kH, kW = self.kernel_size
        padded_in_features = _least_multiple_of(
            orig_value=self.in_channels * kH * kW, multiple=self.default_comp_k * self.default_num_k_unrolls
        )
        padded_out_features = _least_multiple_of(orig_value=self.out_channels, multiple=self.default_warp_n)
        self.linear = SVDQW4A4Linear(
            in_features=padded_in_features,
            out_features=padded_out_features,
            bias=self.bias_flag,
            torch_dtype=orig_conv.weight.dtype,
            device=orig_conv.weight.device,
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        N, C, H, W = x.shape  # TODO
        x_unf = (
            F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
            .transpose(1, 2)
            .contiguous()
        )
        if self.linear.in_features > x_unf.shape[2]:
            x_unf = F.pad(x_unf, (0, self.linear.in_features - x_unf.shape[2]), value=0)
        y = self.linear(x_unf)

        # if self.out_channels < y.shape[2]:
        #     y = y[:, :, :self.out_channels-y.shape[2]]
        # H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        # W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        # y = y.transpose(1, 2).reshape(N, self.out_channels, H_out, W_out)

        return y


# for conv2
class LinearAndTranspose(nn.Module):
    def __init__(self, orig_conv: nn.Conv2d, **kwargs):
        super().__init__()
        self.in_channels = orig_conv.in_channels
        self.out_channels = orig_conv.out_channels
        self.kernel_size = orig_conv.kernel_size
        self.stride = orig_conv.stride
        self.padding = orig_conv.padding
        self.bias_flag = orig_conv.bias is not None

        self.default_warp_n = 128
        self.default_num_k_unrolls = 2
        self.default_comp_k = 256 // 4
        kH, kW = self.kernel_size
        padded_in_features = _least_multiple_of(
            orig_value=self.in_channels * kH * kW, multiple=self.default_comp_k * self.default_num_k_unrolls
        )
        padded_out_features = _least_multiple_of(orig_value=self.out_channels, multiple=self.default_warp_n)
        self.linear = SVDQW4A4Linear(
            in_features=padded_in_features,
            out_features=padded_out_features,
            bias=self.bias_flag,
            torch_dtype=orig_conv.weight.dtype,
            device=orig_conv.weight.device,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, shape: torch.Size):
        N, C, H, W = shape  # TODO
        # x_unf = F.unfold(
        #     x,
        #     kernel_size=self.kernel_size,
        #     stride=self.stride,
        #     padding=self.padding
        # ).transpose(1, 2).contiguous()
        # if self.linear.in_features > x.shape[2]:
        #     x = F.pad(x, (0, self.linear.in_features - x.shape[2]), value=0)
        y = self.linear(x)
        if self.out_channels < y.shape[2]:
            y = y[:, :, : self.out_channels - y.shape[2]]

        H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        y = y.transpose(1, 2).reshape(N, self.out_channels, H_out, W_out)

        return y


class NunchakuSDXLResnetBlock2D(nn.Module):
    """
    try to fuse conv1 and conv2 in ResnetBlock2D to avoid multiple unfold/transpose ops.
    but failed.
    """

    def __init__(self, orig_block: ResnetBlock2D):
        super().__init__()
        self.norm1 = orig_block.norm1
        self.conv1 = UnfoldAndLinear(orig_block.conv1)
        self.norm2 = orig_block.norm2
        self.time_emb_proj = orig_block.time_emb_proj
        self.dropout = orig_block.dropout
        assert self.dropout.p == 0.0
        self.conv2 = LinearAndTranspose(orig_block.conv2)
        self.nonlinearity = orig_block.nonlinearity
        self.conv_shortcut = orig_block.conv_shortcut
        self.output_scale_factor = orig_block.output_scale_factor

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        input_tensor = input_tensor.contiguous()

        hidden_states = self.norm1(input_tensor)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.fuse_conv1_temb_norm2(hidden_states, self.time_emb_proj(temb))

        hidden_states = self.nonlinearity(hidden_states)  # TODO fuse silu into prev conv1_norm2

        print(f">>>>> before conv2, hidden_states: {hidden_states.shape}")
        hidden_states = self.conv2(hidden_states, input_tensor.shape)

        input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

    def fuse_conv1_temb_norm2(self, hidden_states: torch.Tensor, temb: torch.Tensor):
        hidden_states = self.conv1(hidden_states)

        if hidden_states.shape[2] > temb.shape[1]:
            temb = F.pad(temb, (0, hidden_states.shape[2] - temb.shape[1]), value=0)

        hidden_states = hidden_states + temb.unsqueeze(1)  # TODO verify

        n_batch, L, out_channels = hidden_states.shape
        group_size = self.norm2.num_channels // self.norm2.num_groups
        mat_left = torch.full((L,), 1.0 / L, dtype=hidden_states.dtype, device=hidden_states.device).expand(
            n_batch, L, L
        )
        diag = [
            torch.full(
                (group_size, group_size), 1.0 / group_size, dtype=hidden_states.dtype, device=hidden_states.device
            )
        ] * self.norm2.num_groups
        mat_right = torch.block_diag(*diag)
        if hidden_states.shape[-1] > mat_right.shape[0]:
            mat_right = F.pad(
                mat_right,
                (0, hidden_states.shape[-1] - mat_right.shape[0], 0, hidden_states.shape[-1] - mat_right.shape[0]),
                value=0,
            )
        print(f">>>> mat_left: {mat_left.shape}, hidden_state: {hidden_states.shape}, mat_right: {mat_right.shape}")
        expectation = mat_left @ hidden_states @ mat_right
        variant = mat_left @ ((hidden_states - expectation) ** 2) @ mat_right
        normed_hidden_states = (hidden_states - expectation) / torch.sqrt(variant + self.norm2.eps)

        if hidden_states.shape[-1] > self.norm2.weight.shape[0]:
            _w = F.pad(self.norm2.weight, (0, hidden_states.shape[-1] - self.norm2.weight.shape[0]), value=0)
            _b = F.pad(self.norm2.bias, (0, hidden_states.shape[-1] - self.norm2.bias.shape[0]), value=0)
        else:
            _w = self.norm2.weight
            _b = self.norm2.bias
        normed_hidden_states = normed_hidden_states * _w + _b  # TODO verify

        return normed_hidden_states
