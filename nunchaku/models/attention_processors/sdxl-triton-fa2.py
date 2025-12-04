import random
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


class NunchakuSDXLFA2Processor:

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ):
        # Adapted from https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/src/diffusers/models/attention_processor.py#AttnProcessor2_0

        # if len(args) > 0 or kwargs.get("scale", None) is not None:
        #     deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        #     deprecate("scale", "1.0.0", deprecation_message)

        # residual = hidden_states
        # if attn.spatial_norm is not None:
        #     hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # # scaled_dot_product_attention expects attention_mask shape to be
            # # (batch, heads, source_length, target_length)
            # attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            raise NotImplementedError("attention_mask is not supported")

        # if attn.group_norm is not None:
        #     hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        ############# qkv ################
        # query = attn.to_q(hidden_states)

        # if encoder_hidden_states is None:
        #     encoder_hidden_states = hidden_states
        # elif attn.norm_cross:
        #     encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # key = attn.to_k(encoder_hidden_states)
        # value = attn.to_v(encoder_hidden_states)
        if not attn.is_cross_attention:
            qkv = attn.to_qkv(hidden_states)
            query, key, value = qkv.chunk(3, dim=-1)
            # query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        else:
            query, key, value = (
                attn.to_q(hidden_states),
                attn.to_k(encoder_hidden_states),
                attn.to_v(encoder_hidden_states),
            )

        ############# end of qkv ################

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # print(f"BEFORE VEIW(): query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # print(f"AFTER VIEW(): query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")

        # if attn.norm_q is not None:
        #     query = attn.norm_q(query)
        # if attn.norm_k is not None:
        #     key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_t = triton_flash_attn_2(query, key, value)

        print_mismatch(hidden_states, hidden_states_t)

        # print(f"hiddent_states shape: {hidden_states.shape}")

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def print_mismatch(tensor_a: torch.Tensor, tensor_b: torch.Tensor, p=0.001):
    mismatch = (~torch.isclose(tensor_a, tensor_b)).nonzero()
    print(f"mismatch count: {mismatch.shape[0]}")
    for idx in mismatch:
        idx_tuple = tuple(idx.tolist())
        if random.random() < p:
            print(
                f"mismatch at [{idx_tuple}], attn_torch: {tensor_a[idx_tuple].item()}, attn_triton: {tensor_b[idx_tuple].item()}"
            )


def triton_flash_attn_2(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    batch_size_q, num_heads_q, seq_len_q, num_channels_q = query.shape
    batch_size_k, num_heads_k, seq_len_k, num_channels_k = key.shape
    batch_size_v, num_heads_v, seq_len_v, num_channels_v = value.shape

    ######################
    # shape verification #
    ######################
    assert batch_size_q == batch_size_k and batch_size_q == batch_size_v
    assert num_heads_q == num_heads_k and num_heads_q == num_heads_v
    assert seq_len_k == seq_len_v  # seq_len_q does NOT need to be equal to seq_len_k
    assert num_channels_q == num_channels_k  # num_channels_q does NOT need to be equal to num_channels_v

    ########################################################################
    # output shape: (batch_size_q, num_heads_q, seq_len_q, num_channels_v) #
    ########################################################################
    output = torch.empty((batch_size_q, num_heads_q, seq_len_q, num_channels_v), device=query.device, dtype=query.dtype)

    block_size = 64

    # assert seq_len_q % (block_size*2) == 0
    # assert num_channels_q % block_size == 0

    flash_attn_2_kernel[
        (
            batch_size_q,
            num_heads_q,
            triton.cdiv(seq_len_q, block_size),
        )
    ](
        query,
        key,
        value,
        output,
        batch_size_q,
        num_heads_q,
        seq_len_q,
        seq_len_k,
        num_channels_q ** (-0.5),
        BLOCK_Q=tl.constexpr(block_size),
        BLOCK_KV=tl.constexpr(block_size),
        CHN_QK=tl.constexpr(num_channels_q),
        CHN_V=tl.constexpr(num_channels_v),
    )
    return output


@triton.autotune(configs=[triton.Config({}, num_warps=2, num_stages=2)], key=[])
@triton.jit
def flash_attn_2_kernel(
    query,
    key,
    value,
    output,
    BATCH_SIZE,
    NUM_HEADS,
    LEN_Q,
    LEN_KV,
    INV_SQRT_D,
    BLOCK_Q: tl.constexpr,
    # CHANNELS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    CHN_QK: tl.constexpr,
    CHN_V: tl.constexpr,
    # BLOCK_CHANNELS: tl.constexpr
):
    xid = tl.program_id(0)
    yid = tl.program_id(1)
    pid = tl.program_id(2)
    log2_e = 1.44269504

    rowmax_prev = tl.full(shape=(BLOCK_Q, 1), value=-float("inf"), dtype=tl.float32)
    rowsum = tl.zeros(shape=(BLOCK_Q, 1), dtype=tl.float32)
    output_accu = tl.zeros(shape=(BLOCK_Q, CHN_V), dtype=tl.float32)

    for j in range(0, tl.cdiv(LEN_KV, BLOCK_KV)):

        # s = tl.zeros(shape=(BLOCK_Q, BLOCK_KV), dtype=tl.float32)
        # for l in range(0, tl.cdiv(CHN_QK, BLOCK_CHANNELS)):
        q_tile_ptr = tl.make_block_ptr(
            base=query,
            shape=(BATCH_SIZE, NUM_HEADS, LEN_Q, CHN_QK),
            strides=(NUM_HEADS * LEN_Q * CHN_QK, LEN_Q * CHN_QK, CHN_QK, 1),
            offsets=(xid, yid, pid * BLOCK_Q, 0),
            block_shape=(1, 1, BLOCK_Q, CHN_QK),
            order=(3, 2, 1, 0),  # TODO verify
        )
        q_tile_val = tl.reshape(tl.load(q_tile_ptr, boundary_check=(2, 3)).to(dtype=tl.float32), (BLOCK_Q, CHN_QK))
        k_tile_ptr = tl.make_block_ptr(
            base=key,
            shape=(BATCH_SIZE, NUM_HEADS, LEN_KV, CHN_QK),
            strides=(NUM_HEADS * LEN_KV * CHN_QK, LEN_KV * CHN_QK, CHN_QK, 1),
            offsets=(xid, yid, j * BLOCK_KV, 0),
            block_shape=(1, 1, BLOCK_KV, CHN_QK),
            order=(3, 2, 1, 0),  # TODO verify
        )
        k_tile_val = tl.reshape(tl.load(k_tile_ptr, boundary_check=(2, 3)).to(dtype=tl.float32), (BLOCK_KV, CHN_QK))
        # s += tl.dot(q_tile_val, k_tile_val.T)
        s = tl.dot(q_tile_val, k_tile_val.T)
        s *= INV_SQRT_D  # shape is (BLOCK_R, BLOCK_C), type is float32

        rowmax_tile = tl.max(s, axis=1, return_indices=False, keep_dims=True)  # shape is (BLOCK_R, 1), type is float32
        rowmax_curr = tl.maximum(rowmax_prev, rowmax_tile)  # shape is (BLOCK_R, 1), type is float32
        p = tl.math.exp2((s - rowmax_curr) * log2_e)  # shape is (BLOCK_R, BLOCK_C), type is float32
        rowsum_tile = tl.sum(p, axis=1, keep_dims=True)  # shape is (BLOCK_R, 1), type is float32
        scale = tl.math.exp2((rowmax_prev - rowmax_curr) * log2_e)  # shape is (BLOCK_R, 1), type is float32

        v_tile_ptr = tl.make_block_ptr(
            base=value,
            shape=(BATCH_SIZE, NUM_HEADS, LEN_KV, CHN_V),
            strides=(NUM_HEADS * LEN_KV * CHN_V, LEN_KV * CHN_V, CHN_V, 1),
            offsets=(xid, yid, j * BLOCK_KV, 0),
            block_shape=(1, 1, BLOCK_KV, CHN_V),
            order=(3, 2, 1, 0),
        )
        v_tile_val = tl.reshape(
            tl.load(v_tile_ptr).to(dtype=tl.float32), (BLOCK_KV, CHN_V)
        )  # shape is (BLOCK_C, D), type is float32
        pv = tl.dot(p, v_tile_val)  # shape is (BLOCK_R, D), type is float32

        rowmax_prev = rowmax_curr
        rowsum = scale * rowsum + rowsum_tile
        output_accu = scale * output_accu + pv

    output_accu = output_accu / rowsum
    output_tile_ptr = tl.make_block_ptr(
        base=output,
        shape=(BATCH_SIZE, NUM_HEADS, LEN_Q, CHN_V),
        strides=(NUM_HEADS * LEN_Q * CHN_V, LEN_Q * CHN_V, CHN_V, 1),
        offsets=(xid, yid, pid * BLOCK_Q, 0),
        block_shape=(1, 1, BLOCK_Q, CHN_V),
        order=(3, 2, 1, 0),
    )
    tl.store(output_tile_ptr, tl.reshape(output_accu, (1, 1, BLOCK_Q, CHN_V)).to(dtype=tl.float32))


#####################################


@triton.jit
def _block_matmul_kernel(
    a_ptr, b_ptr, o_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Get program IDs for current block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create block pointers for A, B, and the accumulator
    a_tile = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_tile = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension to perform block-wise multiplication
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks from A and B
        a_value = tl.load(a_tile, boundary_check=(0, 1), padding_option="zero")
        b_value = tl.load(b_tile, boundary_check=(0, 1), padding_option="zero")

        # Perform dot product and accumulate
        accumulator += tl.dot(a_value, b_value)

        # Advance block pointers for the next iteration
        a_tile = tl.advance(a_tile, (0, BLOCK_K))
        b_tile = tl.advance(b_tile, (BLOCK_K, 0))

    # Create block pointer for the output
    o_tile = tl.make_block_ptr(
        base=o_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    # Store the accumulated result to global memory
    tl.store(o_tile, accumulator.to(o_tile.dtype.element_ty), boundary_check=(0, 1))


def matmul_block(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    N = b.shape[1]
    o = torch.zeros(M, N, device=a.device, dtype=a.dtype)

    # Define grid for kernel launch
    def _grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    grid = _grid

    # Launch the kernel
    _block_matmul_kernel[grid](a, b, o, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return o


if __name__ == "__main__":

    # MATMUL EXAMPLE
    # A = torch.randn(256, 128, device='cuda', dtype=torch.bfloat16)
    # B = torch.randn(128, 512, device='cuda', dtype=torch.bfloat16)

    # C_triton = matmul_block(A, B)
    # C_torch = torch.matmul(A, B)

    # print(f"Triton result: {C_triton}")
    # print(f"PyTorch result: {C_torch}")
    # print(f"Are results close: {torch.allclose(C_triton, C_torch)}")
    #######################

    # q_shape = (2, 10, 4096, 64)  # typical data size for attention layer in SDXL

    # k_shape = (2, 10, 4096, 64)  # typical data size for attention layer in SDXL
    # v_shape = (2, 10, 4096, 64)  # typical data size for attention layer in SDXL

    # k_shape = (2, 10, 77, 64)  # typical data size for attention layer in SDXL
    # v_shape = (2, 10, 77, 64)  # typical data size for attention layer in SDXL

    q_shape = (1, 1, 64, 64)

    k_shape = (1, 1, 64, 64)
    v_shape = (1, 1, 64, 64)

    q = [torch.randn(q_shape, device="cuda:4", dtype=torch.float32) * 100 + 100 for _ in range(100)]
    k = [torch.randn(k_shape, device="cuda:4", dtype=torch.float32) * 100 + 100 for _ in range(100)]
    v = [torch.randn(v_shape, device="cuda:4", dtype=torch.float32) * 100 + 100 for _ in range(100)]

    runs = 10000

    # warm up
    attn_torch = F.scaled_dot_product_attention(q[0], k[0], v[0], attn_mask=None, dropout_p=0.0, is_causal=False)
    # torch_start_time = time.time()
    # for i in range(runs):
    #     attn_torch = F.scaled_dot_product_attention(
    #         q[i%100], k[i%100], v[i%100], attn_mask=None, dropout_p=0.0, is_causal=False
    #     )
    # torch_time = time.time() - torch_start_time
    # print(f"torch time: {torch_time}")

    # warm up
    attn_triton = triton_flash_attn_2(q[0], k[0], v[0])
    # triton_start_time = time.time()
    # for i in range(runs):
    #     attn_triton = triton_flash_attn_2(q[i%100], k[i%100], v[i%100])
    # triton_time = time.time() - triton_start_time
    # print(f"triton time: {triton_time}")

    # for v1, v2 in zip(attn_torch[0][0][1][:10], attn_triton[0][0][1][:10]):
    #     print(f"attn_torch_value: {v1}, attn_triton_value: {v2}")

    print(f"attn_torch & attn_triton allclose: {torch.allclose(attn_torch, attn_triton)}")

    print_mismatch(attn_torch, attn_triton, p=1.0)

    # diff = attn_torch - attn_triton
    # print(f"diff max: {torch.max(diff)}, min: {torch.min(diff)}, avg: {torch.mean(diff)}, std: {torch.std(diff)}")
