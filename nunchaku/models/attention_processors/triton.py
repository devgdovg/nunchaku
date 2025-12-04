import torch
import triton
import triton.language as tl


@triton.jit
def reverse_kernel(input, N, BLOCK_SIZE: tl.constexpr):
    pid_0 = tl.program_id(0)

    left_start = pid_0 * BLOCK_SIZE
    left_range = tl.arange(0, BLOCK_SIZE) + left_start
    left_mask = left_range < (N // 2)

    right_start = N - (pid_0 * BLOCK_SIZE) - BLOCK_SIZE
    right_range = tl.arange(0, BLOCK_SIZE) + right_start
    right_mask = right_range >= ((N + 1) // 2)

    _left = tl.load(input + left_range, left_mask)
    _right = tl.load(input + right_range, right_mask)

    tl.store(input + left_range, tl.flip(_right), left_mask)
    tl.store(input + right_range, tl.flip(_left), right_mask)


# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 64
    if N == 1:
        return
    n_blocks = triton.cdiv(N, BLOCK_SIZE * 2)
    grid = (n_blocks,)

    reverse_kernel[grid](input, N, BLOCK_SIZE)


if __name__ == "__main__":
    for _len in [1, 2, 3, 20, 21, 50, 51, 100, 101, 1000, 1001, 10000, 100001, 1000000, 1000001, 25000000]:
        # _len = 25000000
        input_tensor = torch.arange(1, _len + 1, device="cuda:4")

        # print(f"before: {input_tensor[:]}")

        test_case = (input_tensor, _len)

        solve(*test_case)

        diff = torch.diff(input_tensor)

        print(
            f"check diff: {torch.all(diff == -1).item()}, len = {_len}, first_ele: {input_tensor[0]}, diff_len: {diff.numel()}"
        )

    # print(f"after: {input_tensor[:]}")
