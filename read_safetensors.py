import argparse
import collections
import os

import torch
from safetensors.torch import load_file


# region for debug
def _print_state_dict(checkpoint_path):

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"model file not exists: {checkpoint_path}")
    state_dict = load_file(checkpoint_path)

    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: dtype={value.dtype}, shape={tuple(value.shape)}")
        else:
            print(f"{key}: (not a tensor, type={type(value)})")
            if isinstance(value, collections.OrderedDict):
                for sub_k, sub_v in value.items():
                    print(f" * {sub_k} -> dtype={sub_v.dtype}, shape={tuple(sub_v.shape)}")
            elif isinstance(value, bool):
                print(f"{key}: type=bool, value={value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="path to the conveted checkpoint file.")
    args = parser.parse_args()
    _print_state_dict(args.model_path)
# endregion
