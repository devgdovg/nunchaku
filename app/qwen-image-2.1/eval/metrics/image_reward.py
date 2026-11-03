"""Image Reward：提示词与生成图片的匹配度。"""

import os
import typing as tp

import torch
from tqdm import tqdm

__all__ = ["compute_image_reward"]


def compute_image_reward(gen_dir: str, manifest: dict[str, dict[str, tp.Any]]) -> float:
    """对 manifest 中的每个样本计算 Image Reward，返回均值。"""
    import ImageReward as RM

    if not manifest:
        raise ValueError("manifest is empty; cannot compute Image Reward")

    model = RM.load("ImageReward-v1.0")
    scores: list[float] = []
    for sample_id, entry in tqdm(manifest.items(), desc="Image Reward", dynamic_ncols=True):
        path = os.path.join(gen_dir, f"{sample_id}.png")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{sample_id}: generated image not found: {path!r}")
        with torch.inference_mode():
            scores.append(float(model.score(entry["prompt"], path)))
    return sum(scores) / len(scores)
