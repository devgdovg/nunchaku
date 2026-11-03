"""LPIPS / PSNR：量化输出相对原版输出的相似度。"""

import os
import typing as tp

import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from tqdm import tqdm

__all__ = ["compute_similarity"]

_METRIC_NAMES = ("lpips", "psnr")


class PairedImageDataset(data.Dataset):
    """按 ``sample_id`` 配对两个输出目录中的同名图片。"""

    def __init__(self, gen_dir: str, ref_dir: str, sample_ids: tp.Sequence[str]) -> None:
        super().__init__()
        self.gen_dir = gen_dir
        self.ref_dir = ref_dir
        self.names = [f"{sample_id}.png" for sample_id in sample_ids]
        missing = [
            name
            for name in self.names
            if not os.path.isfile(os.path.join(gen_dir, name)) or not os.path.isfile(os.path.join(ref_dir, name))
        ]
        if missing:
            raise FileNotFoundError(f"{len(missing)} paired image(s) are missing, e.g. {missing[0]!r}")
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        name = self.names[index]
        gen_image = Image.open(os.path.join(self.gen_dir, name)).convert("RGB")
        ref_image = Image.open(os.path.join(self.ref_dir, name)).convert("RGB")
        if ref_image.size != gen_image.size:
            ref_image = ref_image.resize(gen_image.size, Image.Resampling.BICUBIC)
        return [self.transform(gen_image), self.transform(ref_image)]


def _build_metric(name: str, device: str | tp.Any) -> torch.nn.Module:
    if name == "lpips":
        return LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    if name == "psnr":
        return PeakSignalNoiseRatio(data_range=(0, 1), reduction="elementwise_mean", dim=(1, 2, 3)).to(device)
    raise NotImplementedError(f"Metric {name!r} is not implemented")


def compute_similarity(
    gen_dir: str,
    ref_dir: str,
    sample_ids: tp.Sequence[str],
    metrics: tp.Sequence[str] = _METRIC_NAMES,
    batch_size: int = 64,
    num_workers: int = 8,
    device: str | tp.Any = "cuda",
) -> dict[str, float]:
    """计算 ``gen_dir`` 相对 ``ref_dir`` 的 LPIPS / PSNR。"""
    if len(metrics) == 0:
        return {}

    metric_objects = {name: _build_metric(name, device) for name in metrics}
    dataset = PairedImageDataset(gen_dir, ref_dir, sample_ids)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    with torch.no_grad():
        for gen_tensor, ref_tensor in tqdm(dataloader, desc="Similarity metrics", dynamic_ncols=True):
            gen_tensor = gen_tensor.to(device)
            ref_tensor = ref_tensor.to(device)
            for metric in metric_objects.values():
                metric.update(gen_tensor, ref_tensor)
    return {name: float(metric.compute().item()) for name, metric in metric_objects.items()}
