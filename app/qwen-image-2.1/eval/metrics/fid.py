"""FID（clean-fid），参考集为评估子集的 ground-truth 图片。"""

import hashlib
import os
import typing as tp

import numpy as np

__all__ = ["compute_fid", "materialize_gt_dir"]


def _image_names(folder: str) -> list[str]:
    return sorted(
        name for name in os.listdir(folder) if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    )


def _cache_key(folder: str) -> str:
    names = _image_names(folder)
    digest = hashlib.sha1(os.path.abspath(folder).encode("utf-8")).hexdigest()[:16]
    return f"{digest}-{len(names)}"


def _feature_stats(
    folder: str,
    device: str | tp.Any,
    batch_size: int,
    num_workers: int,
    cache_file: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_file and os.path.isfile(cache_file):
        npz = np.load(cache_file)
        return npz["mu"], npz["sigma"]

    from cleanfid import fid

    num_images = len(_image_names(folder))
    if num_images < 2:
        raise ValueError(f"FID requires at least 2 images, got {num_images} in {folder!r}")

    feat_model = fid.build_feature_extractor("clean", device)
    features = fid.get_folder_features(
        folder,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        verbose=False,
        mode="clean",
        description=f"Extracting FID features from {folder}",
    )
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    if cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez(cache_file, mu=mu, sigma=sigma)
    return mu, sigma


def compute_fid(
    gen_dir: str,
    ref_dir: str,
    device: str | tp.Any = "cuda",
    batch_size: int = 64,
    num_workers: int = 8,
    cache_dir: str | None = None,
) -> float:
    """计算 ``gen_dir`` 与 ``ref_dir`` 两个图片目录之间的 FID。"""
    from cleanfid import fid

    ref_cache = os.path.join(cache_dir, f"fid-{_cache_key(ref_dir)}.npz") if cache_dir else None
    gen_cache = os.path.join(cache_dir, f"fid-{_cache_key(gen_dir)}.npz") if cache_dir else None
    mu_ref, sigma_ref = _feature_stats(ref_dir, device, batch_size, num_workers, ref_cache)
    mu_gen, sigma_gen = _feature_stats(gen_dir, device, batch_size, num_workers, gen_cache)
    return float(fid.frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen))


def materialize_gt_dir(manifest: dict[str, dict[str, tp.Any]], dest_dir: str) -> str:
    """把 manifest 中各样本的 ground-truth 图片以 ``sample_id`` 命名链接到 ``dest_dir``。

    不同样本的 GT 文件名可能重复，按 ``sample_id`` 重命名可避免冲突，并保证
    FID 参考集与本次生成样本严格一一对应。
    """
    os.makedirs(dest_dir, exist_ok=True)
    expected = {
        f"{sample_id}{os.path.splitext(entry['output'])[1].lower() or '.png'}" for sample_id, entry in manifest.items()
    }
    for name in os.listdir(dest_dir):
        if name not in expected:
            os.remove(os.path.join(dest_dir, name))
    for sample_id, entry in manifest.items():
        source = os.path.abspath(entry["output"])
        extension = os.path.splitext(source)[1].lower() or ".png"
        link_path = os.path.join(dest_dir, f"{sample_id}{extension}")
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(source, link_path)
    return dest_dir
