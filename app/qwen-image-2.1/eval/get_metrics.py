"""Qwen Image 2.1 评估指标入口。

绝对指标（FID / Image Reward）对每个输出目录分别计算；对比指标（LPIPS / PSNR）
以第二个目录为基准计算。样本子集以各目录的 ``manifest.jsonl`` 为准。
"""

import argparse
import contextlib
import hashlib
import json
import os
import tempfile

from dataset import manifest_path, read_manifest
from metrics.fid import compute_fid, materialize_gt_dir
from metrics.image_reward import compute_image_reward
from metrics.similarity import compute_similarity

__all__ = ["main"]

_ABSOLUTE_METRICS = ("fid", "image_reward")
_PAIRWISE_METRICS = ("lpips", "psnr")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Qwen Image 2.1 evaluation metrics.")
    parser.add_argument("gen_root", help="Generated image directory containing manifest.jsonl")
    parser.add_argument(
        "ref_root",
        nargs="?",
        default=None,
        help="Reference directory (e.g. bf16) used for LPIPS/PSNR",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        choices=_ABSOLUTE_METRICS + _PAIRWISE_METRICS,
        default=None,
        help="Metrics to compute; defaults to all metrics applicable with the given arguments",
    )
    parser.add_argument("-o", "--output", default="metrics.json", help="Output JSON path")
    parser.add_argument("--device", default="cuda", help="Device used by metric models")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader/FID worker count")
    parser.add_argument("--fid-batch-size", type=int, default=64, help="FID feature extraction batch size")
    parser.add_argument("--fid-cache-dir", default=None, help="Optional directory for cached FID features")
    return parser.parse_args()


def _load_manifest(root: str) -> dict:
    path = manifest_path(root)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"manifest not found: {path!r}; generate images with generate.py first")
    manifest = read_manifest(path)
    if not manifest:
        raise ValueError(f"manifest is empty: {path!r}")
    return manifest


def _root_name(root: str, taken: set[str]) -> str:
    base = os.path.basename(os.path.normpath(root)) or "root"
    name = base
    index = 2
    while name in taken:
        name = f"{base}-{index}"
        index += 1
    taken.add(name)
    return name


def _check_same_samples(gen_manifest: dict, ref_manifest: dict, gen_root: str, ref_root: str) -> None:
    gen_ids = set(gen_manifest)
    ref_ids = set(ref_manifest)
    if gen_ids != ref_ids:
        missing_in_ref = sorted(gen_ids - ref_ids)
        missing_in_gen = sorted(ref_ids - gen_ids)
        raise ValueError(
            f"sample sets differ between {gen_root!r} and {ref_root!r}: "
            f"{len(missing_in_ref)} missing in reference, {len(missing_in_gen)} missing in generation; "
            f"examples: {missing_in_ref[:3]} / {missing_in_gen[:3]}"
        )


def main() -> None:
    args = get_args()
    gen_root = os.path.abspath(args.gen_root)
    ref_root = os.path.abspath(args.ref_root) if args.ref_root else None

    gen_manifest = _load_manifest(gen_root)
    roots = []
    taken: set[str] = set()
    gen_name = _root_name(gen_root, taken)
    roots.append((gen_name, gen_root, gen_manifest))

    ref_manifest = None
    ref_name = None
    if ref_root is not None:
        ref_manifest = _load_manifest(ref_root)
        _check_same_samples(gen_manifest, ref_manifest, gen_root, ref_root)
        ref_name = _root_name(ref_root, taken)
        roots.append((ref_name, ref_root, ref_manifest))

    if args.metrics:
        selected = set(args.metrics)
    else:
        selected = set(_ABSOLUTE_METRICS)
        if ref_root is not None:
            selected.update(_PAIRWISE_METRICS)
    pairwise = tuple(name for name in _PAIRWISE_METRICS if name in selected)
    if pairwise and ref_root is None:
        raise ValueError(f"{pairwise} require a reference directory (second positional argument)")

    results: dict = {}
    with contextlib.ExitStack() as stack:
        if args.fid_cache_dir:
            # Keep the ground-truth links at a stable path so cached FID features
            # for the reference directory remain reusable across runs.
            cache_root = os.path.abspath(args.fid_cache_dir)
            gt_hash = hashlib.sha1(gen_root.encode("utf-8")).hexdigest()[:12]
            gt_dir = os.path.join(cache_root, f"gt-{gt_hash}")
        else:
            gt_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="qwen-image-2.1-gt-"))
        materialize_gt_dir(gen_manifest, gt_dir)
        for name, root, manifest in roots:
            entry: dict = {"num_images": len(manifest)}
            if "fid" in selected:
                entry["fid"] = compute_fid(
                    gen_dir=root,
                    ref_dir=gt_dir,
                    device=args.device,
                    batch_size=args.fid_batch_size,
                    num_workers=args.num_workers,
                    cache_dir=args.fid_cache_dir,
                )
            if "image_reward" in selected:
                entry["image_reward"] = compute_image_reward(root, manifest)
            results[name] = entry
            print(name, json.dumps(entry, sort_keys=True))

    if pairwise and ref_manifest is not None:
        similarity = compute_similarity(
            gen_dir=gen_root,
            ref_dir=ref_root,
            sample_ids=list(gen_manifest),
            metrics=pairwise,
            num_workers=args.num_workers,
            device=args.device,
        )
        results["similarity"] = {f"{gen_name}_vs_{ref_name}": similarity}
        print("similarity", json.dumps(results["similarity"], sort_keys=True))

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
