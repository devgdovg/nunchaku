"""Qwen Image 2.1 评估用批量生图入口（原版 bf16 或 SVDQ fp4）。"""

import argparse
import os

from dataset import EvalSample, append_manifest_entry, load_eval_samples, manifest_path, read_manifest
from pipelines import build_call_kwargs, build_pipeline
from tqdm import tqdm

__all__ = ["main"]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation images with Qwen Image 2.1 (bf16 or nvfp4).")
    parser.add_argument("-p", "--precision", choices=["bf16", "fp4"], required=True, help="Pipeline precision")
    parser.add_argument("--model", default="Qwen/Qwen-Image-2.1", help="Diffusers pipeline directory or repo id")
    parser.add_argument("--ckpt", default="", help="Quantized nvfp4 safetensors (required for fp4)")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset directory containing metadata.jsonl")
    parser.add_argument(
        "-o",
        "--output-root",
        default=None,
        help="Image output directory; defaults to results/qwen-image-2.1/{precision}",
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="Number of samples to use (-1 = full dataset)")
    parser.add_argument("--sample-seed", type=int, default=0, help="Sampling seed for subset selection")
    parser.add_argument("-t", "--num-inference-steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0, help="Qwen 2.1 true CFG scale")
    parser.add_argument(
        "--use-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use the prefix KV cache",
    )
    parser.add_argument("--offload", choices=["model", "sequential"], default="model", help="CPU offload mode")
    parser.add_argument(
        "--chunk-step",
        type=int,
        default=1,
        help="Generate the subset specified by [chunk-start::chunk-step] (for multi-GPU runs)",
    )
    parser.add_argument("--chunk-start", type=int, default=0, help="Index within [0, chunk-step) for this chunk")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing manifest and regenerate every selected sample (single process only)",
    )
    return parser.parse_args()


def _check_existing_entry(sample: EvalSample, existing: dict, args: argparse.Namespace) -> None:
    entry = existing[sample.sample_id]
    expected = {
        "precision": args.precision,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "use_kv_cache": args.use_kv_cache,
    }
    mismatched = {key: (entry.get(key), value) for key, value in expected.items() if entry.get(key) != value}
    if mismatched:
        details = ", ".join(f"{key}={old!r} (current {new!r})" for key, (old, new) in mismatched.items())
        raise ValueError(
            f"{sample.sample_id}: output directory was generated with different settings: {details}; "
            "use a different --output-root or pass --overwrite"
        )


def main() -> None:
    args = get_args()
    assert args.chunk_step > 0
    assert 0 <= args.chunk_start < args.chunk_step
    if args.overwrite and args.chunk_step > 1:
        raise ValueError("--overwrite cannot be combined with --chunk-step > 1")

    output_root = args.output_root or os.path.join("results", "qwen-image-2.1", args.precision)
    os.makedirs(output_root, exist_ok=True)
    manifest_file = manifest_path(output_root)
    if args.overwrite and os.path.isfile(manifest_file):
        os.remove(manifest_file)
    existing = read_manifest(manifest_file)

    samples = load_eval_samples(args.dataset, args.max_samples, args.sample_seed)
    if args.chunk_step > 1:
        samples = samples[args.chunk_start :: args.chunk_step]

    pipeline, transformer_info = build_pipeline(args.precision, args.model, args.ckpt, args.offload)
    print(
        f"[{args.precision}] transformer={transformer_info['transformer_class']} "
        f"svdq_linears={transformer_info['svdq_linears']}"
    )
    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)

    for sample in tqdm(samples, desc="Samples", dynamic_ncols=True):
        image_path = os.path.join(output_root, f"{sample.sample_id}.png")
        if sample.sample_id in existing:
            if os.path.isfile(image_path):
                _check_existing_entry(sample, existing, args)
                continue
            if not args.overwrite:
                _check_existing_entry(sample, existing, args)

        kwargs, opened_images, seed = build_call_kwargs(
            sample, args.num_inference_steps, args.true_cfg_scale, args.use_kv_cache
        )
        try:
            image = pipeline(**kwargs).images[0]
            image.save(image_path)
        finally:
            for opened in opened_images:
                opened.close()
        del image

        append_manifest_entry(
            manifest_file,
            {
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "width": sample.width,
                "height": sample.height,
                "images": list(sample.images),
                "output": sample.output,
                "seed": seed,
                "num_inference_steps": args.num_inference_steps,
                "true_cfg_scale": args.true_cfg_scale,
                "use_kv_cache": args.use_kv_cache,
                "precision": args.precision,
                "transformer_class": transformer_info["transformer_class"],
                "svdq_linears": transformer_info["svdq_linears"],
            },
        )

    print(f"saved {len(samples)} sample(s) to {output_root}")


if __name__ == "__main__":
    main()
