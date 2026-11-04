# Qwen Image 2.1 Quality Evaluation

This directory contains the evaluation tools for the quantized Qwen Image 2.1 transformer. It generates images with both the original BF16 pipeline and the SVDQ NVFP4 pipeline on the same dataset, then computes:

- **FID** and **Image Reward** for each pipeline output set (absolute metrics);
- **LPIPS** and **PSNR** of the quantized output against the BF16 output (comparison metrics).

The evaluation dataset uses the same jsonl layout as the `diffusion-svdq` calibration dataset, with one additional required field: `output`, which points to the ground-truth image used by FID.

## Installation

Image generation uses the regular nunchaku installation. The metrics require a few extra packages:

```shell
pip install clean-fid torchmetrics image-reward clip
```

The FID and Image Reward models download their pretrained weights on first use.

## Dataset Format

The dataset directory must contain a `metadata.jsonl` and the image files it references:

```jsonl
{"prompt": "A sea otter floating on its back", "output": "gt/000001.png", "width": 1024, "height": 1024}
{"prompt": "Change the sign to read 'Qwen Image 2.1'", "output": "gt/000002.png", "image": "inputs/000002.png"}
```

- `prompt` is required for every sample.
- `output` is required and is used as the FID reference image for this sample.
- `image` is optional; when present (a file name or a list of file names) the sample runs through the image-editing path, otherwise it is a text-to-image sample.
- `width` / `height` are optional for text-to-image samples; the pipeline defaults are used when absent.

Samples are identified by their 0-based line number (`sample_00000`, `sample_00001`, ...), so text-to-image and editing samples can be mixed in one dataset and one output directory. Do not reorder or insert lines after generating images: existing outputs and manifests refer to these IDs.

## Image Generation

Generate the BF16 baseline and the quantized outputs in two separate runs:

```shell
# BF16 baseline
python app/qwen-image-2.1/eval/generate.py -p bf16 \
    --model /path/to/Qwen-Image-2.1 \
    -d /path/to/eval-dataset \
    -o results/qwen-image-2.1/bf16

# SVDQ NVFP4 (Blackwell)
python app/qwen-image-2.1/eval/generate.py -p fp4 \
    --model /path/to/Qwen-Image-2.1 \
    --ckpt /path/to/qwen-image-2.1-nvfp4.safetensors \
    -d /path/to/eval-dataset \
    -o results/qwen-image-2.1/fp4
```

Options:

- `-d/--dataset`: dataset directory containing `metadata.jsonl`.
- `-o/--output-root`: output directory; defaults to `results/qwen-image-2.1/{precision}`.
- `--max-samples N`: evaluate a deterministic subset (`-1` means the full dataset).
- `--sample-seed S`: subset sampling seed; the same seed selects the same samples.
- `-t/--num-inference-steps`: diffusion steps (default 25).
- `--true-cfg-scale`: Qwen 2.1 true CFG scale (default 1.0).
- `--no-use-kv-cache`: disable the prefix KV cache.
- `--offload {model,sequential}`: CPU offload mode (default `model`).
- `--chunk-start` / `--chunk-step`: split the selected samples across processes or GPUs.
- `--overwrite`: delete the existing manifest and regenerate the selected samples; use a fresh output root when changing inference parameters.

Each run writes `{sample_id}.png` plus a `manifest.jsonl` that records the prompt, input images, ground-truth path, seed, step count, precision, and the transformer actually used (`transformer_class`, `svdq_linears`). The `fp4` run aborts if the pipeline does not hold the nunchaku transformer or contains no `SVDQW4A4Linear` modules; the `bf16` run aborts if it unexpectedly contains quantized layers. Existing samples are skipped on rerun, so interrupted runs can be resumed. A manifest generated with different inference settings (steps, CFG scale, KV cache, or precision) is rejected; use another output root or `--overwrite`.

## Computing Metrics

The BF16/FP4 directories can be passed to `get_metrics.py`. The first directory is the quantized output; the optional second directory is the reference for LPIPS/PSNR:

```shell
python app/qwen-image-2.1/eval/get_metrics.py \
    results/qwen-image-2.1/fp4 results/qwen-image-2.1/bf16 \
    -o metrics.json
```

Select a subset of metrics with `--metrics`:

```shell
# Absolute metrics only, on the quantized output
python app/qwen-image-2.1/eval/get_metrics.py \
    results/qwen-image-2.1/fp4 --metrics fid image_reward -o metrics.json

# Similarity only
python app/qwen-image-2.1/eval/get_metrics.py \
    results/qwen-image-2.1/fp4 results/qwen-image-2.1/bf16 \
    --metrics lpips psnr -o metrics.json
```

- `fid` / `image_reward` are computed for every provided directory. FID uses the ground-truth images of **the same evaluated subset**, linked by `sample_id`, so the BF16 and FP4 scores are directly comparable.
- `lpips` / `psnr` require the second directory and pair images by `sample_id`; missing or extra samples raise an error.
- `--num-workers`, `--fid-batch-size`, `--fid-cache-dir`, and `--device` control the metric computation.

The result JSON looks like:

```json
{
  "bf16": {"fid": 12.3, "image_reward": 1.01, "num_images": 128},
  "fp4": {"fid": 13.1, "image_reward": 0.98, "num_images": 128},
  "similarity": {"fp4_vs_bf16": {"lpips": 0.08, "psnr": 24.1}}
}
```

## Notes

- FID is sensitive to the number of images; with only a few dozen samples it is best treated as a trend indicator. `num_images` is recorded in the output.
- Both pipelines must be run on the same dataset with the same `--max-samples` / `--sample-seed`. `get_metrics.py` checks that their manifests contain the same sample IDs and identical prompt/seed/steps/CFG/KV-cache settings.
- The BF16 and FP4 runs are intentionally separate to keep VRAM usage low. A single run only holds one pipeline.
- Text-to-image samples use `--width`/`--height` when provided; editing samples derive their output resolution from the input image.
