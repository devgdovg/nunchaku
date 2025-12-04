import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import FluxPipeline

from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision, is_turing
from tests.sdxl.test_sdxl_turbo import plot, run_benchmark


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_flux_time_cost():
    batch_sizes = [1, 2]
    runs = 5
    inference_steps = 20
    guidance_scale = 3.5
    device_name = torch.cuda.get_device_name(0)
    results = {"Diffusers FP8": [], "Nunchaku INT4": []}

    # pipeline_original = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", use_safetensors=True,
    #                                                  torch_dtype=torch.bfloat16)

    pipeline_original = FluxPipeline.from_pretrained("diffusers/FLUX.1-dev-bnb-8bit", torch_dtype=torch.bfloat16).to(
        "cuda"
    )

    for batch_size in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        benchmark_original = run_benchmark(
            pipeline_original, batch_size, guidance_scale, device_name, runs, inference_steps
        )
        results["Diffusers FP8"].append(benchmark_original.mean() * inference_steps)

    del pipeline_original.transformer
    del pipeline_original
    # pipeline_original = None
    gc.collect()
    torch.cuda.empty_cache()

    precision = get_precision()
    quantized_transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    )
    pipeline_quantized = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=quantized_transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    for batch_size in batch_sizes:
        benchmark_quantized = run_benchmark(
            pipeline_quantized, batch_size, guidance_scale, device_name, runs, inference_steps
        )
        results["Nunchaku INT4"].append(benchmark_quantized.mean() * inference_steps)

    ref_root = Path(os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref")))
    plot_save_path = ref_root / "time_cost" / "flux.1-dev"
    os.makedirs(plot_save_path, exist_ok=True)

    plot(batch_sizes, results, device_name, runs, inference_steps, plot_save_path, "Flux.1-dev")
