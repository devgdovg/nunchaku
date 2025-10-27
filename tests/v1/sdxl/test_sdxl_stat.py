import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.resnet import ResnetBlock2D

from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.models.unets.unet_sdxl import (
    NunchakuConv2dAsLinear,
    NunchakuSDXLAttention,
    NunchakuSDXLUNet2DConditionModel,
)
from nunchaku.utils import get_precision, is_turing


class StatHook:
    def __init__(self):
        self.start = []
        self.end = []

    def pre_hook(self, module, input):
        # torch.cuda.synchronize()
        # self.start.append(time.perf_counter())
        start_event = torch.cuda.Event(enable_timing=True)
        self.start.append(start_event)
        start_event.record()

    def post_hook(self, module, input, output):
        # torch.cuda.synchronize()
        # self.end.append(time.perf_counter())
        end_event = torch.cuda.Event(enable_timing=True)
        self.end.append(end_event)
        end_event.record()


def _total(hook: StatHook):
    assert len(hook.start) == len(hook.end)
    # return sum(hook.end) - sum(hook.start)
    return sum([hook.start[i].elapsed_time(hook.end[i]) / 1000 for i in range(len(hook.start))])


def run_stat(pipeline, batch_size, guidance_scale, device, runs, inference_steps):
    prompt = "the University of the Arts in Philadelphia, octane render, volumetric lighting, moody, raining"
    seed = 98765
    generator = torch.Generator().manual_seed(seed)
    # warmup
    _ = pipeline(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=inference_steps,
        num_images_per_prompt=batch_size,
    ).images

    unet = pipeline.unet

    whole_unet_hook = StatHook()

    handle_pre = unet.register_forward_pre_hook(whole_unet_hook.pre_hook)
    handle_post = unet.register_forward_hook(whole_unet_hook.post_hook)

    resnets_hook = StatHook()
    resnets_pre_hooks = []
    resnets_post_hooks = []

    conv2d_hook = StatHook()
    conv2d_pre_hooks = []
    conv2d_post_hooks = []

    conv2d_linear_hook = StatHook()
    conv2d_linear_pre_hooks = []
    conv2d_linear_post_hooks = []

    # transformer_hook = StatHook()
    # transformer_pre_hooks = []
    # transformer_post_hooks = []

    attn1_hook = StatHook()
    attn1_pre_hooks = []
    attn1_post_hooks = []

    attn1_out_hook = StatHook()
    attn1_out_pre_hooks = []
    attn1_out_post_hooks = []

    attn2_hook = StatHook()
    attn2_pre_hooks = []
    attn2_post_hooks = []

    attn2_q_hook = StatHook()
    attn2_q_pre_hooks = []
    attn2_q_post_hooks = []

    # attn2_k_hook = StatHook()
    # attn2_k_pre_hooks = []
    # attn2_k_post_hooks = []

    # attn2_v_hook = StatHook()
    # attn2_v_pre_hooks = []
    # attn2_v_post_hooks = []

    ffnet2_hook = StatHook()
    ffnet2_pre_hooks = []
    ffnet2_post_hooks = []

    attn2_out_hook = StatHook()
    attn2_out_pre_hooks = []
    attn2_out_post_hooks = []

    ffnet_proj_hook = StatHook()
    ffnet_proj_pre_hooks = []
    ffnet_proj_post_hooks = []

    for _name, _module in unet.named_modules():
        if isinstance(_module, ResnetBlock2D):
            resnets_pre_hooks.append(_module.register_forward_pre_hook(resnets_hook.pre_hook))
            resnets_post_hooks.append(_module.register_forward_hook(resnets_hook.post_hook))
        elif (
            ("conv" in _name and "resnets" in _name)
            and ("shortcut" not in _name)
            and isinstance(_module, (nn.Conv2d, NunchakuConv2dAsLinear))
        ):
            conv2d_pre_hooks.append(_module.register_forward_pre_hook(conv2d_hook.pre_hook))
            conv2d_post_hooks.append(_module.register_forward_hook(conv2d_hook.post_hook))
        elif (
            ("conv" in _name and "resnets" in _name and "linear" in _name)
            and ("shortcut" not in _name)
            and isinstance(_module, SVDQW4A4Linear)
        ):
            conv2d_linear_pre_hooks.append(_module.register_forward_pre_hook(conv2d_linear_hook.pre_hook))
            conv2d_linear_post_hooks.append(_module.register_forward_hook(conv2d_linear_hook.post_hook))
        # elif isinstance(_module, (NunchakuSDXLTransformerBlock, BasicTransformerBlock)):
        #     transformer_pre_hooks.append(_module.register_forward_pre_hook(transformer_hook.pre_hook))
        #     transformer_post_hooks.append(_module.register_forward_hook(transformer_hook.post_hook))
        elif "attn1" in _name and isinstance(_module, (NunchakuSDXLAttention, Attention)):
            attn1_pre_hooks.append(_module.register_forward_pre_hook(attn1_hook.pre_hook))
            attn1_post_hooks.append(_module.register_forward_hook(attn1_hook.post_hook))
        elif "attn2" in _name and isinstance(_module, (NunchakuSDXLAttention, Attention)):
            attn2_pre_hooks.append(_module.register_forward_pre_hook(attn2_hook.pre_hook))
            attn2_post_hooks.append(_module.register_forward_hook(attn2_hook.post_hook))
        elif "attn2.to_q" in _name and isinstance(_module, (SVDQW4A4Linear, nn.Linear)):
            attn2_q_pre_hooks.append(_module.register_forward_pre_hook(attn2_q_hook.pre_hook))
            attn2_q_post_hooks.append(_module.register_forward_hook(attn2_q_hook.post_hook))
        # elif "attn2.to_k" in _name and isinstance(_module, (SVDQW4A4Linear, nn.Linear)):
        #     attn2_k_pre_hooks.append(_module.register_forward_pre_hook(attn2_k_hook.pre_hook))
        #     attn2_k_post_hooks.append(_module.register_forward_hook(attn2_k_hook.post_hook))
        # elif "attn2.to_v" in _name and isinstance(_module, (SVDQW4A4Linear, nn.Linear)):
        #     attn2_v_pre_hooks.append(_module.register_forward_pre_hook(attn2_v_hook.pre_hook))
        #     attn2_v_post_hooks.append(_module.register_forward_hook(attn2_v_hook.post_hook))
        elif "ff.net.2" in _name and isinstance(_module, (SVDQW4A4Linear, nn.Linear)):
            ffnet2_pre_hooks.append(_module.register_forward_pre_hook(ffnet2_hook.pre_hook))
            ffnet2_post_hooks.append(_module.register_forward_hook(ffnet2_hook.post_hook))
        elif "attn1.to_out.0" in _name and isinstance(_module, (SVDQW4A4Linear, nn.Linear)):
            attn1_out_pre_hooks.append(_module.register_forward_pre_hook(attn1_out_hook.pre_hook))
            attn1_out_post_hooks.append(_module.register_forward_hook(attn1_out_hook.post_hook))
        elif "attn2.to_out.0" in _name and isinstance(_module, (SVDQW4A4Linear, nn.Linear)):
            attn2_out_pre_hooks.append(_module.register_forward_pre_hook(attn2_out_hook.pre_hook))
            attn2_out_post_hooks.append(_module.register_forward_hook(attn2_out_hook.post_hook))
        elif "ff.net.0.proj" in _name and isinstance(_module, (SVDQW4A4Linear, nn.Linear)):
            ffnet_proj_pre_hooks.append(_module.register_forward_pre_hook(ffnet_proj_hook.pre_hook))
            ffnet_proj_post_hooks.append(_module.register_forward_hook(ffnet_proj_hook.post_hook))

    for _ in range(runs):
        _ = pipeline(
            prompt=prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=inference_steps,
            num_images_per_prompt=batch_size,
        ).images

    handle_pre.remove()
    handle_post.remove()
    for h in resnets_pre_hooks:
        h.remove()
    for h in resnets_post_hooks:
        h.remove()
    for h in conv2d_pre_hooks:
        h.remove()
    for h in conv2d_post_hooks:
        h.remove()
    for h in conv2d_linear_pre_hooks:
        h.remove()
    for h in conv2d_linear_post_hooks:
        h.remove()
    # for h in transformer_pre_hooks:
    #     h.remove()
    # for h in transformer_post_hooks:
    #     h.remove()
    for h in attn1_pre_hooks:
        h.remove()
    for h in attn1_post_hooks:
        h.remove()
    for h in attn2_pre_hooks:
        h.remove()
    for h in attn2_post_hooks:
        h.remove()
    for h in attn2_q_pre_hooks:
        h.remove()
    for h in attn2_q_post_hooks:
        h.remove()
    # for h in attn2_k_pre_hooks:
    #     h.remove()
    # for h in attn2_k_post_hooks:
    #     h.remove()
    # for h in attn2_v_pre_hooks:
    #     h.remove()
    # for h in attn2_v_post_hooks:
    #     h.remove()
    for h in ffnet2_pre_hooks:
        h.remove()
    for h in ffnet2_post_hooks:
        h.remove()
    for h in attn1_out_pre_hooks:
        h.remove()
    for h in attn1_out_post_hooks:
        h.remove()
    for h in attn2_out_pre_hooks:
        h.remove()
    for h in attn2_out_post_hooks:
        h.remove()
    for h in ffnet_proj_pre_hooks:
        h.remove()
    for h in ffnet_proj_post_hooks:
        h.remove()

    return [
        _total(whole_unet_hook),
        _total(resnets_hook),
        _total(conv2d_hook),
        _total(conv2d_linear_hook),
        # _total(transformer_hook),
        _total(attn1_hook),
        _total(attn1_out_hook),
        _total(attn2_hook),
        _total(attn2_q_hook),
        # _total(attn2_k_hook),
        # _total(attn2_v_hook),
        _total(attn2_out_hook),
        _total(ffnet2_hook),
        _total(ffnet_proj_hook),
    ]


def plot_stat(
    title: str = None, plot_save_path: Path = None, stat_data: Dict[str, List] = None, item_names: List = None
):
    x = np.arange(len(item_names))
    width = 0.35

    fig, ax = plt.subplots()

    rects = []

    shift = -1

    for rect_name, rect_data in stat_data.items():
        rect = ax.bar(x + shift * width / 2, rect_data, width, label=rect_name)
        rects.append(rect)
        shift = -shift

    ax.set_ylabel("Time cost (seconds)")
    ax.set_xlabel("Modules")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(item_names, fontsize=6)
    ax.legend()

    def autolabel(rects, fontsize):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=fontsize,
            )

    for r in rects:
        autolabel(r, 6)

    plt.tight_layout()
    plt.savefig(plot_save_path / "stat.png", dpi=300, bbox_inches="tight")


@pytest.mark.skipif(
    is_turing() or get_precision() == "fp4", reason="Skip tests due to using Turing GPUs or FP4 precision"
)
def test_sdxl_time_cost_stat():
    batch_size = 4
    runs = 5
    inference_steps = 30
    guidance_scale = 5.0
    device_name = torch.cuda.get_device_name(0)

    # pipeline_original = StableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, use_safetensors=True, variant="fp16"
    # ).to("cuda")

    # cost_original = run_stat(
    #     pipeline_original, batch_size, guidance_scale, device_name, runs, inference_steps
    # )
    # del pipeline_original.unet
    # del pipeline_original.text_encoder
    # del pipeline_original.text_encoder_2
    # del pipeline_original.vae
    # del pipeline_original

    quantized_unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        # "nunchaku-tech/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors"
        "/data/dongd/quantized-models/merged/sdxl-merged-20251017_1727.safetensors"
    )
    pipeline_quantized = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=quantized_unet,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    cost_quantized = run_stat(pipeline_quantized, batch_size, guidance_scale, device_name, runs, inference_steps)
    del pipeline_quantized.unet
    del pipeline_quantized.text_encoder
    del pipeline_quantized.text_encoder_2
    del pipeline_quantized.vae
    del pipeline_quantized

    # stat_item_names = ["total", "resnets", "conv2d", "t_blocks", "attn1", "attn2", "attn2_q", "attn2_k", "attn2_v", "ff.net.2"]
    # stat_item_names = ["total", "resnets", "attn1",  "attn1\nout", "attn2", "attn2\nq", "attn2\nk", "attn2\nv", "attn2\nout", "ff.net\n2", "ff.net\n0_proj"]
    stat_item_names = [
        "total",
        "resnets",
        "conv2d",
        "conv2d\nlinear",
        "attn1",
        "attn1\nout",
        "attn2",
        "attn2\nq",
        "attn2\nout",
        "ff.net\n2",
        "ff.net\n0_proj",
    ]

    ref_root = Path(os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", "nunchaku-test-cache"))
    plot_save_path = ref_root / "time_stat" / "sdxl"
    os.makedirs(plot_save_path, exist_ok=True)

    plot_stat(
        title=f"SDXL inference time stat\n{runs} runs of {inference_steps} steps each @ batch size {batch_size}\nGPU: {device_name}",
        plot_save_path=plot_save_path,
        stat_data={
            # "original fp16": cost_original,
            "nunchaku int4": cost_quantized
        },
        item_names=stat_item_names,
    )
