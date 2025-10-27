import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from diffusers.models.resnet import ResnetBlock2D
from torch.profiler import ProfilerActivity, profile

from nunchaku.models.unets.resblock import NunchakuSDXLResnetBlock2D
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel


class ProfilerHook:
    def __init__(self, profiler_name):
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler_name = profiler_name

    def pre(self, module, input):
        print(f">>> pre {type(module)}")
        self.profiler.start()
        print(f">>> pre {type(module)} started")
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def post(self, module, input, output):
        self.profiler.stop()
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_event.record()

    def register_as_hook(self, module: nn.Module):
        self.pre_hook = module.register_forward_pre_hook(self.pre)
        self.post_hook = module.register_forward_hook(self.post)

    def export_and_unhook(self):
        self.profiler.export_chrome_trace(f"/data/dongd/sdxl_conv_profiler_trace/{self.profiler_name}_trace.json")
        self.pre_hook.remove()
        self.post_hook.remove()
        print(f"{self.profiler_name} elapsed time: {self.start_event.elapsed_time(self.end_event)}")


def register_profiler_hook(module_name, module: nn.Module):
    profiler_hook = ProfilerHook(module_name)
    profiler_hook.register_as_hook(module)
    return profiler_hook


def resnet_pre(module: ResnetBlock2D, input):
    print(
        f">>>> input[0]: {input[0].shape}, input[1]: {input[1].shape} norm2.weight {module.norm2.weight.shape}, norm2.bias {module.norm2.bias.shape}"
    )


if __name__ == "__main__":
    unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        # "/data/dongd/quantized-models/merged/sdxl-merged-20251017_1727.safetensors"
        "/data/dongd/quantized-models/merged/sdxl-converted-20251017_1727_warp_n_64.safetensors"
    )

    _hooks = []

    for _name, _module in unet.named_modules():
        # if ("resnets" in _name) and ("conv" in _name) and ("shortcut" not in _name) and isinstance(_module, NunchakuConv2dAsLinear):
        #     profiler_hooks.append(register_profiler_hook(_name, _module))
        # elif ("transformer_blocks" in _name) and _name.endswith("ff"):
        #     profiler_hooks.append(register_profiler_hook(_name, _module))
        if ("resnets" in _name) and isinstance(_module, (ResnetBlock2D, NunchakuSDXLResnetBlock2D)):
            _hooks.append(_module.register_forward_pre_hook(resnet_pre))
            # profiler_hooks.append(register_profiler_hook(_name, _module))

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=unet,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

    image = pipeline(prompt=prompt, guidance_scale=5.0, num_inference_steps=50, num_images_per_prompt=1).images[0]

    # for hook in _hooks:
    #     hook.export_and_unhook()

    image.save("sdxl-conv-no-skip.png")
