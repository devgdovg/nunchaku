import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2DModelV2

# from safetensors.torch import load_file

# from examples.mapping import comfy_to_diffusers

# from diffusers.models.transformers.transformer_flux import (
#     FluxSingleTransformerBlock,
#     FluxTransformer2DModel,
#     FluxTransformerBlock,
# )


if __name__ == "__main__":

    seed = 1234567
    generator1 = torch.Generator(device="cuda").manual_seed(seed)
    prompt = "A cat holding a sign that says hello world"

    ####################################################
    ####################################################

    quantized_model = NunchakuFluxTransformer2DModelV2.from_pretrained(
        # "/data/dongd/quantized-models/merged/flux1-dev-custom-converted-f0a56ab51074043f7fb95e86f5246031e2424ea1a54b64a8948f6e612fb52fd6-20250910_0837.safetensors"
        "/data/dongd/quantized-models/merged/svdq-int4_r32-flux.1-dev.safetensors"
    )
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=quantized_model, torch_dtype=torch.bfloat16
    ).to("cuda")
    # file = open("inference_trace_flux_transformer_block_quantized.txt", "a", encoding="utf-8")
    # add_hook(quantized_model, file, NunchakuFluxTransformerBlock)

    # image = pipeline(prompt=prompt, num_inference_steps=20, guidance_scale=3.5, generator=generator1).images[0]
    image = pipeline(prompt=prompt, num_inference_steps=20, guidance_scale=3.5).images[0]
    image.save("cat_quantized.png")

    # file.close()

    #####################################################
    #####################################################

    # path = "/data/dongd/original-models/flux/f0a56ab51074043f7fb95e86f5246031e2424ea1a54b64a8948f6e612fb52fd6"
    # with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
    #     json_config = json.load(f)
    # original_model = FluxTransformer2DModel.from_config(json_config).to(torch.float8_e4m3fn)
    # print(f"original_model on device: {next(original_model.parameters()).device}")
    # checkpoint_file = os.path.basename(path) + ".safetensors"
    # state_dict = comfy_to_diffusers(load_file(os.path.join(path, checkpoint_file)))
    # original_model.load_state_dict(state_dict)
    # pipeline = FluxPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-dev", transformer=original_model, torch_dtype=torch.float8_e4m3fn
    # ).to("cuda:4")
    # file = open("inference_trace_flux_transformer_block_original.txt", "a", encoding="utf-8")
    # add_hook(original_model, file, FluxTransformerBlock)

    # image = pipeline(prompt=prompt, num_inference_steps=20, guidance_scale=3.5, generator=generator1).images[0]
    # image.save("cat_original.png")

    # file.close()
