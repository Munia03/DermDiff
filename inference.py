from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
import os

model_path = "skin_dermdiff_model"

out_dir = 'outdir'
os.makedirs(f'{out_dir}/', exist_ok=True)

skin_tones = ['lighter', 'brown', 'darker']


for chk in range(10):
    chk_str = str((chk+1)*1000)
    os.makedirs(f'{out_dir}/{chk_str}/', exist_ok=True)
    unet = UNet2DConditionModel.from_pretrained(model_path + f"/checkpoint-{chk_str}/unet",
                                                torch_dtype=torch.float16,
                                                use_safetensors=True)

    pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                   unet=unet,
                                                   torch_dtype=torch.float16,
                                                   variant="fp16",
                                                   safety_checker=None,
                                                   use_safetensors=True)
    pipe.to("cuda")

    print(chk_str)
    for i in skin_tones:
        prompt = "benign skin disease on " + str(i) + "skin tone"
        image = pipe(prompt).images[0]
        image.save(f"{out_dir}/{chk_str}/benign-{str(i)}.png")

        prompt = "malignant skin disease on " + str(i) + "skin tone "
        image = pipe(prompt).images[0]
        image.save(f"{out_dir}/{chk_str}/malignant-{str(i)}.png")