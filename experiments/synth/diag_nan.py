"""Locate the NaN source: base SDXL vs ControlNet vs attention slicing on MPS."""
import os

import torch
from diffusers import (AutoencoderKL, ControlNetModel, EulerAncestralDiscreteScheduler,
                       StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline)
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "hf_cache")
dtype = torch.bfloat16
device = "mps"

cn = ControlNetModel.from_pretrained("TheMistoAI/MistoLine", torch_dtype=dtype,
                                     variant="fp16", cache_dir=CACHE)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                    torch_dtype=dtype, cache_dir=CACHE)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "Laxhar/noobai-XL-1.1", controlnet=cn, vae=vae, torch_dtype=dtype, cache_dir=CACHE)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

cond = ImageOps.invert(Image.open(
    os.path.join(HERE, "r2a/renders_v2/seed_san_normal_wave_full/line.png")).convert("L")).convert("RGB")
cond = cond.resize((512, 680))

PROMPT = "masterpiece, best quality, 1boy, solo, sketch, rough sketch, monochrome"
NEG = "worst quality, bad quality"


def nan_cb(p, i, t, kw):
    lat = kw["latents"]
    if torch.isnan(lat).any():
        print(f"  step {i}: NaN in latents ({torch.isnan(lat).float().mean():.2%})", flush=True)
    return kw


def stats(img):
    import numpy as np
    a = np.array(img)
    return f"mean={a.mean():.1f} std={a.std():.1f}"


g = torch.Generator("cpu").manual_seed(1)

print("=== T1: base SDXL, NO controlnet, no slicing ===", flush=True)
base = StableDiffusionXLPipeline(vae=pipe.vae, text_encoder=pipe.text_encoder,
                                 text_encoder_2=pipe.text_encoder_2, tokenizer=pipe.tokenizer,
                                 tokenizer_2=pipe.tokenizer_2, unet=pipe.unet,
                                 scheduler=pipe.scheduler)
img = base(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=10, guidance_scale=5.5,
           width=512, height=680, generator=g, callback_on_step_end=nan_cb).images[0]
print("T1 result:", stats(img), flush=True)
img.save("r2a/diag_t1_base.png")

print("=== T2: + controlnet cs=0.9, no slicing ===", flush=True)
img = pipe(prompt=PROMPT, negative_prompt=NEG, image=cond, controlnet_conditioning_scale=0.9,
           num_inference_steps=10, guidance_scale=5.5, width=512, height=680,
           generator=g, callback_on_step_end=nan_cb).images[0]
print("T2 result:", stats(img), flush=True)
img.save("r2a/diag_t2_cn.png")

print("=== T3: + attention slicing ===", flush=True)
pipe.enable_attention_slicing()
img = pipe(prompt=PROMPT, negative_prompt=NEG, image=cond, controlnet_conditioning_scale=0.9,
           num_inference_steps=10, guidance_scale=5.5, width=512, height=680,
           generator=g, callback_on_step_end=nan_cb).images[0]
print("T3 result:", stats(img), flush=True)
img.save("r2a/diag_t3_slicing.png")
print("DIAG DONE", flush=True)
