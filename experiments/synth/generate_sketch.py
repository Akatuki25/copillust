"""R2a: 3D line render -> sketch-style image via NoobAI-XL + MistoLine ControlNet.

Usage:
  genv/bin/python generate_sketch.py --scenes scene1,scene2 [--styles all]
      [--scales 0.55,0.9] [--steps 28] [--limit N]

Reads r2a/renders/<scene>/line.png as the ControlNet condition (inverted to
white-on-black), writes r2a/gen/<scene>/<style>_cs<scale>.png + meta.json.
"""
import argparse
import gc
import json
import os
import random
import time

import torch
from diffusers import (AutoencoderKL, ControlNetModel, EulerAncestralDiscreteScheduler,
                       StableDiffusionXLControlNetPipeline)
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "hf_cache")
RENDERS = os.environ.get("RENDERS_DIR", os.path.join(HERE, "r2a", "renders_v2"))
OUT = os.path.join(HERE, "r2a", "gen")

QUALITY = "masterpiece, best quality, newest"
NEG = ("worst quality, bad quality, lowres, bad anatomy, bad hands, "
       "photorealistic, 3d, watermark, signature, text")

STYLES = {
    "rough": "sketch, rough sketch, monochrome, greyscale, white background",
    "pencil": "traditional media, colored pencil (medium), sketch, painting (medium)",
    "lineart": "lineart, monochrome, white background",
}


HAIR = ["short hair", "long hair", "twintails", "ponytail", "bob cut", "messy hair"]
HAIR_COLOR = ["black hair", "brown hair", "blonde hair", "silver hair"]
OUTFIT = ["school uniform", "hoodie", "t-shirt", "shorts", "jacket", "casual clothes"]


def character_tags(scene, style, cs, seed):
    """Mannequin renders (fem/masc vroid) get diffusion-invented hair/clothes so
    the pixel domain matches real character sketches (2026-07-04 決定: 方針(b)).
    seed_san already has hair/clothes in the condition lines — inventing different
    ones there only invites geometric drift. Deterministic per (scene,style,cs)."""
    if not scene.startswith(("fem", "masc")):  # fem_vroid/masc_vroid + fem2/masc2
        return []
    rng = random.Random(f"{scene}|{style}|{cs}|{seed}")
    tags = [rng.choice(HAIR), rng.choice(OUTFIT)]
    if style == "pencil":
        tags.insert(1, rng.choice(HAIR_COLOR))
    return tags


def scene_tags(scene):
    subj = "1boy" if ("masc" in scene or "seed" in scene) else "1girl"
    tags = [subj, "solo", "simple background"]
    if "chibi" in scene:
        tags.append("chibi")
    if "_bust" in scene:
        tags.append("upper body")
    else:
        tags.append("full body")
    if "high" in scene:
        tags.append("from above")
    return ", ".join(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="")
    ap.add_argument("--styles", default="rough,pencil,lineart")
    ap.add_argument("--scales", default="0.55,0.9")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        # bf16: fp16 UNet/ControlNet overflows to NaN on MPS
        device, dtype = "mps", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32

    cn = ControlNetModel.from_pretrained("TheMistoAI/MistoLine", torch_dtype=dtype,
                                         variant="fp16", cache_dir=CACHE)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                        torch_dtype=dtype, cache_dir=CACHE)
    vae.config.force_upcast = False
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "Laxhar/noobai-XL-1.1", controlnet=cn, vae=vae, torch_dtype=dtype, cache_dir=CACHE)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    # NOTE: enable_attention_slicing() produces 100% NaN latents on MPS — do not use.
    # On CUDA 16GB neither slicing nor offload is needed.
    if device == "mps":
        pipe.enable_vae_slicing()

    scenes = [s for s in args.scenes.split(",") if s] or sorted(os.listdir(RENDERS))
    styles = args.styles.split(",")
    scales = [float(x) for x in args.scales.split(",")]
    n_total = len(scenes) * len(styles) * len(scales)
    done = 0
    for scene in scenes:
        line_path = os.path.join(RENDERS, scene, "line.png")
        if not os.path.exists(line_path):
            print("skip (no line.png):", scene)
            continue
        cond = ImageOps.invert(Image.open(line_path).convert("L")).convert("RGB")
        os.makedirs(os.path.join(OUT, scene), exist_ok=True)
        for style in styles:
            for cs in scales:
                extra = character_tags(scene, style, cs, args.seed)
                prompt = ", ".join([QUALITY, scene_tags(scene)] + extra + [STYLES[style]])
                name = f"{style}_cs{cs:.2f}".replace(".", "p")
                fp = os.path.join(OUT, scene, name + ".png")
                if os.path.exists(fp):
                    done += 1
                    continue
                t0 = time.time()
                g = torch.Generator("cpu").manual_seed(args.seed)
                img = pipe(prompt=prompt, negative_prompt=NEG, image=cond,
                           controlnet_conditioning_scale=cs,
                           num_inference_steps=args.steps, guidance_scale=args.cfg,
                           width=cond.width, height=cond.height, generator=g).images[0]
                img.save(fp)
                meta = {"scene": scene, "style": style, "cond_scale": cs,
                        "prompt": prompt, "neg": NEG, "steps": args.steps,
                        "cfg": args.cfg, "seed": args.seed, "sec": round(time.time() - t0, 1)}
                with open(fp.replace(".png", ".json"), "w") as f:
                    json.dump(meta, f, indent=1)
                done += 1
                print(f"[{done}/{n_total}] {scene}/{name} {meta['sec']}s", flush=True)
                gc.collect()
                if args.limit and done >= args.limit:
                    print("limit reached")
                    return


if __name__ == "__main__":
    main()
