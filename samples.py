"""
A handy utility for verifying SDXL image generation locally. 
To set up, first run a local cog server using:
   cog run -p 5000 python -m cog.server.http
Then, in a separate terminal, generate samples
   python samples.py
"""


import base64
import os
import sys

import requests


def gen(output_fn, **kwargs):
    if os.path.exists(output_fn):
        return

    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    try:
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        sys.exit(1)

    with open(output_fn, "wb") as f:
        f.write(data)


def main():
    SCHEDULERS = [
        "DDIM",
        "DPMSolverMultistep",
        "HeunDiscrete",
        "KarrasDPM",
        "K_EULER_ANCESTRAL",
        "K_EULER",
        "PNDM",
    ]

    gen(
        f"sample.txt2img.png",
        prompt="A studio portrait photo of a cat",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=1024,
        height=1024,
    )

    for refiner in ["base_image_refiner", "expert_ensemble_refiner", "no_refiner"]:
        gen(
            f"sample.img2img.{refiner}.png",
            prompt="a photo of an astronaut riding a horse on mars",
            image="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png",
            prompt_strength=0.8,
            num_inference_steps=25,
            refine=refiner,
            guidance_scale=7,
            negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
            seed=42,
        )

        gen(
            f"sample.inpaint.{refiner}.png",
            prompt="A majestic tiger sitting on a bench",
            image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
            mask="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png",
            prompt_strength=0.8,
            num_inference_steps=25,
            refine=refiner,
            guidance_scale=7,
            negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
            seed=42,
        )

    for split in range(0, 10):
        split = split / 10.0
        gen(
            f"sample.expert_ensemble_refiner.{split}.txt2img.png",
            prompt="A studio portrait photo of a cat",
            num_inference_steps=25,
            guidance_scale=7,
            refine="expert_ensemble_refiner",
            high_noise_frac=split,
            negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
            seed=1000,
            width=1024,
            height=1024,
        )

    gen(
        f"sample.refine.txt2img.png",
        prompt="A studio portrait photo of a cat",
        num_inference_steps=25,
        guidance_scale=7,
        refine="base_image_refiner",
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=1024,
        height=1024,
    )
    gen(
        f"sample.refine.10.txt2img.png",
        prompt="A studio portrait photo of a cat",
        num_inference_steps=25,
        guidance_scale=7,
        refine="base_image_refiner",
        refine_steps=10,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=1024,
        height=1024,
    )

    gen(
        "samples.2.txt2img.png",
        prompt="A studio portrait photo of a cat",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        scheduler="KarrasDPM",
        num_outputs=2,
        seed=1000,
        width=1024,
        height=1024,
    )

    for s in SCHEDULERS:
        gen(
            f"sample.{s}.txt2img.png",
            prompt="A studio portrait photo of a cat",
            num_inference_steps=25,
            guidance_scale=7,
            negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
            scheduler=s,
            seed=1000,
            width=1024,
            height=1024,
        )


if __name__ == "__main__":
    main()
