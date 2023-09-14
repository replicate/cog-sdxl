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
        print(len(data['output']))
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
    gen(
        f"sample.callbacks.no-lora.png",
        prompt="A watercolor painting of TOK on the beach",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        preview_steps=5,
        seed=1000,
        width=768,
        height=768,
    )
    return
    
    gen(
        f"sample.0.no-lora.png",
        prompt="A watercolor painting of TOK on the beach",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=768,
        height=768,
    )
    gen(
        f"sample.1.dbag-lora.png",
        prompt="A watercolor painting of TOK on the beach",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=768,
        height=768,
        replicate_weights="https://pbxt.replicate.delivery/h8XgfJ4TIfrLIkVrAXgAAn9rGHNOeYfxW9y1UN1ft3ZZVASMC/trained_model.tar",
    )
    gen(
        f"sample.2.no-lora.png",
        prompt="A watercolor painting of TOK on the beach",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=768,
        height=768,
    )
    gen(
        f"sample.3.emoji-lora.png",
        prompt="A watercolor painting of TOK on the beach",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=768,
        height=768,
        replicate_weights="https://pbxt.replicate.delivery/DUxxgRlwU5q3DNhaaEPnH70H6afeUh18iIFTZkbioqVWeoEjA/trained_model.tar"
    )
    gen(
        f"sample.4.emoji-lora.png",
        prompt="A TOK emoji of a dog",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=768,
        height=768,
        replicate_weights="https://pbxt.replicate.delivery/DUxxgRlwU5q3DNhaaEPnH70H6afeUh18iIFTZkbioqVWeoEjA/trained_model.tar"
    )
    gen(
        f"sample.5.dbag-lora.png",
        prompt="A watercolor painting of TOK on the beach",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=768,
        height=768,
        replicate_weights="https://pbxt.replicate.delivery/h8XgfJ4TIfrLIkVrAXgAAn9rGHNOeYfxW9y1UN1ft3ZZVASMC/trained_model.tar",
    )



if __name__ == "__main__":
    main()
