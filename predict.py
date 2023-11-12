import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor
from diffusers import DiffusionPipeline, LCMScheduler, DDIMScheduler

lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"


def download_weights(url, dest, extract=True):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if extract:
        cmd = ["pget", "-x", url, dest]
    else:
        cmd = ["pget", url, dest]
    subprocess.check_call(cmd, close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        self.lora_url = 'setup'  # this allows us to load the weights on the first run

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        print("Loading sdxl txt2img pipeline...")
        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)
        self.txt2img = DiffusionPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
        self.original_scheduler = self.txt2img.scheduler
        if weights:
            self.load_lora_weights(str(weights))
        else:
            self.load_lora_weights(None)

        self.txt2img.to("cuda")

        if not os.path.exists(REFINER_MODEL_CACHE):
            download_weights(REFINER_URL, REFINER_MODEL_CACHE)

        print("Loading refiner pipeline...")
        self.refiner = DiffusionPipeline.from_pretrained(
            REFINER_MODEL_CACHE,
            text_encoder_2=self.txt2img.text_encoder_2,
            vae=self.txt2img.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
        print("setup took: ", time.time() - start)

    def load_lora_weights(self, weights_url, lcm_scale=1.0, style_scale=0.8):
        if weights_url != self.lora_url:
            self.txt2img.unload_lora_weights()

            self.txt2img.load_lora_weights(lcm_lora_id, adapter_name="lcm")

            if weights_url:
                if os.path.exists("style-lora.safetensors"):
                    os.remove("style-lora.safetensors")
                download_weights(weights_url, "style-lora.safetensors", extract=False)
                self.txt2img.load_lora_weights("style-lora.safetensors", adapter_name="style")
                self.lora_url = weights_url
            else:
                self.lora_url = None

        enable_lcm = lcm_scale > 0.0

        if enable_lcm:
            self.txt2img.scheduler = LCMScheduler.from_config(self.original_scheduler.config)
        else:
            # FIXME(ja): allow other schedulers than DDIM
            self.txt2img.scheduler = DDIMScheduler.from_config(self.original_scheduler.config)

        if enable_lcm and weights_url:
            self.txt2img.set_adapters(["lcm", "style"], adapter_weights=[lcm_scale, style_scale])
        elif enable_lcm:
            self.txt2img.set_adapters(["lcm"], adapter_weights=[lcm_scale])
        elif enable_lcm:
            self.txt2img.set_adapters(["style"], adapter_weights=[style_scale])
        else:
            self.txt2img.set_adapters([])


    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=4
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0, le=20, default=2.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "expert_ensemble_refiner", "base_image_refiner"],
            default="no_refiner",
        ),
        high_noise_frac: float = Input(
            description="For expert_ensemble_refiner, the fraction of noise to use",
            default=0.8,
            le=1.0,
            ge=0.0,
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
            default=None,
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False,
        ),
        replicate_weights: str = Input(
            description="safelora weights to use",
            default=None,
        ),
        lcm_scale: float = Input(
            description="Scale for LCM, if 0, the DDIM scheduler is used",
            default=1.0,
        ),
        style_scale: float = Input(
            description="Scale for style LoRA",
            default=0.8,
        )
    ) -> List[Path]:
        """Run a single prediction on the model."""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        self.load_lora_weights(replicate_weights, lcm_scale, style_scale)

        # OOMs can leave vae in bad state
        if self.txt2img.vae.dtype == torch.float32:
            self.txt2img.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height
        pipe = self.txt2img

        if refine == "expert_ensemble_refiner":
            sdxl_kwargs["output_type"] = "latent"
            sdxl_kwargs["denoising_end"] = high_noise_frac
        elif refine == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None
            self.refiner.watermark = None

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = pipe(**common_args, **sdxl_kwargs)

        if refine in ["expert_ensemble_refiner", "base_image_refiner"]:
            refiner_kwargs = {
                "image": output.images,
            }

            if refine == "expert_ensemble_refiner":
                refiner_kwargs["denoising_start"] = high_noise_frac
            if refine == "base_image_refiner" and refine_steps:
                common_args["num_inference_steps"] = refine_steps

            output = self.refiner(**common_args, **refiner_kwargs)

        if not apply_watermark:
            pipe.watermark = watermark_cache
            self.refiner.watermark = watermark_cache

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
