# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File
import hippo_sd
import torch
import tempfile

# set device to cuda seems save gpu memory
torch.cuda.current_device()
from hippo_sd.pipeline_stable_diffusion_xl import replace_pipeline
from hippo_sd.pipeline_stable_diffusion_xl_img2img import replace_img2img_pipeline

replace_pipeline()
replace_img2img_pipeline()

from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

from hippo_sd.pipeline_stable_diffusion import StableDiffusionHippoPipeline


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        import os

        os.system("pwd")
        quantize = False
        self.model = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.model.enable_hippo_engine(quantize)
        self.model.to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Prompt to generate image from"),
        height: int = Input(1024, description="Height of generated image"),
        width: int = Input(1024, description="Width of generated image"),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        refiner_dir = None
        output_type = "latent" if refiner_dir is not None else "pil"
        with torch.autocast("cuda"):
            images = self.model(
                prompt=prompt, height=height, width=width, output_type=output_type
            ).images
            output_path = Path(tempfile.mkdtemp()) / "output.png"
            images[0].save(output_path)
            return Path(output_path)
