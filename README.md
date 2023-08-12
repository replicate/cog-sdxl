# Cog-SDXL

[![Replicate demo and cloud API](https://replicate.com/stability-ai/sdxl/badge)](https://replicate.com/stability-ai/sdxl)

This is an implementation of the [SDXL](https://github.com/Stability-AI/generative-models) as a Cog model. [Cog packages machine learning models as standard containers](https://github.com/replicate/cog).

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

for prediction,

```bash
cog predict -i prompt="a photo of TOK"
```

```bash
cog train -i input_images=@example_datasets/zeke.zip -i use_face_detection_instead=True
```

## Update notes

**2021-08-12**
* Input types are inferred from input name extensions, or from the `input_images_filetype` argument
* Preprocssing are now done with fp16, and if no mask is found, the model will use the whole image

**2023-08-11**
* Default to 768x768 resolution training
* Rank as argument now, default to 32
* Now uses Swin2SR `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` as default, and will upscale + downscale to 768x768