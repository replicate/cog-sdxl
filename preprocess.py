# Have SwinIR upsample
# Have BLIP auto caption
# Have CLIPSeg auto mask concept

import gc
import glob
import mimetypes
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from zipfile import ZipFile

import fire
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    Swin2SRForImageSuperResolution,
    Swin2SRImageProcessor,
)

MODEL_PATH = "./cache"
TEMP_OUT_DIR = "./temp/"
TEMP_IN_DIR = "./temp_in/"


def preprocess(
    input_zip_path: Path,
    caption_text: str,
    mask_target_prompts: str,
    target_size: int,
    crop_based_on_salience: bool,
    use_face_detection_instead: bool,
    temp: float,
    substitution_tokens: List[str],
) -> Path:
    # assert str(files).endswith(".zip"), "files must be a zip file"

    # clear TEMP_IN_DIR first.

    for path in [TEMP_OUT_DIR, TEMP_IN_DIR]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    if str(input_zip_path).endswith(".zip"):
        with ZipFile(str(input_zip_path), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                    "__MACOSX"
                ):
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith("image/"):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, TEMP_IN_DIR)
    else:
        assert str(input_zip_path).endswith(
            ".tar"
        ), "files must be a tar file if not zip"
        with tarfile.open(input_zip_path, "r") as tar_ref:
            for tar_info in tar_ref:
                if tar_info.name[-1] == "/" or tar_info.name.startswith("__MACOSX"):
                    continue

                mt = mimetypes.guess_type(tar_info.name)
                if mt and mt[0] and mt[0].startswith("image/"):
                    tar_info.name = os.path.basename(tar_info.name)
                    tar_ref.extract(tar_info, TEMP_IN_DIR)

    output_dir: str = TEMP_OUT_DIR

    load_and_save_masks_and_captions(
        files=TEMP_IN_DIR,
        output_dir=output_dir,
        caption_text=caption_text,
        mask_target_prompts=mask_target_prompts,
        target_size=target_size,
        crop_based_on_salience=crop_based_on_salience,
        use_face_detection_instead=use_face_detection_instead,
        temp=temp,
        substitution_tokens=substitution_tokens,
    )

    return Path(TEMP_OUT_DIR)


@torch.no_grad()
def swin_ir_sr(
    images: List[Image.Image],
    model_id: Literal[
        "caidas/swin2SR-classical-sr-x2-64", "caidas/swin2SR-classical-sr-x4-48"
    ] = "caidas/swin2SR-classical-sr-x2-64",
    target_size: Optional[Tuple[int, int]] = None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    **kwargs,
) -> List[Image.Image]:
    """
    Upscales images using SwinIR. Returns a list of PIL images.
    If the image is already larger than the target size, it will not be upscaled
    and will be returned as is.

    """

    model = Swin2SRForImageSuperResolution.from_pretrained(
        model_id, cache_dir=MODEL_PATH
    ).to(device)
    processor = Swin2SRImageProcessor()

    out_images = []

    for image in tqdm(images):
        ori_w, ori_h = image.size
        if target_size is not None:
            if ori_w >= target_size[0] and ori_h >= target_size[1]:
                out_images.append(image)
                continue

        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        output = (
            outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        )
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)
        output = Image.fromarray(output)

        out_images.append(output)

    return out_images


@torch.no_grad()
def clipseg_mask_generator(
    images: List[Image.Image],
    target_prompts: Union[List[str], str],
    model_id: Literal[
        "CIDAS/clipseg-rd64-refined", "CIDAS/clipseg-rd16"
    ] = "CIDAS/clipseg-rd64-refined",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    bias: float = 0.01,
    temp: float = 1.0,
    **kwargs,
) -> List[Image.Image]:
    """
    Returns a greyscale mask for each image, where the mask is the probability of the target prompt being present in the image
    """

    if isinstance(target_prompts, str):
        print(
            f'Warning: only one target prompt "{target_prompts}" was given, so it will be used for all images'
        )

        target_prompts = [target_prompts] * len(images)

    processor = CLIPSegProcessor.from_pretrained(model_id, cache_dir=MODEL_PATH)
    model = CLIPSegForImageSegmentation.from_pretrained(
        model_id, cache_dir=MODEL_PATH
    ).to(device)

    masks = []

    for image, prompt in tqdm(zip(images, target_prompts)):
        original_size = image.size

        inputs = processor(
            text=[prompt, ""],
            images=[image] * 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits / temp, dim=0)[0]
        probs = (probs + bias).clamp_(0, 1)
        probs = 255 * probs / probs.max()

        # make mask greyscale
        mask = Image.fromarray(probs.cpu().numpy()).convert("L")

        # resize mask to original size
        mask = mask.resize(original_size)

        masks.append(mask)

    return masks


@torch.no_grad()
def blip_captioning_dataset(
    images: List[Image.Image],
    text: Optional[str] = None,
    model_id: Literal[
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-image-captioning-base",
    ] = "Salesforce/blip-image-captioning-large",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    substitution_tokens: Optional[List[str]] = None,
    **kwargs,
) -> List[str]:
    """
    Returns a list of captions for the given images
    """
    processor = BlipProcessor.from_pretrained(model_id, cache_dir=MODEL_PATH)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id, cache_dir=MODEL_PATH
    ).to(device)
    captions = []
    print(f"Input captioning text: {text}")
    for image in tqdm(images):
        inputs = processor(image, text=text, return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs, max_length=150, do_sample=True, top_k=50, temperature=0.7
        )
        caption = processor.decode(out[0], skip_special_tokens=True)

        # BLIP 2 lowercases all caps tokens. This should properly replace them w/o messing up subwords. I'm sure there's a better way to do this.
        for token in substitution_tokens:
            print(token)
            sub_cap = " " + caption + " "
            print(sub_cap)
            sub_cap = sub_cap.replace(" " + token.lower() + " ", " " + token + " ")
            caption = sub_cap.strip()

        captions.append(caption)
    print("Generated captions", captions)
    return captions


def face_mask_google_mediapipe(
    images: List[Image.Image], blur_amount: float = 80.0, bias: float = 0.05
) -> List[Image.Image]:
    """
    Returns a list of images with mask on the face parts.
    """
    mp_face_detection = mp.solutions.face_detection

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    masks = []
    for image in tqdm(images):
        image = np.array(image)

        results = face_detection.process(image)
        black_image = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        if results.detections:
            for detection in results.detections:
                x_min = int(
                    detection.location_data.relative_bounding_box.xmin * image.shape[1]
                )
                y_min = int(
                    detection.location_data.relative_bounding_box.ymin * image.shape[0]
                )
                width = int(
                    detection.location_data.relative_bounding_box.width * image.shape[1]
                )
                height = int(
                    detection.location_data.relative_bounding_box.height
                    * image.shape[0]
                )

                # draw the colored rectangle
                black_image[y_min : y_min + height, x_min : x_min + width] = 255

        black_image = Image.fromarray(black_image)
        masks.append(black_image)

    return masks


def _crop_to_square(
    image: Image.Image, com: List[Tuple[int, int]], resize_to: Optional[int] = None
):
    cx, cy = com
    width, height = image.size
    if width > height:
        left_possible = max(cx - height / 2, 0)
        left = min(left_possible, width - height)
        right = left + height
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top_possible = max(cy - width / 2, 0)
        top = min(top_possible, height - width)
        bottom = top + width

    image = image.crop((left, top, right, bottom))

    if resize_to:
        image = image.resize((resize_to, resize_to), Image.Resampling.LANCZOS)

    return image


def _center_of_mass(mask: Image.Image):
    """
    Returns the center of mass of the mask
    """
    x, y = np.meshgrid(np.arange(mask.size[0]), np.arange(mask.size[1]))

    x_ = x * np.array(mask)
    y_ = y * np.array(mask)

    x = np.sum(x_) / np.sum(mask)
    y = np.sum(y_) / np.sum(mask)

    return x, y


def load_and_save_masks_and_captions(
    files: Union[str, List[str]],
    output_dir: str = TEMP_OUT_DIR,
    caption_text: Optional[str] = None,
    mask_target_prompts: Optional[Union[List[str], str]] = None,
    target_size: int = 1024,
    crop_based_on_salience: bool = True,
    use_face_detection_instead: bool = False,
    temp: float = 1.0,
    n_length: int = -1,
    substitution_tokens: Optional[List[str]] = None,
):
    """
    Loads images from the given files, generates masks for them, and saves the masks and captions and upscale images
    to output dir. If mask_target_prompts is given, it will generate kinda-segmentation-masks for the prompts and save them as well.

    Example:
    >>> x = load_and_save_masks_and_captions(
                files="./data/images",
                output_dir="./data/masks_and_captions",
                caption_text="a photo of",
                mask_target_prompts="cat",
                target_size=768,
                crop_based_on_salience=True,
                use_face_detection_instead=False,
                temp=1.0,
                n_length=-1,
            )
    """
    os.makedirs(output_dir, exist_ok=True)

    # load images
    if isinstance(files, str):
        # check if it is a directory
        if os.path.isdir(files):
            # get all the .png .jpg in the directory
            files = (
                glob.glob(os.path.join(files, "*.png"))
                + glob.glob(os.path.join(files, "*.jpg"))
                + glob.glob(os.path.join(files, "*.jpeg"))
            )

        if len(files) == 0:
            raise Exception(
                f"No files found in {files}. Either {files} is not a directory or it does not contain any .png or .jpg/jpeg files."
            )
        if n_length == -1:
            n_length = len(files)
        files = sorted(files)[:n_length]

    images = [Image.open(file).convert("RGB") for file in files]

    # captions
    print(f"Generating {len(images)} captions...")
    captions = blip_captioning_dataset(
        images, text=caption_text, substitution_tokens=substitution_tokens
    )

    if mask_target_prompts is None:
        mask_target_prompts = ""
        temp = 999

    print(f"Generating {len(images)} masks...")
    if not use_face_detection_instead:
        seg_masks = clipseg_mask_generator(
            images=images, target_prompts=mask_target_prompts, temp=temp
        )
    else:
        seg_masks = face_mask_google_mediapipe(images=images)

    # find the center of mass of the mask
    if crop_based_on_salience:
        coms = [_center_of_mass(mask) for mask in seg_masks]
    else:
        coms = [(image.size[0] / 2, image.size[1] / 2) for image in images]
    # based on the center of mass, crop the image to a square
    images = [
        _crop_to_square(image, com, resize_to=None) for image, com in zip(images, coms)
    ]

    print(f"Upscaling {len(images)} images...")
    # upscale images anyways
    images = swin_ir_sr(images, target_size=(target_size, target_size))
    images = [
        image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        for image in images
    ]

    seg_masks = [
        _crop_to_square(mask, com, resize_to=target_size)
        for mask, com in zip(seg_masks, coms)
    ]

    data = []

    # clean TEMP_OUT_DIR first
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    os.makedirs(output_dir, exist_ok=True)

    # iterate through the images, masks, and captions and add a row to the dataframe for each
    for idx, (image, mask, caption) in enumerate(zip(images, seg_masks, captions)):
        image_name = f"{idx}.src.jpg"
        mask_file = f"{idx}.mask.png"

        # save the image and mask files
        image.save(output_dir + image_name, quality=99)
        mask.save(output_dir + mask_file)

        # add a new row to the dataframe with the file names and caption
        data.append(
            {"image_path": image_name, "mask_path": mask_file, "caption": caption},
        )

    df = pd.DataFrame(columns=["image_path", "mask_path", "caption"], data=data)
    # save the dataframe to a CSV file
    df.to_csv(os.path.join(output_dir, "captions.csv"), index=False)
