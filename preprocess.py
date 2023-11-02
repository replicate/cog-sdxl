# Have SwinIR upsample
# Have BLIP auto caption
# Have CLIPSeg auto mask concept

import gc
import fnmatch
import mimetypes
import os
import re
import shutil
import tarfile
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from zipfile import ZipFile

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    Swin2SRForImageSuperResolution,
    Swin2SRImageProcessor,
)

from predict import download_weights

# model is fixed to Salesforce/blip-image-captioning-large
BLIP_URL = "https://weights.replicate.delivery/default/blip_large/blip_large.tar"
BLIP_PROCESSOR_URL = "https://weights.replicate.delivery/default/blip_processor/blip_processor.tar"
BLIP_PATH = "./blip-cache"
BLIP_PROCESSOR_PATH = "./blip-proc-cache"

# model is fixed to CIDAS/clipseg-rd64-refined
CLIPSEG_URL = "https://weights.replicate.delivery/default/clip_seg_rd64_refined/clip_seg_rd64_refined.tar"
CLIPSEG_PROCESSOR = "https://weights.replicate.delivery/default/clip_seg_processor/clip_seg_processor.tar"
CLIPSEG_PATH = "./clipseg-cache"
CLIPSEG_PROCESSOR_PATH = "./clipseg-proc-cache"

# model is fixed to caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr
SWIN2SR_URL = "https://weights.replicate.delivery/default/swin2sr_realworld_sr_x4_64_bsrgan_psnr/swin2sr_realworld_sr_x4_64_bsrgan_psnr.tar"
SWIN2SR_PATH = "./swin2sr-cache"

TEMP_OUT_DIR = "./temp/"
TEMP_IN_DIR = "./temp_in/"

CSV_MATCH = "caption"


def preprocess(
    input_images_filetype: str,
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

    caption_csv = None

    if input_images_filetype == "zip" or str(input_zip_path).endswith(".zip"):
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
                if mt and mt[0] and mt[0] == 'text/csv' and CSV_MATCH in zip_info.filename:
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, TEMP_IN_DIR)
                    caption_csv = os.path.join(TEMP_IN_DIR, zip_info.filename)
    elif input_images_filetype == "tar" or str(input_zip_path).endswith(".tar"):
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
                if mt and mt[0] and mt[0] == 'text/csv' and CSV_MATCH in tar_info.name:
                    tar_info.name = os.path.basename(tar_info.name)
                    tar_ref.extract(tar_info, TEMP_IN_DIR)
                    caption_csv = os.path.join(TEMP_IN_DIR, tar_info.name)
    else:
        assert False, "input_images_filetype must be zip or tar"

    output_dir: str = TEMP_OUT_DIR

    load_and_save_masks_and_captions(
        files=TEMP_IN_DIR,
        output_dir=output_dir,
        caption_text=caption_text,
        caption_csv=caption_csv,
        mask_target_prompts=mask_target_prompts,
        target_size=target_size,
        crop_based_on_salience=crop_based_on_salience,
        use_face_detection_instead=use_face_detection_instead,
        temp=temp,
        substitution_tokens=substitution_tokens,
    )

    return Path(TEMP_OUT_DIR)


@torch.no_grad()
@torch.cuda.amp.autocast()
def swin_ir_sr(
    images: List[Image.Image],
    target_size: Optional[Tuple[int, int]] = None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    **kwargs,
) -> List[Image.Image]:
    """
    Upscales images using SwinIR. Returns a list of PIL images.
    If the image is already larger than the target size, it will not be upscaled
    and will be returned as is.

    """
    if not os.path.exists(SWIN2SR_PATH):
        download_weights(SWIN2SR_URL, SWIN2SR_PATH)
    model = Swin2SRForImageSuperResolution.from_pretrained(
        SWIN2SR_PATH
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
@torch.cuda.amp.autocast()
def clipseg_mask_generator(
    images: List[Image.Image],
    target_prompts: Union[List[str], str],
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
    if not os.path.exists(CLIPSEG_PROCESSOR_PATH):
        download_weights(CLIPSEG_PROCESSOR, CLIPSEG_PROCESSOR_PATH)
    if not os.path.exists(CLIPSEG_PATH):
        download_weights(CLIPSEG_URL, CLIPSEG_PATH)
    processor = CLIPSegProcessor.from_pretrained(CLIPSEG_PROCESSOR_PATH)
    model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_PATH).to(device)

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
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    substitution_tokens: Optional[List[str]] = None,
    **kwargs,
) -> List[str]:
    """
    Returns a list of captions for the given images
    """
    if not os.path.exists(BLIP_PROCESSOR_PATH):
        download_weights(BLIP_PROCESSOR_URL, BLIP_PROCESSOR_PATH)
    if not os.path.exists(BLIP_PATH):
        download_weights(BLIP_URL, BLIP_PATH)
    processor = BlipProcessor.from_pretrained(BLIP_PROCESSOR_PATH)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_PATH).to(device)
    captions = []
    text = text.strip()
    print(f"Input captioning text: {text}")
    for image in tqdm(images):
        inputs = processor(image, return_tensors="pt").to("cuda")
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

        captions.append(text + " " + caption)
    print("Generated captions", captions)
    return captions


def face_mask_google_mediapipe(
    images: List[Image.Image], blur_amount: float = 0.0, bias: float = 50.0
) -> List[Image.Image]:
    """
    Returns a list of images with masks on the face parts.
    """
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.1
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.1
    )

    masks = []
    for image in tqdm(images):
        image_np = np.array(image)

        # Perform face detection
        results_detection = face_detection.process(image_np)
        ih, iw, _ = image_np.shape
        if results_detection.detections:
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box

                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                # make sure bbox is within image
                bbox = (
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(iw - bbox[0], bbox[2]),
                    min(ih - bbox[1], bbox[3]),
                )

                print(bbox)

                # Extract face landmarks
                face_landmarks = face_mesh.process(
                    image_np[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                ).multi_face_landmarks

                # https://github.com/google/mediapipe/issues/1615
                # This was def helpful
                indexes = [
                    10,
                    338,
                    297,
                    332,
                    284,
                    251,
                    389,
                    356,
                    454,
                    323,
                    361,
                    288,
                    397,
                    365,
                    379,
                    378,
                    400,
                    377,
                    152,
                    148,
                    176,
                    149,
                    150,
                    136,
                    172,
                    58,
                    132,
                    93,
                    234,
                    127,
                    162,
                    21,
                    54,
                    103,
                    67,
                    109,
                ]

                if face_landmarks:
                    mask = Image.new("L", (iw, ih), 0)
                    mask_np = np.array(mask)

                    for face_landmark in face_landmarks:
                        face_landmark = [face_landmark.landmark[idx] for idx in indexes]
                        landmark_points = [
                            (int(l.x * bbox[2]) + bbox[0], int(l.y * bbox[3]) + bbox[1])
                            for l in face_landmark
                        ]
                        mask_np = cv2.fillPoly(
                            mask_np, [np.array(landmark_points)], 255
                        )

                    mask = Image.fromarray(mask_np)

                    # Apply blur to the mask
                    if blur_amount > 0:
                        mask = mask.filter(ImageFilter.GaussianBlur(blur_amount))

                    # Apply bias to the mask
                    if bias > 0:
                        mask = np.array(mask)
                        mask = mask + bias * np.ones(mask.shape, dtype=mask.dtype)
                        mask = np.clip(mask, 0, 255)
                        mask = Image.fromarray(mask)

                    # Convert mask to 'L' mode (grayscale) before saving
                    mask = mask.convert("L")

                    masks.append(mask)
                else:
                    # If face landmarks are not available, add a black mask of the same size as the image
                    masks.append(Image.new("L", (iw, ih), 255))

        else:
            print("No face detected, adding full mask")
            # If no face is detected, add a white mask of the same size as the image
            masks.append(Image.new("L", (iw, ih), 255))

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
    mask_np = np.array(mask) + 0.01
    x_ = x * mask_np
    y_ = y * mask_np

    x = np.sum(x_) / np.sum(mask_np)
    y = np.sum(y_) / np.sum(mask_np)

    return x, y


def load_and_save_masks_and_captions(
    files: Union[str, List[str]],
    output_dir: str = TEMP_OUT_DIR,
    caption_text: Optional[str] = None,
    caption_csv: Optional[str] = None,
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
                _find_files("*.png", files)
                + _find_files("*.jpg", files)
                + _find_files("*.jpeg", files)
            )

        if len(files) == 0:
            raise Exception(
                f"No files found in {files}. Either {files} is not a directory or it does not contain any .png or .jpg/jpeg files."
            )
        if n_length == -1:
            n_length = len(files)
        files = sorted(files)[:n_length]
        print("Image files: ", files)
    images = [Image.open(file).convert("RGB") for file in files]

    # captions
    if caption_csv:
        print(f"Using provided captions")
        caption_df = pd.read_csv(caption_csv)
        # sort images to be consistent with 'sorted' above
        caption_df = caption_df.sort_values('image_file')
        captions = caption_df['caption'].values
        print("Captions: ", captions)
        if len(captions) != len(images):
            print("Not the same number of captions as images!")
            print(f"Num captions: {len(captions)}, Num images: {len(images)}")
            print("Captions: ", captions)
            print("Images: ", files)
            raise Exception("Not the same number of captions as images! Check that all files passed in have a caption in your caption csv, and vice versa")
                
    else:
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
        image_name = f"{idx}.src.png"
        mask_file = f"{idx}.mask.png"

        # save the image and mask files
        image.save(output_dir + image_name)
        mask.save(output_dir + mask_file)

        # add a new row to the dataframe with the file names and caption
        data.append(
            {"image_path": image_name, "mask_path": mask_file, "caption": caption},
        )

    df = pd.DataFrame(columns=["image_path", "mask_path", "caption"], data=data)
    # save the dataframe to a CSV file
    df.to_csv(os.path.join(output_dir, "captions.csv"), index=False)


def _find_files(pattern, dir="."):
    """Return list of files matching pattern in a given directory, in absolute format.
    Unlike glob, this is case-insensitive.
    """

    rule = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    return [os.path.join(dir, f) for f in os.listdir(dir) if rule.match(f)]
