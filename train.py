import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import shutil
import requests
from huggingface_hub import hf_hub_download

from cog import BaseModel, Input, Path


class TrainingOutput(BaseModel):
    weights: Path


def download_file(url, dest):
    with requests.get(url, stream=True) as r:
        with open(dest, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def train(
    repo_id: str = Input(
        description="huggingface repo to import",
        default=None
    ),
    weights: str = Input(
        description="safetensor weights to import",
        default=None
    ),
    lora_url: str = Input(
        description="full URL of LORA safetensor weights",
        default=None
    ),
) -> TrainingOutput:
    
    dest = 'lora.safetensors'
    if os.path.exists(dest):
        os.remove(dest)

    if repo_id and weights:
        fn = hf_hub_download(repo_id=repo_id, filename=weights)
        shutil.copy(fn, dest)
        os.remove(fn)
    elif lora_url:
        download_file(lora_url, dest)
    else:
        raise ValueError("Must provide either repo_id and weights or lora_url")

    return TrainingOutput(weights=Path(dest))
