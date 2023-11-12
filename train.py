import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import shutil
from huggingface_hub import hf_hub_download

from cog import BaseModel, Input, Path


class TrainingOutput(BaseModel):
    weights: Path


def train(
    repo_id: str = Input(
        description="huggingface repo to import",
    ),
    weights: str = Input(
        description="safetensor weights to import",
    ),
) -> TrainingOutput:
    
    dest = 'lora.safetensors'
    if os.path.exists(dest):
        os.remove(dest)

    fn = hf_hub_download(repo_id=repo_id, filename=weights)

    shutil.copy(fn, dest)
    os.remove(fn)
    return TrainingOutput(weights=Path(dest))
