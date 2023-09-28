import base64
import os
import pytest
import requests
import subprocess
import numpy as np
from PIL import Image
from threading import Thread, Lock
from io import BytesIO

from test_utils import get_image_name, process_log_line, capture_output, wait_for_server_to_be_ready

# Constants
SERVER_URL = "http://localhost:5000/predictions"
HEALTH_CHECK_URL = "http://localhost:5000/health-check"

IMAGE_NAME = "your_image_name"  # replace with your image name
HOST_NAME = "your_host_name"   # replace with your host name


@pytest.fixture(scope="session")
def server():
    image_name = get_image_name()

    command = [
        "docker", "run",
        # "-ti",
        "-p", "5000:5000",
        "--gpus=all",
        image_name
    ]
    print("\n**********************STARTING SERVER**********************")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print_lock = Lock()
    
    stdout_thread = Thread(target=capture_output, args=(process.stdout, print_lock))
    stdout_thread.start()

    stderr_thread = Thread(target=capture_output, args=(process.stderr, print_lock))
    stderr_thread.start()

    wait_for_server_to_be_ready(HEALTH_CHECK_URL)

    yield process

    process.terminate()
    process.wait()


def test_health_check(server):
    response = requests.get(HEALTH_CHECK_URL)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"


def get_image(response):
    data = response.json()
    datauri = data["output"][0]
    base64_encoded_data = datauri.split(",")[1]
    data = base64.b64decode(base64_encoded_data)
    return Image.open(BytesIO(data))

def write_image(response, output_fn):
    if not os.path.exists("tmp/"):
        os.makedirs("tmp")

    img = get_image(response)
    img.save(output_fn)
    return img

def roughly_the_same(img1, img2):
    """
    Assert that pixel RGB values differ by less than 2 across an image
    Handles watermarking variation
    """
    delta = np.array(img1, dtype=np.int32) - np.array(img2, dtype=np.int32)
    return np.abs(np.mean(delta)) < 2 


def test_seeded_prediction(server):
    """
    SDXL w/seed should be deterministic. may need to adjust tolerance for optimized SDXLs
    """
    data = {
        "input": {
            "prompt": "An astronaut riding a rainbow unicorn, cinematic, dramatic",
            "num_inference_steps": 50,
            "width": 1024,
            "height": 1024,
            "scheduler": "DDIM",
            "refine": "expert_ensemble_refiner",
            # Add other parameters here
            "seed": 12103
        }
    }
    response = requests.post(SERVER_URL, json=data)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    img_1 = get_image(response)
    img_1.save("tests/assets/test_out.png")
    img_2 = Image.open("tests/assets/out.png")
    assert roughly_the_same(img_1, img_2)


def test_lora_load_unload(server):
    """
    Tests generation with & without loras
    """
    data = {
        "input": {
            "prompt": "A photo of a dog on the beach",
            "num_inference_steps": 50,
            # Add other parameters here
            "seed": 1234
        }
    }
    response = requests.post(SERVER_URL, json=data)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    write_image(response, "tmp/base_output.png")

    data = {
        "input": {
            "prompt": "A photo of a TOK on the beach",
            "num_inference_steps": 50,
            # Add other parameters here
            "replicate_weights": "https://storage.googleapis.com/dan-scratch-public/tmp/trained_model.tar",
            "seed": 1234
        }
    }
    response = requests.post(SERVER_URL, json=data)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    img_1 = write_image(response, "tmp/lora_output.png")
    response = requests.post(SERVER_URL, json=data)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    img_2 = write_image(response, "tmp/lora_output_again.png")

    assert roughly_the_same(np.array(img_1), np.array(img_2))

    data = {
        "input": {
            "prompt": "A photo of a TOK on the beach",
            "num_inference_steps": 50,
            # Add other parameters here
            "replicate_weights": "https://storage.googleapis.com/dan-scratch-public/tmp/monstertoy_model.tar",
            "seed": 1234
        }
    }
    response = requests.post(SERVER_URL, json=data)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    lora_b = write_image(response, "tmp/lora_output_b.png")
    assert not roughly_the_same(img_1, lora_b)
    
    data = {
        "input": {
            "prompt": "A photo of a dog on the beach",
            "num_inference_steps": 50,
            # Add other parameters here
            "seed": 1234
        }
    }
    response = requests.post(SERVER_URL, json=data)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    write_image(response, "tmp/base_output_again.png")


if __name__ == "__main__":
    pytest.main()

