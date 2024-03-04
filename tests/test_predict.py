import base64
import os
import pickle
import subprocess
import sys
import time
from functools import partial
from io import BytesIO

import numpy as np
import pytest
import replicate
import requests
from PIL import Image, ImageChops

ENV = os.getenv('TEST_ENV', 'local')
LOCAL_ENDPOINT = "http://localhost:5000/predictions"
MODEL = os.getenv('STAGING_MODEL', 'no model configured')

def local_run(model_endpoint: str, model_input: dict):
    response = requests.post(model_endpoint, json={"input": model_input})
    data = response.json()

    try:
        # TODO: this will break if we test batching
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
        return Image.open(BytesIO(data))
    except Exception as e:
        print("Error!")
        print("input:", model_input)
        print(data["logs"])
        raise e


def replicate_run(model: str, version: str, model_input: dict):
    output = replicate.run(
        f"{model}:{version}",
        input=model_input)
    url = output[0]

    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def wait_for_server_to_be_ready(url, timeout=300):
    """
    Waits for the server to be ready.

    Args:
    - url: The health check URL to poll.
    - timeout: Maximum time (in seconds) to wait for the server to be ready.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            data = response.json()

            if data["status"] == "READY":
                return
            elif data["status"] == "SETUP_FAILED":
                raise RuntimeError(
                    "Server initialization failed with status: SETUP_FAILED"
                )

        except requests.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError("Server did not become ready in the expected time.")

        time.sleep(5)  # Poll every 5 seconds


@pytest.fixture(scope="session")
def inference_func():
    """
    local inference uses http API to hit local server; staging inference uses python API b/c it's cleaner. 
    """
    if ENV == 'local':
        return partial(local_run, LOCAL_ENDPOINT)
    elif ENV == 'staging':
        model = replicate.models.get(MODEL)
        version = model.versions.list()[0]
        return partial(replicate_run, MODEL, version.id)
    else:
        raise Exception(f"env should be local or staging but was {ENV}")


@pytest.fixture(scope="session", autouse=True)
def service():
    """
    Spins up local cog server to hit for tests if running locally, no-op otherwise
    """
    if ENV == 'local':
        print("building model")
        # starts local server if we're running things locally
        build_command = 'cog build -t test-model'.split()
        subprocess.run(build_command, check=True)
        container_name = 'cog-test'
        try:
            subprocess.check_output(['docker', 'inspect', '--format="{{.State.Running}}"', container_name])
            print(f"Container '{container_name}' is running. Stopping and removing...")
            subprocess.check_call(['docker', 'stop', container_name])
            subprocess.check_call(['docker', 'rm', container_name])
            print(f"Container '{container_name}' stopped and removed.")
        except subprocess.CalledProcessError:
            # Container not found
            print(f"Container '{container_name}' not found or not running.")

        run_command = f'docker run -d -p 5000:5000 --gpus all --name {container_name} test-model '.split()
        process = subprocess.Popen(run_command, stdout=sys.stdout, stderr=sys.stderr)

        wait_for_server_to_be_ready("http://localhost:5000/health-check")

        yield
        process.terminate()
        process.wait()
        stop_command = "docker stop cog-test".split()
        subprocess.run(stop_command)
    else:
        yield


def image_equal_fuzzy(img_expected, img_actual, test_name='default', tol=5):
    """
    Assert that average pixel values differ by less than tol across image
    """
    img1 = np.array(img_expected, dtype=np.int32)
    img2 = np.array(img_actual, dtype=np.int32)
    
    mean_delta = np.mean(np.abs(img1 - img2))
    imgs_equal = (mean_delta < tol)
    if not imgs_equal:
        # save failures for quick inspection
        save_dir = f"tmp/{test_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_expected.save(os.path.join(save_dir, 'expected.png'))
        img_actual.save(os.path.join(save_dir, 'actual.png'))
        difference = ImageChops.difference(img_expected, img_actual)
        difference.save(os.path.join(save_dir, 'delta.png'))

    return imgs_equal


def test_seeded_prediction(inference_func, request):
    """
    SDXL w/seed should be deterministic. may need to adjust tolerance for optimized SDXLs
    """
    data = {
        "prompt": "An astronaut riding a rainbow unicorn, cinematic, dramatic",
        "num_inference_steps": 50,
        "width": 1024,
        "height": 1024,
        "scheduler": "DDIM",
        "refine": "expert_ensemble_refiner",
        "seed": 12103,
    }
    actual_image = inference_func(data)
    expected_image = Image.open("tests/assets/out.png")
    assert image_equal_fuzzy(actual_image, expected_image, test_name=request.node.name)


def test_lora_load_unload(inference_func, request):
    """
    Tests generation with & without loras. 
    This is checking for some gnarly state issues (can SDXL load / unload LoRAs), so predictions need to run in series.
    """
    SEED = 1234
    base_data = {
        "prompt": "A photo of a dog on the beach",
        "num_inference_steps": 50,
        # Add other parameters here
        "seed": SEED,
    }
    base_img_1 = inference_func(base_data)

    lora_a_data = {
        "prompt": "A photo of a TOK on the beach",
        "num_inference_steps": 50,
        # Add other parameters here
        "replicate_weights": "https://storage.googleapis.com/dan-scratch-public/sdxl/other_model.tar",
        "seed": SEED
    }
    lora_a_img_1 = inference_func(lora_a_data)
    assert not image_equal_fuzzy(lora_a_img_1, base_img_1, test_name=request.node.name)

    lora_a_img_2 = inference_func(lora_a_data)
    assert image_equal_fuzzy(lora_a_img_1, lora_a_img_2, test_name=request.node.name)

    lora_b_data = {
        "prompt": "A photo of a TOK on the beach",
        "num_inference_steps": 50,
        "replicate_weights": "https://storage.googleapis.com/dan-scratch-public/sdxl/monstertoy_model.tar",
        "seed": SEED,
    }
    lora_b_img = inference_func(lora_b_data)
    assert not image_equal_fuzzy(lora_a_img_1, lora_b_img, test_name=request.node.name)
    assert not image_equal_fuzzy(base_img_1, lora_b_img, test_name=request.node.name)
