import time
import pytest
import replicate


@pytest.fixture(scope="module")
def model_name(request):
    return "stability-ai/sdxl"


@pytest.fixture(scope="module")
def model(model_name):
    return replicate.models.get(model_name)


@pytest.fixture(scope="module")
def version(model):
    versions = model.versions.list()
    return versions[0]


@pytest.fixture(scope="module")
def training(model_name, version):
    training_input = {
        "input_images": "https://storage.googleapis.com/replicate-datasets/sdxl-test/monstertoy-captions.tar"
    }
    print(f"Training on {model_name}:{version.id}")
    return replicate.trainings.create(
        version=model_name + ":" + version.id,
        input=training_input,
        destination="replicate-internal/training-scratch",
    )


@pytest.fixture(scope="module")
def prediction_tests():
    return [
        {
            "prompt": "A photo of TOK at the beach",
            "refine": "expert_ensemble_refiner",
        },
    ]


def test_training(training):
    while training.completed_at is None:
        time.sleep(60)
        training.reload()
    assert training.status == "succeeded"


@pytest.fixture(scope="module")
def trained_model_and_version(training):
    trained_model, trained_version = training.output["version"].split(":")
    return trained_model, trained_version


def test_post_training_predictions(trained_model_and_version, prediction_tests):
    trained_model, trained_version = trained_model_and_version
    model = replicate.models.get(trained_model)
    version = model.versions.get(trained_version)
    predictions = [
        replicate.predictions.create(version=version, input=val)
        for val in prediction_tests
    ]

    for val in predictions:
        val.wait()
        assert val.status == "succeeded"
