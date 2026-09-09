import csv
import os
import re
import subprocess
from pathlib import Path

import pytest
import requests

DATA_DIR = Path(__file__).parent / "data"
SMOKE_DIR = DATA_DIR / "smoke"
CONFIDENCE_THRESHOLD = 0.7


def _load_smoke_cases():
    manifest = SMOKE_DIR / "expected.csv"
    if not manifest.exists():
        return []

    with open(manifest, newline="", encoding="utf-8") as f:
        return [(row["image_name"], row["label"]) for row in csv.DictReader(f)]


SMOKE_CASES = _load_smoke_cases()


@pytest.fixture
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8000")


def test_health(base_url):
    response = requests.get(f"{base_url}/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info(base_url):
    response = requests.get(f"{base_url}/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["is_ready"] is True
    assert data["device"] in ["cpu", "cuda"]
    assert data["checkpoint_path"]
    assert data["classes"]


def test_prediction(base_url):
    with open(DATA_DIR / "happy_dog.jpg", "rb") as image:
        response = requests.post(
            f"{base_url}/predict",
            files={"image": ("happy_dog.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"]
    assert data["probabilities"]

    assert 0 <= data["process_time_ms"]


def test_prediction_round_trip(base_url):
    with open(DATA_DIR / "happy_dog.jpg", "rb") as image:
        predicted = requests.post(
            f"{base_url}/predict",
            files={"image": ("happy_dog.jpg", image, "image/jpeg")},
        ).json()
    assert predicted["saved"] is True

    record = requests.get(f"{base_url}/predictions/{predicted['request_id']}")
    assert record.status_code == 200
    assert record.json()["predicted_class"] == predicted["predicted_class"]


@pytest.mark.skipif(not SMOKE_CASES, reason=f"Нет фикстуры {SMOKE_DIR}/expected.csv")
@pytest.mark.parametrize("image_name,expected_label", SMOKE_CASES)
def test_confident_prediction_per_class(base_url, image_name, expected_label):
    with open(SMOKE_DIR / image_name, "rb") as image:
        response = requests.post(
            f"{base_url}/predict", files={"image": (image_name, image, "image/jpeg")}
        )

    assert response.status_code == 200

    data = response.json()
    probabilities = data["probabilities"]

    assert data["predicted_class"] == expected_label, (
        f"{image_name}: ожидался класс '{expected_label}', "
        f"получен '{data['predicted_class']}'. Вероятности: {probabilities}"
    )

    confidence = probabilities[expected_label]
    assert confidence >= CONFIDENCE_THRESHOLD, (
        f"{image_name}: уверенность в классе '{expected_label}' = {confidence:.4f} "
        f"< {CONFIDENCE_THRESHOLD}. Вероятности: {probabilities}"
    )


def _dataset_row_count() -> int:
    """
    Строк в таблице dataset — читает напрямую через cqlsh в контейнере
    Cassandra.
    """
    container, user, password, keyspace = (
        os.environ[name]
        for name in ("CASSANDRA_CONTAINER", "CASSANDRA_USER", "CASSANDRA_PASSWORD", "CASSANDRA_KEYSPACE")
    )
    output = subprocess.run(
        [
            "docker", "exec", container, "cqlsh",
            "-u", user, "-p", password,
            "-e", f"SELECT count(*) FROM {keyspace}.dataset;",
        ],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout

    return int(re.search(r"\d+", output).group())


@pytest.mark.skipif(
    "CASSANDRA_CONTAINER" not in os.environ,
    reason="Нужны CASSANDRA_CONTAINER/USER/PASSWORD/KEYSPACE (задаются в CD)",
)
def test_dataset_loaded_into_cassandra():
    """
    Результат шага CD "Load a dataset sample into Cassandra": фикстура
    tests/functional/data/dataset_sample.csv должна попасть в базу.
    """
    assert _dataset_row_count() > 0, "Таблица dataset пуста: датасет не загружен"
