import configparser
import io
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.model import classifier_service

TEST_CLASSES = ["angry", "happy", "relaxed", "sad"]


class FakeRepository:
    """Подмена CassandraRepository для юнит-тестов."""

    def __init__(self):
        self.rows = {}
        self.is_ready = True

    def connect(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def save_prediction(
        self,
        request_id,
        image_name,
        predicted_class,
        probabilities,
        process_time_ms,
        model_checkpoint,
        device,
        created_at=None,
    ):
        """Записывает результат предсказания в память."""
        created_at = created_at or datetime.now(timezone.utc)

        self.rows[request_id] = {
            "request_id": request_id,
            "created_at": created_at,
            "image_name": image_name,
            "predicted_class": predicted_class,
            "confidence": float(probabilities[predicted_class]),
            "probabilities": probabilities,
            "process_time_ms": process_time_ms,
            "model_checkpoint": model_checkpoint,
            "device": device,
        }

        return created_at


class FakeProducer:
    """Подмена KafkaProducer: запоминает отправленные события."""

    def __init__(self):
        self.sent = []
        self.is_ready = True
        self.fail_on_send = False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, value: dict) -> None:
        if self.fail_on_send:
            raise RuntimeError("Kafka недоступна")

        self.sent.append(value)


@pytest.fixture
def fake_repo(monkeypatch):
    """Подменяет глобальный репозиторий в модуле API."""
    repo = FakeRepository()
    monkeypatch.setattr("src.api.main.cassandra_repository", repo)
    return repo


@pytest.fixture
def fake_producer(monkeypatch):
    """Подменяет глобальный Kafka-продюсер в модуле API."""
    producer = FakeProducer()
    monkeypatch.setattr("src.api.main.kafka_producer", producer)
    return producer


@pytest.fixture(autouse=True)
def setup_mocks(monkeypatch, fake_repo, fake_producer):
    """
    Эта фикстура автоматически применяется ко всем тестам.
    Она подменяет чтение реального config.ini и тяжеловесную ML-модель.
    """

    dummy_cfg = configparser.ConfigParser()
    dummy_cfg.read_dict(
        {"MODEL": {"checkpoint_path": "test-checkpoint-path", "device": "cpu"}}
    )

    monkeypatch.setattr("src.api.main.load_config", lambda *args, **kwargs: dummy_cfg)

    monkeypatch.setattr(
        "src.api.main.checkpoint_path", lambda cfg=None: "test-checkpoint-path"
    )

    def mock_load(checkpoint_path, device="cpu"):
        classifier_service.is_ready = True
        classifier_service.checkpoint_path = checkpoint_path
        classifier_service.device = device
        classifier_service.id2label = {i: c for i, c in enumerate(TEST_CLASSES)}
        classifier_service.label2id = {c: i for i, c in enumerate(TEST_CLASSES)}

    monkeypatch.setattr(classifier_service, "load", mock_load)

    def mock_predict(image):
        probs = {c: 0.1 for c in TEST_CLASSES}
        probs[TEST_CLASSES[0]] = 0.7
        return TEST_CLASSES[0], probs

    monkeypatch.setattr(classifier_service, "predict", mock_predict)


@pytest.fixture
def client():
    """Создаем клиент поверх приложения."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_image():
    """Генерирует тестовую картинку в оперативной памяти."""
    file = io.BytesIO()
    image = Image.new("RGB", (224, 224), color="blue")
    image.save(file, "jpeg")
    file.seek(0)
    return file


def test_health_endpoint(client):
    """Приложение стартует, модель и БД подключены."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "db_connected": True,
        "kafka_connected": True,
    }


def test_predict_success(client, test_image):
    """Основной сценарий: инференс отдаёт ожидаемую форму ответа."""
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", test_image, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] == "angry"
    assert set(data["probabilities"]) == set(TEST_CLASSES)
    assert data["probabilities"]["angry"] == 0.7
    assert "process_time_ms" in data
    assert "X-Process-Time-Ms" in response.headers


def test_predict_invalid_image(client):
    """Отправляем текстовую фигню под видом картинки."""
    response = client.post(
        "/predict",
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Невозможно прочитать файл" in response.json()["detail"]


def test_predict_publishes_event_to_kafka(client, test_image, fake_producer):
    """
    Результат инференса уходит в топик, и в событии есть все поля,
    которые consumer передаёт в save_prediction.
    """
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", test_image, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["published"] is True

    assert len(fake_producer.sent) == 1
    event = fake_producer.sent[0]

    assert event["request_id"] == response.json()["request_id"]
    assert event["predicted_class"] == "angry"
    assert event["image_name"] == "test.jpg"

    assert set(event) >= {
        "request_id",
        "created_at",
        "image_name",
        "predicted_class",
        "probabilities",
        "process_time_ms",
        "model_checkpoint",
        "device",
    }


def test_predict_survives_kafka_failure(client, test_image, fake_producer):
    """
    Ключевой сценарий: инференс уже отработал, и сбой публикации в Kafka
    не должен его обесценивать.
    """
    fake_producer.fail_on_send = True

    response = client.post(
        "/predict",
        files={"image": ("test.jpg", test_image, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["published"] is False
    assert data["predicted_class"] == "angry"
    assert data["probabilities"]["angry"] == 0.7
