import io
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import fastapi
from fastapi.responses import JSONResponse
from PIL import Image

from src.config import checkpoint_path, load_config
from src.db.cassandra_client import CassandraRepository, cassandra_repository
from src.kafka.producer import kafka_producer
from src.logger import get_logger
from src.model import ModelNotLoadedError, classifier_service

from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRecord,
    PredictResponse,
)

logger = get_logger(__name__)


def require_db() -> CassandraRepository:
    """Проверка подключения к Cassandra."""
    if not cassandra_repository.is_ready:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Нет подключения к Cassandra",
        )
    return cassandra_repository


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Артефакт и подключение к БД поднимаются один раз при старте."""
    cfg = load_config()

    ckpt_path = str(checkpoint_path(cfg))
    device = cfg["MODEL"].get("device", "cuda")

    try:
        classifier_service.load(checkpoint_path=ckpt_path, device=device)
        logger.info("Модель успешно загружена из: %s", ckpt_path)
    except Exception:
        logger.exception("Ошибка загрузки модели: %s", ckpt_path)

    try:
        cassandra_repository.connect()
    except Exception:
        logger.exception("Cassandra недоступна, предсказания сохраняться не будут")

    try:
        await kafka_producer.start()
        logger.info("Kafka producer подключён, топик: %s", kafka_producer.topic)
    except Exception:
        logger.exception(
            "Kafka недоступна, предсказания не будут публиковаться в топик"
        )

    yield

    cassandra_repository.shutdown()
    await kafka_producer.stop()
    logger.info("Остановка сервиса и БД, очистка ресурсов.")


app = fastapi.FastAPI(
    title="Dog Emotion Classifier API", version="1.0.0", lifespan=lifespan
)


@app.middleware("http")
async def add_timing_header(request: fastapi.Request, call_next):
    """Добавляет заголовок X-Process-Time-Ms с временем обработки запроса в миллисекундах."""
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = (
        f"{(time.perf_counter() - started) * 1000:.2f}"
    )
    return response


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: fastapi.Request, exc: ModelNotLoadedError):
    """Обработчик исключения, когда модель не загружена."""
    return JSONResponse(
        status_code=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Модель не загружена"},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Проверка состояния сервиса и его зависимостей."""
    return HealthResponse(
        status="ok" if classifier_service.is_ready else "degraded",
        model_loaded=classifier_service.is_ready,
        db_connected=cassandra_repository.is_ready,
        kafka_connected=kafka_producer.is_ready,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info():
    """Информация о модели и её состоянии."""
    if not classifier_service.is_ready:
        logger.error("Модель еще не загружена или не получилось ее загрузить.")
        raise ModelNotLoadedError

    return ModelInfoResponse(
        checkpoint_path=classifier_service.checkpoint_path,
        device=classifier_service.device,
        classes=[
            classifier_service.id2label[i]
            for i in range(len(classifier_service.id2label))
        ],
        is_ready=classifier_service.is_ready,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    image: fastapi.UploadFile,
):
    """Основной метод инференса: результат уходит событием в Kafka;
    В Cassandra его пишет consumer, поэтому запись доступна через
    GET /predictions/{request_id} с задержкой.
    """
    if not classifier_service.is_ready:
        logger.error("Модель еще не загружена или не получилось ее загрузить.")
        raise ModelNotLoadedError

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.exception("Неправильный формат входного изобаражения.")
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail="Невозможно прочитать файл как изображение",
        ) from exc

    started = time.perf_counter()
    try:
        predicted_class, probabilities = classifier_service.predict(image=pil_image)
    except Exception as exc:
        logger.exception("Ошибка инференса")
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка инференса: {exc}",
        ) from exc
    process_time = (time.perf_counter() - started) * 1000

    request_id = uuid.uuid4()
    published = False

    if kafka_producer.is_ready:
        try:
            await kafka_producer.send(
                {
                    "request_id": str(request_id),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "image_name": image.filename or "unknown",
                    "predicted_class": predicted_class,
                    "probabilities": probabilities,
                    "process_time_ms": process_time,
                    "model_checkpoint": classifier_service.checkpoint_path,
                    "device": classifier_service.device,
                }
            )
            published = True
            logger.info("Предсказание %s опубликовано в Kafka", request_id)
        except Exception:
            logger.exception(
                "Не удалось отправить предсказание в Kafka: %s", request_id
            )
    else:
        logger.warning(
            "Kafka producer не запущен, предсказание %s не будет сохранено",
            request_id,
        )

    return PredictResponse(
        request_id=request_id,
        predicted_class=predicted_class,
        probabilities=probabilities,
        process_time_ms=process_time,
        published=published,
    )


@app.get(
    "/predictions/{request_id}", response_model=PredictionRecord, tags=["predictions"]
)
async def read_prediction(request_id: uuid.UUID, repo=fastapi.Depends(require_db)):
    """Точечное чтение результата модели из БД."""
    record = repo.get_prediction(request_id)

    if record is None:
        raise fastapi.HTTPException(404, f"Предсказание {request_id} не найдено")

    return record
