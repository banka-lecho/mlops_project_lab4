import io
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from PIL import Image

from src.config import checkpoint_path, load_config
from src.db.cassandra_client import CassandraRepository, cassandra_repository
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
    if not cassandra_repository.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Нет подключения к Cassandra",
        )
    return cassandra_repository


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    yield

    cassandra_repository.shutdown()
    logger.info("Остановка сервиса и БД, очистка ресурсов.")


app = FastAPI(title="Dog Emotion Classifier API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = (
        f"{(time.perf_counter() - started) * 1000:.2f}"
    )
    return response


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Модель не загружена"},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="ok" if classifier_service.is_ready else "degraded",
        model_loaded=classifier_service.is_ready,
        db_connected=cassandra_repository.is_ready,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info():
    if not classifier_service.is_ready:
        logger.exception("Модель еще не загружена или не получилось ее загрузить.")
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
    image: UploadFile = File(..., description="Изображение для инференса"),
    save: bool = Query(True, description="Сохранять ли результат в Cassandra"),
):
    """Основной метод инференса."""
    if not classifier_service.is_ready:
        logger.exception("Модель еще не загружена или не получилось ее загрузить.")
        raise ModelNotLoadedError

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logger.exception("Неправильный формат входного изобаражения.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Невозможно прочитать файл как изображение",
        )

    started = time.perf_counter()
    try:
        predicted_class, probabilities = classifier_service.predict(image=pil_image)
    except Exception as exc:
        logger.exception("Ошибка инференса")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка инференса: {str(exc)}",
        )
    process_time = (time.perf_counter() - started) * 1000

    request_id = uuid.uuid4()
    saved = False

    if save and cassandra_repository.is_ready:
        try:
            cassandra_repository.save_prediction(
                request_id=request_id,
                image_name=image.filename or "unknown",
                predicted_class=predicted_class,
                probabilities=probabilities,
                process_time_ms=process_time,
                model_checkpoint=classifier_service.checkpoint_path,
                device=classifier_service.device,
            )
            saved = True
        except Exception:
            logger.exception("Не удалось сохранить предсказание: %s", request_id)

    return PredictResponse(
        request_id=request_id,
        predicted_class=predicted_class,
        probabilities=probabilities,
        process_time_ms=process_time,
        saved=saved,
    )


@app.get(
    "/predictions/{request_id}", response_model=PredictionRecord, tags=["predictions"]
)
async def read_prediction(request_id: uuid.UUID, repo=Depends(require_db)):
    """Точечное чтение результата модели из БД."""
    record = repo.get_prediction(request_id)

    if record is None:
        raise HTTPException(404, f"Предсказание {request_id} не найдено")

    return record
