import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Ответ на запрос /health."""

    status: str = Field(..., description="ok или degraded")
    model_loaded: bool
    db_connected: bool = Field(
        False, description="Установлено ли подключение к Cassandra"
    )


class ModelInfoResponse(BaseModel):
    """Ответ на запрос /model/info."""

    checkpoint_path: str
    device: str
    classes: list[str]
    is_ready: bool


class PredictResponse(BaseModel):
    """Ответ на запрос /predict."""

    request_id: uuid.UUID
    predicted_class: str
    probabilities: dict[str, float]
    process_time_ms: float
    published: bool = Field(
        False,
        description="Опубликовано ли предсказание в топик Kafka. "
        "В Cassandra его пишет consumer, поэтому запись появляется позже.",
    )


class PredictionRecord(BaseModel):
    """Запись предсказания в Cassandra."""

    request_id: uuid.UUID
    created_at: datetime
    image_name: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    process_time_ms: float
    model_checkpoint: str
    device: str
