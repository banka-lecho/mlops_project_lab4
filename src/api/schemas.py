import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok или degraded")
    model_loaded: bool
    db_connected: bool = Field(
        False, description="Установлено ли подключение к Cassandra"
    )


class ModelInfoResponse(BaseModel):
    checkpoint_path: str
    device: str
    classes: list[str]
    is_ready: bool


class PredictResponse(BaseModel):
    request_id: uuid.UUID
    predicted_class: str
    probabilities: dict[str, float]
    process_time_ms: float
    saved: bool = Field(False, description="Записано ли предсказание в Cassandra")


class PredictionRecord(BaseModel):
    request_id: uuid.UUID
    created_at: datetime
    image_name: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    process_time_ms: float
    model_checkpoint: str
    device: str
