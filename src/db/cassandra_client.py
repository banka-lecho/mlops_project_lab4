import time
import uuid
from datetime import datetime, timezone
from typing import Any

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent_with_args
from cassandra.query import dict_factory

from src.config import CassandraSettings, cassandra_settings
from src.logger import get_logger

logger = get_logger(__name__)


class CassandraNotConnectedError(Exception):
    """Сервис модели не смог подключиться к базе данных."""


class CassandraSchemaError(Exception):
    """Кластер доступен, но нужной схемы в нём нет."""


class CassandraRepository:
    """Чтение и запись результатов работы модели."""

    def __init__(self, settings: CassandraSettings | None = None):
        self._settings = settings
        self._cluster = None
        self._session = None
        self._statements: dict[str, Any] = {}

    @property
    def is_ready(self) -> bool:
        """Проверка готовности сессии."""
        return self._session is not None and not self._session.is_shutdown

    @property
    def settings(self) -> CassandraSettings:
        """Геттер настроек."""
        if self._settings is None:
            self._settings = cassandra_settings()

        return self._settings

    def connect(self) -> None:
        """Подключение с повторными попытками."""
        if self.is_ready:
            return

        settings = self.settings

        auth_provider = PlainTextAuthProvider(
            username=settings.username,
            password=settings.password,
        )

        cluster = session = None
        last_error = None

        for attempt in range(1, settings.connect_retries + 1):
            try:
                cluster = Cluster(
                    contact_points=list(settings.hosts),
                    port=settings.port,
                    auth_provider=auth_provider,
                )
                session = cluster.connect()

                break

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Попытка %s/%s подключиться к Cassandra не удалась: %s",
                    attempt,
                    settings.connect_retries,
                    exc,
                )

                if attempt < settings.connect_retries:
                    time.sleep(settings.retry_delay_seconds)

        if session is None:
            raise CassandraNotConnectedError(
                f"Не удалось подключиться к Cassandra за {settings.connect_retries} попыток"
            ) from last_error

        if settings.keyspace not in cluster.metadata.keyspaces:
            cluster.shutdown()

            raise CassandraSchemaError(
                f"Keyspace '{settings.keyspace}' не найден. "
                f"Не применена схема src/db/schema.cql "
                f"(docker compose up cassandra-init)"
            )

        session.row_factory = dict_factory
        session.set_keyspace(settings.keyspace)

        self._session = session
        self._cluster = cluster

        try:
            self._prepare_statements()
        except Exception:
            self.shutdown()
            raise

        logger.info(
            "Подключение к Cassandra установлено: keyspace=%s",
            settings.keyspace,
        )

    def shutdown(self) -> None:
        """Закрывает сессии кластера и рвёт пулы TCP-соединений ко всем нодам."""
        if self._cluster is not None:
            self._cluster.shutdown()

        self._cluster = None
        self._session = None
        self._statements = {}

    def _require_session(self):
        if not self.is_ready:
            raise CassandraNotConnectedError("Нет подключения к Cassandra")

        return self._session

    def _prepare_statements(self) -> None:
        """
        Prepared statements: они инициализируются один раз, а значения
        передаются отдельно от текста запроса.
        """
        session = self._session

        self._statements = {
            "insert_prediction": session.prepare(
                """
                INSERT INTO predictions (
                    request_id, created_at, image_name,
                    predicted_class, confidence, probabilities,
                    process_time_ms, model_checkpoint, device
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            ),
            "select_prediction": session.prepare(
                "SELECT * FROM predictions WHERE request_id = ?"
            ),
            "insert_dataset_row": session.prepare(
                """
                INSERT INTO dataset (
                    split, label, image_name, image_path, loaded_at
                ) VALUES (?, ?, ?, ?, ?)
                """
            ),
        }

    def save_prediction(
        self,
        request_id: uuid.UUID,
        image_name: str,
        predicted_class: str,
        probabilities: dict[str, float],
        process_time_ms: float,
        model_checkpoint: str,
        device: str,
        created_at: datetime | None = None,
    ) -> datetime:
        """Записывает результат предсказания."""
        session = self._require_session()

        created_at = created_at or datetime.now(timezone.utc)

        session.execute(
            self._statements["insert_prediction"],
            (
                request_id,
                created_at,
                image_name,
                predicted_class,
                float(probabilities.get(predicted_class, 0.0)),
                {key: float(value) for key, value in probabilities.items()},
                float(process_time_ms),
                model_checkpoint,
                device,
            ),
        )

        logger.info(
            "Результат предсказания сохранён в Cassandra: request_id=%s, класс=%s",
            request_id,
            predicted_class,
        )

        return created_at

    def get_prediction(self, request_id: uuid.UUID) -> dict[str, Any] | None:
        """Одно предсказание по идентификатору запроса."""
        session = self._require_session()

        rows = session.execute(
            self._statements["select_prediction"],
            (request_id,),
        )

        return rows.one()

    def save_dataset_rows(
        self,
        rows: list[dict[str, Any]],
        concurrency: int = 32,
    ) -> int:
        """Батчевая загрузка выборки."""
        session = self._require_session()

        loaded_at = datetime.now(timezone.utc)

        parameters = [
            (
                row["split"],
                row["label"],
                row["image_name"],
                row.get("image_path", ""),
                loaded_at,
            )
            for row in rows
        ]

        results = execute_concurrent_with_args(
            session,
            self._statements["insert_dataset_row"],
            parameters,
            concurrency=concurrency,
            raise_on_first_error=True,
        )

        return sum(1 for success, _ in results if success)


cassandra_repository = CassandraRepository()
