import asyncio
import json
import uuid
from datetime import datetime

from aiokafka import AIOKafkaConsumer

from src.config import KafkaSettings, kafka_settings
from src.db.cassandra_client import CassandraRepository, cassandra_repository
from src.logger import get_logger

logger = get_logger(__name__)


class PredictionConsumer:
    def __init__(
        self,
        repository: CassandraRepository,
        settings: KafkaSettings | None = None,
    ):
        self._settings = settings
        self.topic = None
        self.consumer = None
        self.cassandra_repository = repository

    @property
    def is_ready(self) -> bool:
        """Проверка готовности Kafka consumer."""
        return self.consumer is not None

    @property
    def settings(self) -> KafkaSettings:
        """Геттер настроек."""
        if self._settings is None:
            self._settings = kafka_settings()

        return self._settings

    async def start(self):
        """Инициализация и запуск Kafka consumer."""
        self.topic = self.settings.topic

        consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.settings.bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            group_id=self.settings.consumer_group,
        )
        await consumer.start()

        self.consumer = consumer

        logger.info(
            "Kafka consumer подписан на топик %s, группа %s",
            self.topic,
            self.settings.consumer_group,
        )

    async def stop(self):
        """Остановка Kafka consumer."""
        if self.consumer:
            await self.consumer.stop()
            self.consumer = None
            logger.info("Kafka consumer остановлен")

    async def save_consumed_event(self, event: dict):
        """Сохранение события предсказания в Cassandra."""
        try:
            await asyncio.to_thread(
                self.cassandra_repository.save_prediction,
                request_id=uuid.UUID(event["request_id"]),
                created_at=datetime.fromisoformat(event["created_at"]),
                image_name=event.get("image_name") or "unknown",
                predicted_class=event["predicted_class"],
                probabilities=event["probabilities"],
                process_time_ms=event["process_time_ms"],
                model_checkpoint=event["model_checkpoint"],
                device=event["device"],
            )
        except Exception:
            logger.exception("Не удалось сохранить событие в Cassandra: %r", event)

    async def consume(self):
        """Асинхронное потребление сообщений из Kafka и сохранение их в Cassandra."""
        if not self.consumer:
            raise RuntimeError("Consumer не запущен: сначала вызовите start()")

        async for msg in self.consumer:
            logger.info(
                "Получено событие из %s: партиция %s, оффсет %s",
                msg.topic,
                msg.partition,
                msg.offset,
            )
            await self.save_consumed_event(msg.value)


async def main():
    """Главная функция для запуска консьюмера предсказаний. Сначала БД, потом Kafka."""
    cassandra_repository.connect()
    consumer = PredictionConsumer(repository=cassandra_repository)

    try:
        await consumer.start()
        await consumer.consume()
    except Exception:
        logger.exception("Консьюмер предсказаний остановлен из-за ошибки")
        raise
    finally:
        await consumer.stop()
        cassandra_repository.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
