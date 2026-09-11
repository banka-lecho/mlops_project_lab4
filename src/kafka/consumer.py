import asyncio
import json
import uuid
from datetime import datetime

from aiokafka import AIOKafkaConsumer

from src.config import KafkaSettings, kafka_settings
from src.db.cassandra_client import cassandra_repository
from src.logger import get_logger

logger = get_logger(__name__)


class KafkaConsumer:
    def __init__(
        self,
        settings: KafkaSettings | None = None,
    ):
        self._settings = settings
        self.topic = None
        self.consumer = None
        self.cassandra_repository = cassandra_repository

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

    async def stop(self):
        if self.consumer:
            await self.consumer.stop()
        self.cassandra_repository.shutdown()

    async def save_consumed_event(self, event: dict):
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
            # TODO:: здесь надо залогать нормально
            logger.exception(
                "Не удалось сохранить предсказание: %s", event["request_id"]
            )

    async def consume(self):
        if not self.consumer:
            raise RuntimeError("Consumer is not started. Call start() first.")

        try:
            async for msg in self.consumer:
                event = msg.value
                await self.save_consumed_event(event)
                print(f"Consumed prediction event: {event['request_id']}")
        finally:
            await self.stop()

    async def start(self):
        # TODO:: почему нужно, чтобы consumer стартовал после cassandra_repository.connect()?
        self.cassandra_repository.connect()
        self._settings = self.settings
        self.topic = self.settings.topic
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.settings.bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            group_id=self.settings.consumer_group,
        )
        await self.consumer.start()


kafka_consumer = KafkaConsumer()


async def main():
    await kafka_consumer.start()
    await kafka_consumer.consume()


if __name__ == "__main__":
    asyncio.run(main())
