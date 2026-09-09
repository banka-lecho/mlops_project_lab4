import json

from aiokafka import AIOKafkaProducer

from src.config import KafkaSettings, kafka_settings


class KafkaProducer:
    def __init__(self, settings: KafkaSettings | None = None):
        self._settings = settings
        self.topic = None
        self.producer = None

    @property
    def is_ready(self) -> bool:
        """Проверка готовности Kafka producer."""
        return self.producer is not None

    @property
    def settings(self) -> KafkaSettings:
        """Геттер настроек."""
        if self._settings is None:
            self._settings = kafka_settings()

        return self._settings

    async def send(self, value: dict):
        if self.producer:
            await self.producer.send_and_wait(self.topic, value=value)

    async def stop(self):
        if self.producer:
            await self.producer.stop()
            self.producer = None

    async def start(self):
        self.topic = self.settings.topic

        producer = AIOKafkaProducer(
            bootstrap_servers=self.settings.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
        )
        await producer.start()
        self.producer = producer


kafka_producer = KafkaProducer()
