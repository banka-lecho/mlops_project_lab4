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

    def create_producer(self):
        self._settings = self.settings
        self.topic = self.settings.topic
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.settings.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    async def send(self, value: dict):
        await self.producer.send_and_wait(self.topic, value=value)

    async def stop(self):
        await self.producer.stop()

    async def start(self):
        await self.producer.start()


kafka_producer = KafkaProducer()
