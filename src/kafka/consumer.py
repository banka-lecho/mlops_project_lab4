from kafka import KafkaConsumer
import json

# TODO:: где должен создаватся этот топик
consumer = KafkaConsumer(
    "model-results",  # какой топик читаем
    bootstrap_servers="kafka:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest",  # с какого места читать, если запускаемся впервые
    group_id="model-consumer-group",  # "группа читателей" — важно для распределения нагрузки
)

for message in consumer:
    data = message.value
    print("Получено:", data)
    # тут дальше можно писать в БД, например
