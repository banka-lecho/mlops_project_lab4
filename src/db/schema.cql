-- Схема хранилища результатов модели.
CREATE KEYSPACE IF NOT EXISTS dog_emotion_keyspace
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

-- Результаты инференса: читаются точечно по идентификатору запроса.
CREATE TABLE IF NOT EXISTS dog_emotion_keyspace.predictions (
    request_id       uuid PRIMARY KEY,
    created_at       timestamp,
    image_name       text,
    predicted_class  text,
    confidence       double,
    probabilities    map<text, double>,
    process_time_ms  double,
    model_checkpoint text,
    device           text
);

-- Обучающая и валидационная выборки: партиция на split.
CREATE TABLE IF NOT EXISTS dog_emotion_keyspace.dataset (
    split       text,
    label       text,
    image_name  text,
    image_path  text,
    loaded_at   timestamp,
    PRIMARY KEY ((split), label, image_name)
);
