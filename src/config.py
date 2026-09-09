import configparser
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from src.vault import VaultClient

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.ini"

load_dotenv(ROOT / ".env", override=False)


@lru_cache(maxsize=1)
def load_vault_secrets() -> dict:
    """Секреты из Vault. Ленивая загрузка: только при первом обращении
    к БД, а не при импорте модуля (иначе юнит-тесты требуют Vault)."""
    client = VaultClient()
    path = os.getenv("VAULT_SECRET_PATH", "mlops-app")

    return client.get_secrets(path=path)


class MissingSettingError(RuntimeError):
    """Обязательная переменная окружения не задана."""


def required_env(name: str) -> str:
    """Значение обязательной переменной окружения."""
    value = os.getenv(name, "").strip()

    if not value:
        raise MissingSettingError(
            f"Переменная окружения {name} не задана. "
            f"Скопируйте .env.example в .env и заполните значения "
            f"(в CI/CD — GitHub Secrets)."
        )

    return value


@dataclass(frozen=True)
class CassandraSettings:
    """Параметры подключения к Cassandra."""

    hosts: list[str]
    port: int
    keyspace: str
    username: str = field(repr=False)
    password: str = field(repr=False)
    connect_retries: int = 3
    retry_delay_seconds: float = 2.0


def cassandra_settings() -> CassandraSettings:
    """Настройки подключения к БД из секретов Vault."""

    secrets = load_vault_secrets()

    hosts = secrets["CASSANDRA_HOSTS"].split(",")
    port = int(secrets["CASSANDRA_PORT"])
    keyspace = secrets["CASSANDRA_KEYSPACE"]
    username = secrets["CASSANDRA_USER"]
    password = secrets["CASSANDRA_PASSWORD"]

    return CassandraSettings(
        hosts=hosts,
        port=port,
        keyspace=keyspace,
        username=username,
        password=password,
        connect_retries=int(os.getenv("CASSANDRA_CONNECT_RETRIES", "3")),
        retry_delay_seconds=float(os.getenv("CASSANDRA_RETRY_DELAY", "2.0")),
    )


def load_config(path: Path = None) -> configparser.ConfigParser:
    """Загрузка config.ini: пути к данным и модели, никаких секретов."""
    path = Path(path) if path else CONFIG_FILE

    if not path.exists():
        raise FileNotFoundError(f"config.ini не найден: {path}")

    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")

    return cfg


def path_from_config(
    section: str,
    key: str,
    cfg: configparser.ConfigParser = None,
) -> Path:
    """Путь из config.ini; относительный отсчитывается от корня репозитория."""
    cfg = cfg or load_config()

    raw_path = cfg[section].get(key, "").strip()

    if not raw_path:
        raise ValueError(f"{section}.{key} не указан в config.ini")

    path = Path(raw_path)

    return path if path.is_absolute() else ROOT / path


def checkpoint_path(cfg: configparser.ConfigParser = None) -> Path:
    """Путь к чекпоинту обученного классификатора (.pth)."""
    path = path_from_config("MODEL", "checkpoint_path", cfg)

    if not path.is_file():
        raise FileNotFoundError(f"Чекпоинт модели не найден по пути: {path}")

    return path
