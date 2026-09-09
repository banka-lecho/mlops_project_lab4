import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config, path_from_config
from src.db.cassandra_client import cassandra_repository
from src.logger import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")


def _relative_images_dir() -> str:
    """Каталог изображений так, как он записан в config.ini."""
    return load_config()["DATA"]["images_path"].strip().rstrip("/")


def load_split(
    csv_path: Path | None = None,
    split: str | None = None,
    repository=None,
) -> int:
    """Загружает строки датасета в таблицу dataset."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"Неизвестная выборка {split!r}, ожидается одна из {SPLITS}")

    csv_path = Path(csv_path) if csv_path else path_from_config("DATA", "split_path")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Файл с разбиением не найден: {csv_path}. "
            f"Сначала выполните: python -m src.utils.split_dataset"
        )

    df = pd.read_csv(csv_path)

    missing = {"image_name", "label", "split"} - set(df.columns)

    if missing:
        raise ValueError(
            f"В {csv_path} нет колонок: {sorted(missing)}. "
            f"Похоже, это исходный датасет, а не результат разбиения"
        )

    if split is not None:
        df = df[df["split"] == split]

    if df.empty:
        logger.warning("Нечего загружать: в %s нет строк для split=%s", csv_path, split)

        return 0

    images_dir = _relative_images_dir()

    rows = [
        {
            "split": row.split,
            "label": row.label,
            "image_name": row.image_name,
            "image_path": f"{images_dir}/{row.image_name}",
        }
        for row in df.itertuples(index=False)
    ]

    repository = repository or cassandra_repository
    repository.connect()

    loaded = repository.save_dataset_rows(rows)

    logger.info(
        "Загружено в Cassandra: %s строк из %s (split=%s)",
        loaded,
        csv_path.name,
        split or "все",
    )

    return loaded


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Загрузить разбиение датасета в Cassandra."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="CSV с разбиением (по умолчанию — из config.ini).",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default=None,
        help="Загрузить только одну выборку; по умолчанию все.",
    )

    return parser


def main():
    args = build_arg_parser().parse_args()

    try:
        load_split(csv_path=args.csv_path, split=args.split)
    finally:
        cassandra_repository.shutdown()


if __name__ == "__main__":
    main()
