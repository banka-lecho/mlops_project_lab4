import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.config import path_from_config
from src.logger import get_logger

logger = get_logger(__name__)

BUCKETS = 100


def bucket_of(image_name: str) -> int:
    """Стабильный бакет 0..99 для имени файла."""
    digest = hashlib.md5(image_name.encode("utf-8")).hexdigest()

    return int(digest, 16) % BUCKETS


def assign_split(image_name: str, train_share: int, val_share: int) -> str:
    """Выборка для одной картинки."""
    bucket = bucket_of(image_name)

    if bucket < train_share:
        return "train"

    if bucket < train_share + val_share:
        return "val"

    return "test"


def split_dataset(
    csv_path: Path | None = None,
    out_path: Path | None = None,
    train_share: int = 70,
    val_share: int = 15,
) -> pd.DataFrame:
    """
    Проставляет колонку split и сохраняет результат отдельным файлом.

    Доли задаются в процентах; на test уходит остаток.
    """
    if train_share + val_share >= BUCKETS:
        raise ValueError(
            f"На test не остаётся данных: train={train_share}, val={val_share}"
        )

    csv_path = Path(csv_path) if csv_path else path_from_config("DATA", "csv_path")
    out_path = Path(out_path) if out_path else path_from_config("DATA", "split_path")

    df = pd.read_csv(csv_path)

    missing = {"image_name", "label"} - set(df.columns)

    if missing:
        raise ValueError(f"В {csv_path} нет колонок: {sorted(missing)}")

    duplicates = df["image_name"].duplicated().sum()

    if duplicates:
        raise ValueError(
            f"В {csv_path} есть повторы image_name ({duplicates}): "
            f"одна картинка попадёт в две выборки"
        )

    df["split"] = df["image_name"].map(
        lambda name: assign_split(name, train_share, val_share)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["image_name", "label", "split"]].to_csv(out_path, index=False)

    logger.info("Разбиение сохранено в %s", out_path)

    for split, group in df.groupby("split"):
        logger.info(
            "%s: %s картинок (%.1f%%), по классам: %s",
            split,
            len(group),
            100 * len(group) / len(df),
            group["label"].value_counts().to_dict(),
        )

    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Разбить датасет на train/val/test.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="CSV с таргетами (по умолчанию — из config.ini).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Куда сохранить разбиение (по умолчанию — из config.ini).",
    )
    parser.add_argument(
        "--train-share", type=int, default=70, help="Доля train в процентах."
    )
    parser.add_argument(
        "--val-share",
        type=int,
        default=15,
        help="Доля val в процентах; на test уходит остаток.",
    )

    return parser


def main():
    args = build_arg_parser().parse_args()

    split_dataset(
        csv_path=args.csv_path,
        out_path=args.out_path,
        train_share=args.train_share,
        val_share=args.val_share,
    )


if __name__ == "__main__":
    main()
