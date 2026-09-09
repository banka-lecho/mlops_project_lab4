import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from src.config import checkpoint_path, path_from_config
from src.logger import get_logger

logger = get_logger(__name__)


class ModelNotLoadedError(Exception):
    pass


class DogEmotionDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, img_dir: str, transform=None, label2id: dict | None = None
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        if label2id is None:
            unique_labels = sorted(self.df["label"].unique())
            self.label2id = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label2id = label2id

        self.id2label = {i: label for label, i in self.label2id.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])

        image = Image.open(img_path).convert("RGB")
        label_id = self.label2id[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label_id


class DogEmotionClassifierService:
    def __init__(self):
        self.model = None
        self.id2label = None
        self.label2id = None
        self.device = None
        self.transform = None
        self.checkpoint_path = None
        self.is_ready = False

    def create_model(self, num_classes: int, pretrained: bool = True):
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    def evaluate(self, model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        total_loss = val_loss / len(val_loader.dataset)
        return total_loss, np.array(all_preds), np.array(all_targets)

    def train(self, csv_path: Path, img_path: Path, epochs: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {device}")

        df = pd.read_csv(csv_path)

        if "split" not in df.columns:
            raise ValueError(
                f"В {csv_path} нет колонки split. "
                f"Сначала выполните: python -m src.utils.split_dataset"
            )

        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]

        if train_df.empty or val_df.empty:
            raise ValueError(
                f"Пустая выборка в {csv_path}: train={len(train_df)}, val={len(val_df)}"
            )

        logger.info(
            "Выборки: train=%s, val=%s (test=%s, не используется при обучении)",
            len(train_df),
            len(val_df),
            (df["split"] == "test").sum(),
        )

        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = DogEmotionDataset(
            train_df, img_dir=img_path, transform=train_transform
        )
        val_dataset = DogEmotionDataset(
            val_df,
            img_dir=img_path,
            transform=val_transform,
            label2id=train_dataset.label2id,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=2
        )

        model = self.create_model(num_classes=len(train_dataset.label2id)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        best_f1 = 0.0

        for epoch in tqdm(range(epochs)):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            scheduler.step()
            train_loss = running_loss / len(train_dataset)

            val_loss, preds, targets = self.evaluate(
                model, val_loader, criterion, device
            )
            val_f1_weighted = f1_score(targets, preds, average="weighted")
            val_f1_macro = f1_score(targets, preds, average="macro")

            print(
                f"Epoch [{epoch + 1:02d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1 (Weighted): {val_f1_weighted:.4f} | "
                f"Val F1 (Macro): {val_f1_macro:.4f}"
            )

            if val_f1_weighted > best_f1:
                best_f1 = val_f1_weighted
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "label2id": train_dataset.label2id,
                        "id2label": train_dataset.id2label,
                    },
                    "dog_emotion_efficientnet_best.pth",
                )

        print("\n" + "=" * 50)
        print("Обучение завершено. Оценка лучшей модели на валидации:")
        print("=" * 50)

        checkpoint = torch.load(
            "dog_emotion_efficientnet_best.pth", map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        _, best_preds, best_targets = self.evaluate(
            model, val_loader, criterion, device
        )

        class_names = [
            train_dataset.id2label[i] for i in range(len(train_dataset.id2label))
        ]

        print("\n--- Classification Report ---")
        print(classification_report(best_targets, best_preds, target_names=class_names))

        cm = confusion_matrix(best_targets, best_preds)
        cm_df = pd.DataFrame(
            cm,
            index=[f"True: {c}" for c in class_names],
            columns=[f"Pred: {c}" for c in class_names],
        )
        print("--- Confusion Matrix ---")
        print(cm_df)

    def load(self, checkpoint_path: str, device: str = "cuda"):
        if self.is_ready:
            return

        self.checkpoint_path = str(checkpoint_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.label2id = checkpoint["label2id"]
        self.id2label = checkpoint["id2label"]

        self.model = self.create_model(num_classes=len(self.label2id), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.is_ready = True

    def predict(self, image: Image.Image):
        if not self.is_ready:
            logger.exception("Не удалось загрузить модель: %s", self.checkpoint_path)
            raise ModelNotLoadedError("Модель не загружена")

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        class_probs = {self.id2label[i]: probs[i].item() for i in range(len(probs))}
        predicted_class = self.id2label[probs.argmax().item()]

        return predicted_class, class_probs


classifier_service = DogEmotionClassifierService()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Обучение и инференс классификатора эмоций собак."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Обучить модель.")
    train_parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="CSV с разбиением на train/val/test (по умолчанию — из config.ini).",
    )
    train_parser.add_argument(
        "--img-path",
        type=Path,
        default=None,
        help="Путь к директории с изображениями (по умолчанию — из config.ini).",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Количество эпох обучения."
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Предсказать эмоцию по изображению."
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Путь к чекпоинту модели (.pth), по умолчанию — из config.ini.",
    )
    predict_parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Путь к изображению для классификации.",
    )
    predict_parser.add_argument(
        "--device", default="cuda", help="Устройство для инференса: cuda или cpu."
    )

    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.command == "train":
        csv_path = args.csv_path or path_from_config("DATA", "split_path")
        img_path = args.img_path or path_from_config("DATA", "images_path")

        service = DogEmotionClassifierService()
        service.train(csv_path=csv_path, img_path=img_path, epochs=args.epochs)

    elif args.command == "predict":
        ckpt_path = args.checkpoint or checkpoint_path()

        service = DogEmotionClassifierService()
        service.load(checkpoint_path=str(ckpt_path), device=args.device)

        image = Image.open(args.image)
        predicted_class, class_probs = service.predict(image)

        print(f"Предсказанный класс: {predicted_class}")
        for label, prob in sorted(
            class_probs.items(), key=lambda kv: kv[1], reverse=True
        ):
            print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
