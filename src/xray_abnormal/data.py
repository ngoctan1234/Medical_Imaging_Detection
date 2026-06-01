from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class XrayCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_column: str,
        label_columns: list[str],
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root = self.csv_path.parent
        self.df = pd.read_csv(self.csv_path)
        self.image_column = image_column
        self.label_columns = label_columns

        missing = [c for c in [image_column, *label_columns] if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")

        transform_list: list[transforms.Compose] = [
            transforms.Resize((image_size, image_size)),
        ]
        if augment:
            transform_list.extend(
                [
                    transforms.RandomRotation(7),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        image_path = Path(row[self.image_column])
        if not image_path.is_absolute() and not image_path.exists():
            image_path = self.root / image_path

        image = Image.open(image_path).convert("RGB")
        labels = row[self.label_columns].astype("float32").to_numpy()
        return self.transform(image), torch.tensor(labels, dtype=torch.float32)


def build_inference_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
