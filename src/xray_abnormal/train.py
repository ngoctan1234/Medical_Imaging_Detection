from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from xray_abnormal.data import XrayCsvDataset
from xray_abnormal.model import build_model
from xray_abnormal.utils import ensure_dir, get_device, load_config, set_seed


def compute_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.float32)
    metrics = {"accuracy": float(accuracy_score(targets.reshape(-1), preds.reshape(-1)))}
    try:
        if targets.shape[1] == 1:
            metrics["auc"] = float(roc_auc_score(targets[:, 0], probs[:, 0]))
        else:
            metrics["auc"] = float(roc_auc_score(targets, probs, average="macro"))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    targets_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        probs_list.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets_list.append(targets.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.concatenate(targets_list), np.concatenate(probs_list)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    data_cfg = cfg["data"]
    train_ds = XrayCsvDataset(
        data_cfg["train_csv"],
        data_cfg["image_column"],
        data_cfg["label_columns"],
        data_cfg["image_size"],
        augment=True,
    )
    val_ds = XrayCsvDataset(
        data_cfg["val_csv"],
        data_cfg["image_column"],
        data_cfg["label_columns"],
        data_cfg["image_size"],
        augment=False,
    )

    train_cfg = cfg["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    model = build_model(
        cfg["model"]["name"],
        num_classes=len(data_cfg["label_columns"]),
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    output_dir = ensure_dir(train_cfg["output_dir"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    best_auc = -1.0

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_targets, train_probs = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_targets, val_probs = run_epoch(model, val_loader, criterion, device)
        train_metrics = compute_metrics(train_targets, train_probs, train_cfg["threshold"])
        val_metrics = compute_metrics(val_targets, val_probs, train_cfg["threshold"])

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_auc={train_metrics['auc']:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        current_auc = val_metrics["auc"]
        if np.isnan(current_auc):
            current_auc = -1.0
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "label_columns": data_cfg["label_columns"],
                    "best_auc": best_auc,
                },
                checkpoint_dir / "best.pt",
            )

    print(f"Best checkpoint: {Path(checkpoint_dir / 'best.pt')}")


if __name__ == "__main__":
    main()
