from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class EnsembleResult:
    probabilities: np.ndarray
    weight: float
    agreement_score: float
    agreement_label: str


def weighted_average(
    convnext_probs: np.ndarray,
    raddino_probs: np.ndarray,
    weight: float,
) -> np.ndarray:
    convnext = np.asarray(convnext_probs, dtype=np.float32)
    raddino = np.asarray(raddino_probs, dtype=np.float32)
    if convnext.shape != raddino.shape:
        raise ValueError(f"Probability shapes must match: {convnext.shape} vs {raddino.shape}")
    clipped_weight = float(np.clip(weight, 0.0, 1.0))
    return clipped_weight * convnext + (1.0 - clipped_weight) * raddino


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    valid_scores: list[float] = []
    for class_index in range(y_true.shape[1]):
        class_true = y_true[:, class_index]
        if len(np.unique(class_true)) < 2:
            continue
        valid_scores.append(float(roc_auc_score(class_true, y_prob[:, class_index])))
    if not valid_scores:
        raise ValueError("No class has both positive and negative examples for AUROC.")
    return float(np.mean(valid_scores))


def optimize_ensemble_weight(
    y_true: np.ndarray,
    convnext_probs: np.ndarray,
    raddino_probs: np.ndarray,
    grid_size: int = 101,
) -> tuple[float, float]:
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    best_weight = 0.5
    best_score = -np.inf
    for weight in np.linspace(0.0, 1.0, grid_size):
        probs = weighted_average(convnext_probs, raddino_probs, float(weight))
        score = macro_auroc(y_true, probs)
        if score > best_score:
            best_weight = float(weight)
            best_score = float(score)
    return best_weight, best_score


def agreement_score(convnext_probs: np.ndarray, raddino_probs: np.ndarray) -> float:
    convnext = np.asarray(convnext_probs, dtype=np.float32)
    raddino = np.asarray(raddino_probs, dtype=np.float32)
    if convnext.shape != raddino.shape:
        raise ValueError(f"Probability shapes must match: {convnext.shape} vs {raddino.shape}")
    return float(1.0 - np.mean(np.abs(convnext - raddino)))


def agreement_label(score: float) -> str:
    if score >= 0.90:
        return "High"
    if score >= 0.75:
        return "Medium"
    return "Low"


def run_ensemble(
    convnext_probs: np.ndarray,
    raddino_probs: np.ndarray,
    weight: float,
) -> EnsembleResult:
    probs = weighted_average(convnext_probs, raddino_probs, weight)
    score = agreement_score(convnext_probs, raddino_probs)
    return EnsembleResult(
        probabilities=probs,
        weight=float(np.clip(weight, 0.0, 1.0)),
        agreement_score=score,
        agreement_label=agreement_label(score),
    )


def top_findings(
    labels: list[str],
    probabilities: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 1:
        raise ValueError("top_findings expects one probability vector.")
    if len(labels) != len(probs):
        raise ValueError(f"labels/probabilities length mismatch: {len(labels)} vs {len(probs)}")
    order = np.argsort(probs)[::-1][:top_k]
    return pd.DataFrame(
        [{"label": labels[index], "probability": float(probs[index])} for index in order]
    )
