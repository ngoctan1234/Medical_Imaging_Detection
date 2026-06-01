from __future__ import annotations

import math

import numpy as np


def calibrate_probabilities(probabilities: np.ndarray, temperature: float = 1.4) -> np.ndarray:
    probs = np.clip(probabilities.astype(np.float32), 1e-5, 1 - 1e-5)
    logits = np.log(probs / (1 - probs))
    calibrated = 1 / (1 + np.exp(-(logits / max(temperature, 1e-5))))
    return calibrated.astype(np.float32)


def binary_entropy(probability: float) -> float:
    p = float(np.clip(probability, 1e-5, 1 - 1e-5))
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def uncertainty_label(probability: float) -> str:
    entropy = binary_entropy(probability)
    if entropy < 0.55:
        return "Low"
    if entropy < 0.85:
        return "Medium"
    return "High"


def risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.45:
        return "Medium"
    return "Low"


def severity_level(probability: float) -> str:
    if probability >= 0.80:
        return "Severe"
    if probability >= 0.60:
        return "Moderate"
    if probability >= 0.35:
        return "Mild"
    return "Minimal"
