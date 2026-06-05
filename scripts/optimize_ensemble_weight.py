from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from xray_abnormal.clinical_evidence import VINBIG_CLASSES
from xray_abnormal.ensemble import optimize_ensemble_weight, run_ensemble


def _columns(prefix: str, labels: list[str]) -> list[str]:
    return [f"{prefix}_{label}" for label in labels]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize ConvNeXtV2/RAD-DINO ensemble weight on validation AUROC.")
    parser.add_argument("--input", required=True, help="CSV containing y_*, convnext_*, and raddino_* columns.")
    parser.add_argument("--output", default="outputs/ensemble_weight.yaml")
    parser.add_argument("--grid-size", type=int, default=101)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    label_cols = _columns("y", VINBIG_CLASSES)
    conv_cols = _columns("convnext", VINBIG_CLASSES)
    raddino_cols = _columns("raddino", VINBIG_CLASSES)
    missing = [col for col in label_cols + conv_cols + raddino_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing[:8]}{'...' if len(missing) > 8 else ''}")

    y_true = df[label_cols].to_numpy(dtype=np.float32)
    convnext_probs = df[conv_cols].to_numpy(dtype=np.float32)
    raddino_probs = df[raddino_cols].to_numpy(dtype=np.float32)
    weight, auroc = optimize_ensemble_weight(y_true, convnext_probs, raddino_probs, grid_size=args.grid_size)
    agreement = run_ensemble(convnext_probs, raddino_probs, weight)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                f"convnext_weight: {weight:.4f}",
                f"raddino_weight: {1.0 - weight:.4f}",
                f"validation_macro_auroc: {auroc:.6f}",
                f"agreement_score: {agreement.agreement_score:.6f}",
                f"agreement_label: {agreement.agreement_label}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Best ConvNeXtV2 weight: {weight:.4f}")
    print(f"Validation macro AUROC: {auroc:.6f}")
    print(f"Agreement: {agreement.agreement_label} ({agreement.agreement_score:.3f})")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
