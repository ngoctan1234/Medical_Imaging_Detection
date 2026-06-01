from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from xray_abnormal.pretrained import load_torchxrayvision_model, predict_pretrained_with_heatmap
from xray_abnormal.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/raw/pneumonia_xray_wikimedia.jpg")
    parser.add_argument("--output", default="outputs/pretrained_pneumonia_heatmap.png")
    parser.add_argument("--weights", default="densenet121-res224-all")
    args = parser.parse_args()

    device = get_device("auto")
    model = load_torchxrayvision_model(args.weights, device)
    predictions, _heatmap, overlay = predict_pretrained_with_heatmap(
        model,
        Image.open(args.image),
        device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)

    for item in sorted(predictions, key=lambda x: x["probability"], reverse=True)[:8]:
        print(f"{item['label']}: {item['probability']:.4f}")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
