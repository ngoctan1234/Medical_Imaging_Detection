from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from xray_abnormal.data import build_inference_transform
from xray_abnormal.gradcam import GradCAM, overlay_heatmap
from xray_abnormal.model import build_model, get_gradcam_target_layer, load_checkpoint
from xray_abnormal.utils import get_device, load_config


def load_model_for_inference(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict, list[str]]:
    device = device or get_device("auto")
    checkpoint = load_checkpoint(str(checkpoint_path), device)
    cfg = checkpoint.get("config")
    if cfg is None:
        if config_path is None:
            raise ValueError("Checkpoint has no config. Pass --config.")
        cfg = load_config(config_path)

    label_columns = checkpoint.get("label_columns", cfg["data"]["label_columns"])
    model = build_model(
        cfg["model"]["name"],
        num_classes=len(label_columns),
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, cfg, label_columns


def predict_with_heatmap(
    model: torch.nn.Module,
    cfg: dict,
    image: Image.Image,
    label_columns: list[str],
    device: torch.device,
    class_index: int | None = None,
) -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray]:
    image_size = int(cfg["data"]["image_size"])
    rgb_image = image.convert("RGB").resize((image_size, image_size))
    image_array = np.asarray(rgb_image)
    transform = build_inference_transform(image_size)
    tensor = transform(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    if class_index is None:
        class_index = int(np.argmax(probs))

    target_layer = get_gradcam_target_layer(model, cfg["model"]["name"])
    gradcam = GradCAM(model, target_layer)
    try:
        heatmap = gradcam(tensor, class_index)
    finally:
        gradcam.remove_hooks()

    overlay = overlay_heatmap(image_array, heatmap)
    predictions = [
        {"label": label, "probability": float(prob)}
        for label, prob in zip(label_columns, probs, strict=True)
    ]
    return predictions, heatmap, overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="outputs/heatmap.png")
    parser.add_argument("--class-index", type=int, default=None)
    args = parser.parse_args()

    device = get_device("auto")
    model, cfg, label_columns = load_model_for_inference(args.checkpoint, args.config, device)
    image = Image.open(args.image)
    predictions, _heatmap, overlay = predict_with_heatmap(
        model,
        cfg,
        image,
        label_columns,
        device,
        args.class_index,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)

    for item in predictions:
        print(f"{item['label']}: {item['probability']:.4f}")
    print(f"Saved heatmap overlay: {output_path}")


if __name__ == "__main__":
    main()
