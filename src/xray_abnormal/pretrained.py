from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from xray_abnormal.gradcam import GradCAM, overlay_heatmap


def load_torchxrayvision_model(
    weights: str = "densenet121-res224-all",
    device: torch.device | None = None,
    cache_dir: str = "outputs/model_cache",
) -> torch.nn.Module:
    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise ImportError("Install torchxrayvision to use pretrained X-ray models.") from exc

    model = xrv.models.DenseNet(weights=weights, cache_dir=cache_dir, apply_sigmoid=True)
    model.to(device or torch.device("cpu"))
    model.eval()
    return model


def _preprocess_for_torchxrayvision(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    import torchxrayvision as xrv

    image_array = np.asarray(image.convert("L"))
    image_array = xrv.datasets.normalize(image_array, 255)
    image_array = image_array[None, ...]
    transform = transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(image_size),
        ]
    )
    image_array = transform(image_array)
    return torch.from_numpy(image_array).float().unsqueeze(0)


def predict_pretrained_with_heatmap(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    class_index: int | None = None,
    image_size: int = 224,
) -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray]:
    tensor = _preprocess_for_torchxrayvision(image, image_size).to(device)
    with torch.no_grad():
        outputs = model(tensor)[0].detach().cpu().numpy()

    if class_index is None:
        class_index = int(np.nanargmax(outputs))

    target_layer = model.features.denseblock4
    gradcam = GradCAM(model, target_layer)
    try:
        heatmap = gradcam(tensor, class_index)
    finally:
        gradcam.remove_hooks()

    image_rgb = np.asarray(image.convert("RGB"))
    image_rgb = cv2.resize(image_rgb, (image_size, image_size))
    overlay = overlay_heatmap(image_rgb, heatmap)
    predictions = [
        {"label": label, "probability": float(prob)}
        for label, prob in zip(model.pathologies, outputs, strict=True)
    ]
    return predictions, heatmap, overlay
