from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image
from transformers import pipeline

from Media.device_utils import resolve_device, get_model_cache_path


def load_segmenter(model_name: str, device: Optional[str] = None):
    """Create an image-segmentation pipeline."""
    selected = resolve_device(model_name, preferred=device)
    cache_dir = get_model_cache_path(model_name)
    kwargs: dict[str, Any] = {"model": model_name, "cache_dir": cache_dir}
    if selected.startswith('cuda'):
        try:
            idx = int(selected.split(':', 1)[1])
            kwargs["device"] = idx
        except Exception:
            kwargs["device"] = 0
    else:
        kwargs["device"] = -1
    print(f"Image segmenter: selected device={selected}, cache_dir={cache_dir}")
    return pipeline("image-segmentation", **kwargs)


def apply_transparency(image: Image.Image, masks: Iterable[np.ndarray]) -> Image.Image:
    """Set masked pixels to transparent."""
    rgba = image.convert("RGBA")
    data = np.array(rgba)
    for mask in masks:
        mask_array = np.asarray(mask) > 0
        if mask_array.any():
            data[mask_array] = [0, 0, 0, 0]
    return Image.fromarray(data)


def remove_objects(
    image_path: Path,
    output_path: Path,
    model_name: str,
    device: Optional[str] = None,
) -> Path:
    """Remove segmented objects from an image and persist the result."""
    segmenter = load_segmenter(model_name=model_name, device=device)
    with Image.open(image_path) as img:
        results = segmenter(img)
        masks = (np.array(entry["mask"]) for entry in results if "mask" in entry)
        cleaned = apply_transparency(img, masks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove objects from images using segmentation masks.")
    parser.add_argument("image", type=Path, help="Path to the image that should be processed.")
    parser.add_argument("--output", type=Path, default=Path("modified_image.png"), help="Destination file.")
    parser.add_argument("--model", default="facebook/mask2former-swin-large-coco", help="Segmentation model repo id.")
    parser.add_argument("--device", default=None, help="Torch device identifier (e.g. cuda:0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_path = remove_objects(
        image_path=args.image,
        output_path=args.output,
        model_name=args.model,
        device=args.device,
    )
    print(f"Modified image saved to {target_path}")


if __name__ == "__main__":
    main()