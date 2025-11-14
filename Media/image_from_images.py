from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch

from Media.device_utils import resolve_device, get_cache_dir


def load_pipeline(model_name: str, device: Optional[str] = None) -> StableDiffusionImg2ImgPipeline:
    """Instantiate the Stable Diffusion Img2Img pipeline."""
    selected = resolve_device(model_name, preferred=device)
    cache_dir = get_cache_dir()
    resolved_device = selected
    dtype = torch.float16 if resolved_device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    pipeline_kwargs = {
        "torch_dtype": dtype,
        "cache_dir": cache_dir,
        "local_files_only": False,
    }
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, **pipeline_kwargs)
    print(f"Img2Img pipeline: selected device={selected}, cache_dir={cache_dir}")
    return pipe.to(resolved_device)


def iter_images(image_dir: Path) -> Iterable[Path]:
    supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in image_dir.iterdir():
        if path.suffix.lower() in supported_suffixes:
            yield path


def generate_from_images(
    pipeline: StableDiffusionImg2ImgPipeline,
    source_dir: Path,
    destination_dir: Path,
    prompt: str,
    strength: float,
    guidance_scale: float,
    output_size: int,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    for image_path in iter_images(source_dir):
        with Image.open(image_path) as img:
            init_image = img.convert("RGB").resize((output_size, output_size))
        generated_images = pipeline(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale).images
        if not generated_images:
            continue
        target_path = destination_dir / f"generated_{image_path.stem}.png"
        generated_images[0].save(target_path)
        generated_count += 1
    if generated_count == 0:
        raise ValueError(f"No supported image files found in {source_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from existing images using Stable Diffusion Img2Img.")
    parser.add_argument("source", type=Path, help="Directory with source images.")
    parser.add_argument("destination", type=Path, help="Directory where generated images will be written.")
    parser.add_argument("--prompt", default="A beautiful painting based on this image", help="Text prompt for the generation.")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model repo id.")
    parser.add_argument("--strength", type=float, default=0.75, help="Strength for the Img2Img transformation.")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--size", type=int, default=512, help="Target square size for the input images.")
    parser.add_argument("--device", default=None, help="Torch device identifier (e.g. cuda:0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = load_pipeline(model_name=args.model, device=args.device)
    generate_from_images(
        pipeline=pipe,
        source_dir=args.source,
        destination_dir=args.destination,
        prompt=args.prompt,
        strength=args.strength,
        guidance_scale=args.guidance,
        output_size=args.size,
    )
    print(f"Image generation complete. Results saved to {args.destination}")


if __name__ == "__main__":
    main()