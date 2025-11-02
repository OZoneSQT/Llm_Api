from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from diffusers import StableDiffusionPipeline
import torch


def load_pipeline(model_name: str, device: Optional[str] = None) -> StableDiffusionPipeline:
    """Instantiate a Stable Diffusion text-to-image pipeline."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return pipe.to(resolved_device)


def generate_image(
    prompt: str,
    output_path: Path,
    model_name: str,
    device: Optional[str] = None,
    diffusion_pipeline: Optional[StableDiffusionPipeline] = None,
) -> Path:
    """Generate an image from a text prompt."""
    pipe = diffusion_pipeline or load_pipeline(model_name=model_name, device=device)
    image = pipe(prompt).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using Stable Diffusion.")
    parser.add_argument("prompt", help="Prompt used to guide image generation.")
    parser.add_argument("--output", type=Path, default=Path("generated_image.png"), help="Destination image file.")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model repo id.")
    parser.add_argument("--device", default=None, help="Torch device identifier (e.g. cuda:0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = load_pipeline(model_name=args.model, device=args.device)
    target = generate_image(
        prompt=args.prompt,
        output_path=args.output,
        model_name=args.model,
        device=args.device,
        diffusion_pipeline=pipe,
    )
    print(f"Image saved to {target}")


if __name__ == "__main__":
    main()