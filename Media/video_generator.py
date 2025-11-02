from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import InferenceClient


def build_client(token: str) -> InferenceClient:
    if not token:
        raise EnvironmentError("Set the HUGGINGFACEHUB_API_TOKEN environment variable before running this script.")
    return InferenceClient(token=token)


def generate_video(prompt: str, output_path: Path, model_name: str, num_frames: int, fps: int, token: str) -> Path:
    client = build_client(token=token)
    response = client.text_to_video(model=model_name, inputs={"prompt": prompt, "num_frames": num_frames, "fps": fps})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video using the Hugging Face Inference API.")
    parser.add_argument("prompt", help="Text prompt used for video generation.")
    parser.add_argument("--model", default="stabilityai/stable-video-diffusion-img2vid", help="Inference model repo id.")
    parser.add_argument("--num-frames", type=int, default=24, help="Number of frames in the generated video.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the generated video.")
    parser.add_argument("--output", type=Path, default=Path("generated_video.mp4"), help="Destination video path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    target = generate_video(
        prompt=args.prompt,
        output_path=args.output,
        model_name=args.model,
        num_frames=args.num_frames,
        fps=args.fps,
        token=token,
    )
    print(f"Video saved to {target}")


if __name__ == "__main__":
    main()