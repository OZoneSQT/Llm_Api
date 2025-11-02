from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from transformers import pipeline


def create_tts_pipeline(model_name: str, device: Optional[str] = None):
    """Create a reusable text-to-speech pipeline."""
    pipeline_kwargs = {"model": model_name}
    if device is not None:
        pipeline_kwargs["device"] = device
    return pipeline("text-to-speech", **pipeline_kwargs)


def generate_audio(
    text: str,
    output_path: Path,
    model_name: str,
    device: Optional[str] = None,
    tts_pipeline=None,
) -> Path:
    """Generate audio from text and persist it."""
    tts = tts_pipeline or create_tts_pipeline(model_name=model_name, device=device)
    audio_artifact = tts(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_artifact["audio"])
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audio from text using a Hugging Face TTS model.")
    parser.add_argument("text", help="Text to convert into speech.")
    parser.add_argument("--output", type=Path, default=Path("output.wav"), help="Destination audio file.")
    parser.add_argument("--model", default="facebook/mms-tts-eng", help="Hugging Face repo id of the TTS model.")
    parser.add_argument("--device", default=None, help="Torch device identifier (e.g. cuda:0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tts = create_tts_pipeline(model_name=args.model, device=args.device)
    output_file = generate_audio(
        text=args.text,
        output_path=args.output,
        model_name=args.model,
        device=args.device,
        tts_pipeline=tts,
    )
    print(f"Audio generated and saved to {output_file}")


if __name__ == "__main__":
    main()