from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from transformers import pipeline


def resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_video_pipeline(model_name: str, device: str) -> DiffusionPipeline:
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
    return pipe.to(device)


def load_audio_pipeline(model_name: str, device: str):
    return pipeline("text-to-audio", model=model_name, device=0 if device.startswith("cuda") else -1)


def generate_video_frames(video_pipe: DiffusionPipeline, prompt: str, steps: int) -> list:
    result = video_pipe(prompt, num_inference_steps=steps)
    return result.frames


def save_video(frames, output_path: Path, fps: int) -> Path:
    import importlib

    imageio = importlib.import_module("imageio")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path


def save_audio(audio_output, output_path: Path) -> Path:
    import importlib

    sf = importlib.import_module("soundfile")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio_output["audio"], audio_output["sampling_rate"])
    return output_path


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    import importlib

    moviepy_editor = importlib.import_module("moviepy.editor")
    AudioFileClip = getattr(moviepy_editor, "AudioFileClip")
    VideoFileClip = getattr(moviepy_editor, "VideoFileClip")

    with VideoFileClip(str(video_path)) as video_clip, AudioFileClip(str(audio_path)) as audio_clip:
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a short video with matching audio from text prompts.")
    parser.add_argument("--video-prompt", default="A serene landscape with mountains and a river at sunset")
    parser.add_argument("--audio-prompt", default="Relaxing ambient music with soft piano and nature sounds")
    parser.add_argument("--video-model", default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--audio-model", default="facebook/musicgen-small")
    parser.add_argument("--device", default=None, help="Preferred torch device (e.g. cuda:0).")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps for the video pipeline.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the generated video.")
    parser.add_argument("--tmp-dir", type=Path, default=Path("artifacts"), help="Working directory for intermediate files.")
    parser.add_argument("--output", type=Path, default=Path("final_video_with_audio.mp4"), help="Final merged video path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    video_pipe = load_video_pipeline(model_name=args.video_model, device=device)
    frames = generate_video_frames(video_pipe=video_pipe, prompt=args.video_prompt, steps=args.steps)
    tmp_video = save_video(frames, output_path=args.tmp_dir / "generated_video.mp4", fps=args.fps)

    audio_pipe = load_audio_pipeline(model_name=args.audio_model, device=device)
    audio_output = audio_pipe(args.audio_prompt)
    tmp_audio = save_audio(audio_output, output_path=args.tmp_dir / "generated_audio.wav")

    mux_video_audio(video_path=tmp_video, audio_path=tmp_audio, output_path=args.output)
    print(f"Video with audio saved to {args.output}")


if __name__ == "__main__":
    main()