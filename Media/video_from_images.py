from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Iterable


def load_cv2():
    return importlib.import_module("cv2")


def discover_frames(image_folder: Path) -> Iterable[Path]:
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    return sorted(path for path in image_folder.iterdir() if path.suffix.lower() in supported)


def images_to_video(image_folder: Path, output_path: Path, fps: int) -> Path:
    cv2 = load_cv2()
    frames = list(discover_frames(image_folder))
    if not frames:
        raise ValueError(f"No image files found in {image_folder}")

    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise ValueError(f"Unable to read the first frame at {frames[0]}")
    height, width, _ = first_frame.shape

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Skipping unreadable frame: {frame_path}")
            continue
        writer.write(frame)
    writer.release()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a folder of images into an MP4 video.")
    parser.add_argument("image_folder", type=Path, help="Directory containing source images.")
    parser.add_argument("output", type=Path, help="Destination video file.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = images_to_video(image_folder=args.image_folder, output_path=args.output, fps=args.fps)
    print(f"Video saved to {target}")


if __name__ == "__main__":
    main()