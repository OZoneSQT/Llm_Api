from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class ImageTrainingConfig:
    pretrained_model: str
    data_dir: Path
    output_dir: Path
    captions_file: Optional[Path]
    default_prompt: str
    resolution: int
    batch_size: int
    epochs: int
    learning_rate: float
    gradient_accumulation: int
    mixed_precision: bool
    seed: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config_file(config_path: Optional[Path]) -> tuple[Dict[str, Any], Optional[Path]]:
    if config_path is None:
        return {}, None
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object with parameter names and values.")
    return data, config_path.parent


def resolve_path(value: Optional[Any], base_dir: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    path_value = value if isinstance(value, Path) else Path(str(value))
    if not path_value.is_absolute() and base_dir is not None:
        path_value = base_dir / path_value
    return path_value.resolve()


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def pick_param(
    args_dict: Dict[str, Any],
    config_data: Dict[str, Any],
    name: str,
    default: Any = None,
    *,
    caster: Optional[Callable[[Any], Any]] = None,
    is_path: bool = False,
    base_dir: Optional[Path] = None,
) -> Any:
    if name in args_dict and args_dict[name] is not None:
        value = args_dict[name]
    elif name in config_data and config_data[name] is not None:
        value = config_data[name]
    else:
        value = default
    if value is None:
        return None
    if is_path:
        return resolve_path(value, base_dir)
    if caster is not None:
        return caster(value)
    return value


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        captions: Dict[str, str],
        default_prompt: str,
        resolution: int,
        tokenizer,
        max_length: int,
    ) -> None:
        self.root_dir = root_dir
        self.captions = captions
        self.default_prompt = default_prompt
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        self.image_paths: List[Path] = [
            path
            for path in sorted(root_dir.iterdir())
            if path.suffix.lower() in supported_extensions and path.is_file()
        ]
        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir} with supported extensions {supported_extensions}.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        pixel_values = self.transform(image)
        caption = self.captions.get(image_path.name, self.default_prompt)
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }


def load_captions(captions_file: Optional[Path]) -> Dict[str, str]:
    if captions_file is None:
        return {}
    with captions_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Captions file must contain a JSON object mapping filenames to prompts.")
    return {str(key): str(value) for key, value in data.items()}


def prepare_dataloader(config: ImageTrainingConfig, tokenizer) -> DataLoader:
    captions = load_captions(config.captions_file)
    dataset = ImageCaptionDataset(
        root_dir=config.data_dir,
        captions=captions,
        default_prompt=config.default_prompt,
        resolution=config.resolution,
        tokenizer=tokenizer,
        max_length=tokenizer.model_max_length,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)


def configure_pipeline(config: ImageTrainingConfig, device: torch.device):
    dtype = torch.float16 if config.mixed_precision and device.type == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.unet.train()
    pipeline.text_encoder.eval()
    pipeline.vae.eval()
    for param in pipeline.text_encoder.parameters():
        param.requires_grad_(False)
    for param in pipeline.vae.parameters():
        param.requires_grad_(False)
    return pipeline


def save_pipeline(pipeline: StableDiffusionPipeline, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(output_dir)


def train(config: ImageTrainingConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    pipeline = configure_pipeline(config, device)
    dataloader = prepare_dataloader(config, pipeline.tokenizer)
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(enabled=config.mixed_precision and device.type == "cuda")
    log_path = config.output_dir / "training_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["step", "epoch", "loss"])
        global_step = 0
        for epoch in range(config.epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(dataloader):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                with torch.no_grad():
                    latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipeline.vae.config.scaling_factor
                    encoder_hidden_states = pipeline.text_encoder(input_ids)[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.size(0),), device=device)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                with torch.amp.autocast(enabled=config.mixed_precision and device.type == "cuda"):
                    model_output = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(model_output.float(), noise.float()) / config.gradient_accumulation
                scaler.scale(loss).backward()
                if (step + 1) % config.gradient_accumulation == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    writer.writerow([global_step, epoch, loss.item() * config.gradient_accumulation])
                    log_file.flush()
                epoch_loss += loss.item() * config.gradient_accumulation
            avg_loss = epoch_loss / max(1, len(dataloader))
            print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {avg_loss:.4f}")
    save_pipeline(pipeline, config.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Stable Diffusion model on a local image folder.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file (retrain_image_model_config.json)")
    parser.add_argument("--pretrained-model", type=str, default=None, help="Stable Diffusion checkpoint to fine-tune.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing training images.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory for the fine-tuned model.")
    parser.add_argument("--captions", type=Path, default=None, help="Optional JSON file mapping image filenames to prompts.")
    parser.add_argument("--default-prompt", type=str, default=None, help="Fallback prompt when caption is missing.")
    parser.add_argument("--resolution", type=int, default=None, help="Image resolution for training.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate for optimizer.")
    parser.add_argument("--grad-accumulation", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=None,
        help="Use FP16 mixed precision training.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ImageTrainingConfig:
    config_data, config_dir = load_config_file(args.config)
    args_dict = vars(args)
    pretrained_model = pick_param(args_dict, config_data, "pretrained_model")
    data_dir = pick_param(args_dict, config_data, "data_dir", is_path=True, base_dir=config_dir)
    output_dir = pick_param(args_dict, config_data, "output_dir", is_path=True, base_dir=config_dir)
    if not pretrained_model or data_dir is None or output_dir is None:
        raise SystemExit("Missing required parameters: pretrained_model, data_dir, and output_dir must be provided.")
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"Training data directory does not exist or is not a directory: {data_dir}")
    captions = pick_param(args_dict, config_data, "captions", is_path=True, base_dir=config_dir)
    if captions is not None:
        captions = captions.resolve()
        if not captions.is_file():
            raise SystemExit(f"Captions file not found: {captions}")
    default_prompt = pick_param(args_dict, config_data, "default_prompt", "A photo of the subject")
    resolution = pick_param(args_dict, config_data, "resolution", 512, caster=int)
    if resolution <= 0:
        raise SystemExit("Resolution must be a positive integer.")
    batch_size = pick_param(args_dict, config_data, "batch_size", 1, caster=int)
    if batch_size <= 0:
        raise SystemExit("Batch size must be a positive integer.")
    epochs = pick_param(args_dict, config_data, "epochs", 1, caster=int)
    if epochs <= 0:
        raise SystemExit("Epochs must be a positive integer.")
    learning_rate = pick_param(args_dict, config_data, "learning_rate", 1e-4, caster=float)
    if learning_rate <= 0:
        raise SystemExit("Learning rate must be positive.")
    grad_accumulation = pick_param(args_dict, config_data, "grad_accumulation", 1, caster=int)
    if grad_accumulation <= 0:
        raise SystemExit("Gradient accumulation steps must be a positive integer.")
    mixed_precision = pick_param(args_dict, config_data, "mixed_precision", False, caster=to_bool)
    seed = pick_param(args_dict, config_data, "seed", 42, caster=int)
    return ImageTrainingConfig(
        pretrained_model=str(pretrained_model),
        data_dir=data_dir,
        output_dir=output_dir,
        captions_file=captions,
        default_prompt=str(default_prompt),
        resolution=int(resolution),
        batch_size=int(batch_size),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        gradient_accumulation=int(grad_accumulation),
        mixed_precision=bool(mixed_precision),
        seed=int(seed),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    train(config)


if __name__ == "__main__":
    main()
