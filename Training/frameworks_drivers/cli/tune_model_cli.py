from __future__ import annotations

import argparse
from pathlib import Path

from Training.interface_adapters.controllers.model_tuning_controller import tune


def _default_config_path() -> Path:
    training_dir = Path(__file__).resolve().parents[2]
    return training_dir / 'data' / 'config.json'


def _default_dataset_list_path() -> Path:
    training_dir = Path(__file__).resolve().parents[2]
    return training_dir / 'data' / 'dataset.txt'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Fine-tune a sequence classification model.')
    parser.add_argument('--config', type=Path, default=_default_config_path(), help='Path to tuning configuration JSON.')
    parser.add_argument(
        '--dataset-list',
        type=Path,
        default=_default_dataset_list_path(),
        help='Path to newline-delimited Hugging Face datasets file.',
    )
    parser.add_argument(
        '--no-prefer-sanitized',
        action='store_true',
        help='Use the latest custom dataset even if a sanitized dataset exists.',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = tune(
        config_path=args.config,
        dataset_list_path=args.dataset_list,
        prefer_sanitized=not args.no_prefer_sanitized,
    )

    print(f'Tuned model saved to {result.model_output_dir}')
    print(f'Metrics recorded at {result.metrics_path}')


if __name__ == '__main__':  # pragma: no cover
    main()
