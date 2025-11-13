from __future__ import annotations

import argparse
from pathlib import Path

from Training.frameworks_drivers.logging import get_csv_logger
from Training.interface_adapters.controllers import dataset_controller
from Training.tools import paths as path_utils


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare a small combined dataset for finetuning.')
    sanitized_default = path_utils.latest_dataset('sanitized')
    custom_default = path_utils.latest_dataset('custom')
    parser.add_argument('--datasets', type=str, default='', help='Comma-separated list of HF dataset repo IDs.')
    parser.add_argument('--from-file', type=str, default='./Training/data/dataset.txt', help='Path to a text file listing dataset repo IDs (one per line).')
    parser.add_argument('--include-custom', type=str, default=str(custom_default) if custom_default else '', help='Path to a local dataset directory (defaults to latest custom dataset if available).')
    parser.add_argument('--sanitized', type=str, default=str(sanitized_default) if sanitized_default else '', help='Path to a sanitized dataset directory (defaults to latest sanitized dataset if available).')
    parser.add_argument('--max-per', type=int, default=200, help='Max examples to take from each dataset.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle examples before sampling.')
    parser.add_argument('--output', type=str, default='Training/data/combined_small_dataset.jsonl', help='Output JSONL path for the combined dataset.')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to sample (default: train).')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = get_csv_logger('prepare_datasets_cli')
    logger.info('prepare_datasets_invoked', argv=list(argv) if argv is not None else [])
    exit_code = 0
    try:
        repos = []
        if args.datasets:
            repos.extend([repo.strip() for repo in args.datasets.split(',') if repo.strip()])
        if args.from_file:
            path = Path(args.from_file)
            if path.exists():
                repos.extend([line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip() and not line.strip().startswith('#')])
            else:
                logger.warning('dataset_list_file_missing', path=str(path))

        logger.info('dataset_sources_resolved', repo_count=len(repos))

        request = dataset_controller.build_dataset_request(
            repos=repos,
            split=args.split,
            max_examples=args.max_per,
            sanitized=Path(args.sanitized) if args.sanitized else None,
            custom=Path(args.include_custom) if args.include_custom else None,
            shuffle=args.shuffle,
            output=Path(args.output),
        )

        dataset_controller.prepare_dataset(request, args.max_per)
        print('Wrote combined dataset to:', request.output_path)
        logger.info(
            'prepare_datasets_completed',
            output=str(request.output_path),
            total_sources=len(repos),
            shuffle=bool(args.shuffle),
        )
        exit_code = 0
        return exit_code
    except Exception as exc:  # pragma: no cover - defensive guard for CLI surface
        logger.exception('prepare_datasets_failed', error=str(exc))
        print('Failed to prepare dataset:', exc)
        return 1
    finally:
        logger.close()


if __name__ == '__main__':
    raise SystemExit(main())
