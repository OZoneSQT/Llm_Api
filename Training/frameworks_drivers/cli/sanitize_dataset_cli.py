from __future__ import annotations

import argparse
from pathlib import Path

from Training.interface_adapters.controllers.sanitization_controller import sanitize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sanitize a saved Hugging Face dataset directory.')
    parser.add_argument('dataset_path', type=Path, help='Path to dataset directory created with Dataset.save_to_disk.')
    parser.add_argument(
        '--quarantine-csv',
        type=Path,
        default=None,
        help='Optional explicit path for the quarantine CSV report.',
    )
    parser.add_argument(
        '--no-sanitized-copy',
        action='store_true',
        help='Do not write a sanitized dataset copy to disk.',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = sanitize(
        dataset_path=args.dataset_path,
        quarantine_csv=args.quarantine_csv,
        save_sanitized=not args.no_sanitized_copy,
    )

    print(f'Quarantine report: {result.quarantine_csv}')
    print(f'Flagged records: {len(result.flagged)}')
    if result.sanitized_path:
        print(f'Sanitized dataset saved to {result.sanitized_path}')


if __name__ == '__main__':  # pragma: no cover
    main()
