from __future__ import annotations

import argparse
from pathlib import Path

from Training.interface_adapters.controllers.custom_dataset_controller import generate_custom_dataset
from Training.interface_adapters.controllers.validate_dataset_controller import validate_and_write
from Training.tools import paths as path_utils
from pathlib import Path


def _default_source_dir() -> Path:
    training_dir = Path(__file__).resolve().parents[2]
    return training_dir / 'raw_training_data'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate a custom dataset from local documents.')
    parser.add_argument(
        '--source',
        type=Path,
        default=_default_source_dir(),
        help='Directory containing raw training documents.',
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=path_utils.DATA_ROOT,
        help='Root directory where datasets are stored.',
    )
    parser.add_argument(
        '--skip-sanitize',
        action='store_true',
        help='Skip the sanitization pass.',
    )
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='Optional timestamp suffix for the dataset folder.',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = generate_custom_dataset(
        source_dir=args.source,
        dataset_root=args.dataset_root,
        sanitize=not args.skip_sanitize,
        timestamp=args.timestamp,
    )

    if result.raw_dataset_path is None:
        print('No documents found. Nothing to do.')
        return

    print(f'Raw dataset saved to {result.raw_dataset_path}')
    if result.metadata_path:
        print(f'Metadata written to {result.metadata_path}')
    if result.sanitized_dataset_path:
        print(f'Sanitized dataset saved to {result.sanitized_dataset_path}')
    if result.quarantine_report:
        print(f'Quarantine report written to {result.quarantine_report}')
    print(f'Document count: {result.document_count}')
    print(f'Flagged count: {result.flagged_count}')

    # Validate the resulting dataset (prefer sanitized output when present)
    target = Path(result.sanitized_dataset_path if result.sanitized_dataset_path else result.raw_dataset_path)
    if target and target.exists():
        print('Running dataset validation...')
        val = validate_and_write(target)
        if not val.valid:
            print('Dataset validation failed:')
            for k, v in val.errors.items():
                print(f'- {k}: {v}')
            raise SystemExit(2)
        print('Dataset validation passed; metadata written.')


if __name__ == '__main__':  # pragma: no cover
    main()
