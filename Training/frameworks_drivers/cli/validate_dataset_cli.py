from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Training.interface_adapters.controllers.validate_dataset_controller import validate_and_write


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Validate a dataset saved to disk and optionally write metadata.')
    parser.add_argument('--dataset', required=True, help='Path to dataset folder (saved with datasets.save_to_disk)')
    parser.add_argument('--min-size', type=int, default=1, help='Minimum number of examples')
    parser.add_argument('--required-column', action='append', help='Required column name (can be passed multiple times)')
    args = parser.parse_args(argv)

    path = Path(args.dataset).expanduser().resolve()
    res = validate_and_write(path, required_columns=args.required_column, min_size=args.min_size)
    if not res.valid:
        print('Validation failed:')
        for k, v in res.errors.items():
            print(f'- {k}: {v}')
        return 2

    print('Validation passed')
    print('Metadata written:')
    for k, v in res.metadata.items():
        print(f'- {k}: {v}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
