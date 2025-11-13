from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from datasets import load_from_disk


class ValidationResult:
    def __init__(self, valid: bool, errors: Dict[str, str], metadata: Dict[str, Any] | None = None):
        self.valid = valid
        self.errors = errors
        self.metadata = metadata or {}


def validate_dataset(dataset_path: Path, required_columns: list[str] | None = None, min_size: int = 1) -> ValidationResult:
    """Validate a dataset saved to disk.

    Checks performed:
    - path exists and is a directory
    - dataset can be loaded with `load_from_disk`
    - required columns exist
    - dataset length >= min_size

    Returns a ValidationResult describing any errors.
    """
    errors: Dict[str, str] = {}
    if not dataset_path.exists() or not dataset_path.is_dir():
        errors['path'] = 'Dataset path does not exist or is not a directory'
        return ValidationResult(False, errors)

    try:
        ds = load_from_disk(str(dataset_path))
    except Exception as exc:
        errors['load'] = f'Failed to load dataset: {exc}'
        return ValidationResult(False, errors)

    # Basic checks
    try:
        length = len(ds)
    except Exception:
        # some datasets may be streaming-like; treat as unknown
        length = None

    if length is None:
        # attempt simple iteration to count a small sample
        try:
            length = sum(1 for _ in ds.take(1_000))
        except Exception:
            errors['length'] = 'Unable to determine dataset length'
            return ValidationResult(False, errors)

    if length < min_size:
        errors['size'] = f'Dataset contains {length} examples, fewer than minimum required ({min_size})'

    if required_columns:
        missing = [c for c in required_columns if c not in ds.column_names]
        if missing:
            errors['columns'] = f'Missing required columns: {missing}'

    metadata = {
        'length': length,
        'columns': ds.column_names,
    }

    return ValidationResult(len(errors) == 0, errors, metadata)


def write_metadata(dataset_path: Path, metadata: Dict[str, Any]) -> None:
    out = dataset_path / 'dataset_metadata.json'
    with out.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
