from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from Training.app.use_cases.validate_dataset import validate_dataset, write_metadata


class ValidateResult:
    def __init__(self, valid: bool, errors: Dict[str, str], metadata: Dict[str, Any]):
        self.valid = valid
        self.errors = errors
        self.metadata = metadata


def validate(path: Path, required_columns: list[str] | None = None, min_size: int = 1) -> ValidateResult:
    result = validate_dataset(path, required_columns=required_columns, min_size=min_size)
    return ValidateResult(result.valid, result.errors, result.metadata)


def validate_and_write(path: Path, required_columns: list[str] | None = None, min_size: int = 1) -> ValidateResult:
    result = validate_dataset(path, required_columns=required_columns, min_size=min_size)
    if result.valid:
        write_metadata(path, result.metadata)
    return ValidateResult(result.valid, result.errors, result.metadata)
