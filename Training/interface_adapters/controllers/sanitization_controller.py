from __future__ import annotations

from pathlib import Path

from Training.app.use_cases.sanitization import sanitize_dataset
from Training.domain.entities import SanitizationRequest, SanitizationResult


def sanitize(
    dataset_path: Path,
    quarantine_csv: Path | None = None,
    save_sanitized: bool = True,
) -> SanitizationResult:
    request = SanitizationRequest(
        dataset_path=dataset_path,
        quarantine_csv=quarantine_csv,
        save_sanitized=save_sanitized,
    )
    return sanitize_dataset(request)
