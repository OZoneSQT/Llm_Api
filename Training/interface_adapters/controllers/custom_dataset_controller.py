from __future__ import annotations

from pathlib import Path

from Training.app.use_cases.custom_dataset import build_custom_dataset
from Training.domain.entities import CustomDatasetRequest, CustomDatasetResult


def generate_custom_dataset(
    source_dir: Path,
    dataset_root: Path,
    sanitize: bool = True,
    timestamp: str | None = None,
) -> CustomDatasetResult:
    request = CustomDatasetRequest(
        source_dir=source_dir,
        dataset_root=dataset_root,
        sanitize=sanitize,
        timestamp=timestamp,
    )
    return build_custom_dataset(request)
