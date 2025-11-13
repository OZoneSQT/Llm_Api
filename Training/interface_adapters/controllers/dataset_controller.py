from __future__ import annotations

from pathlib import Path
from typing import List

from Training.app.use_cases import dataset_preparation
from Training.domain.entities import DatasetPreparationRequest, DatasetSpec


def build_dataset_request(
    repos: List[str],
    split: str,
    max_examples: int,
    sanitized: Path | None,
    custom: Path | None,
    shuffle: bool,
    output: Path,
) -> DatasetPreparationRequest:
    specs = [DatasetSpec(repo_id=repo, split=split, max_examples=max_examples) for repo in repos]
    return DatasetPreparationRequest(
        dataset_specs=specs,
        sanitized_path=sanitized,
        custom_path=custom,
        shuffle=shuffle,
        output_path=output,
    )


def prepare_dataset(request: DatasetPreparationRequest, max_examples: int) -> Path:
    return dataset_preparation.prepare_dataset(request, max_examples)
