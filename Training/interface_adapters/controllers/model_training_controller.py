from __future__ import annotations

from pathlib import Path

from Training.app.use_cases.model_training import train_model
from Training.domain.entities import ModelTrainingRequest, ModelTrainingResult


def train(
    config_path: Path,
    dataset_list_path: Path,
    prefer_sanitized: bool = True,
) -> ModelTrainingResult:
    request = ModelTrainingRequest(
        config_path=config_path,
        dataset_list_path=dataset_list_path,
        prefer_sanitized=prefer_sanitized,
    )
    return train_model(request)
