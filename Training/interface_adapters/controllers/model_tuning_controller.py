from __future__ import annotations

from pathlib import Path

from Training.app.use_cases.model_tuning import tune_model
from Training.domain.entities import ModelTuningRequest, ModelTuningResult


def tune(
    config_path: Path,
    dataset_list_path: Path,
    prefer_sanitized: bool = True,
) -> ModelTuningResult:
    request = ModelTuningRequest(
        config_path=config_path,
        dataset_list_path=dataset_list_path,
        prefer_sanitized=prefer_sanitized,
    )
    return tune_model(request)
