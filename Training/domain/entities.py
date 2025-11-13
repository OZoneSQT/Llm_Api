from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelFormatDetection:
    root: Path
    formats: Dict[str, bool]


@dataclass(frozen=True)
class ConversionRequest:
    input_dir: Path
    output_dir: Path
    model_size: Optional[str] = None
    llama_version: Optional[str] = None
    device: str = 'cpu'
    dry_run: bool = False
    confirm: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    split: str
    max_examples: int


@dataclass(frozen=True)
class DatasetPreparationRequest:
    dataset_specs: list[DatasetSpec]
    sanitized_path: Optional[Path]
    custom_path: Optional[Path]
    shuffle: bool
    output_path: Path


@dataclass(frozen=True)
class SanitizationRequest:
    dataset_path: Path
    quarantine_csv: Optional[Path] = None
    save_sanitized: bool = True


@dataclass(frozen=True)
class SanitizationResult:
    flagged: list[tuple[int, list[str], str]]
    quarantine_csv: Path
    sanitized_path: Optional[Path]


@dataclass(frozen=True)
class CustomDatasetRequest:
    source_dir: Path
    dataset_root: Path
    sanitize: bool = True
    timestamp: Optional[str] = None


@dataclass(frozen=True)
class CustomDatasetResult:
    raw_dataset_path: Optional[Path]
    sanitized_dataset_path: Optional[Path]
    metadata_path: Optional[Path]
    document_count: int
    flagged_count: int
    quarantine_report: Optional[Path]


@dataclass(frozen=True)
class ModelTrainingRequest:
    config_path: Path
    dataset_list_path: Path
    prefer_sanitized: bool = True


@dataclass(frozen=True)
class ModelTrainingResult:
    model_output_dir: Path
    log_path: Path
    dataset_size: int


@dataclass(frozen=True)
class ModelTuningRequest:
    config_path: Path
    dataset_list_path: Path
    prefer_sanitized: bool = True


@dataclass(frozen=True)
class ModelTuningResult:
    model_output_dir: Path
    metrics_path: Path


@dataclass(frozen=True)
class MigrationItem:
    source: Path
    destination: Path
    size_bytes: int
    category: str
    action: str


@dataclass(frozen=True)
class MigrationPlan:
    items: list[MigrationItem]
    total_bytes: int
    conflict_policy: str
    confirmed: bool = False
