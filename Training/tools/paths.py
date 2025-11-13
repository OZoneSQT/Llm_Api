"""Centralized paths for model, dataset, cache, and log locations.

Prefer reading values from environment variables so users can override defaults:
- HF_MODEL_ROOT (default: E:\\AI\\Models)
- HF_DATA_ROOT  (default: E:\\AI\\Datasets)
- HF_CACHE_DIR  (default: E:\\AI\\_hf_cache)
- HF_LOG_DIR    (default: E:\\AI\\Logs)

Import these values from scripts that need to read/write model or dataset folders.
"""
from pathlib import Path
from typing import Iterable, Literal

from Training.domain.path_config import PathConfig

_CONFIG = PathConfig.from_env()

MODEL_ROOT = _CONFIG.model_root
DATA_ROOT = _CONFIG.data_root
CACHE_DIR = _CONFIG.cache_dir
LOG_DIR = _CONFIG.log_dir


def safe_repo_name(repo_id: str) -> str:
	return _CONFIG.safe_repo_name(repo_id)


def resolve_model_path(repo_id: str) -> Path:
	return _CONFIG.resolve_repo_path(repo_id, 'model')


def resolve_dataset_path(repo_id: str) -> Path:
	return _CONFIG.resolve_repo_path(repo_id, 'dataset')


def resolve_media_path(repo_id: str) -> Path:
	return _CONFIG.resolve_repo_path(repo_id, 'media')


def find_model_path(repo_id: str) -> Path | None:
	return _CONFIG.find_repo_path(repo_id, 'model')


def find_dataset_path(repo_id: str) -> Path | None:
	return _CONFIG.find_repo_path(repo_id, 'dataset')


def find_media_path(repo_id: str) -> Path | None:
	return _CONFIG.find_repo_path(repo_id, 'media')


def cache_snapshot_dir(repo_id: str, repo_type: str | None = 'model') -> Path:
	return _CONFIG.cache_snapshot_dir(repo_id, repo_type)


def _dataset_prefix(kind: Literal['custom', 'sanitized']) -> str:
	return {
		'custom': 'custom_dataset_',
		'sanitized': 'sanitized_dataset_',
	}[kind]


def list_dataset_dirs(kind: Literal['custom', 'sanitized'] | None = None) -> list[Path]:
	if not DATA_ROOT.exists():
		return []

	if kind is None:
		prefixes: Iterable[str] = (_dataset_prefix('custom'), _dataset_prefix('sanitized'))
	else:
		prefixes = (_dataset_prefix(kind),)

	def matches(path: Path) -> bool:
		return any(path.name.startswith(prefix) for prefix in prefixes)

	try:
		candidates = [p for p in DATA_ROOT.iterdir() if p.is_dir() and matches(p)]
	except FileNotFoundError:
		return []

	def _mtime(path: Path) -> int:
		try:
			return path.stat().st_mtime_ns
		except OSError:
			return 0

	return sorted(candidates, key=_mtime, reverse=True)


def latest_dataset(kind: Literal['custom', 'sanitized']) -> Path | None:
	datasets = list_dataset_dirs(kind)
	return datasets[0] if datasets else None


def latest_dataset_prefer_sanitized() -> Path | None:
	sanitized = latest_dataset('sanitized')
	if sanitized is not None:
		return sanitized
	return latest_dataset('custom')
