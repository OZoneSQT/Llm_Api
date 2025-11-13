from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Literal
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Centralized filesystem locations derived from environment variables."""

    model_root: Path
    data_root: Path
    cache_dir: Path
    log_dir: Path

    @property
    def media_root(self) -> Path:
        """Media assets now reuse the cache directory as their storage root."""

        return self.cache_dir

    @staticmethod
    def _resolve_env(name: str, default: str) -> Path:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
        return Path(default).resolve()

    @classmethod
    def from_env(cls) -> "PathConfig":
        model_root = cls._resolve_env('HF_MODEL_ROOT', r'E:\AI\Models')
        data_root = cls._resolve_env('HF_DATA_ROOT', r'E:\AI\Datasets')
        cache_dir = cls._resolve_env('HF_CACHE_DIR', r'E:\AI\_hf_cache')
        log_dir = cls._resolve_env('HF_LOG_DIR', r'E:\AI\Logs')
        for path in (model_root, data_root, cache_dir, log_dir):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                # best-effort; callers must handle permission failures explicitly
                pass
        return cls(model_root=model_root, data_root=data_root, cache_dir=cache_dir, log_dir=log_dir)

    @staticmethod
    def safe_repo_name(repo_id: str) -> str:
        return repo_id.replace('/', '_')

    @staticmethod
    def _cache_snapshot_dir(cache_dir: Path, repo_id: str, repo_type: Literal['model', 'dataset', 'space'] | None) -> Path:
        prefix_map = {
            None: 'models',
            'model': 'models',
            'dataset': 'datasets',
            'space': 'spaces',
        }
        prefix = prefix_map.get(repo_type, 'models')
        safe_repo = repo_id.replace('/', '--')
        return cache_dir / f"{prefix}--{safe_repo}" / 'snapshots'

    def _custom_roots_for_type(self, repo_type: Literal['model', 'dataset', 'media']) -> Iterable[Path]:
        if repo_type == 'model':
            return (self.model_root,)
        if repo_type == 'dataset':
            return (self.data_root,)
        if repo_type == 'media':
            return (self.cache_dir,)
        return ()

    def _cached_snapshot_candidates(self, repo_id: str, repo_type: Literal['model', 'dataset', 'media'] | None) -> Iterable[Path]:
        snapshot_dir = self._cache_snapshot_dir(self.cache_dir, repo_id, 'dataset' if repo_type == 'dataset' else repo_type)
        if not snapshot_dir.exists():
            return []
        try:
            def _mtime(path: Path) -> int:
                try:
                    return path.stat().st_mtime_ns
                except OSError:
                    return 0

            snapshots = sorted(
                (entry for entry in snapshot_dir.iterdir() if entry.is_dir()),
                key=_mtime,
                reverse=True,
            )
            return snapshots
        except OSError:
            return []

    def find_repo_path(self, repo_id: str, repo_type: Literal['model', 'dataset', 'media'] = 'model') -> Path | None:
        safe_name = self.safe_repo_name(repo_id)
        for root in self._custom_roots_for_type(repo_type):
            candidate = root / safe_name
            if candidate.exists():
                return candidate

        cached_candidates = list(self._cached_snapshot_candidates(repo_id, repo_type))
        if cached_candidates:
            return cached_candidates[0]

        return None

    def resolve_repo_path(self, repo_id: str, repo_type: Literal['model', 'dataset', 'media'] = 'model') -> Path:
        existing = self.find_repo_path(repo_id, repo_type)
        if existing is not None:
            return existing

        safe_name = self.safe_repo_name(repo_id)
        fallback_root = {
            'model': self.model_root,
            'dataset': self.data_root,
            'media': self.cache_dir,
        }.get(repo_type, self.model_root)
        return fallback_root / safe_name

    def cache_snapshot_dir(self, repo_id: str, repo_type: Literal['model', 'dataset', 'media'] | None = 'model') -> Path:
        if repo_type == 'media':
            hf_repo_type: Literal['model', 'dataset', 'space'] | None = 'model'
        elif repo_type == 'dataset':
            hf_repo_type = 'dataset'
        else:
            hf_repo_type = repo_type
        return self._cache_snapshot_dir(self.cache_dir, repo_id, hf_repo_type)