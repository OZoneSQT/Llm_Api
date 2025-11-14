from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


RepoTypeLiteral = Literal['model', 'dataset', 'space', 'media']
RepoTypeDefault = Literal['model', 'dataset', 'media']
RepoType = RepoTypeLiteral | str
RepoTypeOptional = RepoType | None


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
        # Use E:\AI\cache as the canonical cache location (user requested)
        cache_dir = cls._resolve_env('HF_CACHE_DIR', r'E:\\AI\\cache')
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
    def _cache_snapshot_dir(cache_dir: Path, repo_id: str, repo_type: RepoTypeOptional) -> Path:
        prefix_map: dict[RepoTypeOptional, str] = {
            None: 'models',
            'model': 'models',
            'dataset': 'datasets',
            'space': 'spaces',
            'media': 'spaces',
        }
        prefix = prefix_map.get(repo_type, 'models')
        safe_repo = repo_id.replace('/', '--')
        return cache_dir / f"{prefix}--{safe_repo}" / 'snapshots'

    def _custom_roots_for_type(self, repo_type: RepoType | None) -> Iterable[Path]:
        normalized = repo_type if repo_type in ('model', 'dataset', 'media') else 'model'
        if normalized == 'model':
            return (self.model_root,)
        if normalized == 'dataset':
            return (self.data_root,)
        return (self.cache_dir,)

    def _cached_snapshot_candidates(self, repo_id: str, repo_type: RepoType | None) -> Iterable[Path]:
        cache_type: RepoTypeOptional
        if repo_type == 'dataset':
            cache_type = 'dataset'
        elif repo_type == 'media':
            cache_type = 'media'
        else:
            cache_type = repo_type
        snapshot_dir = self._cache_snapshot_dir(self.cache_dir, repo_id, cache_type)
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

    def find_repo_path(self, repo_id: str, repo_type: RepoType | None = 'model') -> Path | None:
        safe_name = self.safe_repo_name(repo_id)
        for root in self._custom_roots_for_type(repo_type):
            candidate = root / safe_name
            if candidate.exists():
                return candidate

        cached_candidates = list(self._cached_snapshot_candidates(repo_id, repo_type))
        if cached_candidates:
            return cached_candidates[0]

        return None

    def resolve_repo_path(self, repo_id: str, repo_type: RepoType | None = 'model') -> Path:
        existing = self.find_repo_path(repo_id, repo_type)
        if existing is not None:
            return existing

        safe_name = self.safe_repo_name(repo_id)
        normalized_type = repo_type if isinstance(repo_type, str) and repo_type in {'model', 'dataset', 'media'} else 'model'
        fallback_root = {
            'model': self.model_root,
            'dataset': self.data_root,
            'media': self.cache_dir,
        }.get(normalized_type, self.model_root)
        return fallback_root / safe_name

    def cache_snapshot_dir(self, repo_id: str, repo_type: RepoType | None = 'model') -> Path:
        if repo_type == 'media':
            hf_repo_type: Literal['model', 'dataset', 'space'] | None = 'model'
        elif repo_type == 'dataset':
            hf_repo_type = 'dataset'
        elif repo_type == 'space':
            hf_repo_type = 'space'
        elif repo_type in (None,):
            hf_repo_type = None
        else:
            hf_repo_type = 'model'
        return self._cache_snapshot_dir(self.cache_dir, repo_id, hf_repo_type)