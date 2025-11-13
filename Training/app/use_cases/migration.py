from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Tuple

from Training.domain.entities import MigrationItem, MigrationPlan
from Training.domain.path_config import PathConfig


LEGACY_PATHS: Tuple[Tuple[str, str], ...] = (
    (r"E:\Models", 'models'),
    (r"E:\Datasets", 'datasets'),
    (r"E:\_hf_cache", 'cache'),
    (r"E:\AI\Models", 'models'),
    (r"E:\AI\Datasets", 'datasets'),
    (r"E:\AI\_hf_cache", 'cache'),
    (r"E:\HF\Models", 'models'),
    (r"E:\HF\Cache", 'cache'),
)


def _sizeof(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except Exception:
            return 0
    total = 0
    for root, _, files in os.walk(str(path)):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except Exception:
                continue
    return total


def collect_legacy_roots() -> List[Tuple[Path, str]]:
    collected: List[Tuple[Path, str]] = []
    seen = set()
    for raw_path, category in LEGACY_PATHS:
        candidate = Path(raw_path)
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            collected.append((candidate, category))
    return collected


def plan_migration(config: PathConfig) -> MigrationPlan:
    items: List[MigrationItem] = []
    total = 0
    for root, category in collect_legacy_roots():
        for entry in root.iterdir():
            if entry.name.startswith('.'):  # skip hidden/system files
                continue
            if category == 'models':
                destination = config.model_root / entry.name
            elif category == 'datasets':
                destination = config.data_root / entry.name
            else:
                destination = config.cache_dir / entry.name
            size_bytes = _sizeof(entry)
            total += size_bytes
            action = 'move' if not destination.exists() else 'skip-conflict'
            items.append(MigrationItem(entry, destination, size_bytes, category, action))
    return MigrationPlan(items=items, total_bytes=total, conflict_policy='skip', confirmed=False)


def _rename_existing(path: Path) -> None:
    timestamp = int(time.time())
    new_path = Path(str(path) + f'.old.{timestamp}')
    path.rename(new_path)


def _merge_directory(src: Path, dest: Path) -> None:
    for root, dirs, files in os.walk(str(src)):
        relative = os.path.relpath(root, str(src))
        target_root = dest / relative if relative != '.' else dest
        target_root.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            source_file = Path(root) / file_name
            target_file = target_root / file_name
            if target_file.exists():
                timestamp = int(time.time())
                alt = target_root / f"{file_name}.migrated.{timestamp}"
                shutil.move(str(source_file), str(alt))
            else:
                shutil.move(str(source_file), str(target_file))
    for root, dirs, files in os.walk(str(src), topdown=False):
        if not dirs and not files:
            Path(root).rmdir()


def execute_migration(plan: MigrationPlan, conflict_policy: str) -> List[Tuple[Path, Path]]:
    performed: List[Tuple[Path, Path]] = []
    for item in plan.items:
        src = item.source
        dest = item.destination
        if dest.exists():
            if conflict_policy == 'skip':
                continue
            if conflict_policy == 'rename':
                _rename_existing(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                performed.append((src, dest))
            elif conflict_policy == 'merge':
                _merge_directory(src, dest)
                performed.append((src, dest))
            elif conflict_policy == 'overwrite':
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink(missing_ok=True)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                performed.append((src, dest))
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            performed.append((src, dest))
    return performed
