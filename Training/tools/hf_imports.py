from __future__ import annotations

import importlib
import importlib.util
import site
import sys
import sysconfig
from pathlib import Path
from types import ModuleType


def _looks_like_site_package(path: Path) -> bool:
    lowered = str(path).lower()
    return 'site-packages' in lowered or 'dist-packages' in lowered


def _module_points_to_shadow(module: ModuleType, shadow_dir: Path) -> bool:
    module_paths = getattr(module, '__path__', None)
    if not module_paths:
        module_file = getattr(module, '__file__', None)
        if not module_file:
            return False
        try:
            return Path(module_file).resolve().is_relative_to(shadow_dir)
        except AttributeError:
            return shadow_dir in Path(module_file).resolve().parents
    for entry in module_paths:
        try:
            if Path(entry).resolve() == shadow_dir:
                return True
        except Exception:
            continue
    return False


def _load_from_init(init_file: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        'datasets',
        init_file,
        submodule_search_locations=[str(init_file.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load datasets package from {init_file}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['datasets'] = module
    spec.loader.exec_module(module)
    return module


def _site_package_dirs() -> list[Path]:
    directories: set[Path] = set()
    for entry in site.getsitepackages() + [site.getusersitepackages()]:
        if not entry:
            continue
        path = Path(entry)
        if path.exists():
            directories.add(path.resolve())
    syscfg = sysconfig.get_paths()
    for key in ('purelib', 'platlib'):
        value = syscfg.get(key)
        if not value:
            continue
        path = Path(value)
        if path.exists():
            directories.add(path.resolve())
    return list(directories)


def _candidate_init_files(shadow_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    search_paths: list[Path] = []
    for entry in list(sys.path):
        if not entry:
            continue
        try:
            base = Path(entry).resolve()
        except Exception:
            continue
        search_paths.append(base)
    search_paths.extend(_site_package_dirs())

    try:
        repo_site_packages = shadow_dir.parent / 'lib' / 'site-packages'
        if repo_site_packages.exists():
            search_paths.append(repo_site_packages.resolve())
    except Exception:
        pass

    seen: set[Path] = set()
    for base in search_paths:
        if base in seen:
            continue
        seen.add(base)
        candidate_dir = base / 'datasets'
        init_file = candidate_dir / '__init__.py'
        if not init_file.exists():
            continue
        try:
            if candidate_dir.resolve() == shadow_dir:
                continue
        except Exception:
            continue
        candidates.append(init_file)
    return candidates


def load_datasets_module() -> ModuleType:
    """Load the Hugging Face ``datasets`` package even when a local ``datasets`` directory exists."""

    module = sys.modules.get('datasets')
    if module is not None:
        repo_root = Path(__file__).resolve().parents[2]
        shadow_dir = (repo_root / 'datasets').resolve()
        if not _module_points_to_shadow(module, shadow_dir):
            return module

    repo_root = Path(__file__).resolve().parents[2]
    shadow_dir = (repo_root / 'datasets').resolve()

    for init_file in _candidate_init_files(shadow_dir):
        if _looks_like_site_package(init_file.parent):
            try:
                return _load_from_init(init_file)
            except ModuleNotFoundError as missing_dependency:
                missing_name = getattr(missing_dependency, 'name', None) or str(missing_dependency)
                raise RuntimeError(
                    f"Hugging Face 'datasets' requires the optional dependency '{missing_name}'. "
                    f"Install it via 'pip install {missing_name}' or 'pip install datasets[{missing_name}]'."
                ) from missing_dependency

    for init_file in _candidate_init_files(shadow_dir):
        try:
            return _load_from_init(init_file)
        except Exception:
            continue

    try:
        module = importlib.import_module('datasets')
    except ModuleNotFoundError as exc:
        raise RuntimeError("Hugging Face 'datasets' package is not installed. Run 'pip install datasets'.") from exc

    if _module_points_to_shadow(module, shadow_dir):
        raise RuntimeError(
            "Local 'datasets' directory shadows the Hugging Face package. "
            "Rename the directory or adjust PYTHONPATH so the installed package is discovered first."
        )

    return module
