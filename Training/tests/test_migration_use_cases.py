from pathlib import Path

from Training.app.use_cases import migration
from Training.domain.entities import MigrationItem, MigrationPlan
from Training.domain.path_config import PathConfig


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_collect_legacy_roots_deduplicates(monkeypatch, tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    monkeypatch.setattr(
        migration,
        "LEGACY_PATHS",
        ((str(legacy), "models"), (str(legacy), "models")),
        raising=False,
    )

    roots = migration.collect_legacy_roots()

    assert len(roots) == 1
    assert roots[0][0] == legacy


def test_plan_migration_builds_items(monkeypatch, tmp_path: Path) -> None:
    models_src = tmp_path / "legacy_models"
    datasets_src = tmp_path / "legacy_datasets"
    cache_src = tmp_path / "legacy_cache"
    for folder in (models_src, datasets_src, cache_src):
        folder.mkdir()

    (models_src / "ModelA").mkdir()
    _write_file(models_src / "ModelA" / "weights.bin", "m")
    _write_file(datasets_src / "DatasetA.json", "d")
    _write_file(cache_src / "tokenizer.model", "c")

    monkeypatch.setattr(
        migration,
        "LEGACY_PATHS",
        (
            (str(models_src), "models"),
            (str(datasets_src), "datasets"),
            (str(cache_src), "cache"),
        ),
        raising=False,
    )

    config = PathConfig(
        model_root=tmp_path / "models",
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )

    plan = migration.plan_migration(config)

    assert plan.total_bytes >= 3
    assert {(item.category, item.action) for item in plan.items} == {
        ("models", "move"),
        ("datasets", "move"),
        ("cache", "move"),
    }
    destinations = {item.destination.parent for item in plan.items}
    assert destinations == {config.model_root, config.data_root, config.cache_dir}


def test_execute_migration_skip_policy(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    dest = tmp_path / "dest.bin"
    _write_file(src, "source")
    _write_file(dest, "destination")

    plan = MigrationPlan(
        items=[MigrationItem(src, dest, size_bytes=6, category="models", action="move")],
        total_bytes=6,
        conflict_policy="skip",
    )

    performed = migration.execute_migration(plan, conflict_policy="skip")

    assert performed == []
    assert src.exists()
    assert dest.exists()


def test_execute_migration_rename_policy(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "rename_src"
    dest = tmp_path / "rename_dest"
    src.mkdir()
    dest.mkdir()
    _write_file(src / "file.txt", "incoming")
    _write_file(dest / "existing.txt", "existing")

    monkeypatch.setattr(migration.time, "time", lambda: 1111, raising=False)

    plan = MigrationPlan(
        items=[MigrationItem(src, dest, size_bytes=8, category="models", action="move")],
        total_bytes=8,
        conflict_policy="rename",
    )

    performed = migration.execute_migration(plan, conflict_policy="rename")

    assert performed == [(src, dest)]
    assert not any(src.rglob("*"))
    assert (dest / "file.txt").exists()
    renamed = dest.parent / "rename_dest.old.1111"
    assert renamed.exists()


def test_execute_migration_merge_policy(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "merge_src"
    dest = tmp_path / "merge_dest"
    (src / "sub").mkdir(parents=True)
    (dest / "sub").mkdir(parents=True)
    _write_file(src / "sub" / "conflict.txt", "incoming")
    _write_file(src / "sub" / "unique.txt", "unique")
    _write_file(dest / "sub" / "conflict.txt", "existing")

    monkeypatch.setattr(migration.time, "time", lambda: 2222, raising=False)

    plan = MigrationPlan(
        items=[MigrationItem(src, dest, size_bytes=12, category="models", action="move")],
        total_bytes=12,
        conflict_policy="merge",
    )

    performed = migration.execute_migration(plan, conflict_policy="merge")

    assert performed == [(src, dest)]
    assert not any(src.rglob("*"))
    conflict_file = dest / "sub" / "conflict.txt"
    migrated_file = dest / "sub" / "conflict.txt.migrated.2222"
    unique_file = dest / "sub" / "unique.txt"
    assert conflict_file.exists()
    assert migrated_file.exists()
    assert unique_file.exists()
    assert conflict_file.read_text(encoding="utf-8") == "existing"
    assert migrated_file.read_text(encoding="utf-8") == "incoming"


def test_execute_migration_overwrite_policy(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "overwrite_src.bin"
    dest = tmp_path / "overwrite_dest.bin"
    _write_file(src, "incoming")
    _write_file(dest, "existing")

    monkeypatch.setattr(migration.time, "time", lambda: 3333, raising=False)

    plan = MigrationPlan(
        items=[MigrationItem(src, dest, size_bytes=8, category="datasets", action="move")],
        total_bytes=8,
        conflict_policy="overwrite",
    )

    performed = migration.execute_migration(plan, conflict_policy="overwrite")

    assert performed == [(src, dest)]
    assert not src.exists()
    assert dest.read_text(encoding="utf-8") == "incoming"


def test_execute_migration_moves_when_destination_missing(tmp_path: Path) -> None:
    src = tmp_path / "fresh_src"
    dest_root = tmp_path / "targets"
    dest = dest_root / "fresh_src"

    src.mkdir()
    _write_file(src / "file.txt", "content")

    plan = MigrationPlan(
        items=[MigrationItem(src, dest, size_bytes=7, category="cache", action="move")],
        total_bytes=7,
        conflict_policy="skip",
    )

    performed = migration.execute_migration(plan, conflict_policy="skip")

    assert performed == [(src, dest)]
    assert not src.exists()
    assert (dest / "file.txt").exists()