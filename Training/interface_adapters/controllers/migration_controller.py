from __future__ import annotations

from typing import List, Tuple

from Training.app.use_cases import migration
from Training.domain.entities import MigrationPlan
from Training.domain.path_config import PathConfig


def build_plan(config: PathConfig) -> MigrationPlan:
    plan = migration.plan_migration(config)
    return plan


def execute_plan(plan: MigrationPlan, conflict_policy: str) -> List[Tuple[str, str]]:
    performed = migration.execute_migration(plan, conflict_policy)
    return [(str(src), str(dest)) for src, dest in performed]
