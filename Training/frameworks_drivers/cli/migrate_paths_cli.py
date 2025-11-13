from __future__ import annotations

import argparse

from Training.domain.path_config import PathConfig
from Training.frameworks_drivers.logging import get_csv_logger
from Training.interface_adapters.controllers import migration_controller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Migrate legacy E: model/dataset/cache folders into centralized roots.')
    parser.add_argument('--yes', action='store_true', help='Execute the migration (default: dry-run).')
    parser.add_argument('--conflict-policy', choices=['skip', 'rename', 'merge', 'overwrite'], default='skip', help='Conflict resolution strategy when destination exists.')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = get_csv_logger('migrate_paths_cli')
    logger.info('migrate_paths_invoked', argv=list(argv) if argv is not None else [])

    config = PathConfig.from_env()
    print('Central targets:')
    print('  MODEL_ROOT:', config.model_root)
    print('  DATA_ROOT :', config.data_root)
    print('  CACHE_DIR :', config.cache_dir)
    print('  LOG_DIR  :', config.log_dir)
    logger.info(
        'central_paths_resolved',
        model_root=str(config.model_root),
        data_root=str(config.data_root),
        cache_dir=str(config.cache_dir),
        log_dir=str(config.log_dir),
    )

    plan = migration_controller.build_plan(config)
    print('\nPlanned actions:')
    for item in plan.items:
        flag = 'MOVE' if item.action != 'skip-conflict' else 'SKIP if conflict'
        print(f" - {flag}: {item.source} -> {item.destination} ({item.size_bytes} bytes) [{item.category}]")
    logger.info('migration_plan_built', total_items=len(plan.items), total_bytes=plan.total_bytes)

    print(f"\nTotal size to move: {plan.total_bytes} bytes")
    print(f"Planned items: {len(plan.items)}")
    print(f"Conflict policy: {args.conflict_policy}")

    if not args.yes:
        print('\nDry-run mode: no filesystem changes will be made. Run with --yes to execute.')
        logger.info('migration_dry_run', conflict_policy=args.conflict_policy)
        logger.close()
        return 0

    confirmation = input("Proceed with moves and apply changes? Type 'yes' to continue: ").strip().lower()
    if confirmation != 'yes':
        print('Aborted by user.')
        logger.warning('migration_aborted_by_user')
        logger.close()
        return 0

    try:
        performed = migration_controller.execute_plan(plan, args.conflict_policy)
        print('\nPerformed moves:')
        for src, dest in performed:
            print(f'  {src} -> {dest}')
        print('Migration complete.')
        logger.info('migration_completed', performed=len(performed), conflict_policy=args.conflict_policy)
        return 0
    except Exception as exc:  # pragma: no cover - defensive guard for CLI surface
        logger.exception('migration_failed', conflict_policy=args.conflict_policy, error=str(exc))
        print('Migration failed:', exc)
        return 1
    finally:
        logger.close()


if __name__ == '__main__':
    raise SystemExit(main())
