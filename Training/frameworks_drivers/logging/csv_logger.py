from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TextIO


def _now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


@dataclass
class StructuredCsvLogger:
    component: str
    log_dir: Path
    run_id: str
    fieldnames: tuple[str, ...] = ("run_id", "timestamp", "level", "component", "message", "context")

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        safe_component = self.component.replace("/", "-")
        self._file_path = self.log_dir / f"{safe_component}_{self.run_id}.csv"
        self._handle: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter[str]] = None

    def _ensure_writer(self) -> csv.DictWriter[str]:
        if self._writer is None:
            self._handle = self._file_path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
            if self._file_path.stat().st_size == 0:
                self._writer.writeheader()
        return self._writer

    def _serialize_context(self, context: Dict[str, Any] | None) -> str:
        if not context:
            return ""
        return json.dumps(context, ensure_ascii=False, sort_keys=True)

    def _write(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        writer = self._ensure_writer()
        payload = {
            "run_id": self.run_id,
            "timestamp": _now_utc(),
            "level": level.upper(),
            "component": self.component,
            "message": message,
            "context": self._serialize_context(context),
        }
        writer.writerow(payload)
        if self._handle:
            self._handle.flush()

    def info(self, message: str, **context: Any) -> None:
        self._write("INFO", message, context or None)

    def warning(self, message: str, **context: Any) -> None:
        self._write("WARN", message, context or None)

    def error(self, message: str, **context: Any) -> None:
        self._write("ERROR", message, context or None)

    def exception(self, message: str, **context: Any) -> None:
        context = context or {}
        context.setdefault("traceback", traceback.format_exc())
        self._write("ERROR", message, context)

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None
            self._writer = None

    def __del__(self) -> None:  # pragma: no cover - defensive close
        self.close()


def get_csv_logger(component: str, *, log_dir: Path | None = None, run_id: str | None = None) -> StructuredCsvLogger:
    if log_dir is None:
        from Training.domain.path_config import PathConfig  # Imported lazily to avoid cycles

        log_dir = PathConfig.from_env().log_dir
    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    return StructuredCsvLogger(component=component, log_dir=Path(log_dir), run_id=run_id)
