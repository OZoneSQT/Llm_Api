from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def load_with_adapter(
    model_path: str,
    family: Optional[str] = None,
    commit_patch: bool = False,
    dtype: Any = None,
    low_cpu_mem_usage: bool = True,
    local_files_only: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]: ...
