"""Small adapter-aware loader helpers used by Training scripts.

These helpers prefer the repository-level `model_adapter` when the environment
variable `USE_ADAPTER` is set to 1/true/yes. They fall back to the Hugging Face
`from_pretrained` APIs when the adapter is not enabled or if the adapter load fails.

Keep these helpers minimal and tolerant so they can be used in many small tools.
"""
from typing import Callable, Optional
import os
import traceback

from Training.app.use_cases import model_loading


def is_use_adapter() -> bool:
    return model_loading.use_adapter_enabled(os.environ)


def load_tokenizer(name_or_path: str, local_files_only: bool = False, **kwargs):
    if is_use_adapter():
        try:
            from Training.tools.model_adapter import load_with_adapter

            tokenizer, model, meta = load_with_adapter(name_or_path, local_files_only=local_files_only)
            if tokenizer is not None:
                return tokenizer
        except Exception:
            traceback.print_exc()

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path, local_files_only=local_files_only, **kwargs)


def load_model(name_or_path: str, loader: Optional[Callable] = None, local_files_only: bool = False, **kwargs):
    return model_loading.load_with_adapter(
        name_or_path=name_or_path,
        loader=loader,
        env=os.environ,
        local_files_only=local_files_only,
        **kwargs,
    )
