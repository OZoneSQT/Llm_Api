from __future__ import annotations

from typing import Callable, Optional

def use_adapter_enabled(env: dict) -> bool:
    value = env.get('USE_ADAPTER', '')
    return value.lower() in ('1', 'true', 'yes')


def load_with_adapter(name_or_path: str, loader: Optional[Callable], env: dict, local_files_only: bool, **kwargs):
    if use_adapter_enabled(env):
        try:
            from Training.tools.model_adapter import load_with_adapter as repo_adapter

            tokenizer, model, meta = repo_adapter(name_or_path, local_files_only=local_files_only)
            if model is not None:
                return model
        except Exception:
            # fall back to provided loader
            pass
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault('local_files_only', local_files_only)
    if loader:
        return loader(name_or_path, **call_kwargs)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(name_or_path, **call_kwargs)
