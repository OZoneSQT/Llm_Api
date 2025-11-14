import os
from pathlib import Path
import pytest

from Training.tools import model_adapter


def test_detect_family_from_path():
    assert model_adapter._detect_family_from_path(Path('E:/Models/meta-llama_Llama-3.2-1B')) == 'meta-llama'
    assert model_adapter._detect_family_from_path(Path('E:/Models/Qwen_Qwen3-1.7B')) == 'qwen'
    assert model_adapter._detect_family_from_path(Path('E:/Models/gemma_Gemma-1B')) == 'gemma'
    assert model_adapter._detect_family_from_path(Path('E:/Models/dphn_Dolphin3.0-Llama3.2-1B')) == 'dolphin'


@pytest.mark.skipif(os.environ.get('RUN_SMOKE_TESTS') != '1', reason='Smoke tests disabled; set RUN_SMOKE_TESTS=1 to enable')
def test_smoke_load_with_adapter_qwen():
    # Smoke test: attempt to load a Qwen model from local path or from HF cache if present
    candidates = [
        Path('E:/Models/Qwen_Qwen3-1.7B'),
        Path('E:/AI/Models/Qwen_Qwen3-1.7B'),
    ]
    # also search the HF cache for any folder with 'qwen' in the name
    # Prefer deterministic cache snapshot discovery via PathConfig if available
    try:
        from Training.domain.path_config import PathConfig
        _cfg = PathConfig.from_env()
        # Attempt to find a cached snapshot for a known repo-id pattern
        snap = _cfg.find_repo_path('Qwen/Qwen3-1.7B', repo_type='model')
        if snap is not None:
            candidates.append(snap)
    except Exception:
        cache_root = Path('E:/AI/cache')
        if cache_root.exists():
            for entry in cache_root.rglob('*'):
                if entry.is_dir() and 'qwen' in entry.name.lower():
                    candidates.append(entry)

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip('Qwen model snapshot not present on this machine')
    # Skip if transformers or tokenizers aren't available/compatible in this environment
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        pytest.skip('transformers package or required classes not available in this environment')

    tokenizer, model, meta = model_adapter.load_with_adapter(str(path), local_files_only=True)
    assert tokenizer is not None and model is not None


@pytest.mark.skipif(os.environ.get('RUN_SMOKE_TESTS') != '1', reason='Smoke tests disabled; set RUN_SMOKE_TESTS=1 to enable')
def test_smoke_load_with_adapter_meta_llama():
    candidates = [
        Path('E:/Models/meta-llama_Llama-3.2-1B'),
        Path('E:/AI/Models/meta-llama_Llama-3.2-1B'),
    ]
    try:
        from Training.domain.path_config import PathConfig
        _cfg = PathConfig.from_env()
        snap = _cfg.find_repo_path('meta-llama/Llama-3.2-1B', repo_type='model')
        if snap is not None:
            candidates.append(snap)
    except Exception:
        cache_root = Path('E:/AI/cache')
        if cache_root.exists():
            for entry in cache_root.rglob('*'):
                if entry.is_dir() and 'meta-llama' in entry.name.lower():
                    candidates.append(entry)

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip('meta-llama snapshot not present on this machine')
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        pytest.skip('transformers package or required classes not available in this environment')

    tokenizer, model, meta = model_adapter.load_with_adapter(str(path), local_files_only=True)
    assert tokenizer is not None and model is not None
