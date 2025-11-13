"""Simple family-level adapter helpers for loading community HF checkpoints.

This module provides a small, safe adapter API to normalize common config/tokenizer
differences across community checkpoints (e.g. rope_scaling variants) and to expose a
single call `load_with_adapter` that returns (tokenizer, model, meta).

Design goals:
- Non-destructive by default: patches are temporary unless commit_patch=True.
- Small and conservative: only normalizes known config issues (rope_scaling) today.
- Family-aware: allows family-specific heuristics (meta-llama, qwen) to be extended.

"""
from __future__ import annotations
from pathlib import Path
import json
import shutil
from typing import Optional, Tuple, Dict, Any


def _detect_gptq(model_path: Path) -> bool:
    """Detect common GPTQ/quantized checkpoint files in the model folder.

    Heuristic: look for files with 'gptq' in the filename or .pt files that are not the standard
    transformers shard names. This is a heuristic and may produce false positives.
    """
    # quick checks
    for p in model_path.rglob('*'):
        name = p.name.lower()
        if 'gptq' in name:
            return True
        if p.suffix.lower() in ('.pt',) and 'pytorch_model' in name and 'gptq' in name:
            return True
    # also detect typical quantization artifacts
    for pat in ('*.pt', '*-gptq-*'):
        for p in model_path.rglob(pat):
            if 'gptq' in p.name.lower():
                return True
    return False


def _gguf_conversion_hint() -> str:
    """Return a short, generic hint explaining conversion options for GGUF/GGML checkpoints.

    We avoid calling external tools; this is just a copyable suggestion for the user.
    """
    return (
        "This model appears to be in GGUF/GGML format. You have two options:\n"
        "  1) Use a GGUF-compatible runtime (e.g. llama.cpp/gguf loaders) for inference.\n"
        "  2) Convert the GGUF/GGML checkpoint to a transformers-compatible checkpoint (safetensors/pytorch). "
        "Common approaches: use the model author's conversion tools or scripts like 'convert_llama_ggml_to_gguf.py' or community converters.\n"
        "Example (conceptual): python convert_gguf_to_transformers.py --input model.gguf --output pytorch_model.safetensors\n"
        "If you need help with a particular model, check the model repo for conversion instructions."
    )


def _patch_config_rope_scaling(model_path: Path) -> Tuple[bool, Optional[Path]]:
    """If config.json contains a non-standard rope_scaling dict, patch it to transformers-accepted shape.

    Returns (patched:bool, backup_path:Path or None).
    """
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        return False, None
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        return False, None

    rs = cfg.get('rope_scaling')
    if not isinstance(rs, dict):
        return False, None

    # If already valid
    if 'type' in rs and 'factor' in rs and isinstance(rs['type'], str):
        return False, None

    # Normalize
    factor = rs.get('factor') or rs.get('high_freq_factor') or rs.get('low_freq_factor') or 1.0
    rope_type = (rs.get('rope_type') or rs.get('type') or 'rope')
    rt = str(rope_type).lower()
    if 'llama' in rt:
        typ = 'dynamic'
    elif rt in ('linear', 'dynamic'):
        typ = rt
    else:
        typ = 'linear'

    cfg['rope_scaling'] = {'type': typ, 'factor': float(factor)}

    # Backup and write
    backup = model_path / 'config.json.bak_adapter'
    try:
        shutil.copy2(cfg_path, backup)
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        return True, backup
    except Exception:
        # On failure, try to restore backup if partially written
        try:
            if backup.exists():
                shutil.copy2(backup, cfg_path)
        except Exception:
            pass
        return False, None


def _restore_backup(backup: Path, cfg_path: Path):
    try:
        if backup and backup.exists():
            shutil.copy2(backup, cfg_path)
            backup.unlink()
    except Exception:
        pass


def _detect_family_from_path(model_path: Path) -> Optional[str]:
    name = model_path.name.lower()
    # Recognize families explicitly. Order matters (dolphin is its own family).
    if 'dolphin' in name or name.startswith('dphn_'):
        return 'dolphin'
    if any(key in name for key in ('meta', 'llama', 'deepseek')):
        return 'meta-llama'
    if 'qwen' in name:
        return 'qwen'
    if 'gemma' in name:
        return 'gemma'
    return 'generic'


def load_with_adapter(model_path: str, family: Optional[str] = None, commit_patch: bool = False,
                      dtype=None, low_cpu_mem_usage: bool = True, local_files_only: bool = True):
    """Load tokenizer and model with family-aware normalization.

    Args:
        model_path: path to local snapshot folder
        family: optional family hint (e.g. 'meta-llama', 'qwen')
        commit_patch: if True, write patches permanently; otherwise restore original files after load
        dtype: torch dtype to pass to from_pretrained (e.g. torch.float16)
        low_cpu_mem_usage: pass-through to transformers.from_pretrained
        local_files_only: avoid network calls

    Returns: (tokenizer, model, meta_dict)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, __version__ as transformers_version
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    import torch

    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model path not found: {mp}")

    fam = family or _detect_family_from_path(mp)

    patched = False
    backup = None
    warnings: list[str] = []
    tokenizer_kwargs: Dict[str, Any] = {'local_files_only': local_files_only}
    model_kwargs: Dict[str, Any] = {
        'local_files_only': local_files_only,
        'low_cpu_mem_usage': low_cpu_mem_usage,
    }
    config_override = None

    # Family-specific normalization
    if fam == 'meta-llama':
        # meta-llama community models often have custom rope_scaling layout
        patched, backup = _patch_config_rope_scaling(mp)
    elif fam == 'dolphin':
        # Dolphin is a Llama-derived family with similar rope_scaling issues but may also
        # include custom tokenizer wrappers. Apply rope_scaling patch and prefer slow tokenizer.
        patched, backup = _patch_config_rope_scaling(mp)
        tokenizer_kwargs['trust_remote_code'] = True
        model_kwargs['trust_remote_code'] = True
        # prefer slow tokenizer by default for Dolphin community checkpoints
        tokenizer_kwargs['use_fast'] = False
    elif fam == 'gemma':
        # Gemma-family models (Gemma/Gemini-like forks) frequently carry transformer config
        # variations and tokenizers that need trust_remote_code or slow-tokenizer fallbacks.
        patched, backup = _patch_config_rope_scaling(mp)
        tokenizer_kwargs['trust_remote_code'] = True
        tokenizer_kwargs['use_fast'] = False
        model_kwargs['trust_remote_code'] = True
    elif fam == 'qwen':
        tokenizer_kwargs['trust_remote_code'] = True
        model_kwargs['trust_remote_code'] = True
        cfg_path = mp / 'config.json'
        cfg_data = None
        if cfg_path.exists():
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg_data = json.load(f)
            except Exception:
                cfg_data = None
        if cfg_data:
            model_type = cfg_data.get('model_type')
            if model_type and model_type not in CONFIG_MAPPING:
                if model_type.startswith('qwen3'):
                    try:
                        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
                        coerced = dict(cfg_data)
                        coerced['model_type'] = 'qwen2'
                        config_override = Qwen2Config(**coerced)
                        model_kwargs['low_cpu_mem_usage'] = False
                        warnings.append('Coerced qwen3 config to Qwen2Config because installed transformers lacks native qwen3 support. Consider upgrading transformers for full compatibility.')
                    except Exception as exc:
                        warnings.append(f"Unable to coerce qwen3 config to Qwen2Config: {exc}. Upgrade transformers >= 4.43 or install transformers-nightly.")
                else:
                    warnings.append(f"Model type '{model_type}' not recognized by transformers {transformers_version}. Consider upgrading transformers or providing a compatible config override.")

    try:
        # Load tokenizer (prefer fixed_tokenizer if present)
        fixed_dir = mp / 'fixed_tokenizer'
        try:
            if fixed_dir.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(fixed_dir), **tokenizer_kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(str(mp), **tokenizer_kwargs)
        except Exception as e:
            msg = str(e)
            # Family-specific tokenizer fallbacks
            if fam == 'qwen' and ('ModelWrapper' in msg or 'did not match any variant' in msg):
                tokenizer_kwargs['use_fast'] = False
                warnings.append('Falling back to use_fast=False for Qwen tokenizer due to ModelWrapper error. Consider upgrading tokenizers >=0.20 or creating a fixed_tokenizer snapshot.')
                if fixed_dir.exists():
                    tokenizer = AutoTokenizer.from_pretrained(str(fixed_dir), **tokenizer_kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(str(mp), **tokenizer_kwargs)
            elif fam == 'gemma' and ('ModelWrapper' in msg or 'did not match any variant' in msg):
                # Gemma tokenizers sometimes require slow tokenizer paths
                tokenizer_kwargs['use_fast'] = False
                tokenizer_kwargs['trust_remote_code'] = True
                warnings.append('Falling back to use_fast=False and trust_remote_code=True for Gemma tokenizer. Consider creating a fixed_tokenizer snapshot.')
                if fixed_dir.exists():
                    tokenizer = AutoTokenizer.from_pretrained(str(fixed_dir), **tokenizer_kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(str(mp), **tokenizer_kwargs)
            elif 'ModelWrapper' in msg or 'did not match any variant' in msg or 'gguf' in msg.lower():
                raise RuntimeError(
                    "Tokenizer load failed and the model folder may contain a non-transformers format (GGUF/GGML/custom). "
                    "Use a GGUF-compatible runtime or convert to a transformers checkpoint. Original error: " + msg
                ) from e
            else:
                raise

        # Determine dtype
        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs['torch_dtype'] = dtype

        # Detect file formats that transformers cannot load (e.g. GGUF/ggml/other binaries)
        gguf_found = any(mp.rglob('*.gguf'))
        ggml_found = any(mp.rglob('*.ggml'))
        if gguf_found or ggml_found:
            hint = _gguf_conversion_hint()
            raise RuntimeError(
                "Model appears to be in GGUF/GGML format which is not loadable with transformers AutoModelForCausalLM. "
                + hint
            )

        # Detect GPTQ/quantized weights and warn the user; adapter will not attempt to dequantize.
        try:
            gptq_found = _detect_gptq(mp)
        except Exception:
            gptq_found = False
        if gptq_found:
            warnings.append('Detected GPTQ/quantized checkpoint files. Transformers cannot always load GPTQ formats directly; consider converting to a supported format or using an appropriate GPTQ loader (e.g., auto-gptq).')

        try:
            if config_override is not None:
                model = AutoModelForCausalLM.from_pretrained(
                    str(mp),
                    config=config_override,
                    **model_kwargs,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    str(mp),
                    **model_kwargs,
                )
        except Exception as e:
            msg = str(e)
            if fam == 'qwen' and 'does not recognize this architecture' in msg:
                raise RuntimeError(
                    f"Transformers {transformers_version} does not recognize the Qwen model architecture in this checkpoint. Upgrade transformers (>=4.43) or provide a compatible config override. Original error: {msg}"
                ) from e
            # Common pattern when model is stored in a non-transformers-friendly format
            if 'ModelWrapper' in msg or 'did not match any variant' in msg or 'invalid' in msg.lower():
                raise RuntimeError(
                    "Model failed to load with transformers. It may be an incompatible or custom-format checkpoint (GPTQ/GGUF/other). "
                    "Inspect the model folder and use a compatible loader or convert the checkpoint to a transformers format. Original error: " + msg
                ) from e
            raise

        meta = {'family': fam, 'patched': patched}
        if warnings:
            meta['warnings'] = warnings
        return tokenizer, model, meta
    finally:
        # restore if we patched and user didn't ask to commit
        if patched and not commit_patch and backup is not None:
            try:
                cfg_path = mp / 'config.json'
                _restore_backup(backup, cfg_path)
            except Exception:
                pass
