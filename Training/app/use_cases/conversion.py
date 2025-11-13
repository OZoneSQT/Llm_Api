from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from Training.domain.entities import ConversionRequest, ModelFormatDetection


def detect_formats(root: Path) -> ModelFormatDetection:
    if not root.exists():
        raise FileNotFoundError(root)
    formats: Dict[str, bool] = {'gguf': False, 'ggml': False, 'gptq': False, 'safetensors': False}
    for item in root.rglob('*'):
        name = item.name.lower()
        if name.endswith('.gguf'):
            formats['gguf'] = True
        if name.endswith('.ggml'):
            formats['ggml'] = True
        if 'gptq' in name or (name.endswith('.pt') and 'gptq' in name):
            formats['gptq'] = True
        if name.endswith('.safetensors'):
            formats['safetensors'] = True
    return ModelFormatDetection(root=root, formats=formats)


def _find_transformers_llama_converter() -> Optional[Path]:
    candidate = Path('lib') / 'transformers' / 'src' / 'transformers' / 'models' / 'llama' / 'convert_llama_weights_to_hf.py'
    if candidate.exists():
        return candidate
    try:
        import inspect
        import transformers  # type: ignore

        modfile = inspect.getsourcefile(transformers)
        if modfile:
            base = Path(modfile).resolve().parents[1]
            pkg_candidate = base / 'models' / 'llama' / 'convert_llama_weights_to_hf.py'
            if pkg_candidate.exists():
                return pkg_candidate
    except Exception:
        return None
    return None


def run_llama_conversion(request: ConversionRequest) -> bool:
    script = _find_transformers_llama_converter()
    if not script:
        return False
    cmd = [
        sys.executable,
        str(script),
        '--input_dir',
        str(request.input_dir),
        '--output_dir',
        str(request.output_dir),
    ]
    if request.model_size:
        cmd += ['--model_size', request.model_size]
    if request.llama_version:
        cmd += ['--llama_version', request.llama_version]
    if request.dry_run or not request.confirm:
        # Skip execution when dry-run or confirmation missing
        return request.dry_run or False
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def is_auto_gptq_available() -> bool:
    try:
        import auto_gptq  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def run_auto_gptq_conversion(request: ConversionRequest) -> bool:
    if not is_auto_gptq_available():
        return False
    cmd = [
        sys.executable,
        '-m',
        'auto_gptq.convert',
        '--model_name_or_path',
        str(request.input_dir),
        '--output_dir',
        str(request.output_dir),
        '--device',
        request.device,
    ]
    if request.dry_run or not request.confirm:
        return request.dry_run or False
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0
