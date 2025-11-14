from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - tests will mock torch where needed
    torch = None
import subprocess
import re

from Training.domain.path_config import PathConfig


MODEL_REQUIREMENTS = {
    # mapping of model name prefix -> (vram_mb_required, ram_mb_required)
    'Wan-AI/Wan2.2-TI2V-5B': (16000, 16000),
    'enhanceaiteam/Flux-Uncensored-V2': (8000, 8000),
    'second-state/FLUX.1-dev-GGUF': (24000, 32000),
    'microsoft/VibeVoice-1.5B': (6000, 8000),
    'Phr00t/WAN2.2-14B-Rapid-AllInOne': (45000, 64000),
    'nvidia/ChronoEdit-14B-Diffusers-Upscaler-Lora': (45000, 64000),
    'dx8152/Qwen-Edit-2509-Multiple-angles': (8000, 16000),
    'dx8152/Qwen-Image-Edit-2509-Light_restoration': (8000, 16000),
    'Wan-AI/Wan2.2-Animate-14B': (45000, 64000),
}
HEADROOM_FRACTION = 0.8


def get_model_requirements(model_name: str) -> Tuple[int, int]:
    # Return (vram_mb, ram_mb). If unknown, return conservative defaults.
    for prefix, req in MODEL_REQUIREMENTS.items():
        if model_name.startswith(prefix):
            return req
    # conservative default for unknown models
    return (12000, 16000)


def _get_available_ram_mb() -> Optional[int]:
    if psutil is None:
        return None
    try:
        vm = psutil.virtual_memory()
        return int(vm.available / 1024 ** 2)
    except Exception:
        return None


def _get_gpu_vram_mb(device_index: int = 0) -> Optional[int]:
    """Return VRAM (MB) for a single GPU index or None if unavailable.

    Tries torch first, then falls back to parsing `nvidia-smi` output.
    """
    # prefer torch when available
    if torch is not None:
        try:
            if not getattr(torch.cuda, 'is_available', lambda: False)():
                return None
            # some torch mocks may not provide device_count; assume 1 if get_device_properties exists
            props = torch.cuda.get_device_properties(device_index)
            return int(props.total_memory / (1024 ** 2))
        except Exception:
            pass

    # fallback: try nvidia-smi (NVIDIA drivers)
    nvidia_output = _run_command(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
    if nvidia_output is not None:
        lines = nvidia_output.strip().splitlines()
        if 0 <= device_index < len(lines):
            return int(lines[device_index].strip())

    rocm_output = _run_command(['rocm-smi', '--showmeminfo', 'vram', '--csv'])
    if rocm_output is not None:
        values = _parse_rocm_vram(rocm_output)
        if values and 0 <= device_index < len(values):
            return values[device_index]

    return None


def _get_all_gpu_vram_mb() -> Optional[List[Optional[int]]]:
    """Return list of VRAM sizes (MB) for all detected GPUs, or None if unknown."""
    if torch is not None:
        try:
            if not getattr(torch.cuda, 'is_available', lambda: False)():
                return []
            # prefer device_count when available
            count = getattr(torch.cuda, 'device_count', lambda: 1)()
            values: List[Optional[int]] = []
            for i in range(count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    values.append(int(props.total_memory / (1024 ** 2)))
                except Exception:
                    values.append(None)
            return values
        except Exception:
            pass

    # fallback to nvidia-smi parsing
    nvidia_output = _run_command(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
    if nvidia_output is not None:
        return [int(l.strip()) for l in nvidia_output.strip().splitlines() if l.strip()]

    rocm_output = _run_command(['rocm-smi', '--showmeminfo', 'vram', '--csv'])
    values = _parse_rocm_vram(rocm_output) if rocm_output is not None else None
    return values


def _run_command(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _parse_rocm_vram(output: str) -> Optional[List[int]]:
    values: List[int] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        candidates = [int(num) for num in re.findall(r"(\d+)", line) if int(num) > 1000]
        if candidates:
            values.append(candidates[-1])
    return values if values else None


def resolve_device(model_name: str, preferred: Optional[str] = None) -> str:
    """Decide device string to use for model loading/runtime.

    - If `preferred` is provided (e.g., 'cuda:0' or 'cpu') it will be honored
      when resources suffice.
    - If GPU is available and has sufficient VRAM for the named model, return
      a CUDA device string (e.g., 'cuda:0'). Otherwise return 'cpu'.
    - If system RAM is insufficient for the model, raise MemoryError.
    """
    vram_required_mb, ram_required_mb = get_model_requirements(model_name)

    # check system RAM first
    avail_ram = _get_available_ram_mb()
    if avail_ram is not None and avail_ram < ram_required_mb:
        raise MemoryError(f"Insufficient system RAM: {avail_ram} MB available, {ram_required_mb} MB required for {model_name}")

    # Honor explicit CPU preference
    if preferred and preferred.lower().startswith('cpu'):
        return 'cpu'

    # Check GPU availability and VRAM across all GPUs
    gpu_vrams = _get_all_gpu_vram_mb()
    if gpu_vrams is not None:
        # honor explicit cuda index if provided
        if preferred and preferred.lower().startswith('cuda'):
            try:
                idx = int(preferred.split(':', 1)[1])
            except Exception:
                idx = 0
            if idx < len(gpu_vrams) and gpu_vrams[idx] is not None:
                total_memory = gpu_vrams[idx]
                if total_memory is not None and _meets_headroom(total_memory, vram_required_mb):
                    return f'cuda:{idx}'
            return 'cpu'

        # best-fit selection: pick GPU with most VRAM that satisfies requirement and headroom
        candidates: List[Tuple[int, int]] = []
        for i, total_memory in enumerate(gpu_vrams):
            if total_memory is None:
                continue
            if _meets_headroom(total_memory, vram_required_mb):
                candidates.append((total_memory, i))

        if candidates:
            _, best_idx = max(candidates, key=lambda pair: pair[0])
            return f'cuda:{best_idx}'
    # Fallback to CPU
    return 'cpu'


def get_cache_dir() -> str:
    # Prefer PathConfig for consistent repo-wide cache location
    try:
        return str(PathConfig.from_env().cache_dir)
    except Exception:
        # fallback to environment variable or sensible default
        return os.environ.get('HF_CACHE_DIR', r'E:\\AI\\cache')


def _meets_headroom(total_memory_mb: int, required_mb: int) -> bool:
    return total_memory_mb * HEADROOM_FRACTION >= required_mb
