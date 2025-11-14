import importlib
import sys
from typing import Iterable, Optional
import types

import pytest


def make_fake_torch(cuda_available: bool, vram_mbs: Iterable[Optional[int]]):
    fake = types.SimpleNamespace()
    fake.cuda = types.SimpleNamespace()
    fake.cuda.is_available = lambda: cuda_available

    vram_list = list(vram_mbs)

    def device_count() -> int:
        return len(vram_list)

    fake.cuda.device_count = device_count

    def get_device_properties(idx: int):
        class Props:
            total_memory = vram_list[idx] * 1024 * 1024

        return Props()

    fake.cuda.get_device_properties = get_device_properties

    # minimal backend marker
    fake.backends = types.SimpleNamespace()
    fake.backends.mps = types.SimpleNamespace()
    fake.backends.mps.is_available = lambda: False
    return fake


def make_fake_psutil(available_mb: int):
    fake = types.SimpleNamespace()

    def virtual_memory():
        return types.SimpleNamespace(available=available_mb * 1024 * 1024)

    fake.virtual_memory = virtual_memory
    return fake


@pytest.mark.parametrize(
    "cuda_available,vram_list,avail_ram_mb,expected,model",
    [
        (True, [20000], 32000, 'cuda:0', 'dx8152/Qwen-Edit-2509-Multiple-angles'),  # GPU has enough vram
        (True, [8000], 32000, 'cpu', 'dx8152/Qwen-Edit-2509-Multiple-angles'),    # headroom prevents 8GB GPU
        (True, [8000], 32000, 'cuda:0', 'microsoft/VibeVoice-1.5B'),             # smaller req satisfied
        (True, [4000], 32000, 'cpu', 'dx8152/Qwen-Edit-2509-Multiple-angles'),  # requirement higher than vram -> cpu
        (False, [], 32000, 'cpu', 'dx8152/Qwen-Edit-2509-Multiple-angles'),      # no GPU -> cpu
    ],
)
def test_resolve_device_basic(monkeypatch, cuda_available, vram_list, avail_ram_mb, expected, model):
    # Ensure device_utils is reloaded with our fake modules
    fake_torch = make_fake_torch(cuda_available, vram_list)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    fake_psutil = make_fake_psutil(avail_ram_mb)
    monkeypatch.setitem(sys.modules, 'psutil', fake_psutil)

    # import device_utils after patching
    device_utils = importlib.reload(importlib.import_module('Media.device_utils'))

    # pick a known model that maps to 12GB default requirement
    dev = device_utils.resolve_device(model)
    dev = device_utils.resolve_device(model)
    assert dev == expected


def test_insufficient_system_ram_raises(monkeypatch):
    fake_torch = make_fake_torch(False, [])
    sys.modules['torch'] = fake_torch
    fake_psutil = make_fake_psutil(1000)  # 1GB available
    sys.modules['psutil'] = fake_psutil
    device_utils = importlib.reload(importlib.import_module('Media.device_utils'))
    with pytest.raises(MemoryError):
        device_utils.resolve_device('Phr00t/WAN2.2-14B-Rapid-AllInOne')


def test_prefer_cpu(monkeypatch):
    fake_torch = make_fake_torch(True, [50000])
    sys.modules['torch'] = fake_torch
    fake_psutil = make_fake_psutil(64000)
    sys.modules['psutil'] = fake_psutil
    device_utils = importlib.reload(importlib.import_module('Media.device_utils'))
    dev = device_utils.resolve_device('Wan-AI/Wan2.2-TI2V-5B', preferred='cpu')
    assert dev == 'cpu'


def test_best_fit_selects_largest_gpu(monkeypatch):
    fake_torch = make_fake_torch(True, [10000, 16000])
    sys.modules['torch'] = fake_torch
    fake_psutil = make_fake_psutil(64000)
    sys.modules['psutil'] = fake_psutil
    device_utils = importlib.reload(importlib.import_module('Media.device_utils'))
    # Default requirement 12GB -> GPU 0 fails headroom, GPU1 meets
    dev = device_utils.resolve_device('unknown/model:default')
    assert dev == 'cuda:1'
