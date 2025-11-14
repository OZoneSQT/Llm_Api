from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, TypeAlias

from Training.tools.hf_imports import load_datasets_module

if TYPE_CHECKING:
    from datasets import Dataset as HFHDataset
else:
    HFHDataset = Any

Dataset: TypeAlias = HFHDataset

_datasets = load_datasets_module()

LoadDataset = Callable[..., Any]
load_dataset: LoadDataset = _datasets.load_dataset

LoadFromDisk = Callable[..., Any]

def _load_from_disk_unavailable(*_args: Any, **_kwargs: Any) -> Any:
    raise AttributeError("The installed 'datasets' package does not provide load_from_disk; upgrade datasets>=2.14")

load_from_disk: LoadFromDisk = getattr(_datasets, "load_from_disk", _load_from_disk_unavailable)

ConcatenateDatasets = Callable[[Iterable[Dataset]], Dataset]

def _concatenate_unavailable(*_args: Any, **_kwargs: Any) -> Any:
    raise AttributeError("The installed 'datasets' package lacks concatenate_datasets; upgrade datasets>=2.14")

concatenate_datasets: ConcatenateDatasets = getattr(_datasets, "concatenate_datasets", _concatenate_unavailable)
