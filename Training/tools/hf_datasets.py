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
load_from_disk: LoadFromDisk = _datasets.load_from_disk

ConcatenateDatasets = Callable[[Iterable[Dataset]], Dataset]
concatenate_datasets: ConcatenateDatasets = _datasets.concatenate_datasets
