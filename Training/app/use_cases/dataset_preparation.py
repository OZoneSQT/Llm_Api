from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from Training.tools.hf_datasets import load_dataset

from Training.domain.entities import DatasetPreparationRequest, DatasetSpec


def _examples_from_iterable(ds: Iterable) -> List[dict]:
    examples = []
    for item in ds:
        if isinstance(item, dict):
            for key in ('text', 'content', 'instruction', 'prompt'):
                value = item.get(key)
                if value:
                    examples.append({'text': str(value)})
                    break
            else:
                values = [str(v) for v in item.values() if isinstance(v, str) and v.strip()]
                if values:
                    examples.append({'text': ' '.join(values)})
        else:
            examples.append({'text': str(item)})
    return examples




def _load_dataset_slice(spec: DatasetSpec) -> List[dict]:
    try:
        sliced = load_dataset(spec.repo_id, split=f"{spec.split}[:{spec.max_examples}]")
        return _examples_from_iterable(sliced)
    except Exception:
        ds = load_dataset(spec.repo_id)
        if spec.split in ds:
            subset = ds[spec.split]
        else:
            subset = ds[list(ds.keys())[0]]
        subset = subset.select(range(min(len(subset), spec.max_examples)))
        return _examples_from_iterable(subset)


def _load_local_path(path: Path, max_examples: int) -> List[dict]:
    if not path.exists():
        return []
    suffix = path.suffix.lower()
    examples: List[dict] = []
    if suffix in ('.jsonl', '.json'):
        with path.open('r', encoding='utf-8') as handle:
            for index, line in enumerate(handle):
                if index >= max_examples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    examples.append({'text': line})
                    continue
                if isinstance(payload, dict):
                    for key in ('text', 'content', 'instruction', 'prompt'):
                        value = payload.get(key)
                        if value:
                            examples.append({'text': str(value)})
                            break
                    else:
                        for value in payload.values():
                            if isinstance(value, str) and value.strip():
                                examples.append({'text': value})
                                break
                else:
                    examples.append({'text': str(payload)})
        return examples

    with path.open('r', encoding='utf-8') as handle:
        for index, line in enumerate(handle):
            if index >= max_examples:
                break
            line = line.strip()
            if line:
                examples.append({'text': line})
    return examples


def prepare_dataset(request: DatasetPreparationRequest, max_examples: int) -> Path:
    aggregated: List[dict] = []
    for spec in request.dataset_specs:
        aggregated.extend(_load_dataset_slice(spec))
    if request.sanitized_path:
        aggregated.extend(_load_local_path(request.sanitized_path, max_examples))
    if request.custom_path:
        aggregated.extend(_load_local_path(request.custom_path, max_examples))

    if request.shuffle:
        import random

        random.shuffle(aggregated)

    max_total = max_examples * max(1, len(request.dataset_specs) + int(bool(request.sanitized_path)) + int(bool(request.custom_path)))
    trimmed = aggregated[:max_total]
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    with request.output_path.open('w', encoding='utf-8') as handle:
        for example in trimmed:
            handle.write(json.dumps(example, ensure_ascii=False) + '\n')
    return request.output_path
