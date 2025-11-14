from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch  # type: ignore
from Training.tools.hf_datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import (  # type: ignore
    AutoModelForSequenceClassification as _AutoModelForSequenceClassification,
    AutoTokenizer as _AutoTokenizer,
    Trainer as _Trainer,
    TrainingArguments as _TrainingArguments,
)

from Training.domain.entities import ModelTuningRequest, ModelTuningResult
from Training.tools import paths as path_utils
from Training.tools.adapter_utils import load_model as adapter_load_model, load_tokenizer as adapter_load_tokenizer

AutoModelForSequenceClassification: Any = _AutoModelForSequenceClassification
AutoTokenizer: Any = _AutoTokenizer
Trainer: Any = _Trainer
TrainingArguments: Any = _TrainingArguments


def _read_config(config_path: Path) -> dict:
    with config_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _load_dataset_names(dataset_list_path: Path) -> list[str]:
    with dataset_list_path.open('r', encoding='utf-8') as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith('#')]


def _load_custom_dataset(prefer_sanitized: bool) -> Dataset | None:
    sanitized = path_utils.latest_dataset('sanitized') if prefer_sanitized else None
    if sanitized:
        return load_from_disk(str(sanitized))
    custom = path_utils.latest_dataset('custom')
    if custom:
        return load_from_disk(str(custom))
    return None


def _load_remote_datasets(dataset_names: list[str]) -> list[Dataset]:
    datasets = []
    for name in dataset_names:
        ds = load_dataset(name)
        if 'train' in ds:
            datasets.append(ds['train'])
        else:
            datasets.append(next(iter(ds.values())))
    return datasets


def _compose_dataset(remote_datasets: list[Dataset], local_dataset: Dataset | None) -> Dataset:
    combined = None
    if remote_datasets:
        combined = concatenate_datasets(remote_datasets)
    if local_dataset is not None:
        combined = concatenate_datasets([combined, local_dataset]) if combined is not None else local_dataset
    if combined is None:
        raise ValueError('The combined dataset is empty. Please check the input datasets.')
    return combined


def _load_model(config: dict) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    model_name = config.get('sequence_model_name', 'distilbert-base-uncased')
    use_adapter = os.environ.get('USE_ADAPTER', '').lower() in ('1', 'true', 'yes')
    tokenizer = None
    model = None

    if use_adapter:
        try:
            from Training.tools.model_adapter import load_with_adapter

            tokenizer, model, _ = load_with_adapter(model_name, local_files_only=False)
        except Exception as exc:  # pragma: no cover - adapter fallback
            print('Adapter load failed, falling back to transformers.from_pretrained:', exc)

    if tokenizer is None or model is None:
        tokenizer = adapter_load_tokenizer(model_name, local_files_only=False)
        model = adapter_load_model(
            model_name,
            loader=AutoModelForSequenceClassification.from_pretrained,
            local_files_only=False,
            num_labels=config.get('sequence_num_labels', 2),
        )
    return tokenizer, model


def tune_model(request: ModelTuningRequest) -> ModelTuningResult:
    config = _read_config(request.config_path)
    dataset_names = _load_dataset_names(request.dataset_list_path)
    local_dataset = _load_custom_dataset(request.prefer_sanitized)
    remote_datasets = _load_remote_datasets(dataset_names) if dataset_names else []
    combined_dataset = _compose_dataset(remote_datasets, local_dataset)

    tokenizer, model = _load_model(config)

    def _tokenize_batch(examples: dict) -> dict:
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=config.get('sequence_max_length', 128),
        )

    split = combined_dataset.train_test_split(test_size=0.1, seed=config.get('seed', 42))
    tokenized_train = split['train'].map(_tokenize_batch, batched=True)
    tokenized_eval = split['test'].map(_tokenize_batch, batched=True)

    output_dir = path_utils.MODEL_ROOT / config.get('tuned_model_ID', 'tuned-model')
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy=config.get('tuning_evaluation_strategy', 'epoch'),
        learning_rate=config.get('tuning_learning_rate', 2e-5),
        per_device_train_batch_size=config.get('tuning_train_batch_size', 8),
        per_device_eval_batch_size=config.get('tuning_eval_batch_size', 8),
        num_train_epochs=config.get('tuning_num_epochs', 2),
        weight_decay=config.get('tuning_weight_decay', 0.01),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    metrics_path = output_dir / 'tuning_metrics.json'
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    trainer.train()
    metrics = trainer.evaluate()
    metrics['start'] = start_time
    metrics['end'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return ModelTuningResult(
        model_output_dir=output_dir,
        metrics_path=metrics_path,
    )
