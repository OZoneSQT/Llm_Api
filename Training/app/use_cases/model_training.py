from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch  # type: ignore
from Training.tools.hf_imports import load_datasets_module
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from Training.domain.entities import ModelTrainingRequest, ModelTrainingResult
from Training.tools import paths as path_utils
from Training.tools.adapter_utils import load_model as adapter_load_model, load_tokenizer as adapter_load_tokenizer


_hf_datasets = load_datasets_module()
Dataset = _hf_datasets.Dataset
concatenate_datasets = _hf_datasets.concatenate_datasets
load_dataset = _hf_datasets.load_dataset
load_from_disk = _hf_datasets.load_from_disk


@dataclass
class _LoadedModel:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM


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


def _load_remote_datasets(dataset_names: Iterable[str]) -> list[Dataset]:
    datasets = []
    for name in dataset_names:
        ds = load_dataset(name)
        if 'train' in ds:
            datasets.append(ds['train'])
        else:
            datasets.append(next(iter(ds.values())))
    return datasets


def _concat_datasets(remote_datasets: list[Dataset], custom_dataset: Dataset | None, generated_dataset: Dataset | None) -> Dataset | None:
    combined = None
    if remote_datasets:
        combined = concatenate_datasets(remote_datasets)
    if generated_dataset is not None:
        combined = concatenate_datasets([combined, generated_dataset]) if combined is not None else generated_dataset
    if custom_dataset is not None:
        combined = concatenate_datasets([combined, custom_dataset]) if combined is not None else custom_dataset
    return combined


def _load_banned_words(path: Path) -> list[str]:
    with path.open('r', encoding='utf-8') as handle:
        return [line.strip().lower() for line in handle if line.strip() and not line.startswith('#')]


def _filter_banned_words(dataset: Dataset, banned_words: list[str]) -> Dataset:
    if not banned_words:
        return dataset

    def _is_safe(example: dict) -> bool:
        text = example.get('text', '').lower()
        return not any(word in text for word in banned_words)

    return dataset.filter(_is_safe)


def _apply_instruction(dataset: Dataset, instruction: str) -> Dataset:
    if not instruction:
        return dataset

    def _add_instruction(example: dict) -> dict:
        example['text'] = instruction + example.get('text', '')
        return example

    return dataset.map(_add_instruction)


def _load_model(config: dict) -> _LoadedModel:
    model_name = config.get('llm_model_name', 'gpt2')
    tokenizer_name = (
        config.get('llm_tokenizer_name')
        or config.get('tokenizer_model')
        or config.get('tokenizer_model_name')
        or model_name
    )
    use_adapter = bool(config.get('use_adapter') or os.environ.get('USE_ADAPTER', '').lower() in ('1', 'true', 'yes'))

    local_model_path = config.get('local_model_path') or config.get('local_model')
    if local_model_path:
        local_model_path = Path(local_model_path).expanduser()
        if not local_model_path.is_absolute():
            local_model_path = (path_utils.MODEL_ROOT / local_model_path).resolve()
        if not local_model_path.exists():
            raise FileNotFoundError(f'Local model path not found: {local_model_path}')
        if use_adapter:
            try:
                from Training.tools.model_adapter import load_with_adapter

                tokenizer, model, _ = load_with_adapter(str(local_model_path), local_files_only=True)
                return _LoadedModel(tokenizer=tokenizer, model=model)
            except Exception as exc:  # pragma: no cover - adapter fallback
                print('Adapter load failed, falling back to transformers.from_pretrained. Adapter error:', exc)
        model = adapter_load_model(
            str(local_model_path),
            loader=AutoModelForCausalLM.from_pretrained,
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer = adapter_load_tokenizer(
            str(local_model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        return _LoadedModel(tokenizer=tokenizer, model=model)

    if use_adapter:
        try:
            from Training.tools.model_adapter import load_with_adapter

            tokenizer, model, _ = load_with_adapter(model_name, local_files_only=False)
            return _LoadedModel(tokenizer=tokenizer, model=model)
        except Exception as exc:  # pragma: no cover - adapter fallback
            print('Adapter load failed for remote name, falling back to transformers.from_pretrained. Adapter error:', exc)

    model = adapter_load_model(
        model_name,
        loader=AutoModelForCausalLM.from_pretrained,
        local_files_only=False,
        trust_remote_code=True,
    )
    tokenizer = adapter_load_tokenizer(
        tokenizer_name,
        local_files_only=False,
        trust_remote_code=True,
    )
    return _LoadedModel(tokenizer=tokenizer, model=model)


def _select_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    if getattr(torch, 'has_mps', False):
        return 'mps'
    if getattr(torch.backends, 'opencl', None) and torch.backends.opencl.is_available():
        return 'opencl'
    return 'cpu'


def _tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize(examples: dict) -> dict:
        tokens = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens

    return dataset.map(_tokenize, batched=True)


def train_model(request: ModelTrainingRequest) -> ModelTrainingResult:
    config = _read_config(request.config_path)
    dataset_names = _load_dataset_names(request.dataset_list_path)
    generated_dataset = None  # Placeholder for future integration of generated datasets
    custom_dataset = _load_custom_dataset(request.prefer_sanitized)
    remote_datasets = _load_remote_datasets(dataset_names) if dataset_names else []
    combined_dataset = _concat_datasets(remote_datasets, custom_dataset, generated_dataset)

    if combined_dataset is None or len(combined_dataset) == 0:
        raise ValueError('The combined dataset is empty. Please check the input datasets.')

    repo_root = request.config_path.resolve().parent.parent.parent
    banned_words_path = repo_root / 'Training' / 'data' / 'banned_words.txt'
    banned_words = _load_banned_words(banned_words_path)
    combined_dataset = _filter_banned_words(combined_dataset, banned_words)

    instruction = config.get('instruction', config.get('instruction_prefix', ''))
    add_instruction = config.get('add_instruction', bool(instruction))
    if add_instruction and instruction:
        combined_dataset = _apply_instruction(combined_dataset, instruction)

    loaded = _load_model(config)
    device = _select_device()
    loaded.model.to(device)

    instruction_length = len(loaded.tokenizer(instruction)['input_ids']) if instruction else 0
    max_total_length = instruction_length + config.get('max_sequence_length', 256)
    tokenized_dataset = _tokenize_dataset(combined_dataset, loaded.tokenizer, max_total_length)

    training_args = TrainingArguments(
        output_dir=str(path_utils.MODEL_ROOT / config.get('model_ID', 'my-model')),
        num_train_epochs=config.get('num_train_epochs', 2),
        per_device_train_batch_size=config.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', 4),
        save_strategy=config.get('save_strategy', 'epoch'),
        logging_steps=config.get('logging_steps', 10),
        fp16=config.get('fp16', torch.cuda.is_available()),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        warmup_steps=config.get('warmup_steps', 0),
        weight_decay=config.get('weight_decay', 0.0),
        lr_scheduler_type=config.get('lr_scheduler_type', 'linear'),
        eval_steps=config.get('eval_steps', None),
        logging_strategy=config.get('logging_strategy', 'epoch'),
        seed=config.get('seed', 42),
        learning_rate=config.get('learning_rate', 2e-5),
    )

    split = tokenized_dataset.train_test_split(test_size=0.1)
    data_collator = DataCollatorForLanguageModeling(tokenizer=loaded.tokenizer, mlm=False)

    trainer = Trainer(
        model=loaded.model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        data_collator=data_collator,
    )

    model_output_dir = Path(training_args.output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = model_output_dir / 'buildlog.log'
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    logfile_path.write_text(f'Start timestamp: {start_time}\n', encoding='utf-8')
    with logfile_path.open('a', encoding='utf-8') as handle:
        handle.write(f'Combined dataset size: {len(combined_dataset)}\n')

    trainer.train()

    loaded.model.save_pretrained(str(model_output_dir))
    loaded.tokenizer.save_pretrained(str(model_output_dir))

    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with logfile_path.open('a', encoding='utf-8') as handle:
        handle.write(f'End timestamp: {end_time}\n')

    return ModelTrainingResult(
        model_output_dir=model_output_dir,
        log_path=logfile_path,
        dataset_size=len(combined_dataset),
    )
