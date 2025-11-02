from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


@dataclass
class ReasoningConfig:
    model_name: str
    text_column: str
    label_column: str
    max_length: int
    output_dir: Path
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    num_train_epochs: float
    weight_decay: float


def load_data(train_file: Path, test_file: Path, text_column: str, label_column: str) -> DatasetDict:
    """Load dataset from CSV files and validate required columns."""
    data_files: Dict[str, str] = {"train": str(train_file), "test": str(test_file)}
    dataset = load_dataset("csv", data_files=data_files)
    missing_columns = {
        split: [column for column in (text_column, label_column) if column not in dataset[split].column_names]
        for split in dataset
    }
    missing = {split: cols for split, cols in missing_columns.items() if cols}
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return dataset


def encode_labels(dataset: DatasetDict, label_column: str):
    """Ensure labels are encoded as integers suitable for classification."""
    labels = dataset["train"][label_column]
    if not labels:
        raise ValueError(f"No labels found in column '{label_column}'.")
    if labels and isinstance(labels[0], str):
        unique_labels = sorted(set(labels))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}

        def map_labels(example):
            example[label_column] = label2id[example[label_column]]
            return example

        dataset = dataset.map(map_labels)
        num_labels = len(unique_labels)
    else:
        if labels and not isinstance(labels[0], int):
            def cast_label(example):
                example[label_column] = int(example[label_column])
                return example

            try:
                dataset = dataset.map(cast_label)
            except (TypeError, ValueError):
                pass
        num_labels = len(set(labels))
    return dataset, num_labels


def tokenize_dataset(dataset: DatasetDict, tokenizer, text_column: str, label_column: str, max_length: int):
    """Tokenize dataset with truncation and padding."""

    def preprocess_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(preprocess_function, batched=True)
    tokenized = tokenized.rename_column(text_column, "text")
    tokenized = tokenized.rename_column(label_column, "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def train_reasoning_model(config: ReasoningConfig, train_file: Path, test_file: Path) -> None:
    dataset = load_data(train_file=train_file, test_file=test_file, text_column=config.text_column, label_column=config.label_column)
    dataset, num_labels = encode_labels(dataset, config.label_column)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized_datasets = tokenize_dataset(dataset, tokenizer, config.text_column, config.label_column, config.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        evaluation_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
    num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a sequence classification model for reasoning tasks.")
    parser.add_argument("train_file", type=Path, help="Path to the training CSV file.")
    parser.add_argument("test_file", type=Path, help="Path to the test CSV file.")
    parser.add_argument("--text-column", default="text", help="Column containing the text input.")
    parser.add_argument("--label-column", default="label", help="Column containing the target label.")
    parser.add_argument("--model", default="distilbert-base-uncased", help="Base model to fine-tune.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--output-dir", type=Path, default=Path("reasoning_model"), help="Directory to store the trained model.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ReasoningConfig(
        model_name=args.model,
        text_column=args.text_column,
        label_column=args.label_column,
        max_length=args.max_length,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
    )
    train_reasoning_model(config=config, train_file=args.train_file, test_file=args.test_file)


if __name__ == "__main__":
    main()