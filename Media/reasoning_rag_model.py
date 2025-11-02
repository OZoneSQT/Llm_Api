from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import RagRetriever, RagSequenceForGeneration, RagTokenizer


@dataclass
class RagConfig:
    model_name: str
    passages_path: Path
    index_path: Path
    index_name: str = "custom"


def build_retriever(config: RagConfig) -> RagRetriever:
    """Initialise a RAG retriever from local passages and FAISS index."""
    return RagRetriever.from_pretrained(
        config.model_name,
        index_name=config.index_name,
        passages_path=str(config.passages_path),
        index_path=str(config.index_path),
    )


def load_rag_model(config: RagConfig) -> RagSequenceForGeneration:
    retriever = build_retriever(config)
    return RagSequenceForGeneration.from_pretrained(config.model_name, retriever=retriever)


def ask_question(model: RagSequenceForGeneration, tokenizer: RagTokenizer, question: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(input_ids=inputs["input_ids"], max_new_tokens=max_new_tokens)
    answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return answers[0] if answers else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a Retrieval-Augmented Generation model.")
    parser.add_argument("question", help="Question to ask the model.")
    parser.add_argument("--model", default="facebook/rag-sequence-nq", help="RAG model repo id.")
    parser.add_argument("--passages", type=Path, required=True, help="Path to passages JSON/CSV file used by the retriever.")
    parser.add_argument("--index", type=Path, required=True, help="Path to the FAISS index.")
    parser.add_argument("--index-name", default="custom", help="Retriever index name.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum tokens to generate in the answer.")
    parser.add_argument("--dataset", type=Path, help="Optional CSV dataset used for further fine-tuning.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RagConfig(
        model_name=args.model,
        passages_path=args.passages,
        index_path=args.index,
        index_name=args.index_name,
    )
    tokenizer = RagTokenizer.from_pretrained(config.model_name)
    model = load_rag_model(config)
    answer = ask_question(model=model, tokenizer=tokenizer, question=args.question, max_new_tokens=args.max_new_tokens)
    print(f"Answer: {answer}")

    if args.dataset:
        dataset = load_dataset("csv", data_files={"train": str(args.dataset)})
        print(f"Loaded dataset with {len(dataset['train'])} records for potential fine-tuning.")


if __name__ == "__main__":
    main()