from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from datasets import concatenate_datasets

import subprocess
import json

# Read config from config.json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


######################
### Setup datasets ###
######################

# Load RAG dataset
useRagDate = config.get('useRagDate', True)

if useRagDate:
    subprocess.run(["python", "./Training/build_dk.py"], check=True)
    rag_dataset = load_dataset("json", data_files={"train": "./Training/data/docs.json"})

# Login using e.g. `huggingface-cli login` to access this dataset

# Load each dataset from "dataset.txt", one per line
with open("./Training/data/dataset.txt", "r", encoding="utf-8") as f:
    dataset_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Concatenate them
datasets = [load_dataset(name) for name in dataset_names]
combined_dataset = concatenate_datasets(datasets)

# Add RAG dataset if needed
if useRagDate:
    combined_dataset = concatenate_datasets([combined_dataset, rag_dataset["train"]])

# Guard against empty datasets
if len(combined_dataset) == 0:
    raise ValueError("The combined dataset is empty. Please check the input datasets.")


###################
### Filter data ###
###################

# Filter out data using a wordlist from a file
def load_banned_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip() and not line.startswith('#')]

banned_words = load_banned_words('./Training/data/banned_words.txt')
def is_safe(example):
    text = example.get("text", "").lower()
    return not any(bad_word in text for bad_word in banned_words)

if banned_words:
    combined_dataset = combined_dataset.filter(is_safe)


################################
### Set instruction datasets ###
################################

# Add instruction prefix to each sample
instruction = config.get("instruction_prefix", "Instruction: Answer as a drunk sailor.\n")
def add_instruction(example):
    example["text"] = instruction + example.get("text", "")
    return example

if instruction:
    combined_dataset = combined_dataset.map(add_instruction)


#################################
### Setup model and tokenizer ###
#################################

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(config.get("llm_model_name", "gpt2"))
tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_model", "gpt2"))

# Define the tokenize function before using it
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Tokenize the combined dataset
tokenized_combined = combined_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="my-model",
    num_train_epochs=config.get("num_train_epochs", 1),
    per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
    per_device_eval_batch_size=config.get("per_device_eval_batch_size", 2),
    evaluation_strategy=config.get("evaluation_strategy", "epoch"),
    save_strategy=config.get("save_strategy", "epoch"),
    logging_steps=config.get("logging_steps", 10),
    fp16_full_eval=True
)

# Split tokenized_combined into train and eval sets
split = tokenized_combined.train_test_split(test_size=0.1)
train_ds = split['train']
eval_ds = split['test']

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)


###############
### Execute ###
###############

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("my-model")
tokenizer.save_pretrained("my-model")