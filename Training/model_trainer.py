from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import Dataset
import os
import json
import torch
import time

# model_trainer.py


# Read config from config.json
with open("data/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


######################
### Setup datasets ###
######################

# Local data


script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "data")
os.makedirs(output_dir, exist_ok=True)

# Load custom dataset from disk if exists
custom_loaded_dataset = None
custom_dataset_dirs = config.get("local_dataset_path", "")
if config.get("add_local_dataset", False) and os.path.exists(custom_dataset_dirs):
    custom_loaded_dataset = Dataset.load_from_disk(custom_dataset_dirs)


# Create Hugging Face Dataset
# Load datasets from Hugging Face

# Login using e.g. `huggingface-cli login` to access this dataset

# Load each dataset from "dataset.txt", one per line
with open("data/dataset.txt", "r", encoding="utf-8") as f:
    dataset_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]

datasets = [load_dataset(name) for name in dataset_names]


# Concatenate datasets
combined_dataset = None

if datasets:
    train_datasets = [ds['train'] for ds in datasets]
    combined_dataset = concatenate_datasets(train_datasets)


if custom_loaded_dataset is not None:
    combined_dataset = concatenate_datasets([combined_dataset, custom_loaded_dataset]) if (datasets) else custom_loaded_dataset

# Guard against empty datasets
if combined_dataset is None or len(combined_dataset) == 0:
    raise ValueError("The combined dataset is empty. Please check the input datasets.")


###################
### Filter data ###
###################

# Filter out data using a wordlist from a file
def load_banned_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip() and not line.startswith('#')]

banned_words = load_banned_words('data/banned_words.txt')

def is_safe(example):
    text = example.get("text", "").lower()
    return not any(bad_word in text for bad_word in banned_words)

if banned_words:
    combined_dataset = combined_dataset.filter(is_safe)


# Remove empty lines and lines starting with "#"
def is_valid_line(example):
    text = example.get("text", "").strip()
    return text and not (text.startswith("#"))

combined_dataset = combined_dataset.filter(is_valid_line)


################################
### Set instruction datasets ###
################################

# Add instruction prefix to each sample
instruction = config.get("instruction", "Instruction: Answer as a drunk sailor.\n")
add_instruction_enabled = config.get("add_instruction", False)

def add_instruction(example):
    example["text"] = instruction + example.get("text", "")
    return example

if add_instruction_enabled:
    combined_dataset = combined_dataset.map(add_instruction)


#################################
### Setup model and tokenizer ###
#################################

# Retrieve model and tokenizer names
model_name = config.get("llm_model_name", "gpt2")
tokenizer_name = config.get("llm_tokenizer_name", "gpt2")
model_type = config.get("llm_model_type", "causal_lm")

# Masked Language Modeling
IsMaskedLanguageModel = False

# Set model, tokenizer, and initialize additional libraries
if model_name.startswith("marian"):
    from transformers import MarianMTModel, MarianTokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    IsMaskedLanguageModel = True

elif model_type == "causal_lm":
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    IsMaskedLanguageModel = False

elif model_type == "seq2seq_lm":
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    IsMaskedLanguageModel = True

elif model_type == "classification":
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    IsMaskedLanguageModel = True


#########################################################################################################################

# Check CUDA availability and set device
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # For Apple Silicon (M1/M2)
elif torch.has_mps:
    device = "mps"  # Alternative check for MPS
elif torch.backends.opencl.is_available():
    device = "opencl"  # For some AMD/Intel GPUs (experimental, rarely supported)
else:
    device = "cpu"
model.to(device)

instruction_token_length = len(tokenizer(instruction)["input_ids"])
max_token_length = instruction_token_length + 256  # or more if needed

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding='max_length',
        max_length=max_token_length,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # or use tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize the combined dataset
tokenized_combined = combined_dataset.map(tokenize_function, batched=True)


# Training arguments
# Note: On a RTX3080 it utiliztise around 4.9 GB VRAM on 1.3 M dataset
training_args = TrainingArguments(
    output_dir=config.get("model_ID", "my-model"),                              # Directory to save model checkpoints and final model (default my-model if not in config).
    num_train_epochs=config.get("num_train_epochs", 2),                         # Number of training epochs (default 2 if not in config).
    per_device_train_batch_size=config.get("per_device_train_batch_size", 4),   # Batch size per device (GPU/CPU) during training. Default is 4.
    per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),     # Batch size per device during evaluation. Default is 4.
    save_strategy=config.get("save_strategy", "epoch"),                         # When to save checkpoints: "epoch" saves at the end of each epoch.
    logging_steps=config.get("logging_steps", 10),                              # Log metrics every N steps. Default is 10.
    fp16=True,                                                                  # Enables mixed precision (FP16).
    learning_rate=config.get("learning_rate", 2e-5),                            # Learning rate for the optimizer.    
    weight_decay=config.get("weight_decay", 0.01),                              # Weight decay for regularization.
)

# Split tokenized_combined into train and eval sets
split = tokenized_combined.train_test_split(test_size=0.1)
train_ds = split['train']
eval_ds = split['test']

# Setup data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=IsMaskedLanguageModel  # For causal LM (GPT-2, Llama, etc.)
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds.shuffle(seed=42).select(range(2000)),
    data_collator=data_collator.shuffle(seed=42).select(range(500)),
)


###########################
### Execute and logging ###
###########################

# Logging
logfile_path = f"{config.get("model_ID", "my-model")}/buildlog.log"
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# Log Dataset info
with open(logfile_path, "w", encoding="utf-8") as logf:
    logf.write(f"Start timestamp: {start_time}\n")

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("models/latest/")
tokenizer.save_pretrained("models/latest/")

# Log end time
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
with open(logfile_path, "a", encoding="utf-8") as logf:
    logf.write(f"End timestamp: {end_time}\n")