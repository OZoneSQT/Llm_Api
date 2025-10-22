from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from datasets import concatenate_datasets
import glob
import os

# model_tuner.py


######################
### Setup datasets ###
######################

# Local data

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "data")
os.makedirs(output_dir, exist_ok=True)


# Load custom dataset from disk if exists
# if os.path.exists("custom_dataset"):
# Find the latest custom_dataset_* folder and load it
custom_dataset_dirs = glob.glob("./datasets/custom_dataset_*")
if custom_dataset_dirs:
    latest_dir = max(custom_dataset_dirs, key=os.path.getmtime)
    custom_loaded_dataset = Dataset.load_from_disk(latest_dir)


# Create Hugging Face Dataset
# Load datasets from Hugging Face

# Login using e.g. `huggingface-cli login` to access this dataset

# Load each dataset from "dataset.txt", one per line
with open("./Training/data/dataset.txt", "r", encoding="utf-8") as f:
    dataset_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]

datasets = [load_dataset(name) for name in dataset_names]


# Concatenate datasets
if datasets:
    train_datasets = [ds['train'] for ds in datasets]
    combined_dataset = concatenate_datasets(train_datasets)

if custom_dataset_dirs:
    combined_dataset = concatenate_datasets([combined_dataset, custom_loaded_dataset]) if (datasets) else custom_loaded_dataset

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
instruction = "Instruction: Answer as a drunk sailor.\n"
def add_instruction(example):
    example["text"] = instruction + example.get("text", "")
    return example

if instruction:
    combined_dataset = combined_dataset.map(add_instruction)


#################################
### Setup model and tokenizer ###
#################################

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = combined_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(2000)),  # Small subset for demo
    eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(500)),
)


###############
### Execute ###
###############

# Train the model
trainer.train()

# Save the tuned model
model.save_pretrained("./tuned_model")
tokenizer.save_pretrained("./tuned_model")