from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from datasets import concatenate_datasets

# model_tuner.py


######################
### Setup datasets ###
######################

# Load your dataset
# Load your local JSON file
rag_dataset = load_dataset("json", data_files={"train": "./Training/data/docs.json"})

if len(rag_dataset) > 0:
    # If convert to dicts with a 'text' field
    if isinstance(rag_dataset["train"][0], str):
        dataset = rag_dataset.map(lambda x: {"text": x} if isinstance(x, str) else x)

# Load a public dataset, e.g. IMDb
dataset = load_dataset("imdb")

# Login using e.g. `huggingface-cli login` to access this dataset

# Load each dataset from "dataset.txt", one per line
with open("./Training/data/dataset.txt", "r", encoding="utf-8") as f:
    dataset_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Concatenate them
datasets = [load_dataset(name) for name in dataset_names]
combined_dataset = concatenate_datasets(datasets)

if len(rag_dataset) > 0:
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