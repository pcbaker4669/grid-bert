import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "google-bert/bert-base-uncased"

TRAIN_FILE = Path(r"D:\GridBERT\training\train.jsonl")
VALID_FILE = Path(r"D:\GridBERT\training\validation.jsonl")

OUTPUT_DIR = Path(r"D:\GridBERT\models\GridBERT-v0.1")

NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MLM_PROBABILITY = 0.15

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class GridTextDataset(Dataset):

    def __init__(self, filename):

        self.examples = []

        print(f"Loading {filename}...")

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                record = json.loads(line)

                self.examples.append({
                    "input_ids": record["input_ids"],
                    "attention_mask": record["attention_mask"]
                })

        print(f"Loaded {len(self.examples):,} sequences.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


train_dataset = GridTextDataset(TRAIN_FILE)
validation_dataset = GridTextDataset(VALID_FILE)


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ---------------------------------------------------------
# Load Base BERT
# ---------------------------------------------------------

print("\nLoading Base BERT...")

model = AutoModelForMaskedLM.from_pretrained(
    MODEL_NAME
)


# ---------------------------------------------------------
# MLM Data Collator
#
# Randomly masks 15% of tokens during each batch.
# ---------------------------------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROBABILITY,
    seed=42
)


# ---------------------------------------------------------
# Hardware
# ---------------------------------------------------------

has_cuda = torch.cuda.is_available()

if has_cuda:

    print("\nGPU detected:")
    print(torch.cuda.get_device_name(0))

    train_batch_size = 8
    eval_batch_size = 8

else:

    print("\nWARNING: No CUDA GPU detected.")
    print("Training will run on the CPU and may be slow.")

    train_batch_size = 2
    eval_batch_size = 2


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

training_args = TrainingArguments(

    output_dir=str(OUTPUT_DIR),

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=100,

    save_total_limit=3,

    fp16=has_cuda,

    seed=42,

    report_to="none"
)


# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    data_collator=data_collator,

    processing_class=tokenizer
)


# ---------------------------------------------------------
# Evaluate Base BERT before training
# ---------------------------------------------------------

print("\nEvaluating Base BERT before GridBERT training...")

base_results = trainer.evaluate()

base_loss = base_results["eval_loss"]
base_perplexity = math.exp(base_loss)

print(f"Base BERT validation loss: {base_loss:.4f}")
print(f"Base BERT perplexity:      {base_perplexity:.2f}")


# ---------------------------------------------------------
# Train GridBERT
# ---------------------------------------------------------

print("\nStarting GridBERT v0.1 training...\n")

trainer.train()


# ---------------------------------------------------------
# Evaluate GridBERT
# ---------------------------------------------------------

print("\nEvaluating GridBERT...")

grid_results = trainer.evaluate()

grid_loss = grid_results["eval_loss"]
grid_perplexity = math.exp(grid_loss)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Base BERT loss:        {base_loss:.4f}")
print(f"GridBERT loss:         {grid_loss:.4f}")

print(f"Base BERT perplexity:  {base_perplexity:.2f}")
print(f"GridBERT perplexity:   {grid_perplexity:.2f}")


# ---------------------------------------------------------
# Save final model
# ---------------------------------------------------------

FINAL_DIR = OUTPUT_DIR / "final"

trainer.save_model(str(FINAL_DIR))

tokenizer.save_pretrained(
    str(FINAL_DIR)
)

print()
print(f"GridBERT saved to:")
print(FINAL_DIR)