"""
train_reliability_classifier.py

Purpose
-------
Fine-tunes either Base BERT or GridBERT v0.3 for binary
electric-grid reliability classification.

The two models use:

- identical training data
- identical validation data
- identical tokenizer
- identical hyperparameters
- identical random seed
- identical model-selection criterion

The validation F1 score is used to select the best checkpoint.

The internal test set and external unseen test set are NOT
used by this script.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from config_loader import load_config


# ============================================================
# Command line
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    choices=[
        "basebert",
        "gridbert",
    ],
)

args = parser.parse_args()


# ============================================================
# Configuration
# ============================================================

config = load_config()

DATA_ROOT = Path(
    config["paths"]["data_root"]
)

training_config = config[
    "classification_training"
]


def data_root_path(value):

    path = Path(value)

    if path.is_absolute():
        return path

    return DATA_ROOT / path


TRAIN_FILE = data_root_path(
    config["classification"]["train_file"]
)

VALIDATION_FILE = data_root_path(
    config["classification"]["validation_file"]
)


MAX_LENGTH = int(
    training_config["max_length"]
)

BATCH_SIZE = int(
    training_config["batch_size"]
)

EPOCHS = int(
    training_config["epochs"]
)

LEARNING_RATE = float(
    training_config["learning_rate"]
)

WEIGHT_DECAY = float(
    training_config["weight_decay"]
)

WARMUP_RATIO = float(
    training_config["warmup_ratio"]
)

GRADIENT_CLIP_NORM = float(
    training_config["gradient_clip_norm"]
)

RANDOM_SEED = int(
    training_config["seed"]
)


# ============================================================
# Select model
# ============================================================

if args.model == "basebert":

    MODEL_SOURCE = (
        config["models"][
            "reference_model"
        ]
    )

    MODEL_LABEL = "Base BERT"

    OUTPUT_DIR = data_root_path(
        config["classification"][
            "basebert_output"
        ]
    )

else:

    MODEL_SOURCE = data_root_path(
        config["paths"][
            "grid_model_dir"
        ]
    )

    MODEL_LABEL = "GridBERT v0.3"

    OUTPUT_DIR = data_root_path(
        config["classification"][
            "gridbert_output"
        ]
    )


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(
    RANDOM_SEED
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 72)
print("GRIDBERT RELIABILITY CLASSIFIER")
print("=" * 72)

print()
print(f"Model:       {MODEL_LABEL}")
print(f"Model source: {MODEL_SOURCE}")
print(f"Device:      {device}")

print()
print(f"Training:    {TRAIN_FILE}")
print(f"Validation:  {VALIDATION_FILE}")

print()


# ============================================================
# Load datasets
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)

validation_df = pd.read_csv(
    VALIDATION_FILE
)


required_columns = {
    "text",
    "label",
}


for name, dataframe in [

    ("train", train_df),
    ("validation", validation_df),

]:

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:

        raise ValueError(
            f"{name} dataset is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    if dataframe["text"].isna().any():

        raise ValueError(
            f"{name} contains missing text."
        )

    if dataframe["label"].isna().any():

        raise ValueError(
            f"{name} contains missing labels."
        )


train_df["label"] = (
    train_df["label"].astype(int)
)

validation_df["label"] = (
    validation_df[
        "label"
    ].astype(int)
)


valid_labels = {
    0,
    1,
}


if not set(
    train_df["label"].unique()
).issubset(valid_labels):

    raise ValueError(
        "Training data contains labels "
        "other than 0 and 1."
    )


if not set(
    validation_df[
        "label"
    ].unique()
).issubset(valid_labels):

    raise ValueError(
        "Validation data contains labels "
        "other than 0 and 1."
    )


print(
    f"Training passages:   "
    f"{len(train_df):,}"
)

print(
    f"Validation passages: "
    f"{len(validation_df):,}"
)


print()
print("Training labels:")

print(
    train_df[
        "label"
    ].value_counts().sort_index()
)


print()
print("Validation labels:")

print(
    validation_df[
        "label"
    ].value_counts().sort_index()
)


# ============================================================
# Tokenizer
# ============================================================

TOKENIZER_MODEL = (
    config["models"][
        "tokenizer_model"
    ]
)


print()
print(
    f"Loading tokenizer: "
    f"{TOKENIZER_MODEL}"
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL
    )
)


# ============================================================
# PyTorch Dataset
# ============================================================

class ReliabilityDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length,
    ):

        self.texts = (
            dataframe["text"]
            .astype(str)
            .tolist()
        )

        self.labels = (
            dataframe["label"]
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer

        self.max_length = (
            max_length
        )


    def __len__(self):

        return len(
            self.texts
        )


    def __getitem__(
        self,
        index,
    ):

        encoding = (
            self.tokenizer(
                self.texts[index],
                truncation=True,
                max_length=(
                    self.max_length
                ),
            )
        )

        encoding["labels"] = (
            self.labels[index]
        )

        return encoding


train_dataset = ReliabilityDataset(
    train_df,
    tokenizer,
    MAX_LENGTH,
)

validation_dataset = ReliabilityDataset(
    validation_df,
    tokenizer,
    MAX_LENGTH,
)


# ============================================================
# Dynamic padding
# ============================================================

data_collator = (
    DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0,
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
)


# ============================================================
# Load classification model
# ============================================================

print()
print(
    f"Loading model: "
    f"{MODEL_SOURCE}"
)


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_SOURCE,
        num_labels=2,

        id2label={
            0: "NOT_RELIABILITY",
            1: "RELIABILITY",
        },

        label2id={
            "NOT_RELIABILITY": 0,
            "RELIABILITY": 1,
        },
    )
)

# ============================================================
# Restore Base BERT pooler for GridBERT
# ============================================================

if args.model == "gridbert":

    from transformers import AutoModel

    print()
    print(
        "Loading original Base BERT pooler "
        "for GridBERT..."
    )

    base_model = AutoModel.from_pretrained(
        config["models"]["reference_model"]
    )

    model.bert.pooler.load_state_dict(
        base_model.pooler.state_dict()
    )

    del base_model

    print(
        "Base BERT pooler loaded into GridBERT."
    )

model.to(
    device
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


total_training_steps = (
    len(train_loader)
    * EPOCHS
)


warmup_steps = int(
    total_training_steps
    * WARMUP_RATIO
)


scheduler = (
    get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(
            warmup_steps
        ),
        num_training_steps=(
            total_training_steps
        ),
    )
)


# ============================================================
# Validation
# ============================================================

def evaluate(
    model,
    data_loader,
):

    model.eval()

    predictions = []
    labels = []

    total_loss = 0.0


    with torch.no_grad():

        for batch in data_loader:

            batch = {
                key: value.to(
                    device
                )
                for key, value
                in batch.items()
            }

            outputs = model(
                **batch
            )

            total_loss += (
                outputs.loss.item()
            )

            predicted = (
                torch.argmax(
                    outputs.logits,
                    dim=1,
                )
            )

            predictions.extend(
                predicted.cpu().tolist()
            )

            labels.extend(
                batch["labels"]
                .cpu()
                .tolist()
            )


    average_loss = (
        total_loss
        / len(data_loader)
    )


    metrics = {

        "loss":
            average_loss,

        "accuracy":
            accuracy_score(
                labels,
                predictions,
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0,
            ),
    }


    return metrics


# ============================================================
# Training
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


history = []

best_f1 = -1.0
best_epoch = None
best_validation_metrics = None


print()
print("=" * 72)
print("TRAINING")
print("=" * 72)


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    total_train_loss = 0.0


    for batch in train_loader:

        batch = {
            key: value.to(
                device
            )
            for key, value
            in batch.items()
        }


        optimizer.zero_grad()


        outputs = model(
            **batch
        )


        loss = outputs.loss

        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
        )


        optimizer.step()

        scheduler.step()


        total_train_loss += (
            loss.item()
        )


    average_train_loss = (
        total_train_loss
        / len(train_loader)
    )


    validation_metrics = evaluate(
        model,
        validation_loader,
    )


    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"  Train loss: "
        f"{average_train_loss:.4f}"
    )

    print(
        f"  Validation loss: "
        f"{validation_metrics['loss']:.4f}"
    )

    print(
        f"  Accuracy: "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"  Recall: "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"  F1: "
        f"{validation_metrics['f1']:.4f}"
    )


    history.append({

        "epoch":
            epoch,

        "train_loss":
            average_train_loss,

        "validation_loss":
            validation_metrics[
                "loss"
            ],

        "accuracy":
            validation_metrics[
                "accuracy"
            ],

        "precision":
            validation_metrics[
                "precision"
            ],

        "recall":
            validation_metrics[
                "recall"
            ],

        "f1":
            validation_metrics[
                "f1"
            ],
    })


    # --------------------------------------------------------
    # Save best checkpoint by validation F1
    # --------------------------------------------------------

    if (
        validation_metrics["f1"]
        > best_f1
    ):

        best_f1 = (
            validation_metrics["f1"]
        )

        best_epoch = epoch

        best_validation_metrics = (
            validation_metrics.copy()
        )


        print(
            "  *** New best validation F1 ***"
        )


        model.save_pretrained(
            OUTPUT_DIR
        )

        tokenizer.save_pretrained(
            OUTPUT_DIR
        )


# ============================================================
# Save training history
# ============================================================

history_file = (
    OUTPUT_DIR
    / "training_history.csv"
)


with open(
    history_file,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [

        "epoch",
        "train_loss",
        "validation_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(history)


# ============================================================
# Save reproducibility information
# ============================================================

summary = {

    "model":
        MODEL_LABEL,

    "model_source":
        str(MODEL_SOURCE),

    "tokenizer":
        TOKENIZER_MODEL,

    "train_file":
        str(TRAIN_FILE),

    "validation_file":
        str(VALIDATION_FILE),

    "training_passages":
        len(train_df),

    "validation_passages":
        len(validation_df),

    "training_label_counts": {
        str(key): int(value)
        for key, value
        in train_df[
            "label"
        ].value_counts().items()
    },

    "validation_label_counts": {
        str(key): int(value)
        for key, value
        in validation_df[
            "label"
        ].value_counts().items()
    },

    "hyperparameters": {

        "max_length":
            MAX_LENGTH,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "warmup_ratio":
            WARMUP_RATIO,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "seed":
            RANDOM_SEED,

        "class_weights":
            None,

        "resampling":
            False,
    },

    "model_selection":
        "highest validation F1",

    "best_epoch":
        best_epoch,

    "best_validation_metrics":
        best_validation_metrics,
}


summary_file = (
    OUTPUT_DIR
    / "training_summary.json"
)


with open(
    summary_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=2,
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 72)
print("TRAINING COMPLETE")
print("=" * 72)

print()
print(
    f"Model: "
    f"{MODEL_LABEL}"
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation F1: "
    f"{best_f1:.4f}"
)

print()
print(
    f"Best model saved:"
)

print(
    OUTPUT_DIR
)

print()
print(
    f"Training history:"
)

print(
    history_file
)

print()
print(
    f"Training summary:"
)

print(
    summary_file
)