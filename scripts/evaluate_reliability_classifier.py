"""
evaluate_reliability_classifier.py

Purpose
-------
Evaluates a previously fine-tuned reliability classifier on either:

1. The internal held-out test set
2. The external unseen-document test set

No training or model selection occurs in this script.

The same evaluation procedure is used for Base BERT and GridBERT v0.3.
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
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

parser.add_argument(
    "--dataset",
    required=True,
    choices=[
        "internal",
        "external",
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

MAX_LENGTH = int(
    training_config["max_length"]
)

BATCH_SIZE = int(
    training_config["batch_size"]
)


def data_root_path(value):

    path = Path(value)

    if path.is_absolute():
        return path

    return DATA_ROOT / path


# ============================================================
# Select model
# ============================================================

if args.model == "basebert":

    MODEL_DIR = data_root_path(
        config["classification"][
            "basebert_output"
        ]
    )

    MODEL_LABEL = "Base BERT"

else:

    MODEL_DIR = data_root_path(
        config["classification"][
            "gridbert_output"
        ]
    )

    MODEL_LABEL = "GridBERT v0.3"


# ============================================================
# Select evaluation dataset
# ============================================================

if args.dataset == "internal":

    TEST_FILE = data_root_path(
        config["classification"][
            "test_file"
        ]
    )

    RESULTS_DIR = data_root_path(
        config["classification"][
            "results_dir"
        ]
    )

    DATASET_LABEL = "internal"

else:

    TEST_FILE = data_root_path(
        config["classification"][
            "external_binary_file"
        ]
    )

    RESULTS_DIR = data_root_path(
        config["classification"][
            "external_results_dir"
        ]
    )

    DATASET_LABEL = "external"


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
print("GRIDBERT RELIABILITY EVALUATION")
print("=" * 72)

print()
print(
    f"Model:       {MODEL_LABEL}"
)

print(
    f"Model path:  {MODEL_DIR}"
)

print(
    f"Dataset:     {DATASET_LABEL}"
)

print(
    f"Test file:   {TEST_FILE}"
)

print(
    f"Device:      {device}"
)


# ============================================================
# Load evaluation data
# ============================================================

test_df = pd.read_csv(
    TEST_FILE
)

required_columns = {
    "text",
    "label",
}

missing = (
    required_columns
    - set(test_df.columns)
)

if missing:

    raise ValueError(
        "Test dataset missing columns: "
        + ", ".join(
            sorted(missing)
        )
    )


if test_df["text"].isna().any():

    raise ValueError(
        "Test dataset contains missing text."
    )


if test_df["label"].isna().any():

    raise ValueError(
        "Test dataset contains missing labels."
    )


test_df["label"] = (
    test_df["label"].astype(int)
)


if not set(
    test_df["label"].unique()
).issubset({0, 1}):

    raise ValueError(
        "Test dataset contains labels other than 0 and 1."
    )


print()
print(
    f"Test passages: "
    f"{len(test_df):,}"
)

print()
print("Test labels:")

print(
    test_df[
        "label"
    ].value_counts().sort_index()
)


# ============================================================
# Load saved tokenizer and model
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_DIR
    )
)


print("Loading classifier...")

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR
    )
)

model.to(
    device
)

model.eval()


# ============================================================
# Dataset
# ============================================================

class ReliabilityDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length,
    ):

        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.tokenizer = tokenizer

        self.max_length = (
            max_length
        )


    def __len__(self):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index,
    ):

        row = (
            self.dataframe.iloc[
                index
            ]
        )

        encoding = (
            self.tokenizer(
                str(row["text"]),
                truncation=True,
                max_length=(
                    self.max_length
                ),
            )
        )

        encoding["labels"] = int(
            row["label"]
        )

        encoding["row_index"] = (
            index
        )

        return encoding


dataset = ReliabilityDataset(
    test_df,
    tokenizer,
    MAX_LENGTH,
)


data_collator = (
    DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )
)


def collate_batch(features):

    row_indexes = [
        feature.pop(
            "row_index"
        )
        for feature in features
    ]

    batch = data_collator(
        features
    )

    batch["row_indexes"] = (
        torch.tensor(
            row_indexes,
            dtype=torch.long,
        )
    )

    return batch


data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=0,
)


# ============================================================
# Evaluate
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []
all_row_indexes = []

total_loss = 0.0


with torch.no_grad():

    for batch in data_loader:

        row_indexes = batch.pop(
            "row_indexes"
        )

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

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=1,
            )
        )

        predictions = (
            torch.argmax(
                probabilities,
                dim=1,
            )
        )

        all_labels.extend(
            batch["labels"]
            .cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions
            .cpu()
            .tolist()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .tolist()
        )

        all_row_indexes.extend(
            row_indexes.tolist()
        )


average_loss = (
    total_loss
    / len(data_loader)
)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0,
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0,
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0,
)


matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1],
)


tn, fp, fn, tp = (
    matrix.ravel()
)


report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=[
        "NOT_RELIABILITY",
        "RELIABILITY",
    ],
    output_dict=True,
    zero_division=0,
)


# ============================================================
# Console results
# ============================================================

print()
print("=" * 72)
print("RESULTS")
print("=" * 72)

print()
print(
    f"Loss:      "
    f"{average_loss:.4f}"
)

print(
    f"Accuracy:  "
    f"{accuracy:.4f}"
)

print(
    f"Precision: "
    f"{precision:.4f}"
)

print(
    f"Recall:    "
    f"{recall:.4f}"
)

print(
    f"F1:        "
    f"{f1:.4f}"
)

print()
print("Confusion matrix:")

print()
print(
    f"TN: {tn:3d}    "
    f"FP: {fp:3d}"
)

print(
    f"FN: {fn:3d}    "
    f"TP: {tp:3d}"
)


# ============================================================
# Output paths
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


model_file_label = (
    "basebert"
    if args.model == "basebert"
    else "gridbert_v03"
)


prefix = (
    f"{model_file_label}_"
    f"{DATASET_LABEL}"
)


metrics_file = (
    RESULTS_DIR
    / f"{prefix}_metrics.json"
)

predictions_file = (
    RESULTS_DIR
    / f"{prefix}_predictions.csv"
)

confusion_file = (
    RESULTS_DIR
    / f"{prefix}_confusion_matrix.csv"
)


# ============================================================
# Save metrics
# ============================================================

metrics = {

    "model":
        MODEL_LABEL,

    "dataset":
        DATASET_LABEL,

    "model_path":
        str(MODEL_DIR),

    "test_file":
        str(TEST_FILE),

    "test_passages":
        len(test_df),

    "test_label_counts": {
        str(key): int(value)
        for key, value
        in test_df[
            "label"
        ].value_counts().items()
    },

    "max_length":
        MAX_LENGTH,

    "batch_size":
        BATCH_SIZE,

    "loss":
        average_loss,

    "accuracy":
        accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "f1":
        f1,

    "confusion_matrix": {

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_positive":
            int(tp),
    },

    "classification_report":
        report,
}


with open(
    metrics_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=2,
    )


# ============================================================
# Save predictions
# ============================================================

prediction_rows = []


for (
    row_index,
    true_label,
    predicted_label,
    probability,
) in zip(
    all_row_indexes,
    all_labels,
    all_predictions,
    all_probabilities,
):

    original = (
        test_df.iloc[
            row_index
        ]
    )

    prediction_rows.append({

        "id":
            original.get(
                "id",
                row_index + 1,
            ),

        "source":
            original.get(
                "source",
                "",
            ),

        "source_document":
            original.get(
                "source_document",
                "",
            ),

        "true_label":
            true_label,

        "predicted_label":
            predicted_label,

        "reliability_probability":
            probability,

        "correct":
            int(
                true_label
                == predicted_label
            ),

        "text":
            original["text"],
    })


with open(
    predictions_file,
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    fieldnames = [

        "id",
        "source",
        "source_document",

        "true_label",
        "predicted_label",

        "reliability_probability",
        "correct",

        "text",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        prediction_rows
    )


# ============================================================
# Save confusion matrix
# ============================================================

with open(
    confusion_file,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([
        "",
        "Predicted_NOT",
        "Predicted_RELIABILITY",
    ])

    writer.writerow([
        "Actual_NOT",
        int(tn),
        int(fp),
    ])

    writer.writerow([
        "Actual_RELIABILITY",
        int(fn),
        int(tp),
    ])


# ============================================================
# Finished
# ============================================================

print()
print("Metrics:")
print(metrics_file)

print()
print("Predictions:")
print(predictions_file)

print()
print("Confusion matrix:")
print(confusion_file)