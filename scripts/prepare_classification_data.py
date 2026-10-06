from pathlib import Path
import json

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# Helper functions
# ============================================================

def normalize_path(value):
    """
    Normalize Windows/Unix path differences for comparison.
    """
    if pd.isna(value):
        return ""

    value = str(value).strip().replace("\\", "/").lower()

    while "//" in value:
        value = value.replace("//", "/")

    return value


def manifest_relative_path(path):
    """
    Convert:

    D:/GridBERT_v0_3/text/nerc/ltra/report.txt

    into:

    nerc/ltra/report.txt
    """
    path = normalize_path(path)

    marker = "/text/"

    if marker in path:
        path = path.split(marker, 1)[1]

    return path


def load_jsonl_source_documents(jsonl_file):
    """
    Read a GridBERT training/validation JSONL file and return
    the unique normalized source_file values.

    Expected JSONL field:
        "source_file": "nerc\\ltra\\report.txt"
    """
    source_documents = set()

    with open(jsonl_file, "r", encoding="utf-8") as infile:

        for line_number, line in enumerate(infile, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_file} "
                    f"at line {line_number}: {exc}"
                )

            if "source_file" not in record:
                raise ValueError(
                    f"'source_file' missing from "
                    f"{jsonl_file} at line {line_number}"
                )

            source_documents.add(
                normalize_path(record["source_file"])
            )

    return source_documents


# ============================================================
# Load configuration
# ============================================================

config_path = Path("config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

root = Path(config["paths"]["data_root"])

labeled_file = (
    root / config["classification"]["labeled_file"]
)

train_output = (
    root / config["classification"]["train_file"]
)

validation_output = (
    root / config["classification"]["validation_file"]
)

test_output = (
    root / config["classification"]["test_file"]
)

manifest_file = (
    root / config["paths"]["corpus_manifest"]
)

mlm_train_file = (
    root / config["paths"]["train_file"]
)

mlm_validation_file = (
    root / config["paths"]["validation_file"]
)

seed = config["dataset"]["seed"]


# ============================================================
# Load binary classification dataset
# ============================================================

print("\n========================================")
print("LOADING CLASSIFICATION DATA")
print("========================================")

print(f"\nFile: {labeled_file}")

df = pd.read_csv(labeled_file)

required_columns = [
    "source_document",
    "text",
    "label",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["text"].isna().any():
    raise ValueError(
        "Dataset contains missing text values."
    )

if df["label"].isna().any():
    raise ValueError(
        "Dataset contains missing labels."
    )

print(f"Sequences: {len(df):,}")
print(
    f"Unique documents: "
    f"{df['source_document'].nunique():,}"
)

print("\nLabels:")
print(
    df["label"]
    .value_counts()
    .sort_index()
)

print("\nLabel percentages:")
print(
    (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(1)
)


# ============================================================
# Normalize classification document paths
# ============================================================

df["source_document_normalized"] = (
    df["source_document"].apply(normalize_path)
)


# ============================================================
# Load GridBERT v0.3 manifest
# ============================================================

print("\n========================================")
print("LOADING GRIDBERT V0.3 MANIFEST")
print("========================================")

print(f"\nFile: {manifest_file}")

manifest = pd.read_csv(manifest_file)

if "text_file" not in manifest.columns:
    raise ValueError(
        "Expected column 'text_file' "
        "in corpus manifest."
    )

print(
    f"Manifest rows: {len(manifest):,}"
)

manifest["source_document_normalized"] = (
    manifest["text_file"].apply(
        manifest_relative_path
    )
)

corpus_documents = set(
    manifest["source_document_normalized"]
)

print(
    f"Unique corpus documents: "
    f"{len(corpus_documents):,}"
)


# ============================================================
# Load ACTUAL MLM training documents
# ============================================================

print("\n========================================")
print("LOADING ACTUAL MLM TRAINING DATA")
print("========================================")

print(f"\nTraining JSONL:")
print(mlm_train_file)

training_documents = (
    load_jsonl_source_documents(
        mlm_train_file
    )
)

print(
    f"\nUnique MLM training documents: "
    f"{len(training_documents):,}"
)


# ============================================================
# Load ACTUAL MLM validation documents
# ============================================================

print("\nValidation JSONL:")
print(mlm_validation_file)

mlm_validation_documents = (
    load_jsonl_source_documents(
        mlm_validation_file
    )
)

print(
    f"\nUnique MLM validation documents: "
    f"{len(mlm_validation_documents):,}"
)


# ============================================================
# Sanity checks on the pretraining corpus
# ============================================================

print("\n========================================")
print("PRETRAINING SANITY CHECKS")
print("========================================")


# ------------------------------------------------------------
# Training and validation should not overlap
# ------------------------------------------------------------

overlap = (
    training_documents
    & mlm_validation_documents
)

if overlap:
    print(
        "\nERROR: Some documents appear in BOTH "
        "MLM training and validation:"
    )

    for document in sorted(overlap):
        print(document)

    raise ValueError(
        "MLM training/validation document leakage detected."
    )

print(
    "\nNo overlap between MLM training "
    "and validation documents."
)


# ------------------------------------------------------------
# Combined JSONL documents should match manifest
# ------------------------------------------------------------

jsonl_documents = (
    training_documents
    | mlm_validation_documents
)

print(
    f"\nDocuments represented in JSONL files: "
    f"{len(jsonl_documents):,}"
)

missing_from_jsonl = (
    corpus_documents - jsonl_documents
)

extra_in_jsonl = (
    jsonl_documents - corpus_documents
)

if missing_from_jsonl:

    print(
        "\nWARNING: Manifest documents not "
        "represented in JSONL files:"
    )

    for document in sorted(
        missing_from_jsonl
    ):
        print(document)

if extra_in_jsonl:

    print(
        "\nWARNING: JSONL documents not "
        "found in manifest:"
    )

    for document in sorted(
        extra_in_jsonl
    ):
        print(document)


if (
    not missing_from_jsonl
    and not extra_in_jsonl
):
    print(
        "Manifest and JSONL document sets match."
    )


# ============================================================
# Compare with configured fixed validation documents
# ============================================================

configured_validation_documents = {
    normalize_path(document)
    for document in config["dataset"].get(
        "fixed_validation_documents",
        []
    )
}

print("\nConfigured fixed validation documents:")
print(
    len(configured_validation_documents)
)

if (
    configured_validation_documents
    == mlm_validation_documents
):
    print(
        "Configured validation documents match "
        "the actual validation JSONL."
    )

else:

    print(
        "\nWARNING: Configured fixed validation "
        "documents do not exactly match "
        "validation.jsonl."
    )

    only_in_config = (
        configured_validation_documents
        - mlm_validation_documents
    )

    only_in_jsonl = (
        mlm_validation_documents
        - configured_validation_documents
    )

    if only_in_config:

        print("\nOnly in config:")

        for document in sorted(
            only_in_config
        ):
            print(document)

    if only_in_jsonl:

        print("\nOnly in validation JSONL:")

        for document in sorted(
            only_in_jsonl
        ):
            print(document)


# ============================================================
# Add pretraining provenance to classification data
# ============================================================

df["in_v03_corpus"] = (
    df["source_document_normalized"]
    .isin(corpus_documents)
    .astype(int)
)

df["used_in_mlm_training"] = (
    df["source_document_normalized"]
    .isin(training_documents)
    .astype(int)
)

df["used_in_mlm_validation"] = (
    df["source_document_normalized"]
    .isin(mlm_validation_documents)
    .astype(int)
)


# ============================================================
# Sanity check classification provenance
# ============================================================

invalid_rows = df[
    (df["used_in_mlm_training"] == 1)
    &
    (df["used_in_mlm_validation"] == 1)
]

if len(invalid_rows) > 0:
    raise ValueError(
        "Classification rows found whose documents "
        "appear in both MLM training and validation."
    )


# ============================================================
# Report pretraining exposure
# ============================================================

print("\n========================================")
print("CLASSIFICATION PRETRAINING EXPOSURE")
print("========================================")

print("\nSequences in v0.3 corpus:")
print(
    df["in_v03_corpus"]
    .value_counts()
    .sort_index()
)

print("\nSequences from MLM TRAINING documents:")
print(
    df["used_in_mlm_training"]
    .value_counts()
    .sort_index()
)

print("\nSequences from MLM VALIDATION documents:")
print(
    df["used_in_mlm_validation"]
    .value_counts()
    .sort_index()
)


# ============================================================
# Document-level exposure report
# ============================================================

document_status = (
    df[
        [
            "source_document",
            "source_document_normalized",
            "in_v03_corpus",
            "used_in_mlm_training",
            "used_in_mlm_validation",
        ]
    ]
    .drop_duplicates()
)

print("\nUnique classification documents:")

print(
    document_status[
        [
            "in_v03_corpus",
            "used_in_mlm_training",
            "used_in_mlm_validation",
        ]
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# Print truly unseen documents
# ============================================================

unseen_documents = (
    document_status.loc[
        document_status[
            "in_v03_corpus"
        ] == 0,
        "source_document"
    ]
    .sort_values()
)

print("\n========================================")
print("DOCUMENTS NEVER SEEN IN GRIDBERT V0.3")
print("========================================")

if len(unseen_documents) == 0:

    print("\nNone.")

else:

    for document in unseen_documents:
        print(document)


# ============================================================
# Print documents used only for MLM validation
# ============================================================

mlm_validation_classifier_docs = (
    document_status.loc[
        document_status[
            "used_in_mlm_validation"
        ] == 1,
        "source_document"
    ]
    .sort_values()
)

print("\n========================================")
print("MLM VALIDATION DOCUMENTS")
print("FOUND IN CLASSIFICATION DATA")
print("========================================")

if len(
    mlm_validation_classifier_docs
) == 0:

    print("\nNone.")

else:

    for document in (
        mlm_validation_classifier_docs
    ):
        print(document)


# ============================================================
# Create document-level supervised splits
#
# 70% train
# 15% validation
# 15% test
# ============================================================

print("\n========================================")
print("CREATING SUPERVISED DOCUMENT SPLITS")
print("========================================")


# ------------------------------------------------------------
# 70% supervised training
# 30% temporary
# ------------------------------------------------------------

gss1 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.70,
    random_state=seed,
)

train_indices, temp_indices = next(
    gss1.split(
        df,
        groups=(
            df[
                "source_document_normalized"
            ]
        ),
    )
)

train_df = (
    df.iloc[train_indices]
    .copy()
)

temp_df = (
    df.iloc[temp_indices]
    .copy()
)


# ------------------------------------------------------------
# Split remaining 30% equally
#
# 15% validation
# 15% test
# ------------------------------------------------------------

gss2 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.50,
    random_state=seed,
)

validation_indices, test_indices = next(
    gss2.split(
        temp_df,
        groups=(
            temp_df[
                "source_document_normalized"
            ]
        ),
    )
)

validation_df = (
    temp_df.iloc[
        validation_indices
    ].copy()
)

test_df = (
    temp_df.iloc[
        test_indices
    ].copy()
)


# ============================================================
# Add split column
# ============================================================

train_df["split"] = "train"

validation_df["split"] = (
    "validation"
)

test_df["split"] = "test"


# ============================================================
# Verify no supervised document leakage
# ============================================================

train_docs = set(
    train_df[
        "source_document_normalized"
    ]
)

validation_docs = set(
    validation_df[
        "source_document_normalized"
    ]
)

test_docs = set(
    test_df[
        "source_document_normalized"
    ]
)

assert train_docs.isdisjoint(
    validation_docs
)

assert train_docs.isdisjoint(
    test_docs
)

assert validation_docs.isdisjoint(
    test_docs
)

print(
    "\nNo source-document leakage "
    "between supervised splits."
)


# ============================================================
# Report supervised split statistics
# ============================================================

def print_split_stats(
    name,
    frame
):

    print("\n----------------------------------------")
    print(name.upper())
    print("----------------------------------------")

    print(
        f"Sequences: {len(frame):,}"
    )

    print(
        f"Documents: "
        f"{frame['source_document'].nunique():,}"
    )

    print("\nLabels:")

    print(
        frame["label"]
        .value_counts()
        .sort_index()
    )

    print("\nLabel percentages:")

    print(
        (
            frame["label"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        ).round(1)
    )

    print(
        "\nSequences from MLM "
        "training documents:"
    )

    print(
        frame[
            "used_in_mlm_training"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nSequences from MLM "
        "validation documents:"
    )

    print(
        frame[
            "used_in_mlm_validation"
        ]
        .value_counts()
        .sort_index()
    )


print_split_stats(
    "Training Set",
    train_df,
)

print_split_stats(
    "Validation Set",
    validation_df,
)

print_split_stats(
    "Test Set",
    test_df,
)


# ============================================================
# Remove internal helper column
# ============================================================

for frame in [
    train_df,
    validation_df,
    test_df,
]:

    frame.drop(
        columns=[
            "source_document_normalized"
        ],
        inplace=True,
    )


# ============================================================
# Create output directories
# ============================================================

train_output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

validation_output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

test_output.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Save supervised datasets
# ============================================================

train_df.to_csv(
    train_output,
    index=False,
)

validation_df.to_csv(
    validation_output,
    index=False,
)

test_df.to_csv(
    test_output,
    index=False,
)


# ============================================================
# Save complete classification dataset
# ============================================================

complete_output = (
    labeled_file.parent
    / "reliability_binary_with_pretraining_status.csv"
)

complete_df = pd.concat(
    [
        train_df,
        validation_df,
        test_df,
    ],
    ignore_index=True,
)

complete_df.to_csv(
    complete_output,
    index=False,
)


# ============================================================
# Final report
# ============================================================

print("\n========================================")
print("FILES WRITTEN")
print("========================================")

print(f"\nTrain:")
print(train_output)

print(f"\nValidation:")
print(validation_output)

print(f"\nTest:")
print(test_output)

print(f"\nComplete dataset:")
print(complete_output)

print("\nPreparation complete.")