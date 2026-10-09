# this will build a manifest of the external unseen documents, including provenance information
# we will check if the documents were used in the v0.3 manifest, and if they were used in MLM training or validation

import csv
import json
import re
from pathlib import Path

import pymupdf as fitz
from transformers import AutoTokenizer

from config_loader import load_config


# ============================================================
# Configuration
# ============================================================

config = load_config()

DATA_ROOT = Path(config["paths"]["data_root"])

RAW_ROOT = (
    DATA_ROOT
    / config["classification"]["external_raw_dir"]
)

TEXT_ROOT = (
    DATA_ROOT
    / config["classification"]["external_text_dir"]
)

OUTPUT_FILE = (
    DATA_ROOT
    / config["classification"]["external_manifest_file"]
)

V03_MANIFEST = (
    DATA_ROOT
    / config["paths"]["corpus_manifest"]
)

MLM_TRAIN_FILE = (
    DATA_ROOT
    / config["paths"]["train_file"]
)

MLM_VALIDATION_FILE = (
    DATA_ROOT
    / config["paths"]["validation_file"]
)

TOKENIZER_MODEL = config["models"]["tokenizer_model"]


# ============================================================
# Known document metadata
# ============================================================

DOCUMENT_METADATA = {

    "2025-rtep-baseline-assessment.pdf": {
        "document_date": "2026-03-18",
        "document_type": "Baseline Reliability Assessment",
    },

    "item-03b---1-system-operations-report---presentation.pdf": {
        "document_date": "2026-09-28",
        "document_type": "System Operations Report",
    },

    "20260605-item-04---load-adjustment-accuracy-report.pdf": {
        "document_date": "2026-06-05",
        "document_type": "Data Center Accuracy Report",
    },

    "nerc_sra_2026.pdf": {
        "document_date": "2026-05",
        "document_type": "Summer Reliability Assessment",
    },

    "demand-response-issues-and-performance-2025-sep-21-2026.pdf": {
        "document_date": "2026-09-21",
        "document_type": "Demand Response Performance Report",
    },
}


# ============================================================
# Helpers
# ============================================================

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def normalize_path(path):
    return str(path).replace("\\", "/").lower()


def load_mlm_documents(jsonl_file):

    documents = set()

    with open(
        jsonl_file,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            source_file = record.get(
                "source_file"
            )

            if source_file:
                documents.add(
                    normalize_path(source_file)
                )

    return documents


def make_mlm_keys(documents):

    keys = set()

    for document in documents:

        parts = document.split("/")

        if len(parts) >= 2:

            source = parts[0]
            filename = parts[-1]

            keys.add(
                (source, filename)
            )

    return keys


# ============================================================
# Load tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL
)


# ============================================================
# Load GridBERT v0.3 manifest
# ============================================================

v03_manifest_keys = set()

with open(
    V03_MANIFEST,
    "r",
    encoding="utf-8",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        source = (
            row["source"]
            .strip()
            .lower()
        )

        filename = (
            row["filename"]
            .strip()
            .lower()
        )

        v03_manifest_keys.add(
            (source, filename)
        )


# ============================================================
# Load actual MLM provenance
# ============================================================

mlm_train_documents = load_mlm_documents(
    MLM_TRAIN_FILE
)

mlm_validation_documents = load_mlm_documents(
    MLM_VALIDATION_FILE
)

mlm_train_keys = make_mlm_keys(
    mlm_train_documents
)

mlm_validation_keys = make_mlm_keys(
    mlm_validation_documents
)


# ============================================================
# Build external manifest
# ============================================================

rows = []

pdf_files = sorted(
    RAW_ROOT.rglob("*.pdf")
)

print()
print(
    f"Found {len(pdf_files)} external PDFs."
)


for pdf_path in pdf_files:

    relative_pdf = pdf_path.relative_to(
        RAW_ROOT
    )

    source = relative_pdf.parts[0].lower()

    relative_text = (
        relative_pdf.with_suffix(".txt")
    )

    text_path = (
        TEXT_ROOT
        / relative_text
    )

    print()
    print(f"Processing: {relative_pdf}")

    if not text_path.exists():

        print(
            "WARNING: matching text file "
            "was not found."
        )

        continue

    # ----------------------------------------
    # PDF page count
    # ----------------------------------------

    document = fitz.open(pdf_path)

    page_count = len(document)

    document.close()

    # ----------------------------------------
    # Text statistics
    # ----------------------------------------

    text = text_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    clean_text = normalize_whitespace(
        text
    )

    tokens = tokenizer(
        clean_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # ----------------------------------------
    # Provenance checks
    # ----------------------------------------

    pdf_key = (
        source,
        pdf_path.name.lower(),
    )

    txt_key = (
        source,
        text_path.name.lower(),
    )

    in_v03_manifest = int(
        pdf_key in v03_manifest_keys
    )

    used_in_mlm_training = int(
        txt_key in mlm_train_keys
    )

    used_in_mlm_validation = int(
        txt_key in mlm_validation_keys
    )

    metadata = DOCUMENT_METADATA.get(
        pdf_path.name.lower(),
        {},
    )

    rows.append({

        "source":
            source,

        "filename":
            pdf_path.name,

        "document_date":
            metadata.get(
                "document_date",
                "",
            ),

        "document_type":
            metadata.get(
                "document_type",
                "",
            ),

        "pages":
            page_count,

        "characters":
            len(clean_text),

        "words":
            len(clean_text.split()),

        "bert_tokens":
            len(tokens),

        "text_file":
            str(text_path),

        "in_v03_manifest":
            in_v03_manifest,

        "used_in_mlm_training":
            used_in_mlm_training,

        "used_in_mlm_validation":
            used_in_mlm_validation,
    })


# ============================================================
# Save manifest
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fieldnames = [

    "source",
    "filename",
    "document_date",
    "document_type",

    "pages",
    "characters",
    "words",
    "bert_tokens",

    "text_file",

    "in_v03_manifest",
    "used_in_mlm_training",
    "used_in_mlm_validation",
]


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 75)
print("EXTERNAL UNSEEN DOCUMENT MANIFEST")
print("=" * 75)

for row in rows:

    print()
    print(row["filename"])

    print(
        f"  Source: "
        f"{row['source']}"
    )

    print(
        f"  Pages: "
        f"{row['pages']:,}"
    )

    print(
        f"  Words: "
        f"{row['words']:,}"
    )

    print(
        f"  BERT tokens: "
        f"{row['bert_tokens']:,}"
    )

    print(
        f"  In v0.3 manifest: "
        f"{row['in_v03_manifest']}"
    )

    print(
        f"  Used in MLM training: "
        f"{row['used_in_mlm_training']}"
    )

    print(
        f"  Used in MLM validation: "
        f"{row['used_in_mlm_validation']}"
    )


print()
print("=" * 75)

print(
    f"Documents: "
    f"{len(rows)}"
)

print(
    f"Total pages: "
    f"{sum(row['pages'] for row in rows):,}"
)

print(
    f"Total words: "
    f"{sum(row['words'] for row in rows):,}"
)

print(
    f"Total BERT tokens: "
    f"{sum(row['bert_tokens'] for row in rows):,}"
)

print()
print("Saved:")
print(OUTPUT_FILE)