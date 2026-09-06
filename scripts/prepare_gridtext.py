from pathlib import Path
import json
import random

from transformers import AutoTokenizer


TEXT_DIR = Path(r"D:\GridBERT\text")
TRAIN_DIR = Path(r"D:\GridBERT\training")

TRAIN_FILE = TRAIN_DIR / "train.jsonl"
VALID_FILE = TRAIN_DIR / "validation.jsonl"

MODEL_NAME = "google-bert/bert-base-uncased"

CONTENT_LENGTH = 254
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.10

TRAIN_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

random.seed(RANDOM_SEED)


# ---------------------------------------------------------
# Find all extracted text documents
# ---------------------------------------------------------

documents = sorted(TEXT_DIR.rglob("*.txt"))
print(f"Found {len(documents)} text documents.")


# ---------------------------------------------------------
# Split DOCUMENTS, not individual chunks
#
# This prevents chunks from the same report appearing
# in both training and validation data.
# ---------------------------------------------------------

random.shuffle(documents)

validation_count = max(
    1,
    round(len(documents) * VALIDATION_FRACTION)
)

validation_documents = documents[:validation_count]
training_documents = documents[validation_count:]


print(f"Training documents: {len(training_documents)}")
print(f"Validation documents: {len(validation_documents)}")


def create_chunks(document_path):

    text = document_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    # Tokenize without adding [CLS] and [SEP] yet
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    for start in range(0, len(token_ids), CONTENT_LENGTH):

        chunk = token_ids[
            start:start + CONTENT_LENGTH
        ]

        # Ignore very tiny final fragments
        if len(chunk) < 32:
            continue

        chunk = [
            tokenizer.cls_token_id,
            *chunk,
            tokenizer.sep_token_id
        ]

        chunks.append({
            "input_ids": chunk,
            "attention_mask": [1] * len(chunk),
            "source_file": str(
                document_path.relative_to(TEXT_DIR)
            )
        })

    return chunks


def write_dataset(documents, output_file):

    total_chunks = 0

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as outfile:

        for document in documents:

            print(f"Processing: {document.name}")

            chunks = create_chunks(document)

            for chunk in chunks:

                outfile.write(
                    json.dumps(chunk) + "\n"
                )

                total_chunks += 1

    return total_chunks


print("\nCreating training dataset...")

train_chunks = write_dataset(
    training_documents,
    TRAIN_FILE
)

print("\nCreating validation dataset...")

validation_chunks = write_dataset(
    validation_documents,
    VALID_FILE
)


print("\nFinished.")
print(f"Training chunks: {train_chunks:,}")
print(f"Validation chunks: {validation_chunks:,}")
print(f"Training file: {TRAIN_FILE}")
print(f"Validation file: {VALID_FILE}")