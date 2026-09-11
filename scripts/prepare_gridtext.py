from pathlib import Path
import json
import random
from config_loader import load_config, data_path
from transformers import AutoTokenizer

config = load_config()

TEXT_DIR = data_path(config, "text_dir")
TRAIN_DIR = data_path(config, "training_dir")

TRAIN_FILE = data_path(config, "train_file")
VALID_FILE = data_path(config, "validation_file")

MODEL_NAME = config["models"]["tokenizer_model"]

CONTENT_LENGTH = config["dataset"]["content_length"]
MIN_FINAL_FRAGMENT = config["dataset"]["min_final_fragment"]
RANDOM_SEED = config["dataset"]["seed"]
VALIDATION_FRACTION = config["dataset"]["validation_fraction"]


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
# ---------------------------------------------------------

FIXED_VALIDATION = config["dataset"].get(
    "fixed_validation_documents",
    []
)

print("Fixed validation documents configured:", len(FIXED_VALIDATION))
print(FIXED_VALIDATION)

if FIXED_VALIDATION:

    # Map relative paths to actual document paths
    document_map = {
        document.relative_to(TEXT_DIR).as_posix(): document
        for document in documents
    }

    # Make sure every configured validation document exists
    missing = [
        path
        for path in FIXED_VALIDATION
        if path not in document_map
    ]

    if missing:
        raise FileNotFoundError(
            "Configured validation documents not found:\n"
            + "\n".join(missing)
        )

    validation_documents = [
        document_map[path]
        for path in FIXED_VALIDATION
    ]

    validation_set = set(validation_documents)

    training_documents = [
        document
        for document in documents
        if document not in validation_set
    ]

else:

    # Fallback to reproducible random document split
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
        if len(chunk) < MIN_FINAL_FRAGMENT:
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