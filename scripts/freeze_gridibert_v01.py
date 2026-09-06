from pathlib import Path
import hashlib
import json
from datetime import datetime

ROOT = Path(r"D:\GridBERT")
EXP_DIR = ROOT / "experiments" / "GridBERT-v0.1"
EXP_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = ROOT / "training" / "train.jsonl"
VALID_FILE = ROOT / "training" / "validation.jsonl"
CORPUS_MANIFEST = ROOT / "gridtext_manifest.csv"
MODEL_DIR = ROOT / "models" / "GridBERT-v0.1" / "final"


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)

    return h.hexdigest()


def source_documents(jsonl_file):
    sources = set()

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if "source_file" in record:
                sources.add(record["source_file"])

    return sorted(sources)


manifest = {
    "experiment": "GridBERT-v0.1",
    "created": datetime.now().isoformat(),

    "base_model": "google-bert/bert-base-uncased",

    "corpus": {
        "documents": 65,
        "bert_tokens": 3958771
    },

    "dataset": {
        "sequence_length": 256,
        "content_tokens": 254,

        "train_documents": 59,
        "validation_documents": 6,

        "train_sequences": 14265,
        "validation_sequences": 1343,

        "train_sources": source_documents(TRAIN_FILE),
        "validation_sources": source_documents(VALID_FILE),

        "train_sha256": sha256(TRAIN_FILE),
        "validation_sha256": sha256(VALID_FILE)
    },

    "training": {
        "epochs": 3,
        "learning_rate": 2e-5,
        "mlm_probability": 0.15,
        "weight_decay": 0.01,
        "seed": 42
    },

    "results": {
        "base_validation_loss": 2.6675,
        "base_perplexity": 14.40,

        "gridbert_validation_loss": 1.4490,
        "gridbert_perplexity": 4.26
    },

    "paths": {
        "model": str(MODEL_DIR),
        "train": str(TRAIN_FILE),
        "validation": str(VALID_FILE)
    }
}

if CORPUS_MANIFEST.exists():
    manifest["corpus"]["manifest_sha256"] = sha256(CORPUS_MANIFEST)

output = EXP_DIR / "experiment_manifest.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)

print(f"Experiment frozen:")
print(output)