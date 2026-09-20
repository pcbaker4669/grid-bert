import csv
import random
import re
from pathlib import Path

from config_loader import load_config, data_path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

config = load_config()

TEXT_ROOT = data_path(config, "text_dir")
DATA_ROOT = Path(config["paths"]["data_root"])

OUTPUT_FILE = (
    DATA_ROOT
    / "classification"
    / "data"
    / "reliability_candidates.csv"
)

RANDOM_SEED = 42
TARGET_CANDIDATES = 500

# Approximate mix of candidate types
TARGET_RELIABILITY_SIGNAL = 300
TARGET_GENERAL_GRID = 200

RELIABILITY_TERMS = [ 
    "reliability", "reliable", "resource adequacy", "reserve margin", "capacity shortfall",
    "capacity shortage", "generation shortage", "thermal overload", "overloaded",
    "thermal violation", "voltage violation", "low voltage", "voltage stability", 
    "transient stability", "contingency", "n-1", "n-1-1", "loss of load", "load shed",
    "load shedding", "outage", "emergency", "operating reserve", "transmission constraint",
    "transfer capability", "deliverability", "generator retirement", "generation retirement",
    "resource retirement", "corrective action plan",
]

GRID_TERMS = [
    "transmission", "generation", "generator", "capacity", "load", "demand", "market",
    "congestion", "dispatch", "reserve", "interconnection", "substation", "transformer",
    "power flow", "electric grid", "bulk electric system", "rto", "iso", "ferc", "nerc",
    "pjm", "miso", "caiso", "spp", "iso-ne",
]

# ---------------------------------------------------------
# Text processing
# ---------------------------------------------------------
def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    """
    Simple sentence splitter suitable for the already-cleaned
    GridText files. No additional NLP package is required.
    """
    text = normalize_whitespace(text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def make_passages(sentences):
    """
    Build passages of one or two sentences.

    Short passages provide more context than isolated sentences
    while remaining well below BERT's input limit.
    """

    passages = []

    for i, sentence in enumerate(sentences):
        # Single sentence
        if 80 <= len(sentence) <= 700:
            passages.append(sentence)

        # Two-sentence passage
        if i + 1 < len(sentences):
            passage = sentence + " " + sentences[i + 1]
            if 120 <= len(passage) <= 1000:
                passages.append(passage)

    return passages

def contains_term(text, terms):
    lower_text = text.lower()
    return any(term in lower_text for term in terms)

# ---------------------------------------------------------
# Read GridText corpus
# ---------------------------------------------------------

rng = random.Random(RANDOM_SEED)

reliability_candidates = []
general_candidates = []
seen_text = set()


for text_file in TEXT_ROOT.rglob("*.txt"):
    relative_path = text_file.relative_to(TEXT_ROOT)

    if len(relative_path.parts) > 1:
        source = relative_path.parts[0]
    else:
        source = "unknown"

    try:
        text = text_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"Unable to read {text_file}: {exc}")
        continue

    sentences = split_sentences(text)
    passages = make_passages(sentences)

    for passage in passages:
        normalized = normalize_whitespace(passage)

        # Remove duplicate passages
        duplicate_key = normalized.lower()

        if duplicate_key in seen_text:
            continue

        seen_text.add(duplicate_key)

        record = {"source": source, "source_document": str(relative_path),"text": normalized}

        if contains_term(normalized, RELIABILITY_TERMS):
            reliability_candidates.append(record)

        elif contains_term(normalized, GRID_TERMS):
            general_candidates.append(record)

  # ---------------------------------------------------------
# Sample candidate passages
# ---------------------------------------------------------

rng.shuffle(reliability_candidates)
rng.shuffle(general_candidates)

selected_reliability = reliability_candidates[:TARGET_RELIABILITY_SIGNAL]
selected_general = general_candidates[:TARGET_GENERAL_GRID]
selected = (selected_reliability + selected_general)

rng.shuffle(selected)     


# ---------------------------------------------------------
# Write annotation CSV
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

fieldnames = ["id", "source", "source_document", "text", "label", "notes"]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()

    for index, record in enumerate(selected, start=1):
        writer.writerow({
            "id": index,
            "source": record["source"],
            "source_document": record["source_document"],
            "text": record["text"],
            "label": "",
            "notes": "",
        })

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("=" * 60)
print("RELIABILITY CLASSIFICATION CANDIDATES")
print("=" * 60)

print(
    f"Reliability-signal pool: "
    f"{len(reliability_candidates):,}"
)

print(
    f"General-grid pool:       "
    f"{len(general_candidates):,}"
)

print(
    f"Candidates selected:     "
    f"{len(selected):,}"
)

print()
print(f"Saved to:")
print(OUTPUT_FILE)