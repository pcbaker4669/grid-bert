from pathlib import Path
import json
import math
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM


BASE_MODEL = "google-bert/bert-base-uncased"
GRID_MODEL = r"D:\GridBERT\models\GridBERT-v0.1\final"

VALID_FILE = Path(r"D:\GridBERT\training\validation.jsonl")

OUTPUT_DIR = Path(
    r"D:\GridBERT\experiments\GridBERT-v0.1"
)

FIXED_FILE = OUTPUT_DIR / "fixed_validation.jsonl"
RESULT_FILE = OUTPUT_DIR / "fixed_mask_results.json"

SEED = 42
MLM_PROBABILITY = 0.15
BATCH_SIZE = 2

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

random.seed(SEED)


# ---------------------------------------------------------
# Create fixed masks
# ---------------------------------------------------------

def create_fixed_dataset():

    print("Creating fixed masked validation dataset...")

    with open(VALID_FILE, "r", encoding="utf-8") as infile, \
         open(FIXED_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:

            record = json.loads(line)

            ids = record["input_ids"].copy()

            labels = [-100] * len(ids)

            candidates = [
                i for i, token_id in enumerate(ids)
                if token_id not in {
                    tokenizer.cls_token_id,
                    tokenizer.sep_token_id,
                    tokenizer.pad_token_id
                }
            ]

            random.shuffle(candidates)

            n_mask = max(
                1,
                round(len(candidates) * MLM_PROBABILITY)
            )

            selected = candidates[:n_mask]

            for i in selected:

                original = ids[i]
                labels[i] = original

                r = random.random()

                # Standard BERT MLM procedure
                if r < 0.80:
                    ids[i] = tokenizer.mask_token_id

                elif r < 0.90:
                    ids[i] = random.randrange(
                        tokenizer.vocab_size
                    )

                # remaining 10% unchanged

            outfile.write(
                json.dumps({
                    "input_ids": ids,
                    "labels": labels
                }) + "\n"
            )

    print(f"Saved: {FIXED_FILE}")


# ---------------------------------------------------------
# Load fixed dataset
# ---------------------------------------------------------

def load_dataset():

    records = []

    with open(FIXED_FILE, "r", encoding="utf-8") as f:

        for line in f:
            records.append(json.loads(line))

    return records


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate(model_name, records):

    print(f"\nLoading {model_name}")

    model = AutoModelForMaskedLM.from_pretrained(
        model_name
    )

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():

        for start in range(
            0,
            len(records),
            BATCH_SIZE
        ):

            batch = records[
                start:start + BATCH_SIZE
            ]

            max_len = max(
                len(x["input_ids"])
                for x in batch
            )

            input_ids = []
            attention_masks = []
            labels = []

            for item in batch:

                length = len(item["input_ids"])
                padding = max_len - length

                input_ids.append(
                    item["input_ids"]
                    + [tokenizer.pad_token_id] * padding
                )

                attention_masks.append(
                    [1] * length
                    + [0] * padding
                )

                labels.append(
                    item["labels"]
                    + [-100] * padding
                )

            input_ids = torch.tensor(input_ids)
            attention_masks = torch.tensor(
                attention_masks
            )
            labels = torch.tensor(labels)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks
            )

            logits = outputs.logits

            mask = labels != -100

            loss = F.cross_entropy(
                logits[mask],
                labels[mask],
                reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += mask.sum().item()

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "masked_tokens": total_tokens
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if not FIXED_FILE.exists():
    create_fixed_dataset()
else:
    print("Using existing fixed mask dataset:")
    print(FIXED_FILE)

records = load_dataset()

print(f"Validation sequences: {len(records):,}")

base_results = evaluate(
    BASE_MODEL,
    records
)

grid_results = evaluate(
    GRID_MODEL,
    records
)

results = {
    "base_bert": base_results,
    "gridbert": grid_results
}

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(results, f, indent=4)


print()
print("=" * 60)
print("FIXED MASK RESULTS")
print("=" * 60)

print(
    f"Base BERT loss:       "
    f"{base_results['loss']:.4f}"
)

print(
    f"GridBERT loss:        "
    f"{grid_results['loss']:.4f}"
)

print(
    f"Base BERT perplexity: "
    f"{base_results['perplexity']:.2f}"
)

print(
    f"GridBERT perplexity:  "
    f"{grid_results['perplexity']:.2f}"
)

print()
print(f"Results saved to:")
print(RESULT_FILE)