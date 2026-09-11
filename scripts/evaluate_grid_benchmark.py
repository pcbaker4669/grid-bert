import json
import csv
from pathlib import Path
from collections import defaultdict
from config_loader import load_config, data_path
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


config = load_config()

BASE_MODEL = config["models"]["reference_model"]

GRID_MODEL = str(
    data_path(config, "trained_model_dir")
)

BENCHMARK_FILE = data_path(
    config,
    "benchmark_file"
)

CSV_FILE = data_path(
    config,
    "benchmark_results_csv"
)

JSON_FILE = data_path(
    config,
    "benchmark_summary_json"
)


# ---------------------------------------------------------
# Load benchmark
# ---------------------------------------------------------

def load_benchmark():
    records = []

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


# ---------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------

def evaluate_model(model_name, benchmark, tokenizer):

    print(f"\nLoading model: {model_name}")

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    results = []

    with torch.no_grad():

        for i, item in enumerate(benchmark, start=1):

            prompt = item["prompt"]
            expected = item["expected"]
            category = item["category"]

            # Verify expected answer is one BERT token
            expected_ids = tokenizer.encode(
                expected,
                add_special_tokens=False
            )

            if len(expected_ids) != 1:
                raise ValueError(
                    f"Expected word '{expected}' is not a "
                    f"single BERT token: {expected_ids}"
                )

            expected_id = expected_ids[0]

            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            )

            mask_positions = (
                inputs["input_ids"] == tokenizer.mask_token_id
            ).nonzero(as_tuple=True)

            if len(mask_positions[1]) != 1:
                raise ValueError(
                    f"Prompt must contain exactly one [MASK]: {prompt}"
                )

            mask_index = mask_positions[1].item()

            outputs = model(**inputs)

            logits = outputs.logits[
                0,
                mask_index
            ]

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            # Sort entire vocabulary by probability
            sorted_ids = torch.argsort(
                probabilities,
                descending=True
            )

            expected_position = (
                sorted_ids == expected_id
            ).nonzero(as_tuple=True)[0].item()

            rank = expected_position + 1

            expected_probability = (
                probabilities[expected_id].item()
            )

            top5_ids = sorted_ids[:5].tolist()

            top5_tokens = tokenizer.convert_ids_to_tokens(
                top5_ids
            )

            top5_probs = [
                probabilities[token_id].item()
                for token_id in top5_ids
            ]

            results.append({
                "category": category,
                "prompt": prompt,
                "expected": expected,
                "rank": rank,
                "hit_at_1": int(rank == 1),
                "hit_at_5": int(rank <= 5),
                "reciprocal_rank": 1.0 / rank,
                "expected_probability": expected_probability,
                "top1": top5_tokens[0],
                "top1_probability": top5_probs[0],
                "top5": top5_tokens,
                "top5_probabilities": top5_probs
            })

            print(
                f"{i:2d}/50  "
                f"rank={rank:5d}  "
                f"expected={expected:15s}  "
                f"top1={top5_tokens[0]}"
            )

    return results


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

def summarize(results):

    n = len(results)

    overall = {
        "count": n,
        "hits_at_1": sum(
            r["hit_at_1"] for r in results
        ) / n,
        "hits_at_5": sum(
            r["hit_at_5"] for r in results
        ) / n,
        "mrr": sum(
            r["reciprocal_rank"] for r in results
        ) / n
    }

    category_groups = defaultdict(list)

    for result in results:
        category_groups[result["category"]].append(result)

    categories = {}

    for category, group in category_groups.items():

        count = len(group)

        categories[category] = {
            "count": count,
            "hits_at_1": sum(
                r["hit_at_1"] for r in group
            ) / count,
            "hits_at_5": sum(
                r["hit_at_5"] for r in group
            ) / count,
            "mrr": sum(
                r["reciprocal_rank"] for r in group
            ) / count
        }

    return {
        "overall": overall,
        "categories": categories
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

benchmark = load_benchmark()

print(f"Loaded {len(benchmark)} benchmark prompts.")

base_results = evaluate_model(
    BASE_MODEL,
    benchmark,
    tokenizer
)

grid_results = evaluate_model(
    GRID_MODEL,
    benchmark,
    tokenizer
)

base_summary = summarize(base_results)
grid_summary = summarize(grid_results)


# ---------------------------------------------------------
# Save detailed CSV
# ---------------------------------------------------------

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "category",
        "prompt",
        "expected",

        "base_rank",
        "base_hit1",
        "base_hit5",
        "base_probability",
        "base_top1",

        "grid_rank",
        "grid_hit1",
        "grid_hit5",
        "grid_probability",
        "grid_top1"
    ])

    for base, grid in zip(
        base_results,
        grid_results
    ):

        writer.writerow([
            base["category"],
            base["prompt"],
            base["expected"],

            base["rank"],
            base["hit_at_1"],
            base["hit_at_5"],
            base["expected_probability"],
            base["top1"],

            grid["rank"],
            grid["hit_at_1"],
            grid["hit_at_5"],
            grid["expected_probability"],
            grid["top1"]
        ])


# ---------------------------------------------------------
# Save summary JSON
# ---------------------------------------------------------

summary = {
    "benchmark_prompts": len(benchmark),
    "base_bert": base_summary,
    "gridbert": grid_summary
}

with open(
    JSON_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print()
print("=" * 65)
print("GRID DOMAIN BENCHMARK RESULTS")
print("=" * 65)

print("\nOVERALL")

print(
    f"{'Metric':<12}"
    f"{'Base BERT':>12}"
    f"{'GridBERT':>12}"
)

print(
    f"{'Hits@1':<12}"
    f"{base_summary['overall']['hits_at_1']:>12.3f}"
    f"{grid_summary['overall']['hits_at_1']:>12.3f}"
)

print(
    f"{'Hits@5':<12}"
    f"{base_summary['overall']['hits_at_5']:>12.3f}"
    f"{grid_summary['overall']['hits_at_5']:>12.3f}"
)

print(
    f"{'MRR':<12}"
    f"{base_summary['overall']['mrr']:>12.3f}"
    f"{grid_summary['overall']['mrr']:>12.3f}"
)


print("\nBY CATEGORY")

print(
    f"{'Category':<20}"
    f"{'Base H@1':>10}"
    f"{'Grid H@1':>10}"
    f"{'Base H@5':>10}"
    f"{'Grid H@5':>10}"
)

for category in base_summary["categories"]:

    b = base_summary["categories"][category]
    g = grid_summary["categories"][category]

    print(
        f"{category:<20}"
        f"{b['hits_at_1']:>10.2f}"
        f"{g['hits_at_1']:>10.2f}"
        f"{b['hits_at_5']:>10.2f}"
        f"{g['hits_at_5']:>10.2f}"
    )


print()
print(f"Detailed results: {CSV_FILE}")
print(f"Summary results:  {JSON_FILE}")