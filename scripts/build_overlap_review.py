import csv
from pathlib import Path

from config_loader import load_config


# ============================================================
# Configuration
# ============================================================

config = load_config()

DATA_ROOT = Path(
    config["paths"]["data_root"]
)

CANDIDATE_FILE = (
    DATA_ROOT
    / config["classification"]["external_candidates_file"]
)

OVERLAP_FILE = (
    DATA_ROOT
    / config["classification"]["external_overlap_file"]
)

OUTPUT_FILE = (
    DATA_ROOT
    / "classification"
    / "external_test"
    / "provenance"
    / "overlap_review.csv"
)


# ============================================================
# Load candidate text
# ============================================================

candidate_text = {}

with open(
    CANDIDATE_FILE,
    "r",
    encoding="utf-8-sig",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        candidate_text[row["id"]] = row["text"]


# ============================================================
# Load flagged overlap records
# ============================================================

review_rows = []

with open(
    OVERLAP_FILE,
    "r",
    encoding="utf-8-sig",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row["review_status"] == "CLEAR":
            continue

        review_rows.append({

            "id":
                row["id"],

            "source":
                row["source"],

            "source_document":
                row["source_document"],

            "candidate_text":
                candidate_text.get(
                    row["id"],
                    "",
                ),

            "review_status":
                row["review_status"],

            "near_duplicate_score":
                row["near_duplicate_score"],

            "matching_5grams":
                row["matching_5grams"],

            "candidate_5grams":
                row["candidate_5grams"],

            "closest_v03_document":
                row["closest_v03_document"],

            "closest_v03_text":
                row["closest_v03_text"],

            "decision":
                "",

            "notes":
                "",
        })


# ============================================================
# Save review file
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fieldnames = [

    "id",
    "source",
    "source_document",

    "candidate_text",

    "review_status",
    "near_duplicate_score",
    "matching_5grams",
    "candidate_5grams",

    "closest_v03_document",
    "closest_v03_text",

    "decision",
    "notes",
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
    writer.writerows(review_rows)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("EXTERNAL OVERLAP REVIEW")
print("=" * 70)

print(
    f"Flagged passages: "
    f"{len(review_rows)}"
)

print()

for row in review_rows:

    print(
        f"ID {row['id']}: "
        f"{row['review_status']} "
        f"(score={row['near_duplicate_score']})"
    )

    print(
        f"  External: "
        f"{row['source_document']}"
    )

    print(
        f"  Closest v0.3: "
        f"{row['closest_v03_document']}"
    )

    print()


print("Saved:")
print(OUTPUT_FILE)