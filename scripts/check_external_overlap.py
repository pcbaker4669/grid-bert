"""
check_external_overlap.py

Purpose
-------
Checks external-test candidate passages against the frozen
GridText v0.3 corpus for possible textual overlap.

The script performs two checks:

1. Exact normalized word match
   Determines whether the complete external passage appears
   word-for-word in a GridText v0.3 document.

2. Near-duplicate screening
   Breaks each external passage into 5-word shingles and
   determines what proportion of those shingles occur in the
   same GridText v0.3 document.

No candidates are automatically removed.

Results are written to overlap_check.csv for manual review.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

from config_loader import load_config


# ============================================================
# Configuration
# ============================================================

config = load_config()

DATA_ROOT = Path(
    config["paths"]["data_root"]
)

GRIDTEXT_ROOT = (
    DATA_ROOT
    / config["paths"]["text_dir"]
)

CANDIDATE_FILE = (
    DATA_ROOT
    / config["classification"][
        "external_candidates_file"
    ]
)

OUTPUT_FILE = (
    DATA_ROOT
    / config["classification"][
        "external_overlap_file"
    ]
)


# ============================================================
# Similarity settings
# ============================================================

SHINGLE_SIZE = 5

# 80% or more of the candidate's 5-word shingles occurring
# in one v0.3 document is considered strong near duplication.
NEAR_DUPLICATE_THRESHOLD = 0.80

# 50% to 80% is retained but flagged for manual inspection.
POSSIBLE_OVERLAP_THRESHOLD = 0.50


# ============================================================
# Text normalization
# ============================================================

def tokenize_words(text):
    """
    Lowercase word tokenization used for both the external
    candidates and GridText v0.3.

    Punctuation differences are ignored.
    """

    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?",
        text.lower(),
    )


def make_shingles(words, size=SHINGLE_SIZE):

    if len(words) < size:
        return []

    return [
        tuple(words[i:i + size])
        for i in range(
            len(words) - size + 1
        )
    ]


def make_context(text, seed_phrase, width=450):
    """
    Extract a short portion of the closest GridText document
    around one matching 5-word shingle.
    """

    position = text.find(
        seed_phrase
    )

    if position == -1:
        return ""

    start = max(
        0,
        position - width,
    )

    end = min(
        len(text),
        position
        + len(seed_phrase)
        + width,
    )

    return text[start:end]


# ============================================================
# Load external candidates
# ============================================================

candidates = []

with open(
    CANDIDATE_FILE,
    "r",
    encoding="utf-8-sig",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        words = tokenize_words(
            row["text"]
        )

        word_string = " ".join(
            words
        )

        shingle_list = make_shingles(
            words
        )

        shingle_set = set(
            shingle_list
        )

        candidates.append({

            "id":
                row["id"],

            "source":
                row["source"],

            "source_document":
                row["source_document"],

            "text":
                row["text"],

            "word_string":
                word_string,

            "shingle_list":
                shingle_list,

            "shingle_set":
                shingle_set,

            "exact_match":
                False,

            "exact_match_document":
                "",

            "best_score":
                0.0,

            "best_match_count":
                0,

            "best_document":
                "",

            "best_context":
                "",
        })


print()
print(
    f"Loaded {len(candidates)} external candidates."
)


# ============================================================
# Build candidate shingle index
# ============================================================

shingle_to_candidates = defaultdict(
    set
)

for index, candidate in enumerate(
    candidates
):

    for shingle in candidate[
        "shingle_set"
    ]:

        shingle_to_candidates[
            shingle
        ].add(index)


print(
    f"Unique candidate 5-word shingles: "
    f"{len(shingle_to_candidates):,}"
)


# ============================================================
# Find GridText documents
# ============================================================

gridtext_files = sorted(
    GRIDTEXT_ROOT.rglob("*.txt")
)

print(
    f"GridText v0.3 documents: "
    f"{len(gridtext_files)}"
)

print()
print(
    "Checking external candidates "
    "against GridText v0.3..."
)


# ============================================================
# Scan GridText documents
# ============================================================

for document_number, text_file in enumerate(
    gridtext_files,
    start=1,
):

    relative_path = (
        text_file
        .relative_to(GRIDTEXT_ROOT)
        .as_posix()
    )

    print(
        f"[{document_number:03d}/"
        f"{len(gridtext_files):03d}] "
        f"{relative_path}"
    )

    try:

        text = text_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception as exc:

        print(
            f"  Unable to read: {exc}"
        )

        continue


    # --------------------------------------------------------
    # Normalize document
    # --------------------------------------------------------

    document_words = tokenize_words(
        text
    )

    document_word_string = " ".join(
        document_words
    )


    # --------------------------------------------------------
    # Exact normalized-word matches
    # --------------------------------------------------------

    for candidate in candidates:

        if candidate["exact_match"]:
            continue

        candidate_string = (
            candidate["word_string"]
        )

        if (
            candidate_string
            and candidate_string
            in document_word_string
        ):

            candidate[
                "exact_match"
            ] = True

            candidate[
                "exact_match_document"
            ] = relative_path

            candidate[
                "best_score"
            ] = 1.0

            candidate[
                "best_match_count"
            ] = len(
                candidate[
                    "shingle_set"
                ]
            )

            candidate[
                "best_document"
            ] = relative_path

            candidate[
                "best_context"
            ] = make_context(
                document_word_string,
                candidate_string[
                    :min(
                        len(candidate_string),
                        150,
                    )
                ],
            )


    # --------------------------------------------------------
    # Find candidate shingles contained in this document
    # --------------------------------------------------------

    found_by_candidate = defaultdict(
        set
    )

    if len(document_words) >= SHINGLE_SIZE:

        for i in range(
            len(document_words)
            - SHINGLE_SIZE
            + 1
        ):

            shingle = tuple(
                document_words[
                    i:i + SHINGLE_SIZE
                ]
            )

            candidate_indexes = (
                shingle_to_candidates.get(
                    shingle
                )
            )

            if not candidate_indexes:
                continue

            for candidate_index in (
                candidate_indexes
            ):

                found_by_candidate[
                    candidate_index
                ].add(shingle)


    # --------------------------------------------------------
    # Calculate containment score
    # --------------------------------------------------------

    for candidate_index, matched_shingles in (
        found_by_candidate.items()
    ):

        candidate = candidates[
            candidate_index
        ]

        # Exact match already has maximum score.
        if candidate["exact_match"]:
            continue

        candidate_shingles = candidate[
            "shingle_set"
        ]

        if not candidate_shingles:
            continue

        score = (
            len(matched_shingles)
            / len(candidate_shingles)
        )

        if score > candidate[
            "best_score"
        ]:

            candidate[
                "best_score"
            ] = score

            candidate[
                "best_match_count"
            ] = len(
                matched_shingles
            )

            candidate[
                "best_document"
            ] = relative_path


            # Find a matching shingle in candidate order
            seed_shingle = None

            for shingle in candidate[
                "shingle_list"
            ]:

                if shingle in matched_shingles:

                    seed_shingle = shingle
                    break


            if seed_shingle:

                seed_phrase = " ".join(
                    seed_shingle
                )

                candidate[
                    "best_context"
                ] = make_context(
                    document_word_string,
                    seed_phrase,
                )


# ============================================================
# Assign review status
# ============================================================

for candidate in candidates:

    score = candidate[
        "best_score"
    ]

    if candidate[
        "exact_match"
    ]:

        status = "EXACT_MATCH"

    elif (
        score
        >= NEAR_DUPLICATE_THRESHOLD
    ):

        status = (
            "REVIEW_NEAR_DUPLICATE"
        )

    elif (
        score
        >= POSSIBLE_OVERLAP_THRESHOLD
    ):

        status = (
            "REVIEW_POSSIBLE_OVERLAP"
        )

    else:

        status = "CLEAR"

    candidate[
        "review_status"
    ] = status


# ============================================================
# Save overlap report
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fieldnames = [

    "id",
    "source",
    "source_document",

    "exact_match",
    "near_duplicate_score",

    "matching_5grams",
    "candidate_5grams",

    "closest_v03_document",
    "closest_v03_text",

    "review_status",
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

    for candidate in candidates:

        writer.writerow({

            "id":
                candidate["id"],

            "source":
                candidate["source"],

            "source_document":
                candidate[
                    "source_document"
                ],

            "exact_match":
                int(
                    candidate[
                        "exact_match"
                    ]
                ),

            "near_duplicate_score":
                round(
                    candidate[
                        "best_score"
                    ],
                    4,
                ),

            "matching_5grams":
                candidate[
                    "best_match_count"
                ],

            "candidate_5grams":
                len(
                    candidate[
                        "shingle_set"
                    ]
                ),

            "closest_v03_document":
                candidate[
                    "best_document"
                ],

            "closest_v03_text":
                candidate[
                    "best_context"
                ],

            "review_status":
                candidate[
                    "review_status"
                ],
        })


# ============================================================
# Summary
# ============================================================

exact_count = sum(
    candidate["review_status"]
    == "EXACT_MATCH"
    for candidate in candidates
)

near_count = sum(
    candidate["review_status"]
    == "REVIEW_NEAR_DUPLICATE"
    for candidate in candidates
)

possible_count = sum(
    candidate["review_status"]
    == "REVIEW_POSSIBLE_OVERLAP"
    for candidate in candidates
)

clear_count = sum(
    candidate["review_status"]
    == "CLEAR"
    for candidate in candidates
)


print()
print("=" * 78)
print("EXTERNAL PASSAGE OVERLAP CHECK")
print("=" * 78)

print(
    f"External candidates:          "
    f"{len(candidates):>4}"
)

print(
    f"Exact matches:                "
    f"{exact_count:>4}"
)

print(
    f"Near duplicates >= 0.80:     "
    f"{near_count:>4}"
)

print(
    f"Possible overlap 0.50-0.80:  "
    f"{possible_count:>4}"
)

print(
    f"Clear < 0.50:                 "
    f"{clear_count:>4}"
)

print()
print("Saved:")
print(OUTPUT_FILE)