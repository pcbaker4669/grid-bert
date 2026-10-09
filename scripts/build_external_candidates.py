"""
build_external_candidates.py

Purpose
-------
Build candidate passages for the completely external,
unseen-document evaluation of the GridBERT reliability
classification experiment.

The primary candidate-generation procedure is intentionally
consistent with build_reliability_candidates.py.

For slide-oriented documents that produce fewer than 50
usable sentence-based candidates, a bullet-text fallback is
also used.

Candidate-selection keywords are used only to construct the
annotation sample. They do NOT determine ground-truth labels.
"""

import csv
import random
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

TEXT_ROOT = (
    DATA_ROOT
    / config["classification"]["external_text_dir"]
)

OUTPUT_FILE = (
    DATA_ROOT
    / config["classification"]["external_candidates_file"]
)

PROVENANCE_FILE = (
    DATA_ROOT
    / "classification"
    / "external_test"
    / "provenance"
    / "external_candidate_selection.csv"
)


RANDOM_SEED = 42

TARGET_CANDIDATES = 125

# Initial target per document
TARGET_PER_DOCUMENT = 25

# Approximate 60/40 candidate mix
TARGET_RELIABILITY_PER_DOCUMENT = 15
TARGET_GENERAL_PER_DOCUMENT = 10


# ============================================================
# Candidate-selection terms
# ============================================================

RELIABILITY_TERMS = [

    "reliability",
    "reliable",
    "resource adequacy",
    "reserve margin",
    "capacity shortfall",
    "capacity shortage",
    "generation shortage",
    "thermal overload",
    "overloaded",
    "thermal violation",
    "voltage violation",
    "low voltage",
    "voltage stability",
    "transient stability",
    "contingency",
    "n-1",
    "n-1-1",
    "loss of load",
    "load shed",
    "load shedding",
    "outage",
    "emergency",
    "operating reserve",
    "transmission constraint",
    "transfer capability",
    "deliverability",
    "generator retirement",
    "generation retirement",
    "resource retirement",
    "corrective action plan",
]


GRID_TERMS = [

    "transmission",
    "generation",
    "generator",
    "capacity",
    "load",
    "demand",
    "market",
    "congestion",
    "dispatch",
    "reserve",
    "interconnection",
    "substation",
    "transformer",
    "power flow",
    "electric grid",
    "bulk electric system",
    "rto",
    "iso",
    "ferc",
    "nerc",
    "pjm",
    "miso",
    "caiso",
    "spp",
    "iso-ne",
]


# ============================================================
# Text processing
# ============================================================

def normalize_whitespace(text):

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def split_sentences(text):
    """
    Same basic sentence splitter used by the original
    reliability candidate script.
    """

    text = normalize_whitespace(
        text
    )

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def make_passages(sentences):
    """
    Same one- and two-sentence passage construction used
    by the original reliability candidate script.
    """

    passages = []

    for i, sentence in enumerate(
        sentences
    ):

        # Single sentence
        if 80 <= len(sentence) <= 700:

            passages.append({
                "text": sentence,
                "passage_method": "sentence",
            })

        # Two-sentence passage
        if i + 1 < len(sentences):

            passage = (
                sentence
                + " "
                + sentences[i + 1]
            )

            if 120 <= len(passage) <= 1000:

                passages.append({
                    "text": passage,
                    "passage_method":
                        "two_sentence",
                })

    return passages


# ============================================================
# Slide / bullet fallback
# ============================================================

def is_noise_line(line):

    text = line.strip()
    lower = text.lower()

    if not text:
        return True

    if re.fullmatch(r"\d+", text):
        return True

    if "www.pjm.com" in lower:
        return True

    if lower.startswith("pjm ©"):
        return True

    if lower == "iso public":
        return True

    return False


def make_bullet_passages(text):
    """
    Fallback for slide-oriented PDFs.

    Wrapped bullet lines are combined into passages.
    This is used only when the normal sentence method
    generates fewer than 50 grid-related candidates.
    """

    passages = []

    current = []

    def flush():

        nonlocal current

        if not current:
            return

        passage = normalize_whitespace(
            " ".join(current)
        )

        if 80 <= len(passage) <= 1000:

            passages.append({
                "text": passage,
                "passage_method":
                    "bullet_fallback",
            })

        current = []

    for raw_line in text.splitlines():

        line = normalize_whitespace(
            raw_line
        )

        if is_noise_line(line):
            continue

        # Start of a new bullet
        if re.match(
            r"^[•▪◦–—]\s*",
            line,
        ):

            flush()

            line = re.sub(
                r"^[•▪◦–—]\s*",
                "",
                line,
            )

            current = [line]

        elif current:

            # Wrapped continuation of current bullet
            current.append(line)

    flush()

    return passages


def contains_term(text, terms):

    lower_text = text.lower()

    return any(
        term in lower_text
        for term in terms
    )


# ============================================================
# Build candidate pools by document
# ============================================================

rng = random.Random(
    RANDOM_SEED
)

document_pools = {}

global_seen_text = set()


text_files = sorted(
    TEXT_ROOT.rglob("*.txt")
)

print()
print(
    f"Found {len(text_files)} external text files."
)


for text_file in text_files:

    relative_path = (
        text_file.relative_to(
            TEXT_ROOT
        )
    )

    if len(relative_path.parts) > 1:

        source = (
            relative_path.parts[0]
        )

    else:

        source = "unknown"

    text = text_file.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    sentences = split_sentences(
        text
    )

    raw_passages = make_passages(
        sentences
    )

    reliability_candidates = []
    general_candidates = []

    local_seen = set()

    # ----------------------------------------
    # Normal sentence-based passages
    # ----------------------------------------

    for passage_record in raw_passages:

        normalized = normalize_whitespace(
            passage_record["text"]
        )

        duplicate_key = (
            normalized.lower()
        )

        if duplicate_key in local_seen:
            continue

        local_seen.add(
            duplicate_key
        )

        if contains_term(
            normalized,
            RELIABILITY_TERMS,
        ):

            reliability_candidates.append({
                "source": source,
                "source_document":
                    relative_path.as_posix(),
                "text": normalized,
                "candidate_pool":
                    "RELIABILITY_SIGNAL",
                "passage_method":
                    passage_record[
                        "passage_method"
                    ],
            })

        elif contains_term(
            normalized,
            GRID_TERMS,
        ):

            general_candidates.append({
                "source": source,
                "source_document":
                    relative_path.as_posix(),
                "text": normalized,
                "candidate_pool":
                    "GENERAL_GRID",
                "passage_method":
                    passage_record[
                        "passage_method"
                    ],
            })


    # ----------------------------------------
    # Bullet fallback for slide-oriented docs
    # ----------------------------------------

    normal_candidate_count = (
        len(reliability_candidates)
        + len(general_candidates)
    )

    if normal_candidate_count < 50:

        bullet_passages = (
            make_bullet_passages(text)
        )

        for passage_record in bullet_passages:

            normalized = (
                normalize_whitespace(
                    passage_record["text"]
                )
            )

            duplicate_key = (
                normalized.lower()
            )

            if duplicate_key in local_seen:
                continue

            local_seen.add(
                duplicate_key
            )

            if contains_term(
                normalized,
                RELIABILITY_TERMS,
            ):

                reliability_candidates.append({
                    "source":
                        source,

                    "source_document":
                        relative_path.as_posix(),

                    "text":
                        normalized,

                    "candidate_pool":
                        "RELIABILITY_SIGNAL",

                    "passage_method":
                        "bullet_fallback",
                })

            elif contains_term(
                normalized,
                GRID_TERMS,
            ):

                general_candidates.append({
                    "source":
                        source,

                    "source_document":
                        relative_path.as_posix(),

                    "text":
                        normalized,

                    "candidate_pool":
                        "GENERAL_GRID",

                    "passage_method":
                        "bullet_fallback",
                })


    # ----------------------------------------
    # Remove cross-document exact duplicates
    # ----------------------------------------

    clean_reliability = []
    clean_general = []

    for record in (
        reliability_candidates
        + general_candidates
    ):

        duplicate_key = (
            record["text"].lower()
        )

        if duplicate_key in global_seen_text:
            continue

        global_seen_text.add(
            duplicate_key
        )

        if (
            record["candidate_pool"]
            == "RELIABILITY_SIGNAL"
        ):

            clean_reliability.append(
                record
            )

        else:

            clean_general.append(
                record
            )


    rng.shuffle(
        clean_reliability
    )

    rng.shuffle(
        clean_general
    )

    document_pools[
        relative_path.as_posix()
    ] = {

        "reliability":
            clean_reliability,

        "general":
            clean_general,
    }


# ============================================================
# Initial document-balanced selection
# ============================================================

selected = []

remaining = {}


for document_name in sorted(
    document_pools
):

    reliability_pool = (
        document_pools[
            document_name
        ]["reliability"]
    )

    general_pool = (
        document_pools[
            document_name
        ]["general"]
    )


    selected_reliability = (
        reliability_pool[
            :TARGET_RELIABILITY_PER_DOCUMENT
        ]
    )

    selected_general = (
        general_pool[
            :TARGET_GENERAL_PER_DOCUMENT
        ]
    )


    document_selected = (
        selected_reliability
        + selected_general
    )


    remaining_reliability = (
        reliability_pool[
            len(selected_reliability):
        ]
    )

    remaining_general = (
        general_pool[
            len(selected_general):
        ]
    )


    # ----------------------------------------
    # Fill document to approximately 25
    # if one candidate pool is small
    # ----------------------------------------

    needed = (
        TARGET_PER_DOCUMENT
        - len(document_selected)
    )

    fill_pool = (
        remaining_reliability
        + remaining_general
    )

    rng.shuffle(fill_pool)

    additional = (
        fill_pool[:needed]
    )

    document_selected.extend(
        additional
    )


    additional_ids = {
        id(record)
        for record in additional
    }

    remaining_reliability = [
        record
        for record
        in remaining_reliability
        if id(record)
        not in additional_ids
    ]

    remaining_general = [
        record
        for record
        in remaining_general
        if id(record)
        not in additional_ids
    ]


    selected.extend(
        document_selected
    )


    remaining[document_name] = {

        "reliability":
            remaining_reliability,

        "general":
            remaining_general,
    }


# ============================================================
# Fill remaining slots while favoring reliability candidates
# ============================================================

TARGET_RELIABILITY_TOTAL = round(
    TARGET_CANDIDATES * 0.60
)


def count_reliability(records):

    return sum(
        record["candidate_pool"]
        == "RELIABILITY_SIGNAL"
        for record in records
    )


document_names = sorted(
    remaining
)


# Round-robin reliability fill

while (
    len(selected) < TARGET_CANDIDATES
    and
    count_reliability(selected)
    < TARGET_RELIABILITY_TOTAL
):

    added_any = False

    for document_name in document_names:

        pool = (
            remaining[
                document_name
            ]["reliability"]
        )

        if pool:

            selected.append(
                pool.pop()
            )

            added_any = True

        if (
            len(selected)
            >= TARGET_CANDIDATES
        ):
            break

    if not added_any:
        break


# ============================================================
# Fill any final remaining slots from either pool
# ============================================================

while len(selected) < TARGET_CANDIDATES:

    added_any = False

    for document_name in document_names:

        rel_pool = (
            remaining[
                document_name
            ]["reliability"]
        )

        gen_pool = (
            remaining[
                document_name
            ]["general"]
        )

        available = (
            rel_pool
            if rel_pool
            else gen_pool
        )

        if available:

            selected.append(
                available.pop()
            )

            added_any = True

        if (
            len(selected)
            >= TARGET_CANDIDATES
        ):
            break

    if not added_any:
        break


# Final reproducible shuffle

rng.shuffle(
    selected
)


# ============================================================
# Write annotation CSV
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fieldnames = [
    "id",
    "source",
    "source_document",
    "text",
    "label",
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

    for index, record in enumerate(
        selected,
        start=1,
    ):

        writer.writerow({

            "id":
                index,

            "source":
                record["source"],

            "source_document":
                record["source_document"],

            "text":
                record["text"],

            "label":
                "",

            "notes":
                "",
        })


# ============================================================
# Write candidate-selection provenance
# ============================================================

PROVENANCE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    PROVENANCE_FILE,
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    fieldnames = [

        "id",
        "source",
        "source_document",
        "candidate_pool",
        "passage_method",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for index, record in enumerate(
        selected,
        start=1,
    ):

        writer.writerow({

            "id":
                index,

            "source":
                record["source"],

            "source_document":
                record["source_document"],

            "candidate_pool":
                record["candidate_pool"],

            "passage_method":
                record["passage_method"],
        })


# ============================================================
# Summary
# ============================================================

print()
print("=" * 80)
print("EXTERNAL RELIABILITY CANDIDATES")
print("=" * 80)

print()

for document_name in sorted(
    document_pools
):

    pools = document_pools[
        document_name
    ]

    selected_for_document = [
        record
        for record in selected
        if (
            record["source_document"]
            == document_name
        )
    ]

    selected_rel = sum(
        record["candidate_pool"]
        == "RELIABILITY_SIGNAL"
        for record
        in selected_for_document
    )

    selected_general = sum(
        record["candidate_pool"]
        == "GENERAL_GRID"
        for record
        in selected_for_document
    )

    print(document_name)

    print(
        f"  Reliability pool: "
        f"{len(pools['reliability']):,}"
    )

    print(
        f"  General-grid pool: "
        f"{len(pools['general']):,}"
    )

    print(
        f"  Selected: "
        f"{len(selected_for_document):,}"
    )

    print(
        f"    Reliability signal: "
        f"{selected_rel:,}"
    )

    print(
        f"    General grid: "
        f"{selected_general:,}"
    )

    print()


total_reliability = (
    count_reliability(selected)
)

total_general = (
    len(selected)
    - total_reliability
)


print("-" * 80)

print(
    f"Total selected: "
    f"{len(selected):,}"
)

print(
    f"Reliability-signal: "
    f"{total_reliability:,}"
)

print(
    f"General-grid: "
    f"{total_general:,}"
)

print()

print("Annotation file:")
print(OUTPUT_FILE)

print()

print("Selection provenance:")
print(PROVENANCE_FILE)