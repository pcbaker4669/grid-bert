import csv
from collections import defaultdict

from config_loader import load_config, data_path


config = load_config()

MANIFEST_FILE = data_path(config, "corpus_manifest")
OUTPUT_FILE = MANIFEST_FILE.parent / "corpus_summary.csv"


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


stats = defaultdict(lambda: {
    "documents": 0,
    "bert_tokens": 0,
    "words": 0,
    "pages": 0,
})

with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    required_columns = {"source", "bert_tokens"}
    missing_columns = required_columns - set(reader.fieldnames or [])

    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    for row in reader:
        source = (row.get("source") or "unknown").strip().lower()

        stats[source]["documents"] += 1
        stats[source]["bert_tokens"] += safe_int(row.get("bert_tokens"))
        stats[source]["words"] += safe_int(row.get("words"))
        stats[source]["pages"] += safe_int(row.get("pages"))


total_documents = sum(x["documents"] for x in stats.values())
total_tokens = sum(x["bert_tokens"] for x in stats.values())
total_words = sum(x["words"] for x in stats.values())
total_pages = sum(x["pages"] for x in stats.values())


rows = []

for source, values in stats.items():
    percent = (
        (values["bert_tokens"] / total_tokens) * 100
        if total_tokens
        else 0
    )

    rows.append({
        "source": source,
        "documents": values["documents"],
        "bert_tokens": values["bert_tokens"],
        "percent_of_corpus": percent,
        "words": values["words"],
        "pages": values["pages"],
    })


rows.sort(
    key=lambda row: row["bert_tokens"],
    reverse=True
)


print()
print("=" * 82)
print("GRIDBERT CORPUS SUMMARY")
print("=" * 82)

print(
    f"{'Source':<18}"
    f"{'Documents':>10}"
    f"{'BERT Tokens':>16}"
    f"{'% Corpus':>12}"
    f"{'Words':>14}"
    f"{'Pages':>10}"
)

print("-" * 82)

for row in rows:
    print(
        f"{row['source']:<18}"
        f"{row['documents']:>10,}"
        f"{row['bert_tokens']:>16,}"
        f"{row['percent_of_corpus']:>11.2f}%"
        f"{row['words']:>14,}"
        f"{row['pages']:>10,}"
    )

print("-" * 82)

print(
    f"{'TOTAL':<18}"
    f"{total_documents:>10,}"
    f"{total_tokens:>16,}"
    f"{100.00:>11.2f}%"
    f"{total_words:>14,}"
    f"{total_pages:>10,}"
)

print()
print(f"Manifest: {MANIFEST_FILE}")


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "source",
        "documents",
        "bert_tokens",
        "percent_of_corpus",
        "words",
        "pages",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in rows:
        output_row = row.copy()
        output_row["percent_of_corpus"] = round(
            output_row["percent_of_corpus"],
            2
        )
        writer.writerow(output_row)


print(f"Summary saved: {OUTPUT_FILE}")