from pathlib import Path
import csv
import re

import pymupdf as fitz
from transformers import AutoTokenizer


RAW_DIR = Path(r"D:\GridBERT\raw")
TEXT_DIR = Path(r"D:\GridBERT\text")
OUTPUT_CSV = Path(r"D:\GridBERT\gridtext_manifest.csv")

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


rows = []

pdf_files = list(RAW_DIR.rglob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.\n")

for pdf_path in pdf_files:

    relative_path = pdf_path.relative_to(RAW_DIR)

    output_path = (
        TEXT_DIR /
        relative_path.with_suffix(".txt")
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Processing: {relative_path}")

    try:
        document = fitz.open(pdf_path)

        pages = []

        for page in document:
            pages.append(page.get_text())

        document.close()

        text = "\n".join(pages)
        text = clean_text(text)

        output_path.write_text(
            text,
            encoding="utf-8"
        )

        tokens = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False
        )["input_ids"]

        rows.append({
            "source": relative_path.parts[0],
            "filename": pdf_path.name,
            "pages": len(pages),
            "characters": len(text),
            "words": len(text.split()),
            "bert_tokens": len(tokens),
            "text_file": str(output_path)
        })

    except Exception as e:

        print(f"ERROR: {pdf_path}")
        print(e)


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "source",
            "filename",
            "pages",
            "characters",
            "words",
            "bert_tokens",
            "text_file"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


total_tokens = sum(
    row["bert_tokens"]
    for row in rows
)

print("\nFinished.")
print(f"Documents processed: {len(rows)}")
print(f"Total BERT tokens: {total_tokens:,}")
print(f"Manifest: {OUTPUT_CSV}")