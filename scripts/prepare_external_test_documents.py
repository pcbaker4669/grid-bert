from pathlib import Path
import fitz
import yaml


# ============================================================
# Load configuration
# ============================================================

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

root = Path(config["paths"]["data_root"])

raw_root = root / config["classification"]["external_raw_dir"]
text_root = root / config["classification"]["external_text_dir"]


# ============================================================
# Extract one PDF
# ============================================================

def extract_pdf(pdf_path, txt_path):

    document = fitz.open(pdf_path)

    text_parts = []

    for page in document:
        text_parts.append(page.get_text())

    full_text = "\n".join(text_parts)

    txt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    txt_path.write_text(
        full_text,
        encoding="utf-8",
    )

    return {
        "pages": len(document),
        "characters": len(full_text),
        "words": len(full_text.split()),
    }


# ============================================================
# Find all external PDFs
# ============================================================

pdf_files = sorted(
    raw_root.rglob("*.pdf")
)

print(
    f"Found {len(pdf_files)} external PDFs."
)


# ============================================================
# Extract each PDF
# ============================================================

for pdf_path in pdf_files:

    relative_path = pdf_path.relative_to(
        raw_root
    )

    txt_path = (
        text_root
        / relative_path
    ).with_suffix(".txt")

    print("\nProcessing:")
    print(pdf_path)

    stats = extract_pdf(
        pdf_path,
        txt_path,
    )

    print(
        f"  Pages: {stats['pages']}"
    )

    print(
        f"  Characters: "
        f"{stats['characters']:,}"
    )

    print(
        f"  Words: "
        f"{stats['words']:,}"
    )

    print(
        f"  Output: {txt_path}"
    )


print("\nFinished.")