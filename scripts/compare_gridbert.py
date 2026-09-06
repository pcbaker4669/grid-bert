from transformers import pipeline

BASE_MODEL = "google-bert/bert-base-uncased"
GRID_MODEL = r"D:\GridBERT\models\GridBERT-v0.1\final"

print("Loading Base BERT...")
base_bert = pipeline(
    "fill-mask",
    model=BASE_MODEL
)

print("Loading GridBERT...")
grid_bert = pipeline(
    "fill-mask",
    model=GRID_MODEL
)


sentences = [
    "PJM operates a wholesale electricity [MASK].",
    "NERC establishes reliability [MASK].",
    "The generator retirement created a reliability [MASK].",
    "Transmission congestion can increase electricity [MASK].",
    "The power plant was no longer economically [MASK].",
    "The utility must maintain adequate generation [MASK].",
    "The transmission line experienced thermal [MASK].",
    "The grid operator dispatched additional [MASK].",
    "A shortage of generation can threaten grid [MASK].",
    "Electricity demand is also known as electrical [MASK]."
]


for sentence in sentences:

    print("\n" + "=" * 80)
    print(sentence)

    base_results = base_bert(sentence, top_k=5)
    grid_results = grid_bert(sentence, top_k=5)

    print("\nBASE BERT")
    for result in base_results:
        print(
            f"{result['token_str']:20s}"
            f"{result['score']:.4f}"
        )

    print("\nGRIDBERT")
    for result in grid_results:
        print(
            f"{result['token_str']:20s}"
            f"{result['score']:.4f}"
        )