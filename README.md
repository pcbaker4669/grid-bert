# GridBERT

GridBERT is a proof-of-concept domain-adapted BERT model designed to better represent the specialized language of the U.S. electric-power grid, including grid reliability, transmission planning, wholesale electricity markets, resource adequacy, generation, dispatch, and energy regulation.

GridBERT v0.1 was created by continuing masked-language-model pretraining of `google-bert/bert-base-uncased` on a curated corpus of public U.S. electric-grid documents from:

- U.S. Department of Energy (DOE)
- Federal Energy Regulatory Commission (FERC)
- North American Electric Reliability Corporation (NERC)
- PJM Interconnection

The primary research question was:

> **How well does Base BERT understand specialized U.S. electric-grid policy language, and can domain-adaptive pretraining improve that understanding?**

GridBERT v0.1 substantially outperformed Base BERT on both held-out masked-language modeling and a 50-prompt grid-domain benchmark.

---

# 1. Project Motivation

General-purpose BERT was pretrained on broad English-language corpora. It therefore understands many ordinary language relationships, but specialized technical terms can have meanings that differ substantially from general usage.

Examples from the electric-grid domain include:

- dispatch
- load
- congestion
- thermal limits
- resource adequacy
- reserves
- reliability
- capacity
- transmission constraints
- wholesale electricity markets

For example, Base BERT initially interpreted:

```text
The grid operator dispatched additional [MASK].
```

using ordinary-language concepts such as:

```text
workers
vehicles
troops
firefighters
volunteers
```

In electric-grid operations, however, `dispatch` normally refers to generation, units, resources, reserves, or power.

GridBERT tests whether continued domain-specific pretraining can shift BERT's contextual representations toward electric-grid meanings.

---

# 2. GridBERT v0.1 Overview

GridBERT v0.1 uses:

```text
Base model:
google-bert/bert-base-uncased
```

The model architecture and tokenizer were retained.

GridBERT was created using **domain-adaptive pretraining**, also called continued pretraining.

The model was not trained from scratch.

The basic process was:

```text
Base BERT
    |
    v
Public electric-grid documents
    |
    v
PDF text extraction
    |
    v
GridText corpus
    |
    v
BERT tokenization
    |
    v
256-token training sequences
    |
    v
Masked Language Model training
    |
    v
GridBERT v0.1
    |
    v
Fixed-mask validation
    |
    v
Grid-domain benchmark
```

---

# 3. Data Sources

The GridText v0.1 corpus contains public documents from four major U.S. grid institutions.

## DOE

Examples include:

- National Transmission Planning Study
- National Transmission Needs Study
- Section 202(c) emergency orders
- emergency-order applications and related filings
- grid reliability and transmission policy reports

## FERC

Examples include:

- Energy Markets Primer
- Summer Energy Market and Electric Reliability Assessments
- Winter Energy Market and Electric Reliability Assessments
- Winter Storm Elliott review
- Arctic storm performance reviews
- demand-response reports
- State of the Markets reports
- Interregional Transfer Capability Study

## NERC

The corpus contains multiple Long-Term Reliability Assessments covering several decades.

These documents provide extensive terminology related to:

- resource adequacy
- generation capacity
- reserve margins
- reliability
- transmission
- electricity demand
- regional reliability risks

## PJM

Examples include:

- State of the Market reports
- Base Residual Auction reports
- capacity-market reports
- planning-period parameters

---

# 4. Corpus Statistics

The final GridText v0.1 corpus contained:

| Measure | Value |
|---|---:|
| Documents | 65 |
| BERT tokens | 3,958,771 |
| Training documents | 59 |
| Validation documents | 6 |
| Training sequences | 14,265 |
| Validation sequences | 1,343 |
| Maximum sequence length | 256 |

Documents were split at the **document level**, rather than randomly splitting individual text chunks.

This prevents sections from the same source document from appearing in both training and validation data.

A fixed random seed was used:

```python
RANDOM_SEED = 42
```

Documents were sorted before shuffling to make the split reproducible.

---

# 5. Text Extraction

PDF files are recursively processed and converted into plain text.

The extraction pipeline:

1. Locates PDFs in the raw document directory.
2. Extracts text using PyMuPDF.
3. Cleans extracted whitespace.
4. Saves text files while preserving source-directory structure.
5. Counts tokens using the Base BERT tokenizer.
6. Produces a corpus manifest.

Example structure:

```text
raw/
├── doe/
├── ferc/
├── nerc/
│   └── ltra/
└── pjm/

text/
├── doe/
├── ferc/
├── nerc/
│   └── ltra/
└── pjm/
```

The corpus manifest records information about the processed documents.

---

# 6. Preparing the Training Dataset

The extracted text is divided into sequences suitable for BERT training.

GridBERT v0.1 uses:

```text
254 content tokens
+ [CLS]
+ [SEP]
----------------
256 total tokens
```

The dataset preparation script generates:

```text
training/train.jsonl
training/validation.jsonl
```

Each JSONL record contains:

```json
{
  "input_ids": [],
  "attention_mask": [],
  "source_file": ""
}
```

Very short trailing fragments containing fewer than 32 tokens are excluded.

The train-validation split occurs at the document level.

---

# 7. Validation Documents

The reproducible v0.1 validation set contains six documents:

```text
99ras.txt
nerc_ltra_2021.txt
NationalTransmissionPlanningStudy-Chapter1.txt
A3_State of the Markets 2023_Presentation_0320_1715.txt
ltra2004.txt
2013_ltra_final.txt
```

These documents were not included in GridBERT MLM training.

---

# 8. Training GridBERT

GridBERT v0.1 was initialized from:

```text
google-bert/bert-base-uncased
```

Training used masked-language modeling.

Approximately 15% of eligible tokens were dynamically selected for the MLM task.

## Training Parameters

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Learning rate | 2e-5 |
| MLM probability | 0.15 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Sequence length | 256 |
| Training sequences | 14,265 |

Training was performed using Hugging Face Transformers and PyTorch.

The original experiment was trained on CPU and required approximately nine hours.

The final model was saved locally as:

```text
models/GridBERT-v0.1/final/
```

---

# 9. Validation Loss During Training

Validation loss improved across all three epochs:

| Model / Epoch | Validation Loss |
|---|---:|
| Base BERT | 2.6675 |
| GridBERT Epoch 1 | 1.606 |
| GridBERT Epoch 2 | 1.478 |
| GridBERT Epoch 3 | 1.449 |

The continued reduction in validation loss provided no immediate indication that the third epoch was degrading held-out performance.

---

# 10. Fixed-Mask Evaluation

Standard MLM evaluation can dynamically select different masked tokens during different runs.

To create a stricter comparison, a separate fixed-mask validation dataset was generated.

The same:

- input sequences
- masked positions
- original target tokens

were supplied to both Base BERT and GridBERT.

The fixed dataset contains all 1,343 validation sequences.

## Results

| Model | MLM Loss | Perplexity |
|---|---:|---:|
| Base BERT | 2.6431 | 14.06 |
| **GridBERT v0.1** | **1.4324** | **4.19** |

GridBERT therefore reduced held-out fixed-mask perplexity from:

```text
14.06 -> 4.19
```

This confirms that the improvement observed during training was not simply caused by differences in randomly selected masked tokens.

---

# 11. Domain-Specific Probe Examples

Before training, several manually constructed prompts were used to identify possible domain mismatch.

## PJM Market

Prompt:

```text
PJM operates a wholesale electricity [MASK].
```

Base BERT:

```text
business       0.2006
network        0.1306
system         0.1187
company        0.0924
market         0.0825
```

GridBERT:

```text
market         0.9237
system         0.0377
marketplace    0.0144
```

GridBERT developed a much stronger relationship between PJM and the wholesale electricity market.

## Grid Reliability

Prompt:

```text
A shortage of generation can threaten grid [MASK].
```

Base BERT:

```text
capacity       0.1157
reliability    0.0717
access         0.0549
supply         0.0495
```

GridBERT:

```text
reliability    0.9094
security       0.0493
operations     0.0137
stability      0.0053
```

## Dispatch

Prompt:

```text
The grid operator dispatched additional [MASK].
```

Base BERT:

```text
workers
vehicles
troops
firefighters
volunteers
```

GridBERT:

```text
resources
reinforcements
customers
power
staff
```

The GridBERT predictions show a shift away from the ordinary-language interpretation of `dispatch` and toward grid-related concepts.

## Generation Adequacy

Prompt:

```text
The utility must maintain adequate generation [MASK].
```

Base BERT strongly predicted:

```text
capacity
```

GridBERT produced:

```text
reserves
capacity
resources
capability
supply
```

This suggests a broader relationship between generation and resource-adequacy terminology.

---

# 12. Remaining Weaknesses

Domain adaptation did not improve every term equally.

For example:

```text
The transmission line experienced thermal [MASK].
```

GridBERT continued producing predictions such as:

```text
collapse
issues
failure
problems
```

rather than strongly predicting concepts such as:

```text
limit
constraint
overload
```

This suggests that GridBERT v0.1 still has weak representations for some highly specialized transmission terminology.

---

# 13. 50-Prompt Grid-Domain Benchmark

A larger diagnostic benchmark was developed containing 50 masked-language prompts.

The benchmark contains five prompts from each of ten categories:

1. Markets
2. Reliability
3. Resource adequacy
4. Transmission
5. Congestion
6. Dispatch
7. Load
8. Generation
9. Reserves
10. Institutions

Each prompt specifies one expected domain term.

Example:

```json
{
  "category": "reliability",
  "prompt": "A shortage of generation can threaten grid [MASK].",
  "expected": "reliability"
}
```

---

# 14. Benchmark Metrics

Three ranking metrics are calculated.

## Hits@1

The proportion of prompts where the expected domain term is the model's highest-ranked prediction.

## Hits@5

The proportion of prompts where the expected term appears anywhere among the five highest-ranked predictions.

## Mean Reciprocal Rank

MRR measures how highly the expected term is ranked.

Examples:

```text
Rank 1  -> 1.00
Rank 2  -> 0.50
Rank 5  -> 0.20
Rank 10 -> 0.10
```

---

# 15. Domain Benchmark Results

## Overall Results

| Metric | Base BERT | GridBERT |
|---|---:|---:|
| Hits@1 | 0.280 | **0.720** |
| Hits@5 | 0.600 | **0.900** |
| MRR | 0.430 | **0.804** |

The expected grid-domain term was Base BERT's first prediction 28% of the time.

For GridBERT, it was the first prediction 72% of the time.

The expected term appeared within GridBERT's top five predictions on 90% of benchmark prompts, compared with 60% for Base BERT.

---

# 16. Results by Domain

| Category | Base H@1 | GridBERT H@1 | Base H@5 | GridBERT H@5 |
|---|---:|---:|---:|---:|
| Markets | 0.40 | **0.60** | 0.80 | **1.00** |
| Reliability | 0.40 | **1.00** | 0.80 | **1.00** |
| Resource adequacy | 0.60 | 0.60 | 0.60 | **1.00** |
| Transmission | 0.20 | **0.60** | 0.40 | **0.80** |
| Congestion | 0.00 | **0.60** | 0.60 | 0.60 |
| Dispatch | 0.20 | **0.80** | 0.60 | **1.00** |
| Load | 0.40 | 0.40 | 0.80 | 0.80 |
| Generation | 0.40 | **0.80** | 0.60 | **0.80** |
| Reserves | 0.20 | **0.80** | 0.40 | **1.00** |
| Institutions | 0.00 | **1.00** | 0.40 | **1.00** |

The largest gains appeared in:

- reliability
- institutional terminology
- dispatch
- reserves
- generation
- transmission
- congestion

`load` remained one of the weakest categories.

---

# 17. Main Result

The primary GridBERT v0.1 result is:

> **Domain-adaptive pretraining increased Hits@1 on a 50-prompt electric-grid language benchmark from 28% for Base BERT to 72% for GridBERT.**

This result is supported independently by the fixed-mask MLM evaluation:

> **GridBERT reduced held-out masked-language perplexity from 14.06 to 4.19.**

Together, these results provide evidence that continued domain-specific pretraining substantially changed BERT's representation of specialized electric-grid language.

---

# 18. Repository Structure

A suggested repository structure is:

```text
GridBERT/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── extract_gridtext.py
│   ├── prepare_gridtext.py
│   ├── train_gridbert.py
│   ├── compare_gridbert.py
│   ├── freeze_gridbert_v01.py
│   ├── fixed_mask_eval.py
│   ├── create_grid_benchmark.py
│   └── evaluate_grid_benchmark.py
│
├── benchmarks/
│   └── grid_domain_benchmark.jsonl
│
├── results/
│   ├── fixed_mask_results.json
│   ├── grid_domain_benchmark_results.csv
│   └── grid_domain_benchmark_summary.json
│
└── docs/
    └── experiment_notes/
```

Large source documents, extracted text, training datasets, checkpoints, and model weights should generally not be committed directly to GitHub.

---

# 19. Scripts

## `extract_gridtext.py`

Processes the raw PDF corpus.

Responsibilities:

- recursively locate PDFs
- extract text using PyMuPDF
- clean whitespace
- save extracted text
- calculate BERT token counts
- generate corpus manifest

## `prepare_gridtext.py`

Creates the MLM training dataset.

Responsibilities:

- locate extracted text files
- sort documents for reproducibility
- shuffle using seed 42
- perform document-level train-validation split
- tokenize documents
- create 256-token sequences
- generate JSONL datasets

Outputs:

```text
train.jsonl
validation.jsonl
```

## `train_gridbert.py`

Performs continued masked-language-model pretraining.

Responsibilities:

- load Base BERT
- load GridText datasets
- dynamically mask training tokens
- evaluate Base BERT
- train for three epochs
- evaluate GridBERT
- save the final model

## `compare_gridbert.py`

Runs the original manually constructed masked-word probes against both Base BERT and GridBERT.

This script is primarily intended for qualitative comparison and demonstration.

## `freeze_gridbert_v01.py`

Creates a reproducibility manifest for the GridBERT v0.1 experiment.

The manifest records information including:

- model
- corpus size
- train-validation split
- training parameters
- file hashes
- experimental results

## `fixed_mask_eval.py`

Creates a permanent fixed-mask validation dataset and evaluates both models against exactly the same masked tokens.

Outputs include:

```text
fixed_validation.jsonl
fixed_mask_results.json
```

## `create_grid_benchmark.py`

Creates the 50-prompt GridBERT domain benchmark.

The benchmark contains 50 prompts, 10 categories, and 5 prompts per category.

## `evaluate_grid_benchmark.py`

Evaluates Base BERT and GridBERT against the grid-domain benchmark.

Metrics include:

```text
Hits@1
Hits@5
Mean Reciprocal Rank
```

The script also creates prompt-level CSV results and category-level summary results.

---

# 20. Installation

The original experiment was developed using Python 3.12.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the principal dependencies:

```powershell
python -m pip install transformers torch pymupdf
```

The original development environment used Hugging Face Transformers 5.16.1.

For exact reproducibility, use the versions recorded in `requirements.txt`.

---

# 21. Running the Experiment

The basic workflow is:

## Step 1: Extract corpus text

```powershell
python .\scripts\extract_gridtext.py
```

## Step 2: Prepare training data

```powershell
python .\scripts\prepare_gridtext.py
```

## Step 3: Train GridBERT

```powershell
python .\scripts\train_gridbert.py
```

## Step 4: Compare diagnostic prompts

```powershell
python .\scripts\compare_gridbert.py
```

## Step 5: Freeze the experiment

```powershell
python .\scripts\freeze_gridbert_v01.py
```

## Step 6: Run fixed-mask evaluation

```powershell
python .\scripts\fixed_mask_eval.py
```

## Step 7: Create domain benchmark

```powershell
python .\scripts\create_grid_benchmark.py
```

## Step 8: Evaluate domain benchmark

```powershell
python .\scripts\evaluate_grid_benchmark.py
```

---

# 22. Reproducibility

Several measures were added to improve reproducibility.

## Fixed random seed

```python
seed = 42
```

## Deterministic document ordering

Documents are sorted before the train-validation shuffle.

## Document-level splitting

Individual reports cannot contribute chunks to both training and validation sets.

## Fixed-mask validation

A permanent masked validation dataset allows future GridBERT versions to be tested against exactly the same prediction task.

## File hashes

The experiment manifest records SHA-256 hashes for important dataset files.

## Frozen benchmark

The GridBERT v0.1 50-prompt benchmark should not be modified after results are reported.

Future benchmark revisions should receive new version identifiers.

---

# 23. Limitations

GridBERT v0.1 is a proof-of-concept experiment.

Important limitations include:

### Corpus size

The approximately 4-million-token corpus is small compared with the original BERT pretraining corpus.

### Institutional coverage

The current corpus emphasizes DOE, FERC, NERC, and PJM.

Future versions should include additional RTO/ISOs, utilities, state public utility commissions, and other grid institutions.

### Benchmark construction

The current 50-prompt benchmark was manually constructed and uses one expected term per prompt.

Natural language often permits multiple technically correct completions.

For example:

```text
capacity
resources
reserves
supply
```

could all be reasonable under certain contexts.

The benchmark should therefore be interpreted as a diagnostic benchmark rather than a definitive measure of domain knowledge.

### Expert validation

The benchmark has not yet undergone independent expert review.

### Terminology coverage

GridBERT remains weaker in several areas, including:

- load
- thermal limits
- some congestion terminology
- transmission constraints

### Model architecture

GridBERT v0.1 uses the original BERTBASE architecture.

Future work may compare domain adaptation using newer encoder architectures.

---

# 24. Future Work

Potential GridBERT v0.2 improvements include:

- expand the GridText corpus
- improve institutional diversity
- add more transmission-planning documents
- increase coverage of grid operations
- specifically target weak terminology
- construct expert-reviewed benchmarks
- permit multiple acceptable benchmark answers
- develop downstream classification tasks
- perform named entity recognition
- identify generation-retirement events
- classify reliability concerns
- detect emergency regulatory actions
- detect transmission constraints
- classify institutional framing
- compare policy language among DOE, FERC, NERC, RTO/ISOs, utilities, and state regulators

Longer-term research could investigate whether GridBERT can convert large collections of regulatory and technical documents into structured data suitable for computational social science and policy analysis.

---

# 25. Research Direction

GridBERT is intended primarily as a domain-specific text measurement and information-extraction model rather than as a general conversational language model.

Potential applications include automatically identifying:

- generator retirements
- reliability concerns
- resource-adequacy warnings
- transmission constraints
- capacity shortages
- emergency actions
- market interventions
- regulatory disputes
- federal-state jurisdictional issues
- institutional framing of reliability problems

One longer-term application is a **Grid Reliability Policy Monitor** that uses GridBERT to identify emerging reliability and governance issues across thousands of regulatory and technical documents.

---

# 26. Model Portability

Once trained, GridBERT can be copied to another computer without retraining.

A saved model can be loaded locally using:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = r"path\to\GridBERT-v0.1\final"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
```

Python, PyTorch, and Transformers are required on the destination computer.

Internet access is not required when loading a complete locally saved model.

---

# 27. Repository Data Policy

The GitHub repository should contain:

- source code
- benchmark definitions
- experiment metadata
- small result files
- documentation

The repository should generally exclude:

- original PDF corpora
- extracted document text
- training JSONL files
- model checkpoints
- final model weights
- Hugging Face cache files
- Python virtual environments

These files are either large, reproducible from source documents, or inappropriate for normal Git version control.

Model weights can later be distributed through a model repository such as Hugging Face if desired.

---

# 28. Status

**GridBERT v0.1: Complete**

Current experimental results:

```text
Corpus:
65 documents
3,958,771 BERT tokens

Training:
59 documents
14,265 sequences

Validation:
6 documents
1,343 sequences

Fixed-mask evaluation:
Base BERT perplexity: 14.06
GridBERT perplexity:   4.19

50-prompt benchmark:
Base BERT Hits@1: 28%
GridBERT Hits@1:  72%

Base BERT Hits@5: 60%
GridBERT Hits@5:  90%

Base BERT MRR: 0.430
GridBERT MRR:  0.804
```

GridBERT v0.1 therefore provides an initial reproducible proof of concept that continued domain-adaptive pretraining can substantially improve BERT's representation of specialized U.S. electric-grid language.
