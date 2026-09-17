# GridBERT v0.3

GridBERT v0.3 is the next development version of GridBERT, a domain-adapted BERT model designed to better represent the specialized language of the U.S. electric-power grid.

The project focuses on technical, regulatory, market, and policy language related to:

- grid reliability
- transmission planning
- wholesale electricity markets
- resource adequacy
- generation
- dispatch
- operating reserves
- congestion
- load
- emergency grid actions
- grid governance and regulation

GridBERT v0.3 is being developed as an expanded and more systematic successor to GridBERT v0.2.

---

# 1. Research Goal

The main research question remains:

> **Can domain-adaptive pretraining improve BERT's representation of specialized U.S. electric-grid language?**

GridBERT v0.3 extends this question by asking whether a larger and more diverse grid-domain corpus can improve areas where GridBERT v0.1 remained comparatively weak.

Current target areas include:

- transmission constraints
- congestion
- load
- dispatch
- operating reserves
- resource adequacy
- grid institutional language

---

# 2. Relationship to GridBERT v0.1

GridBERT v0.1 demonstrated that continued masked-language-model pretraining on a curated electric-grid corpus could substantially improve Base BERT's performance on grid-domain language.

GridBERT v0.1 used:

- 65 documents
- 3,958,771 BERT tokens
- 59 training documents
- 6 validation documents
- 14,265 training sequences
- 1,343 validation sequences

Its main evaluation results were:

| Metric | Base BERT | GridBERT v0.1 |
|---|---:|---:|
| Fixed-mask perplexity | 14.06 | **4.19** |
| Hits@1 | 28% | **72%** |
| Hits@5 | 60% | **90%** |
| MRR | 0.430 | **0.804** |

These v0.1 results serve as the baseline for continued development.

GridBERT v0.2 will be trained independently from Base BERT rather than by continuing training from the v0.1 model. This makes comparisons between v0.1 and v0.2 easier to interpret because both begin from the same pretrained Base BERT starting point.

---

# 3. Base Model

GridBERT v0.2 starts from:

```text
google-bert/bert-base-uncased
```

The model architecture and tokenizer are retained.

GridBERT v0.2 is created using **domain-adaptive pretraining** with masked language modeling.

The model is not trained from scratch.

---

# 4. Data Repository

The GridBERT v0.2 data repository is separate from the v0.1 data repository.

```text
D:\GridBERT_v0_2
```

This separation protects the frozen GridBERT v0.1 corpus, experiment files, and trained model.

Recommended data structure:

```text
D:\GridBERT_v0_2\
├── raw\
│   ├── academic\
│   ├── doe\
│   ├── ferc\
│   ├── nerc\
│   └── pjm\
│
├── text\
├── training\
├── experiments\
├── models\
└── gridtext_manifest_v02.csv
```

---

# 5. Corpus Expansion Strategy

GridBERT v0.2 expands the original corpus rather than simply adding more documents at random.

The primary objective is to improve coverage of areas where v0.1 showed weaker performance.

Priority topics include:

- transmission constraints
- transmission capability
- congestion
- load and load forecasting
- dispatch
- operating reserves
- resource adequacy
- market design
- reliability-must-run concepts
- grid institutional responsibilities

The v0.2 corpus includes the original types of technical and regulatory sources used in v0.1, while adding more scholarly and policy-oriented material.

---

# 6. Source Categories

## U.S. Department of Energy

Potential and existing document types include:

- National Transmission Planning Study
- National Transmission Needs Study
- Federal Power Act Section 202(c) materials
- emergency grid orders
- reliability and transmission planning documents

## Federal Energy Regulatory Commission

Potential and existing document types include:

- electric power market materials
- transmission siting materials
- reliability assessments
- market assessments
- RTO/ISO materials
- reliability-must-run materials
- transmission and congestion studies
- major market and regulatory orders

Thin FERC landing pages should generally not be treated as full training documents unless they contain substantive explanatory text.

Whenever possible, the substantive linked order, report, filing, or study should be preferred.

## North American Electric Reliability Corporation

NERC material provides language related to:

- resource adequacy
- reserve margins
- reliability
- load growth
- transmission
- generation capacity
- regional reliability risks

The v0.2 corpus should avoid unnecessary duplication of NERC material already heavily represented in v0.1 unless the new material adds useful coverage.

## PJM Interconnection

PJM material provides language related to:

- capacity markets
- reliability
- market design
- generation
- demand growth
- constrained supply
- resource adequacy
- grid operations

## Academic Literature

GridBERT v0.2 adds an `academic` source category.

Initial scholarly additions include:

- Macey, Welton, and Wiseman (2024), *Grid Reliability in the Electric Era*
- Wiseman (2022), *Regional Cooperative Federalism and the U.S. Electric Grid*
- Wolak (2022), *Long-Term Resource Adequacy in Wholesale Electricity Markets with Significant Intermittent Renewables*

Academic papers are stored under:

```text
D:\GridBERT_v0_2\raw\academic
```

These papers add sustained scholarly language related to:

- grid governance
- reliability
- resource adequacy
- electricity markets
- federal-state relationships
- RTO/ISO institutions

---

# 7. Corpus Management

The v0.2 corpus should be tracked more systematically than v0.1.

The corpus manifest should record, at minimum:

- source organization
- source category
- document title
- filename
- year
- author or issuing organization
- document type
- URL or DOI
- pages
- word count
- BERT token count
- local text-file location

The v0.2 manifest is stored as:

```text
D:\GridBERT_v0_2\gridtext_manifest_v02.csv
```

The manifest should be treated as the authoritative record of the training corpus.

---

# 8. Text Extraction

The extraction pipeline:

1. Recursively locates PDF files under the configured raw-data directory.
2. Extracts text using PyMuPDF.
3. Cleans whitespace.
4. Saves extracted text while preserving source-directory structure.
5. Counts tokens using the Base BERT tokenizer.
6. Produces the corpus manifest.

Example:

```text
raw/
├── academic/
├── doe/
├── ferc/
├── nerc/
└── pjm/

text/
├── academic/
├── doe/
├── ferc/
├── nerc/
└── pjm/
```

---

# 9. Training Dataset Preparation

GridBERT v0.2 uses BERT training sequences with a maximum length of 256 tokens.

```text
254 content tokens
+ [CLS]
+ [SEP]
----------------
256 total tokens
```

The dataset preparation script produces:

```text
training/train.jsonl
training/validation.jsonl
```

Each sequence records:

```json
{
  "input_ids": [],
  "attention_mask": [],
  "source_file": ""
}
```

Very small trailing fragments are excluded according to the configuration.

Documents are split at the document level to prevent chunks from the same report from appearing in both training and validation sets.

---

# 10. Validation Strategy

GridBERT v0.2 should preserve a clean held-out validation set.

The original six v0.1 validation documents should remain excluded from v0.2 training if they are used for direct cross-version comparison.

Those documents were:

```text
99ras.txt
nerc_ltra_2021.txt
NationalTransmissionPlanningStudy-Chapter1.txt
A3_State of the Markets 2023_Presentation_0320_1715.txt
ltra2004.txt
2013_ltra_final.txt
```

Maintaining these holdouts allows more meaningful comparison among:

```text
Base BERT
GridBERT v0.1
GridBERT v0.2
```

---

# 11. Project Configuration

GridBERT v0.2 uses a single root-level `config.yaml` file.

This keeps paths, model references, training parameters, and evaluation settings out of individual Python scripts.

Example:

```yaml
project:
  name: "GridBERT"
  version: "v0.2"

paths:
  data_root: "D:/GridBERT_v0_2"

  raw_dir: "raw"
  text_dir: "text"
  training_dir: "training"

  train_file: "training/train.jsonl"
  validation_file: "training/validation.jsonl"
  corpus_manifest: "gridtext_manifest_v02.csv"

  model_output_dir: "models/GridBERT-v0.2"
  trained_model_dir: "models/GridBERT-v0.2/final"

  experiment_dir: "experiments/GridBERT-v0.2"
  benchmark_file: "experiments/GridBERT-v0.2/grid_domain_benchmark.jsonl"
  benchmark_results_csv: "experiments/GridBERT-v0.2/grid_domain_benchmark_results.csv"
  benchmark_summary_json: "experiments/GridBERT-v0.2/grid_domain_benchmark_summary.json"
  fixed_validation_file: "experiments/GridBERT-v0.2/fixed_validation.jsonl"
  fixed_mask_results_file: "experiments/GridBERT-v0.2/fixed_mask_results.json"

models:
  tokenizer_model: "google-bert/bert-base-uncased"
  training_start_model: "google-bert/bert-base-uncased"
  reference_model: "google-bert/bert-base-uncased"
  trained_model_label: "GridBERT v0.2"

dataset:
  content_length: 254
  min_final_fragment: 32
  validation_fraction: 0.10
  seed: 42

training:
  epochs: 3
  learning_rate: 2.0e-5
  mlm_probability: 0.15
  weight_decay: 0.01
  seed: 42

evaluation:
  fixed_mask_seed: 42
  fixed_mask_probability: 0.15
  fixed_mask_batch_size: 2
  benchmark_top_k: 5
```

The helper script:

```text
scripts/config_loader.py
```

loads this configuration and resolves paths relative to `paths.data_root`.

---

# 12. Training

GridBERT v0.2 will use masked-language-model pretraining starting from Base BERT.

Current default configuration:

| Parameter | Value |
|---|---:|
| Starting model | `google-bert/bert-base-uncased` |
| Epochs | 3 |
| Learning rate | 2e-5 |
| MLM probability | 0.15 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Maximum sequence length | 256 |

These values can be changed in `config.yaml` without editing the training script.

The trained model will be written to:

```text
D:\GridBERT_v0_2\models\GridBERT-v0.2\final
```

---

# 13. Evaluation Plan

GridBERT v0.2 should be evaluated using the same types of tests used for v0.1.

## Fixed-Mask Evaluation

Both Base BERT and GridBERT v0.2 receive identical masked validation tokens.

Metrics:

- MLM loss
- perplexity

## Grid-Domain Benchmark

The frozen 50-prompt diagnostic benchmark can be reused to compare versions.

Metrics:

- Hits@1
- Hits@5
- Mean Reciprocal Rank

The original benchmark covers:

- markets
- reliability
- resource adequacy
- transmission
- congestion
- dispatch
- load
- generation
- reserves
- institutions

The v0.1 benchmark should remain unchanged so that v0.2 results can be compared directly.

---

# 14. Downstream Proof of Concept

A major next step for GridBERT is demonstrating usefulness beyond masked-language prediction.

The initial proposed downstream application is a:

> **Grid Reliability Sentence Classifier**

The classifier would determine whether a sentence describes a grid reliability concern.

Example positive sentence:

```text
Retirements and load growth could create capacity shortfalls.
```

Example negative sentence:

```text
PJM operates a competitive wholesale electricity market.
```

The experiment would compare:

```text
Base BERT classifier
vs.
GridBERT classifier
```

using the same labeled training and test data.

Possible evaluation metrics include:

- accuracy
- precision
- recall
- F1 score

This provides a practical test of whether domain-adaptive pretraining improves performance on an applied grid-policy NLP task.

---

# 15. Repository Structure

Recommended GitHub project structure:

```text
GridBERT/
│
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
│
├── scripts/
│   ├── config_loader.py
│   ├── extract_gridtext.py
│   ├── prepare_gridtext.py
│   ├── train_gridbert.py
│   ├── compare_gridbert.py
│   ├── fixed_mask_eval.py
│   ├── create_grid_benchmark.py
│   └── evaluate_grid_benchmark.py
│
├── benchmarks/
├── results/
├── experiments/
└── docs/
```

Large corpus files and model weights should remain outside the normal Git repository.

---

# 16. Scripts

## `config_loader.py`

Loads `config.yaml` and resolves configured data paths.

## `extract_gridtext.py`

- locates PDFs
- extracts text with PyMuPDF
- cleans text
- saves extracted text
- counts BERT tokens
- builds the corpus manifest

## `prepare_gridtext.py`

- loads extracted text
- performs document-level train-validation splitting
- tokenizes documents
- creates 256-token sequences
- writes training and validation JSONL files

## `train_gridbert.py`

- loads Base BERT
- loads GridText training data
- performs dynamic MLM masking
- trains GridBERT
- evaluates validation performance
- saves the trained model

## `compare_gridbert.py`

Runs qualitative masked-word comparisons between Base BERT and GridBERT.

## `fixed_mask_eval.py`

Evaluates Base BERT and GridBERT on identical masked validation tokens.

## `create_grid_benchmark.py`

Creates the frozen 50-prompt grid-domain benchmark.

## `evaluate_grid_benchmark.py`

Calculates:

- Hits@1
- Hits@5
- Mean Reciprocal Rank
- category-level results

---

# 17. Installation

The project was developed using Python 3.12.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install transformers torch accelerate pymupdf pyyaml
```

For exact package versions, use the project `requirements.txt`.

---

# 18. Running the Pipeline

Before running any script, verify the paths and settings in:

```text
config.yaml
```

## Extract corpus text

```powershell
python .\scripts\extract_gridtext.py
```

## Prepare training sequences

```powershell
python .\scripts\prepare_gridtext.py
```

## Train GridBERT v0.2

```powershell
python .\scripts\train_gridbert.py
```

## Run qualitative comparison

```powershell
python .\scripts\compare_gridbert.py
```

## Create domain benchmark

```powershell
python .\scripts\create_grid_benchmark.py
```

## Evaluate domain benchmark

```powershell
python .\scripts\evaluate_grid_benchmark.py
```

## Run fixed-mask evaluation

```powershell
python .\scripts\fixed_mask_eval.py
```

---

# 19. Reproducibility

GridBERT v0.2 uses several reproducibility controls.

## Configuration File

All important paths and experiment parameters are stored in `config.yaml`.

## Fixed Random Seed

```python
seed = 42
```

## Deterministic Document Ordering

Documents are sorted before any random split.

## Document-Level Splitting

Chunks from the same report cannot appear in both training and validation data.

## Frozen v0.1 Benchmark

The original 50-prompt benchmark remains unchanged for cross-version comparison.

## Held-Out Documents

The original v0.1 validation reports should remain excluded from v0.2 training when used for direct comparison.

---

# 20. Limitations

GridBERT v0.2 is still under development.

Current limitations include:

- the expanded corpus is not yet finalized
- v0.2 has not yet been trained
- v0.2 evaluation results are not yet available
- institutional coverage remains incomplete
- academic-source coverage is still small
- the 50-prompt benchmark uses one expected answer per prompt
- the benchmark has not yet undergone independent expert validation

No performance claims should be made for GridBERT v0.2 until training and evaluation are complete.

---

# 21. Planned Improvements

Current priorities include:

- expand the corpus with carefully selected scholarly and institutional sources
- improve transmission-language coverage
- improve congestion-language coverage
- improve load and load-forecasting coverage
- improve dispatch and reserve terminology
- improve resource-adequacy coverage
- improve corpus metadata and provenance tracking
- create a downstream reliability classifier
- develop more rigorous expert-reviewed benchmarks
- compare Base BERT, GridBERT v0.1, and GridBERT v0.2

---

# 22. Long-Term Research Direction

GridBERT is intended as a domain-specific measurement and information-extraction model for electric-grid policy and operations.

Potential applications include identifying:

- generator retirements
- reliability concerns
- resource-adequacy warnings
- transmission constraints
- capacity shortages
- emergency actions
- market interventions
- regulatory disputes
- institutional responsibilities
- federal-state jurisdictional issues
- reliability framing

A longer-term application is a:

> **Grid Reliability Policy Monitor**

Such a system could automatically scan technical and regulatory documents and convert unstructured grid-policy language into structured data for computational social science and policy analysis.

---

# 23. Repository Data Policy

The GitHub repository should contain:

- source code
- `config.yaml`
- benchmark definitions
- experiment metadata
- corpus manifest
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

Model weights may later be distributed through a dedicated model repository if desired.

---

# 24. Status

**GridBERT v0.2: In Development**

Current status:

```text
Base model:
google-bert/bert-base-uncased

Data repository:
D:\GridBERT_v0_2

Corpus:
Expansion in progress

New source category:
Academic literature

Initial scholarly additions:
Macey, Welton & Wiseman (2024)
Wiseman (2022)
Wolak (2022)

Configuration:
Single root-level config.yaml

Training:
Not yet started

Evaluation:
Pending
```

GridBERT v0.2 is designed to provide a cleaner, larger, and more systematically managed follow-up to the successful GridBERT v0.1 proof-of-concept.
