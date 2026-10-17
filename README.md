# GridBERT v0.3

GridBERT v0.3 is a domain-adapted BERT model for U.S. electric-grid text. It was created by continuing masked-language-model (MLM) pretraining from `google-bert/bert-base-uncased` on a corpus of electric-grid documents and then evaluating whether domain adaptation improves downstream reliability classification.

The main downstream research question is:

> Does GridBERT v0.3 outperform Base BERT when both models are fine-tuned on the same relatively small electric-grid reliability classification dataset?

The project compares:

- **Base BERT:** `google-bert/bert-base-uncased`
- **GridBERT v0.3:** Base BERT after domain-adaptive MLM pretraining on GridText v0.3

Both models are fine-tuned using the same supervised data, tokenizer, hyperparameters, validation procedure, and test sets.

---

## Project Status

The GridBERT v0.3 experiment is complete and should be treated as frozen.

Completed stages include:

- GridText v0.3 corpus construction
- GridBERT v0.3 masked-language-model pretraining
- Fixed-mask MLM evaluation
- Frozen grid-domain benchmark evaluation
- Manual construction of a reliability-classification dataset
- Document-level train/validation/test partitioning
- Base BERT fine-tuning
- GridBERT v0.3 fine-tuning
- Internal held-out evaluation
- External unseen-document evaluation
- Exact McNemar testing
- Paired bootstrap confidence intervals
- Per-document and disagreement analysis

---

## GridText v0.3

GridText v0.3 contains:

- **160 PDF documents**
- **20,521 pages**
- **7,267,913 words**
- **12,192,533 BERT tokens**

Primary sources include:

- NERC
- FERC
- DOE
- PJM
- CAISO
- MISO
- SPP
- ISO-NE
- academic sources

The corpus includes material on:

- resource adequacy
- transmission planning
- generation retirements
- system operations
- reliability assessments
- electricity markets
- emergency operations
- grid policy

Prepared MLM data:

- **52,052 training sequences**
- **1,343 validation sequences**
- **6 fixed validation documents**
- maximum sequence length: **256 tokens**

---

## GridBERT v0.3 Pretraining

GridBERT v0.3 was initialized from:

```text
google-bert/bert-base-uncased
```

Main MLM training settings:

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Learning rate | 2e-5 |
| MLM probability | 0.15 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Maximum sequence length | 256 |

Final model location:

```text
D:/GridBERT_v0_3/models/GridBERT-v0.3/final
```

### Fixed-mask evaluation

| Model | Loss | Perplexity |
|---|---:|---:|
| Base BERT | 2.6431 | 14.06 |
| GridBERT v0.1 | 1.4324 | 4.19 |
| GridBERT v0.2 | 1.4301 | 4.18 |
| GridBERT v0.3 | **1.3880** | **4.01** |

### Frozen grid-domain benchmark

| Model | Hit@1 | Hit@5 | MRR |
|---|---:|---:|---:|
| Base BERT | 0.280 | 0.600 | 0.430 |
| GridBERT v0.3 | **0.760** | **0.920** | **0.845** |

---

## Reliability Classification Dataset

The initial candidate set contained:

- **300 reliability-signal candidates**
- **200 general-grid candidates**
- **500 total candidates**
- random seed: **42**

Keyword pools were used only to select passages for manual review. They did not determine the final labels.

Manual labels:

```text
0 = NOT RELIABILITY
1 = RELIABILITY
2 = MIXED
3 = UNCERTAIN
4 = BAD TRAINING SEQUENCE
```

Only labels 0 and 1 were retained for the primary binary experiment.

Final binary dataset:

- **416 passages**
- **255 RELIABILITY**
- **161 NOT RELIABILITY**
- **106 unique source documents**

### Document-level split

The split was performed by source document rather than by individual passage.

| Split | Passages | Source Documents |
|---|---:|---:|
| Training | 260 | 74 |
| Validation | 100 | 16 |
| Internal test | 56 | 16 |
| Total | 416 | 106 |

No source document appears in more than one supervised partition.

---

## Downstream Fine-Tuning

Both Base BERT and GridBERT v0.3 were fully fine-tuned for binary sequence classification.

Labels:

```text
0 = NOT RELIABILITY
1 = RELIABILITY
```

Both models used:

- tokenizer: `google-bert/bert-base-uncased`
- maximum sequence length: 256
- the same training and validation passages
- the same hyperparameters
- the same random seed
- no class weighting
- no resampling

### Fine-tuning settings

| Parameter | Value |
|---|---:|
| Maximum sequence length | 256 |
| Batch size | 8 |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.10 |
| Gradient clipping norm | 1.0 |
| Random seed | 42 |
| Class weights | None |
| Resampling | None |

AdamW optimization and a linear learning-rate schedule with warmup were used.

Training batches were shuffled each epoch. Validation batches were not shuffled.

The best version of each model was selected using validation F1:

- **Base BERT:** epoch 4
- **GridBERT v0.3:** epoch 5

No further model changes were made before internal or external testing.

### GridBERT pooler handling

GridBERT v0.3 was saved following MLM training, so its checkpoint did not contain the BERT pooler used for sequence classification.

Before supervised fine-tuning:

- GridBERT's domain-adapted encoder was retained
- the MLM prediction head was discarded
- the original `bert-base-uncased` pooler was restored
- a new two-class classification head was initialized

This gave both models the same starting pooler while preserving the domain-adapted GridBERT encoder.

---

## External Unseen-Document Test Set

A second evaluation set was created from five newer documents from:

- PJM
- NERC
- CAISO

All five documents were programmatically verified as absent from:

- `gridtext_manifest_v03.csv`
- `training/train.jsonl`
- `training/validation.jsonl`

For every external document:

```text
in_v03_manifest = 0
used_in_mlm_training = 0
used_in_mlm_validation = 0
```

Initial external candidate set:

- **125 passages**
- **73 reliability-signal**
- **52 general-grid**

A passage-level overlap screen against GridText v0.3 identified:

- 3 exact matches
- 2 near duplicates
- 6 possible overlaps

All 11 flagged passages were manually reviewed and removed.

After annotation and final quality control, the external binary set contained:

- **92 passages**
- **56 RELIABILITY**
- **36 NOT RELIABILITY**

The external test set was not used for:

- training
- validation
- checkpoint selection
- threshold selection
- hyperparameter tuning

---

## Final Downstream Results

### Internal held-out test

| Metric | Base BERT | GridBERT v0.3 |
|---|---:|---:|
| Accuracy | 0.7321 | **0.8036** |
| Precision | 0.7500 | **0.8438** |
| Recall | 0.8182 | 0.8182 |
| F1 | 0.7826 | **0.8308** |
| Specificity | 0.6087 | **0.7826** |
| Macro F1 | 0.7169 | **0.7984** |

### External unseen-document test

| Metric | Base BERT | GridBERT v0.3 |
|---|---:|---:|
| Accuracy | 0.7826 | **0.8478** |
| Precision | 0.7813 | **0.9038** |
| Recall | **0.8929** | 0.8393 |
| F1 | 0.8333 | **0.8704** |
| Specificity | 0.6111 | **0.8611** |
| Macro F1 | 0.7604 | **0.8431** |

A recurring result was a reduction in false-positive reliability classifications.

Internal test false positives:

```text
Base BERT:      9
GridBERT v0.3:  5
```

External test false positives:

```text
Base BERT:     14
GridBERT v0.3:  5
```

---

## Statistical Analysis

The final statistical comparison used:

- two-sided exact McNemar tests
- Holm correction across the two McNemar tests
- paired percentile bootstrap
- 10,000 bootstrap samples
- 95% confidence intervals
- bootstrap seed 42

### McNemar results

| Dataset | Base Only Correct | GridBERT Only Correct | Exact p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Internal test | 2 | 6 | 0.2891 | 0.4202 |
| External unseen test | 5 | 11 | 0.2101 | 0.4202 |

The overall paired differences did not reach the conventional `p < .05` threshold.

The strongest external bootstrap results were:

- **Precision difference:** +12.26 percentage points  
  95% CI: +4.83 to +20.98 points

- **Specificity difference:** +25.00 percentage points  
  95% CI: +11.43 to +40.00 points

The results should be interpreted as consistent evidence of improved downstream performance, not proof of statistical superiority.

---

# Repository Structure

A typical project layout is:

```text
GridBERT-v0.3/
│
├── config.yaml
├── config_loader.py
│
├── scripts/
│   ├── extract_gridtext.py
│   ├── prepare_gridtext.py
│   ├── train_gridbert.py
│   ├── fixed_mask_eval.py
│   ├── create_grid_benchmark.py
│   ├── evaluate_grid_benchmark.py
│   ├── compare_gridbert.py
│   ├── freeze_gridibert_v01.py
│   ├── build_reliability_candidates.py
│   ├── prepare_classification_data.py
│   ├── build_external_manifest.py
│   ├── build_external_candidates.py
│   ├── check_external_overlap.py
│   ├── build_overlap_review.py
│   ├── train_reliability_classifier.py
│   ├── evaluate_reliability_classifier.py
│   └── analyze_reliability_results.py
│
└── classification/
    ├── data/
    ├── models/
    ├── results/
    └── external_test/
        ├── raw/
        ├── text/
        ├── candidates/
        ├── provenance/
        └── results/
```

The large data and model directories may be stored outside the Git repository and referenced through `config.yaml`.

---

# Script Reference

## `scripts/extract_gridtext.py`

Extracts text from the original PDF corpus.

Used before any model training.

## `scripts/prepare_gridtext.py`

Builds MLM training and validation sequences from extracted text.

Produces the authoritative:

```text
training/train.jsonl
training/validation.jsonl
```

## `scripts/train_gridbert.py`

Continues MLM pretraining from `google-bert/bert-base-uncased` using GridText v0.3.

## `scripts/fixed_mask_eval.py`

Evaluates Base BERT and GridBERT versions using the same fixed masked tokens.

## `scripts/create_grid_benchmark.py`

Creates the frozen grid-domain masked-word benchmark.

## `scripts/evaluate_grid_benchmark.py`

Evaluates models on the frozen grid-domain benchmark.

## `scripts/compare_gridbert.py`

Supports model-version comparisons during GridBERT development.

## `scripts/freeze_gridibert_v01.py`

Used during earlier model-version preservation.

## `scripts/build_reliability_candidates.py`

Builds the original 500-passage manual annotation candidate set.

Uses reliability-signal and general-grid keyword pools for candidate selection only.

## `scripts/prepare_classification_data.py`

Creates the final binary classification dataset and document-level train/validation/test split.

## `scripts/build_external_manifest.py`

Verifies that selected external documents were absent from GridText v0.3 and the actual MLM training/validation files.

Produces:

```text
classification/external_test/provenance/external_unseen_manifest.csv
```

## `scripts/build_external_candidates.py`

Creates external candidate passages from the unseen documents.

Includes a bullet-text fallback for presentation documents that do not contain conventional sentence punctuation.

## `scripts/check_external_overlap.py`

Screens external candidates for exact and near-duplicate text against GridText v0.3.

## `scripts/build_overlap_review.py`

Creates a review file containing flagged overlap cases for manual inspection.

## `scripts/train_reliability_classifier.py`

Fine-tunes either Base BERT or GridBERT v0.3 for binary reliability classification.

Usage:

```powershell
python scripts/train_reliability_classifier.py --model basebert
```

or:

```powershell
python scripts/train_reliability_classifier.py --model gridbert
```

## `scripts/evaluate_reliability_classifier.py`

Evaluates a frozen classifier on the internal or external test set.

Usage:

```powershell
python scripts/evaluate_reliability_classifier.py --model basebert --dataset internal
python scripts/evaluate_reliability_classifier.py --model gridbert --dataset internal

python scripts/evaluate_reliability_classifier.py --model basebert --dataset external
python scripts/evaluate_reliability_classifier.py --model gridbert --dataset external
```

## `scripts/analyze_reliability_results.py`

Performs the final paired statistical comparison using saved predictions.

Outputs include:

```text
final_model_comparison.csv
paired_statistical_tests.csv
external_per_document_results.csv
internal_disagreements.csv
external_disagreements.csv
statistical_analysis_summary.json
```

---

# Installation

The project was developed under Windows 11 using a Python virtual environment.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages.

At minimum, the downstream scripts require:

```text
torch
transformers
pandas
numpy
scikit-learn
PyYAML
PyMuPDF
```

For exact reproduction, save and use a frozen environment file:

```powershell
pip freeze > requirements_frozen.txt
```

Then restore with:

```powershell
pip install -r requirements_frozen.txt
```

---

# Check Hardware

To determine whether PyTorch can use CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CPU and CUDA training are both valid, but Base BERT and GridBERT should be trained under the same environment and preferably on the same hardware for a controlled comparison.

---

# Reproduction Workflows

There are three useful levels of reproduction.

## 1. Full rebuild

Use this when starting from the original PDF collection:

```text
Raw PDFs
  ↓
Extract text
  ↓
Prepare GridText
  ↓
Train GridBERT
  ↓
Evaluate MLM model
  ↓
Prepare supervised dataset
  ↓
Fine-tune both classifiers
  ↓
Run internal and external tests
  ↓
Run statistical analysis
```

## 2. Downstream reproduction

Use this when the frozen GridBERT v0.3 model and labeled datasets already exist:

```powershell
python scripts/train_reliability_classifier.py --model basebert
python scripts/train_reliability_classifier.py --model gridbert
python scripts/evaluate_reliability_classifier.py --model basebert --dataset internal
python scripts/evaluate_reliability_classifier.py --model gridbert --dataset internal
python scripts/evaluate_reliability_classifier.py --model basebert --dataset external
python scripts/evaluate_reliability_classifier.py --model gridbert --dataset external
python scripts/analyze_reliability_results.py
```

## 3. Evaluation-only reproduction

Use this when both trained classifiers already exist:

```powershell
python scripts/evaluate_reliability_classifier.py --model basebert --dataset internal
python scripts/evaluate_reliability_classifier.py --model gridbert --dataset internal
python scripts/evaluate_reliability_classifier.py --model basebert --dataset external
python scripts/evaluate_reliability_classifier.py --model gridbert --dataset external
python scripts/analyze_reliability_results.py
```

This is the safest way to reproduce the final reported results without retraining either model.

---

# Important Reproducibility Checks

Before retraining, verify:

```text
Training passages:   260
Validation passages: 100
```

Training labels:

```text
0 = 106
1 = 154
```

Validation labels:

```text
0 = 32
1 = 68
```

External test:

```text
92 passages
0 = 36
1 = 56
```

The external binary test file used in the final experiment is:

```text
classification/external_test/candidates/external_unseen_binary.csv
```

If exact reproduction is the goal, do not rebuild or relabel this file.

---

# Preserving the Frozen Experiment

Do not overwrite the original GridBERT v0.3 model, classifier outputs, prediction files, or statistical results when testing new changes.

A separate directory should be used for new experiments, for example:

```text
D:/GridBERT_v0_4/
```

or:

```text
D:/GridBERT_v0_3_reproduction/
```

The following should be treated as permanent v0.3 artifacts:

- frozen GridBERT v0.3 checkpoint
- manually labeled binary dataset
- train/validation/test split
- external unseen binary test set
- Base BERT classifier
- GridBERT v0.3 classifier
- saved prediction files
- confusion matrices
- final statistical-analysis outputs

---

# Interpretation of Results

GridBERT v0.3 achieved higher accuracy and F1 than Base BERT on:

- validation
- internal held-out testing
- external unseen-document testing

The most consistent improvement was in identifying passages that were **not** primarily about reliability.

This suggests that domain-adaptive pretraining helped GridBERT distinguish reliability-specific electric-grid language from more general electric-grid language.

The paired McNemar tests did not reach conventional statistical significance, so the results should be described as consistent evidence of improved performance rather than proof that GridBERT is statistically superior.

---

# Citation

If this repository accompanies a paper or preprint, add the final citation here once available.

Example:

```text
Author. "GridBERT: Domain-Adaptive BERT for Electric-Grid Reliability Text Classification." Year.
```

---

# Notes

This repository documents the GridBERT v0.3 experiment as performed. Future model versions should use separate versioned data, model, and result directories so that v0.3 remains reproducible.
