"""
analyze_reliability_results.py

Final statistical analysis for the GridBERT reliability experiment.

Compares frozen Base BERT and GridBERT v0.3 predictions on:

1. Internal held-out test set
2. External unseen-document test set

Outputs:
- Final model comparison table
- Exact McNemar tests
- Paired bootstrap 95% confidence intervals
- External per-document performance
- Internal disagreement cases
- External disagreement cases
- Reproducibility summary

No model training or inference occurs here.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config_loader import load_config


# ============================================================
# Configuration
# ============================================================

config = load_config()

DATA_ROOT = Path(
    config["paths"]["data_root"]
)

SEED = 42
BOOTSTRAP_SAMPLES = 10000


def root_path(value):

    path = Path(value)

    if path.is_absolute():
        return path

    return DATA_ROOT / path


CLASSIFICATION_RESULTS = root_path(
    config["classification"]["results_dir"]
)

EXTERNAL_RESULTS = root_path(
    config["classification"]["external_results_dir"]
)

BASE_MODEL_DIR = root_path(
    config["classification"]["basebert_output"]
)

GRID_MODEL_DIR = root_path(
    config["classification"]["gridbert_output"]
)


OUTPUT_DIR = (
    CLASSIFICATION_RESULTS
    / "final_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Input files
# ============================================================

INTERNAL_BASE = (
    CLASSIFICATION_RESULTS
    / "basebert_internal_predictions.csv"
)

INTERNAL_GRID = (
    CLASSIFICATION_RESULTS
    / "gridbert_v03_internal_predictions.csv"
)

EXTERNAL_BASE = (
    EXTERNAL_RESULTS
    / "basebert_external_predictions.csv"
)

EXTERNAL_GRID = (
    EXTERNAL_RESULTS
    / "gridbert_v03_external_predictions.csv"
)

BASE_SUMMARY = (
    BASE_MODEL_DIR
    / "training_summary.json"
)

GRID_SUMMARY = (
    GRID_MODEL_DIR
    / "training_summary.json"
)


# ============================================================
# Utility functions
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "specificity":
            specificity,

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


# ============================================================
# Exact McNemar test
# ============================================================

def exact_mcnemar(
    base_correct,
    grid_correct,
):

    base_correct = np.asarray(
        base_correct,
        dtype=bool,
    )

    grid_correct = np.asarray(
        grid_correct,
        dtype=bool,
    )

    both_correct = int(
        np.sum(
            base_correct
            & grid_correct
        )
    )

    base_only = int(
        np.sum(
            base_correct
            & ~grid_correct
        )
    )

    grid_only = int(
        np.sum(
            ~base_correct
            & grid_correct
        )
    )

    both_wrong = int(
        np.sum(
            ~base_correct
            & ~grid_correct
        )
    )

    discordant = (
        base_only
        + grid_only
    )

    if discordant == 0:

        p_value = 1.0

    else:

        smaller = min(
            base_only,
            grid_only,
        )

        lower_tail = sum(
            math.comb(
                discordant,
                k,
            )
            for k
            in range(
                smaller + 1
            )
        ) / (
            2 ** discordant
        )

        p_value = min(
            1.0,
            2.0 * lower_tail,
        )

    return {

        "both_correct":
            both_correct,

        "base_only_correct":
            base_only,

        "grid_only_correct":
            grid_only,

        "both_wrong":
            both_wrong,

        "discordant_pairs":
            discordant,

        "mcnemar_exact_p":
            p_value,
    }


# ============================================================
# Paired bootstrap
# ============================================================

def paired_bootstrap(
    y_true,
    base_pred,
    grid_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    base_pred = np.asarray(
        base_pred,
        dtype=int,
    )

    grid_pred = np.asarray(
        grid_pred,
        dtype=int,
    )

    n = len(
        y_true
    )

    rng = np.random.default_rng(
        SEED
    )

    metric_names = [

        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "macro_f1",
    ]

    differences = {
        metric: []
        for metric
        in metric_names
    }

    for _ in range(
        BOOTSTRAP_SAMPLES
    ):

        indexes = rng.integers(
            0,
            n,
            size=n,
        )

        sample_true = (
            y_true[indexes]
        )

        sample_base = (
            base_pred[indexes]
        )

        sample_grid = (
            grid_pred[indexes]
        )

        base_metrics = (
            calculate_metrics(
                sample_true,
                sample_base,
            )
        )

        grid_metrics = (
            calculate_metrics(
                sample_true,
                sample_grid,
            )
        )

        for metric in metric_names:

            differences[
                metric
            ].append(

                grid_metrics[
                    metric
                ]
                -
                base_metrics[
                    metric
                ]
            )

    results = {}

    for metric in metric_names:

        values = np.asarray(
            differences[
                metric
            ]
        )

        results[metric] = {

            "difference_mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "ci_95_low":
                float(
                    np.percentile(
                        values,
                        2.5,
                    )
                ),

            "ci_95_high":
                float(
                    np.percentile(
                        values,
                        97.5,
                    )
                ),
        }

    return results


# ============================================================
# Load paired prediction files
# ============================================================

def load_prediction_pair(
    base_file,
    grid_file,
    dataset_name,
):

    base = pd.read_csv(
        base_file
    )

    grid = pd.read_csv(
        grid_file
    )

    if len(base) != len(grid):

        raise ValueError(
            f"{dataset_name}: "
            "Base/Grid row counts differ."
        )

    required = {

        "true_label",
        "predicted_label",
        "text",
    }

    for name, dataframe in [
        ("Base", base),
        ("Grid", grid),
    ]:

        missing = (
            required
            - set(
                dataframe.columns
            )
        )

        if missing:

            raise ValueError(
                f"{dataset_name} "
                f"{name} missing columns: "
                f"{sorted(missing)}"
            )

    if not np.array_equal(
        base["true_label"]
        .astype(int)
        .to_numpy(),

        grid["true_label"]
        .astype(int)
        .to_numpy(),
    ):

        raise ValueError(
            f"{dataset_name}: "
            "true labels do not align."
        )

    if not np.array_equal(
        base["text"]
        .fillna("")
        .astype(str)
        .to_numpy(),

        grid["text"]
        .fillna("")
        .astype(str)
        .to_numpy(),
    ):

        raise ValueError(
            f"{dataset_name}: "
            "passage text does not align."
        )

    print(
        f"{dataset_name}: "
        f"{len(base)} paired predictions verified."
    )

    return base, grid


# ============================================================
# Analyze one dataset
# ============================================================

def analyze_dataset(
    dataset_name,
    base,
    grid,
):

    y_true = (
        base[
            "true_label"
        ]
        .astype(int)
        .to_numpy()
    )

    base_pred = (
        base[
            "predicted_label"
        ]
        .astype(int)
        .to_numpy()
    )

    grid_pred = (
        grid[
            "predicted_label"
        ]
        .astype(int)
        .to_numpy()
    )

    base_metrics = (
        calculate_metrics(
            y_true,
            base_pred,
        )
    )

    grid_metrics = (
        calculate_metrics(
            y_true,
            grid_pred,
        )
    )

    base_correct = (
        y_true
        == base_pred
    )

    grid_correct = (
        y_true
        == grid_pred
    )

    mcnemar = exact_mcnemar(
        base_correct,
        grid_correct,
    )

    bootstrap = paired_bootstrap(
        y_true,
        base_pred,
        grid_pred,
    )

    return {

        "dataset":
            dataset_name,

        "n":
            len(y_true),

        "base":
            base_metrics,

        "grid":
            grid_metrics,

        "mcnemar":
            mcnemar,

        "bootstrap":
            bootstrap,
    }


# ============================================================
# Load predictions
# ============================================================

print()
print("=" * 72)
print("GRIDBERT FINAL STATISTICAL ANALYSIS")
print("=" * 72)
print()

internal_base, internal_grid = (
    load_prediction_pair(
        INTERNAL_BASE,
        INTERNAL_GRID,
        "Internal test",
    )
)

external_base, external_grid = (
    load_prediction_pair(
        EXTERNAL_BASE,
        EXTERNAL_GRID,
        "External test",
    )
)


# ============================================================
# Analyze test datasets
# ============================================================

print()
print(
    "Running paired bootstrap "
    f"({BOOTSTRAP_SAMPLES:,} samples)..."
)

internal_analysis = (
    analyze_dataset(
        "Internal Test",
        internal_base,
        internal_grid,
    )
)

external_analysis = (
    analyze_dataset(
        "External Unseen Test",
        external_base,
        external_grid,
    )
)


# ============================================================
# Holm correction for two McNemar tests
# ============================================================

analyses = [
    internal_analysis,
    external_analysis,
]

sorted_analyses = sorted(
    analyses,
    key=lambda x:
        x["mcnemar"][
            "mcnemar_exact_p"
        ],
)

previous_adjusted = 0.0

for rank, analysis in enumerate(
    sorted_analyses,
    start=1,
):

    raw_p = (
        analysis[
            "mcnemar"
        ][
            "mcnemar_exact_p"
        ]
    )

    remaining = (
        len(sorted_analyses)
        - rank
        + 1
    )

    adjusted = min(
        1.0,
        raw_p * remaining,
    )

    adjusted = max(
        adjusted,
        previous_adjusted,
    )

    analysis[
        "mcnemar"
    ][
        "holm_adjusted_p"
    ] = adjusted

    previous_adjusted = (
        adjusted
    )


# ============================================================
# Final comparison table
# ============================================================

comparison_rows = []


# Validation results

with open(
    BASE_SUMMARY,
    "r",
    encoding="utf-8",
) as file:

    base_summary = json.load(
        file
    )


with open(
    GRID_SUMMARY,
    "r",
    encoding="utf-8",
) as file:

    grid_summary = json.load(
        file
    )


for model_name, summary in [

    ("Base BERT", base_summary),
    ("GridBERT v0.3", grid_summary),

]:

    metrics = summary[
        "best_validation_metrics"
    ]

    comparison_rows.append({

        "dataset":
            "Validation",

        "model":
            model_name,

        "n":
            summary[
                "validation_passages"
            ],

        "accuracy":
            metrics[
                "accuracy"
            ],

        "precision":
            metrics[
                "precision"
            ],

        "recall":
            metrics[
                "recall"
            ],

        "f1":
            metrics[
                "f1"
            ],

        "specificity":
            np.nan,

        "macro_f1":
            np.nan,

        "loss":
            metrics[
                "loss"
            ],
    })


# Test results

for analysis in analyses:

    for model_key, model_name in [

        ("base", "Base BERT"),
        ("grid", "GridBERT v0.3"),

    ]:

        metrics = (
            analysis[
                model_key
            ]
        )

        comparison_rows.append({

            "dataset":
                analysis[
                    "dataset"
                ],

            "model":
                model_name,

            "n":
                analysis[
                    "n"
                ],

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "precision":
                metrics[
                    "precision"
                ],

            "recall":
                metrics[
                    "recall"
                ],

            "f1":
                metrics[
                    "f1"
                ],

            "specificity":
                metrics[
                    "specificity"
                ],

            "macro_f1":
                metrics[
                    "macro_f1"
                ],

            "loss":
                np.nan,
        })


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_file = (
    OUTPUT_DIR
    / "final_model_comparison.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False,
)


# ============================================================
# Statistical comparison table
# ============================================================

stat_rows = []

for analysis in analyses:

    row = {

        "dataset":
            analysis[
                "dataset"
            ],

        "n":
            analysis[
                "n"
            ],

        "both_correct":
            analysis[
                "mcnemar"
            ][
                "both_correct"
            ],

        "base_only_correct":
            analysis[
                "mcnemar"
            ][
                "base_only_correct"
            ],

        "grid_only_correct":
            analysis[
                "mcnemar"
            ][
                "grid_only_correct"
            ],

        "both_wrong":
            analysis[
                "mcnemar"
            ][
                "both_wrong"
            ],

        "discordant_pairs":
            analysis[
                "mcnemar"
            ][
                "discordant_pairs"
            ],

        "mcnemar_exact_p":
            analysis[
                "mcnemar"
            ][
                "mcnemar_exact_p"
            ],

        "holm_adjusted_p":
            analysis[
                "mcnemar"
            ][
                "holm_adjusted_p"
            ],
    }

    for metric in [

        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "macro_f1",

    ]:

        base_value = (
            analysis[
                "base"
            ][
                metric
            ]
        )

        grid_value = (
            analysis[
                "grid"
            ][
                metric
            ]
        )

        ci = (
            analysis[
                "bootstrap"
            ][
                metric
            ]
        )

        row[
            f"base_{metric}"
        ] = base_value

        row[
            f"grid_{metric}"
        ] = grid_value

        row[
            f"difference_{metric}"
        ] = (
            grid_value
            - base_value
        )

        row[
            f"{metric}_ci95_low"
        ] = (
            ci[
                "ci_95_low"
            ]
        )

        row[
            f"{metric}_ci95_high"
        ] = (
            ci[
                "ci_95_high"
            ]
        )

    stat_rows.append(
        row
    )


stat_df = pd.DataFrame(
    stat_rows
)

stat_file = (
    OUTPUT_DIR
    / "paired_statistical_tests.csv"
)

stat_df.to_csv(
    stat_file,
    index=False,
)


# ============================================================
# External per-document analysis
# ============================================================

external_document_rows = []

document_values = (
    external_base[
        "source_document"
    ]
    .fillna(
        "(missing)"
    )
    .astype(str)
)


for document in sorted(
    document_values.unique()
):

    indexes = np.where(
        document_values.to_numpy()
        == document
    )[0]

    y_true = (
        external_base.iloc[
            indexes
        ][
            "true_label"
        ]
        .astype(int)
        .to_numpy()
    )

    for dataframe, model_name in [

        (
            external_base,
            "Base BERT",
        ),

        (
            external_grid,
            "GridBERT v0.3",
        ),

    ]:

        y_pred = (
            dataframe.iloc[
                indexes
            ][
                "predicted_label"
            ]
            .astype(int)
            .to_numpy()
        )

        metrics = (
            calculate_metrics(
                y_true,
                y_pred,
            )
        )

        external_document_rows.append({

            "source_document":
                document,

            "model":
                model_name,

            "passages":
                len(
                    indexes
                ),

            "not_reliability":
                int(
                    np.sum(
                        y_true
                        == 0
                    )
                ),

            "reliability":
                int(
                    np.sum(
                        y_true
                        == 1
                    )
                ),

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "precision":
                metrics[
                    "precision"
                ],

            "recall":
                metrics[
                    "recall"
                ],

            "f1":
                metrics[
                    "f1"
                ],

            "specificity":
                metrics[
                    "specificity"
                ],

            "macro_f1":
                metrics[
                    "macro_f1"
                ],

            "tn":
                metrics[
                    "tn"
                ],

            "fp":
                metrics[
                    "fp"
                ],

            "fn":
                metrics[
                    "fn"
                ],

            "tp":
                metrics[
                    "tp"
                ],
        })


external_document_df = (
    pd.DataFrame(
        external_document_rows
    )
)

external_document_file = (
    OUTPUT_DIR
    / "external_per_document_results.csv"
)

external_document_df.to_csv(
    external_document_file,
    index=False,
)


# ============================================================
# Disagreement files
# ============================================================

def save_disagreements(
    base,
    grid,
    dataset_name,
    output_file,
):

    output = base.copy()

    output = output.rename(
        columns={
            "predicted_label":
                "base_prediction",

            "reliability_probability":
                "base_reliability_probability",

            "correct":
                "base_correct",
        }
    )

    output[
        "grid_prediction"
    ] = (
        grid[
            "predicted_label"
        ]
        .astype(int)
    )

    output[
        "grid_reliability_probability"
    ] = (
        grid[
            "reliability_probability"
        ]
    )

    output[
        "grid_correct"
    ] = (
        grid[
            "correct"
        ]
        .astype(int)
    )

    disagreement = output[
        output[
            "base_prediction"
        ]
        !=
        output[
            "grid_prediction"
        ]
    ].copy()

    disagreement[
        "which_model_correct"
    ] = np.where(

        disagreement[
            "grid_correct"
        ]
        >
        disagreement[
            "base_correct"
        ],

        "GridBERT v0.3",

        "Base BERT",
    )

    disagreement.insert(
        0,
        "dataset",
        dataset_name,
    )

    disagreement.to_csv(
        output_file,
        index=False,
    )


save_disagreements(

    internal_base,
    internal_grid,

    "Internal Test",

    OUTPUT_DIR
    / "internal_disagreements.csv",
)


save_disagreements(

    external_base,
    external_grid,

    "External Unseen Test",

    OUTPUT_DIR
    / "external_disagreements.csv",
)


# ============================================================
# JSON reproducibility summary
# ============================================================

summary = {

    "analysis": (
        "Paired comparison of frozen "
        "Base BERT and GridBERT v0.3 "
        "reliability classifiers."
    ),

    "bootstrap": {

        "method":
            "paired percentile bootstrap",

        "samples":
            BOOTSTRAP_SAMPLES,

        "confidence_level":
            0.95,

        "seed":
            SEED,
    },

    "mcnemar": {

        "method":
            "two-sided exact McNemar test",

        "multiple_test_correction":
            "Holm correction",

        "number_of_tests":
            2,
    },

    "internal_test":
        internal_analysis,

    "external_test":
        external_analysis,
}


summary_file = (
    OUTPUT_DIR
    / "statistical_analysis_summary.json"
)


with open(
    summary_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=2,
    )


# ============================================================
# Console summary
# ============================================================

print()
print("=" * 72)
print("FINAL RESULTS")
print("=" * 72)


for analysis in analyses:

    print()
    print(
        analysis[
            "dataset"
        ]
    )

    print("-" * 72)

    print(
        "Base BERT:"
    )

    print(
        f"  Accuracy: "
        f"{analysis['base']['accuracy']:.4f}"
    )

    print(
        f"  F1:       "
        f"{analysis['base']['f1']:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{analysis['base']['macro_f1']:.4f}"
    )

    print(
        f"  Specificity: "
        f"{analysis['base']['specificity']:.4f}"
    )

    print()

    print(
        "GridBERT v0.3:"
    )

    print(
        f"  Accuracy: "
        f"{analysis['grid']['accuracy']:.4f}"
    )

    print(
        f"  F1:       "
        f"{analysis['grid']['f1']:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{analysis['grid']['macro_f1']:.4f}"
    )

    print(
        f"  Specificity: "
        f"{analysis['grid']['specificity']:.4f}"
    )

    print()

    print(
        "Paired comparison:"
    )

    print(
        f"  Base only correct: "
        f"{analysis['mcnemar']['base_only_correct']}"
    )

    print(
        f"  Grid only correct: "
        f"{analysis['mcnemar']['grid_only_correct']}"
    )

    print(
        f"  Exact McNemar p: "
        f"{analysis['mcnemar']['mcnemar_exact_p']:.6f}"
    )

    print(
        f"  Holm adjusted p: "
        f"{analysis['mcnemar']['holm_adjusted_p']:.6f}"
    )

    print()

    for metric in [
        "accuracy",
        "f1",
        "specificity",
        "macro_f1",
    ]:

        actual_difference = (
            analysis[
                "grid"
            ][
                metric
            ]
            -
            analysis[
                "base"
            ][
                metric
            ]
        )

        ci = (
            analysis[
                "bootstrap"
            ][
                metric
            ]
        )

        print(
            f"  {metric} difference: "
            f"{actual_difference:+.4f} "
            f"[95% CI "
            f"{ci['ci_95_low']:+.4f}, "
            f"{ci['ci_95_high']:+.4f}]"
        )


print()
print("=" * 72)
print("OUTPUT FILES")
print("=" * 72)

print()
print(
    comparison_file
)

print(
    stat_file
)

print(
    external_document_file
)

print(
    OUTPUT_DIR
    / "internal_disagreements.csv"
)

print(
    OUTPUT_DIR
    / "external_disagreements.csv"
)

print(
    summary_file
)