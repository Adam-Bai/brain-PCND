from pathlib import Path

import os
import pandas as pd


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

BASE = (
    ROOT
    / "results"
    / "clinical_v9"
    / "v3f_final_strict"
)

SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 9)
]


frames = []

for sub in SUBJECTS:

    f = (
        BASE
        / sub
        / "fold_summary.csv"
    )

    if not f.exists():
        raise FileNotFoundError(f)

    frames.append(
        pd.read_csv(f)
    )


folds = pd.concat(
    frames,
    ignore_index=True,
)

folds.to_csv(
    BASE
    / "all_available_fold_summaries.csv",
    index=False,
)


group = (
    folds
    .groupby(
        [
            "context_k",
            "condition",
            "target",
        ],
        as_index=False,
    )
    .agg(
        n_subjects=(
            "heldout",
            "nunique",
        ),

        pearson_mean=(
            "pearson",
            "mean",
        ),

        pearson_std=(
            "pearson",
            "std",
        ),

        spearman_mean=(
            "spearman",
            "mean",
        ),

        mae_mean=(
            "mae",
            "mean",
        ),

        top10_mean=(
            "top10_recall",
            "mean",
        ),
    )
)


group.to_csv(
    BASE
    / "group_summary.csv",
    index=False,
)


print()
print("=" * 120)
print("V3f FINAL STRICT — GROUP SUMMARY")
print("=" * 120)

print(
    group
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# NESTED GAINS
# ============================================================

wide = folds.pivot_table(
    index=[
        "heldout",
        "context_k",
        "target",
    ],

    columns="condition",

    values=[
        "pearson",
        "spearman",
        "mae",
    ],
).reset_index()


wide.columns = [
    "_".join(c).strip("_")
    if isinstance(c, tuple)
    else c
    for c in wide.columns
]


for metric in [
    "pearson",
    "spearman",
]:

    if (
        f"{metric}_global"
        in wide
        and
        f"{metric}_population"
        in wide
    ):

        wide[
            f"gain_global_{metric}"
        ] = (
            wide[
                f"{metric}_global"
            ]
            -
            wide[
                f"{metric}_population"
            ]
        )

    if (
        f"{metric}_node"
        in wide
        and
        f"{metric}_global"
        in wide
    ):

        wide[
            f"gain_node_{metric}"
        ] = (
            wide[
                f"{metric}_node"
            ]
            -
            wide[
                f"{metric}_global"
            ]
        )

    if (
        f"{metric}_relation"
        in wide
        and
        f"{metric}_node"
        in wide
    ):

        wide[
            f"gain_relation_{metric}"
        ] = (
            wide[
                f"{metric}_relation"
            ]
            -
            wide[
                f"{metric}_node"
            ]
        )

    if (
        f"{metric}_relation"
        in wide
        and
        f"{metric}_relation_perm"
        in wide
    ):

        wide[
            f"gain_correspondence_{metric}"
        ] = (
            wide[
                f"{metric}_relation"
            ]
            -
            wide[
                f"{metric}_relation_perm"
            ]
        )


if (
    "mae_population" in wide
    and
    "mae_global" in wide
):

    wide[
        "gain_global_mae"
    ] = (
        wide[
            "mae_population"
        ]
        -
        wide[
            "mae_global"
        ]
    )


if (
    "mae_global" in wide
    and
    "mae_node" in wide
):

    wide[
        "gain_node_mae"
    ] = (
        wide[
            "mae_global"
        ]
        -
        wide[
            "mae_node"
        ]
    )


if (
    "mae_node" in wide
    and
    "mae_relation" in wide
):

    wide[
        "gain_relation_mae"
    ] = (
        wide[
            "mae_node"
        ]
        -
        wide[
            "mae_relation"
        ]
    )


wide.to_csv(
    BASE
    / "patient_level_nested_gains.csv",
    index=False,
)


focus_cols = [
    "heldout",
    "context_k",
    "target",
]

focus_cols += [
    c
    for c in [
        "gain_global_pearson",
        "gain_node_pearson",
        "gain_relation_pearson",
        "gain_correspondence_pearson",

        "gain_global_spearman",
        "gain_node_spearman",
        "gain_relation_spearman",
        "gain_correspondence_spearman",

        "gain_global_mae",
        "gain_node_mae",
        "gain_relation_mae",
    ]
    if c in wide.columns
]


print()
print("=" * 120)
print("PATIENT-LEVEL NESTED GAINS")
print("=" * 120)

print(
    wide[
        focus_cols
    ]
    .round(4)
    .to_string(
        index=False
    )
)
