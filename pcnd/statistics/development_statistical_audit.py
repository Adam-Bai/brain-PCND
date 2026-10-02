from __future__ import annotations

from pathlib import Path
import os
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


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

OUT = (
    BASE
    / "final_statistical_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 9)
]


METRICS = [
    "pearson",
    "spearman",
    "mae",
    "top10_recall",
]


HIGHER_IS_BETTER = {
    "pearson": True,
    "spearman": True,
    "mae": False,
    "top10_recall": True,
}


BOOTSTRAP_N = 100_000
BASE_SEED = 20260926


# ============================================================
# LOAD RAW REPEAT-LEVEL METRICS
# ============================================================

frames = []

for sub in SUBJECTS:

    f = (
        BASE
        / sub
        / "test_metrics_raw.csv"
    )

    if not f.exists():
        raise FileNotFoundError(
            f
        )

    x = pd.read_csv(f)

    required = {
        "heldout",
        "context_k",
        "repeat",
        "condition",
        "target",
        *METRICS,
    }

    missing = (
        required
        -
        set(x.columns)
    )

    if missing:
        raise RuntimeError(
            f"{sub}: missing columns "
            f"{sorted(missing)}"
        )

    frames.append(x)


raw = pd.concat(
    frames,
    ignore_index=True,
)


# ============================================================
# PATIENT IS THE INFERENCE UNIT
#
# First average random-context repeats WITHIN patient.
# No pair-level or repeat-level pseudoreplication.
# ============================================================

patient_condition = (
    raw
    .groupby(
        [
            "heldout",
            "context_k",
            "condition",
            "target",
        ],
        as_index=False,
    )
    .agg(
        n_repeats=(
            "repeat",
            "nunique",
        ),

        pearson=(
            "pearson",
            "mean",
        ),

        spearman=(
            "spearman",
            "mean",
        ),

        mae=(
            "mae",
            "mean",
        ),

        top10_recall=(
            "top10_recall",
            "mean",
        ),
    )
)


patient_condition.to_csv(
    OUT
    / "patient_condition_summary.csv",
    index=False,
)


# ============================================================
# CONTRAST DEFINITIONS
#
# Positive delta ALWAYS means the more complex / intended
# condition is better.
# ============================================================

CONTRAST_SPECS = []


# P -> P+C
for k in [
    1,
    3,
    5,
    10,
]:

    CONTRAST_SPECS.append({
        "contrast":
            "global_vs_population",

        "context_k":
            k,

        "condition_a":
            "global",

        "condition_b":
            "population",
    })


# P+C -> P+C+A
for k in [
    1,
    3,
    5,
    10,
]:

    CONTRAST_SPECS.append({
        "contrast":
            "node_vs_global",

        "context_k":
            k,

        "condition_a":
            "node",

        "condition_b":
            "global",
    })


# P+C+A -> P+C+A+T
#
# Relation is deliberately not interpreted at k=1.
for k in [
    3,
    5,
    10,
]:

    CONTRAST_SPECS.append({
        "contrast":
            "relation_vs_node",

        "context_k":
            k,

        "condition_a":
            "relation",

        "condition_b":
            "node",
    })


# STRICT relational correspondence control:
#
# Same patient.
# Same recording node.
# Same node susceptibility.
# Same response marginal.
# Stimulation identity permuted within recording channel.
for k in [
    3,
    5,
    10,
]:

    CONTRAST_SPECS.append({
        "contrast":
            "relation_vs_relation_perm",

        "context_k":
            k,

        "condition_a":
            "relation",

        "condition_b":
            "relation_perm",
    })


# ============================================================
# HELPERS
# ============================================================

def stable_seed(
    *parts,
):

    s = "|".join(
        map(
            str,
            parts,
        )
    )

    h = hashlib.sha256(
        s.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        BASE_SEED
        +
        int(
            h[:8],
            16,
        )
    ) % (
        2**32
    )


def percentile_bootstrap(
    values,
    statistic="mean",
    n_boot=BOOTSTRAP_N,
    seed=0,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    n = len(x)

    if n == 0:

        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        n,
        size=(
            n_boot,
            n,
        ),
    )

    samples = x[
        indices
    ]

    if statistic == "mean":

        boot = samples.mean(
            axis=1
        )

    elif statistic == "median":

        boot = np.median(
            samples,
            axis=1,
        )

    else:

        raise ValueError(
            statistic
        )

    low, high = np.quantile(
        boot,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(low),
        float(high),
    )


def exact_signflip_p(
    values,
):

    """
    Two-sided exact paired sign-flip/randomization test
    on the patient-level MEAN difference.

    With n=8 there are only 2^8 = 256 assignments.
    """

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    # Exact zeros carry no sign information.
    x = x[
        np.abs(x)
        >
        1e-15
    ]

    n = len(x)

    if n == 0:

        return 1.0

    obs = abs(
        x.mean()
    )

    if n <= 20:

        count = 0
        total = 0

        for signs in itertools.product(
            [
                -1.0,
                1.0,
            ],
            repeat=n,
        ):

            signs = np.asarray(
                signs,
                dtype=float,
            )

            stat = abs(
                (
                    signs
                    *
                    x
                ).mean()
            )

            if stat >= (
                obs
                -
                1e-15
            ):

                count += 1

            total += 1

        return float(
            count
            /
            total
        )

    # Fallback for a future larger cohort.
    rng = np.random.default_rng(
        BASE_SEED
    )

    n_perm = 200_000

    signs = rng.choice(
        [
            -1.0,
            1.0,
        ],
        size=(
            n_perm,
            n,
        ),
    )

    stats = np.abs(
        (
            signs
            *
            x[
                None,
                :
            ]
        ).mean(
            axis=1
        )
    )

    return float(
        (
            1
            +
            np.sum(
                stats
                >= obs
            )
        )
        /
        (
            n_perm
            +
            1
        )
    )


def wilcoxon_p(
    values,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

        return np.nan

    if np.all(
        np.abs(x)
        <
        1e-15
    ):

        return 1.0

    try:

        result = wilcoxon(
            x,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            method="auto",
        )

        return float(
            result.pvalue
        )

    except Exception:

        return np.nan


def rank_biserial_from_differences(
    values,
):

    """
    Paired rank-biserial correlation:
      (W+ - W-) / (W+ + W-)
    """

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    x = x[
        np.abs(x)
        >
        1e-15
    ]

    if len(x) == 0:

        return 0.0

    abs_x = np.abs(x)

    order = abs_x.argsort()

    ranks = np.empty(
        len(x),
        dtype=float,
    )

    ranks[
        order
    ] = np.arange(
        1,
        len(x) + 1,
        dtype=float,
    )

    # Handle ties with average ranks.
    unique_vals = np.unique(
        abs_x
    )

    for v in unique_vals:

        idx = np.where(
            abs_x == v
        )[0]

        if len(idx) > 1:

            ranks[
                idx
            ] = ranks[
                idx
            ].mean()

    w_pos = ranks[
        x > 0
    ].sum()

    w_neg = ranks[
        x < 0
    ].sum()

    denom = (
        w_pos
        +
        w_neg
    )

    if denom == 0:

        return 0.0

    return float(
        (
            w_pos
            -
            w_neg
        )
        /
        denom
    )


def holm_adjust(
    pvalues,
):

    p = np.asarray(
        pvalues,
        dtype=float,
    )

    out = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid = np.where(
        np.isfinite(p)
    )[0]

    if len(valid) == 0:

        return out

    pv = p[
        valid
    ]

    order = np.argsort(
        pv
    )

    m = len(pv)

    adjusted_sorted = np.zeros(
        m,
        dtype=float,
    )

    running = 0.0

    for rank, idx in enumerate(
        order
    ):

        candidate = (
            m
            -
            rank
        ) * pv[idx]

        running = max(
            running,
            candidate,
        )

        adjusted_sorted[
            rank
        ] = min(
            1.0,
            running,
        )

    for rank, idx in enumerate(
        order
    ):

        out[
            valid[idx]
        ] = adjusted_sorted[
            rank
        ]

    return out


# ============================================================
# BUILD PATIENT-LEVEL CONTRASTS
# ============================================================

patient_delta_rows = []
summary_rows = []


for spec in CONTRAST_SPECS:

    k = spec[
        "context_k"
    ]

    cond_a = spec[
        "condition_a"
    ]

    cond_b = spec[
        "condition_b"
    ]

    contrast = spec[
        "contrast"
    ]

    for target in [
        "early",
        "late",
    ]:

        a = patient_condition[
            (
                patient_condition[
                    "context_k"
                ]
                == k
            )
            &
            (
                patient_condition[
                    "condition"
                ]
                == cond_a
            )
            &
            (
                patient_condition[
                    "target"
                ]
                == target
            )
        ].copy()

        b = patient_condition[
            (
                patient_condition[
                    "context_k"
                ]
                == k
            )
            &
            (
                patient_condition[
                    "condition"
                ]
                == cond_b
            )
            &
            (
                patient_condition[
                    "target"
                ]
                == target
            )
        ].copy()

        for metric in METRICS:

            aa = a[
                [
                    "heldout",
                    metric,
                ]
            ].rename(
                columns={
                    metric:
                        "value_a"
                }
            )

            bb = b[
                [
                    "heldout",
                    metric,
                ]
            ].rename(
                columns={
                    metric:
                        "value_b"
                }
            )

            merged = aa.merge(
                bb,
                on="heldout",
                how="inner",
                validate="one_to_one",
            )

            if len(merged) == 0:

                continue

            if HIGHER_IS_BETTER[
                metric
            ]:

                delta = (
                    merged[
                        "value_a"
                    ]
                    -
                    merged[
                        "value_b"
                    ]
                )

                delta_definition = (
                    f"{cond_a} - "
                    f"{cond_b}"
                )

            else:

                # Positive ALWAYS means A is better.
                delta = (
                    merged[
                        "value_b"
                    ]
                    -
                    merged[
                        "value_a"
                    ]
                )

                delta_definition = (
                    f"{cond_b} - "
                    f"{cond_a}"
                    " (MAE reduction)"
                )

            merged[
                "delta"
            ] = delta

            for _, row in (
                merged.iterrows()
            ):

                patient_delta_rows.append({
                    "heldout":
                        row[
                            "heldout"
                        ],

                    "context_k":
                        k,

                    "target":
                        target,

                    "contrast":
                        contrast,

                    "condition_a":
                        cond_a,

                    "condition_b":
                        cond_b,

                    "metric":
                        metric,

                    "value_a":
                        row[
                            "value_a"
                        ],

                    "value_b":
                        row[
                            "value_b"
                        ],

                    "delta":
                        row[
                            "delta"
                        ],
                })

            d = delta.to_numpy(
                dtype=float
            )

            d = d[
                np.isfinite(d)
            ]

            n = len(d)

            if n == 0:

                continue

            wins = int(
                np.sum(
                    d > 1e-12
                )
            )

            losses = int(
                np.sum(
                    d < -1e-12
                )
            )

            ties = int(
                n
                -
                wins
                -
                losses
            )

            mean_delta = float(
                np.mean(d)
            )

            median_delta = float(
                np.median(d)
            )

            ci_mean_low, ci_mean_high = (
                percentile_bootstrap(
                    d,
                    statistic="mean",
                    seed=stable_seed(
                        contrast,
                        k,
                        target,
                        metric,
                        "mean",
                    ),
                )
            )

            (
                ci_median_low,
                ci_median_high,
            ) = percentile_bootstrap(
                d,
                statistic="median",
                seed=stable_seed(
                    contrast,
                    k,
                    target,
                    metric,
                    "median",
                ),
            )

            std_delta = (
                float(
                    np.std(
                        d,
                        ddof=1,
                    )
                )
                if n > 1
                else np.nan
            )

            dz = (
                mean_delta
                /
                std_delta

                if (
                    n > 1
                    and
                    std_delta
                    > 1e-15
                )

                else np.nan
            )

            signflip_p = (
                exact_signflip_p(
                    d
                )
            )

            wx_p = wilcoxon_p(
                d
            )

            non_tied = (
                wins
                +
                losses
            )

            win_fraction = (
                wins
                /
                non_tied

                if non_tied > 0
                else np.nan
            )

            rbc = (
                rank_biserial_from_differences(
                    d
                )
            )

            relative_mae_improvement = (
                float(
                    np.mean(
                        (
                            merged[
                                "value_b"
                            ].to_numpy(
                                dtype=float
                            )
                            -
                            merged[
                                "value_a"
                            ].to_numpy(
                                dtype=float
                            )
                        )
                        /
                        np.maximum(
                            np.abs(
                                merged[
                                    "value_b"
                                ].to_numpy(
                                    dtype=float
                                )
                            ),
                            1e-12,
                        )
                    )
                    *
                    100.0
                )
                if metric == "mae"
                else np.nan
            )

            summary_rows.append({
                "contrast":
                    contrast,

                "context_k":
                    k,

                "target":
                    target,

                "metric":
                    metric,

                "condition_a":
                    cond_a,

                "condition_b":
                    cond_b,

                "delta_definition":
                    delta_definition,

                "n_subjects":
                    n,

                "mean_delta":
                    mean_delta,

                "median_delta":
                    median_delta,

                "std_delta":
                    std_delta,

                "bootstrap_mean_ci_low":
                    ci_mean_low,

                "bootstrap_mean_ci_high":
                    ci_mean_high,

                "bootstrap_median_ci_low":
                    ci_median_low,

                "bootstrap_median_ci_high":
                    ci_median_high,

                "wins":
                    wins,

                "ties":
                    ties,

                "losses":
                    losses,

                "win_fraction_non_tied":
                    win_fraction,

                "paired_dz":
                    dz,

                "rank_biserial":
                    rbc,

                "exact_signflip_p":
                    signflip_p,

                "wilcoxon_p":
                    wx_p,

                "mean_relative_mae_improvement_pct":
                    relative_mae_improvement,
            })


patient_deltas = pd.DataFrame(
    patient_delta_rows
)

summary = pd.DataFrame(
    summary_rows
)


patient_deltas.to_csv(
    OUT
    / "patient_level_paired_deltas.csv",
    index=False,
)


# ============================================================
# MULTIPLICITY
#
# HAPwave is DEVELOPMENT DATA.
# These adjusted p values are descriptive / exploratory.
#
# We nevertheless audit the same family that will later be
# pre-specified for the independent lockbox.
# ============================================================

summary[
    "is_future_lockbox_primary"
] = (
    (
        summary[
            "contrast"
        ]
        ==
        "relation_vs_relation_perm"
    )
    &
    (
        summary[
            "context_k"
        ]
        ==
        10
    )
    &
    (
        summary[
            "target"
        ]
        ==
        "late"
    )
    &
    (
        summary[
            "metric"
        ]
        ==
        "spearman"
    )
)


# Key secondary family planned for INDEPENDENT LOCKBOX.
#
# Primary is NOT included here because it is tested separately.
secondary_mask = (
    (
        (
            summary[
                "contrast"
            ]
            ==
            "relation_vs_node"
        )
        &
        (
            summary[
                "context_k"
            ]
            ==
            10
        )
        &
        (
            summary[
                "target"
            ]
            ==
            "late"
        )
        &
        (
            summary[
                "metric"
            ]
            ==
            "spearman"
        )
    )
    |
    (
        (
            summary[
                "contrast"
            ]
            ==
            "relation_vs_relation_perm"
        )
        &
        (
            summary[
                "context_k"
            ]
            ==
            10
        )
        &
        (
            summary[
                "target"
            ]
            ==
            "late"
        )
        &
        (
            summary[
                "metric"
            ].isin(
                [
                    "pearson",
                    "mae",
                ]
            )
        )
    )
    |
    (
        (
            summary[
                "contrast"
            ]
            ==
            "relation_vs_relation_perm"
        )
        &
        (
            summary[
                "context_k"
            ].isin(
                [
                    3,
                    5,
                ]
            )
        )
        &
        (
            summary[
                "target"
            ]
            ==
            "late"
        )
        &
        (
            summary[
                "metric"
            ]
            ==
            "spearman"
        )
    )
    |
    (
        (
            summary[
                "contrast"
            ]
            ==
            "relation_vs_relation_perm"
        )
        &
        (
            summary[
                "context_k"
            ]
            ==
            10
        )
        &
        (
            summary[
                "target"
            ]
            ==
            "early"
        )
        &
        (
            summary[
                "metric"
            ]
            ==
            "spearman"
        )
    )
)


summary[
    "is_future_lockbox_key_secondary"
] = secondary_mask


summary[
    "holm_key_secondary_p"
] = np.nan


if secondary_mask.sum() > 0:

    adjusted = holm_adjust(
        summary.loc[
            secondary_mask,
            "exact_signflip_p",
        ].to_numpy()
    )

    summary.loc[
        secondary_mask,
        "holm_key_secondary_p",
    ] = adjusted


# Extra conservative development-wide family:
# all strict correspondence tests, all endpoints,
# k = 3 / 5 / 10, both early and late.
all_corr_mask = (
    (
        summary[
            "contrast"
        ]
        ==
        "relation_vs_relation_perm"
    )
    &
    (
        summary[
            "context_k"
        ].isin(
            [
                3,
                5,
                10,
            ]
        )
    )
)


summary[
    "holm_all_correspondence_p"
] = np.nan


if all_corr_mask.sum() > 0:

    adjusted = holm_adjust(
        summary.loc[
            all_corr_mask,
            "exact_signflip_p",
        ].to_numpy()
    )

    summary.loc[
        all_corr_mask,
        "holm_all_correspondence_p",
    ] = adjusted


summary.to_csv(
    OUT
    / "paired_contrast_statistical_summary.csv",
    index=False,
)


# ============================================================
# CORRESPONDENCE TRAJECTORY
# ============================================================

corr_traj = summary[
    (
        summary[
            "contrast"
        ]
        ==
        "relation_vs_relation_perm"
    )
    &
    (
        summary[
            "metric"
        ].isin(
            [
                "pearson",
                "spearman",
                "mae",
            ]
        )
    )
].copy()


corr_traj = corr_traj[
    [
        "context_k",
        "target",
        "metric",
        "n_subjects",
        "mean_delta",
        "median_delta",
        "bootstrap_mean_ci_low",
        "bootstrap_mean_ci_high",
        "wins",
        "ties",
        "losses",
        "exact_signflip_p",
        "wilcoxon_p",
        "holm_all_correspondence_p",
    ]
].sort_values(
    [
        "target",
        "metric",
        "context_k",
    ]
)


corr_traj.to_csv(
    OUT
    / "correspondence_trajectory.csv",
    index=False,
)


# ============================================================
# PRIMARY-LIKE DEVELOPMENT SNAPSHOT
#
# IMPORTANT:
# This is NOT confirmatory because HAPwave was repeatedly used
# during architecture development.
# ============================================================

primary_like = summary[
    summary[
        "is_future_lockbox_primary"
    ]
].copy()


primary_like.to_csv(
    OUT
    / "future_lockbox_primary_endpoint_development_snapshot.csv",
    index=False,
)


# ============================================================
# PAPER-READY NESTED TABLE
# ============================================================

group = (
    patient_condition
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

        spearman_std=(
            "spearman",
            "std",
        ),

        mae_mean=(
            "mae",
            "mean",
        ),

        mae_std=(
            "mae",
            "std",
        ),

        top10_mean=(
            "top10_recall",
            "mean",
        ),
    )
)


group.to_csv(
    OUT
    / "paper_ready_group_metrics.csv",
    index=False,
)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report = []

report.append(
    "# V3f Final Strict Statistical Audit"
)

report.append("")

report.append(
    "**Status:** HAPwave is a development cohort, "
    "not an independent confirmatory lockbox."
)

report.append("")

report.append(
    "All inferential statistics use the patient as the "
    "independent unit. Random context repeats are averaged "
    "within patient before hypothesis testing."
)

report.append("")

report.append(
    "Positive paired deltas always indicate improvement "
    "of the intended/more complete condition. For MAE, "
    "the sign is reversed so that positive means lower MAE."
)

report.append("")

report.append(
    "## Future independent-lockbox primary endpoint"
)

report.append("")

if len(primary_like):

    r = primary_like.iloc[0]

    report.append(
        "- Contrast: relation vs within-recording-channel "
        "stimulation permutation"
    )

    report.append(
        "- Response: late"
    )

    report.append(
        "- Context budget: k = 10"
    )

    report.append(
        "- Metric: patient-level Spearman map correlation"
    )

    report.append(
        f"- HAPwave development mean delta: "
        f"{r['mean_delta']:.4f}"
    )

    report.append(
        f"- 95% patient bootstrap CI: "
        f"[{r['bootstrap_mean_ci_low']:.4f}, "
        f"{r['bootstrap_mean_ci_high']:.4f}]"
    )

    report.append(
        f"- Wins / ties / losses: "
        f"{int(r['wins'])}/"
        f"{int(r['ties'])}/"
        f"{int(r['losses'])}"
    )

    report.append(
        f"- Exact sign-flip p (development only): "
        f"{r['exact_signflip_p']:.6f}"
    )

report.append("")

report.append(
    "## Late-response strict correspondence trajectory"
)

report.append("")

late_corr = corr_traj[
    (
        corr_traj[
            "target"
        ]
        ==
        "late"
    )
    &
    (
        corr_traj[
            "metric"
        ].isin(
            [
                "pearson",
                "spearman",
            ]
        )
    )
]


for _, r in (
    late_corr.iterrows()
):

    report.append(
        f"- k={int(r['context_k'])}, "
        f"{r['metric']}: "
        f"mean Δ={r['mean_delta']:.4f}, "
        f"95% CI "
        f"[{r['bootstrap_mean_ci_low']:.4f}, "
        f"{r['bootstrap_mean_ci_high']:.4f}], "
        f"W/T/L="
        f"{int(r['wins'])}/"
        f"{int(r['ties'])}/"
        f"{int(r['losses'])}, "
        f"exact p={r['exact_signflip_p']:.6f}"
    )


report.append("")

report.append(
    "## Claim boundary"
)

report.append("")

report.append(
    "These development results support predictive information "
    "in stimulation-response correspondence after controlling "
    "for patient-wide calibration and persistent "
    "recording-node susceptibility. They do not by themselves "
    "establish direct anatomical connectivity, polysynaptic "
    "mechanisms, causal clinical benefit, or epileptogenic-zone "
    "ground truth."
)

report.append("")

report.append(
    "Independent external validation is required for the "
    "confirmatory claim because HAPwave was repeatedly examined "
    "during model development."
)


(
    OUT
    / "STATISTICAL_AUDIT.md"
).write_text(
    "\n".join(report),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print(
    "=" * 125
)
print(
    "V3f FINAL STRICT — STATISTICAL AUDIT"
)
print(
    "=" * 125
)


print()
print(
    "FUTURE LOCKBOX PRIMARY ENDPOINT "
    "(HAPWAVE DEVELOPMENT SNAPSHOT)"
)
print(
    "-" * 125
)

if len(primary_like):

    cols = [
        "context_k",
        "target",
        "metric",
        "n_subjects",
        "mean_delta",
        "median_delta",
        "bootstrap_mean_ci_low",
        "bootstrap_mean_ci_high",
        "wins",
        "ties",
        "losses",
        "rank_biserial",
        "exact_signflip_p",
        "wilcoxon_p",
    ]

    print(
        primary_like[
            cols
        ]
        .round(5)
        .to_string(
            index=False
        )
    )


print()
print(
    "STRICT CORRESPONDENCE — LATE"
)
print(
    "-" * 125
)

print(
    corr_traj[
        (
            corr_traj[
                "target"
            ]
            ==
            "late"
        )
        &
        (
            corr_traj[
                "metric"
            ].isin(
                [
                    "pearson",
                    "spearman",
                    "mae",
                ]
            )
        )
    ]
    .round(5)
    .to_string(
        index=False
    )
)


print()
print(
    "Saved:",
    OUT
)
