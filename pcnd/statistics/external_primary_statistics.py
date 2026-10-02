from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime, timezone

import hashlib
import json

import numpy as np
import pandas as pd

from scipy.stats import (
    beta,
    binomtest,
    wilcoxon,
)


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

OUT = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_frozen_inference_v1"
)

PATIENT = (
    OUT
    / "FROZEN_PATIENT_METRIC_SUMMARY.csv"
)

INFERENCE_MANIFEST = (
    OUT
    / "FROZEN_INFERENCE_MANIFEST.json"
)

STAT_PROTOCOL = (
    OUT
    / "PREINFERENCE_STATISTICAL_OPERATIONALIZATION.json"
)

PRIMARY_CSV = (
    OUT
    / "PRIMARY_PATIENT_DELTAS.csv"
)

SECONDARY_CSV = (
    OUT
    / "SECONDARY_FAMILY_RESULTS.csv"
)

RESULT_JSON = (
    OUT
    / "PRIMARY_EXTERNAL_RESULT.json"
)


def sha256_file(p):

    h = hashlib.sha256()

    with open(p, "rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


if (
    PRIMARY_CSV.exists()
    or
    SECONDARY_CSV.exists()
    or
    RESULT_JSON.exists()
):

    raise RuntimeError(
        "External statistical result "
        "already exists; refusing overwrite."
    )


protocol = json.loads(
    STAT_PROTOCOL.read_text()
)

if (
    protocol[
        "external_model_performance_observed_before_freeze"
    ]
    is not False
):

    raise RuntimeError(
        "Statistical operationalization "
        "was not frozen outcome-blind."
    )


m = json.loads(
    INFERENCE_MANIFEST.read_text()
)

if (
    sha256_file(
        PATIENT
    )
    !=
    m[
        "patient_summary_sha256"
    ]
):

    raise RuntimeError(
        "Patient summary hash mismatch."
    )


df = pd.read_csv(
    PATIENT
)


def one_condition(
    k,
    target,
    condition,
):

    x = df[
        (
            df[
                "context_k"
            ]
            ==
            k
        )
        &
        (
            df[
                "target"
            ]
            ==
            target
        )
        &
        (
            df[
                "condition"
            ]
            ==
            condition
        )
    ].copy()

    if (
        len(x)
        !=
        74
    ):

        raise RuntimeError(
            f"Expected 74 patient rows for "
            f"k={k}, target={target}, "
            f"condition={condition}; "
            f"got {len(x)}"
        )

    return (
        x
        .set_index(
            "subject"
        )
        .sort_index()
    )


def paired_delta(
    k,
    target,
    metric,
    a,
    b,
    lower_is_better=False,
):

    A = one_condition(
        k,
        target,
        a,
    )

    B = one_condition(
        k,
        target,
        b,
    )

    if not A.index.equals(
        B.index
    ):

        raise RuntimeError(
            "Patient alignment mismatch."
        )


    if lower_is_better:

        # positive = A better than B
        delta = (
            B[
                metric
            ]
            -
            A[
                metric
            ]
        )

    else:

        delta = (
            A[
                metric
            ]
            -
            B[
                metric
            ]
        )


    return pd.DataFrame({
        "subject":
            A.index,

        "a":
            A[
                metric
            ].to_numpy(),

        "b":
            B[
                metric
            ].to_numpy(),

        "delta":
            delta.to_numpy(),
    })


def bootstrap_mean_ci(
    delta,
    B,
    seed,
    batch=10000,
):

    delta = np.asarray(
        delta,
        dtype=float,
    )

    n = len(
        delta
    )

    rng = np.random.default_rng(
        seed
    )

    vals = np.empty(
        B,
        dtype=np.float64,
    )

    pos = 0

    while pos < B:

        b = min(
            batch,
            B - pos,
        )

        idx = rng.integers(
            0,
            n,
            size=(
                b,
                n,
            ),
        )

        vals[
            pos:
            pos+b
        ] = delta[
            idx
        ].mean(
            axis=1
        )

        pos += b


    return (
        float(
            np.quantile(
                vals,
                0.025,
            )
        ),

        float(
            np.quantile(
                vals,
                0.975,
            )
        ),
    )


def mc_signflip(
    delta,
    B,
    seed,
    batch=50000,
):

    delta = np.asarray(
        delta,
        dtype=np.float64,
    )

    n = len(
        delta
    )

    obs = abs(
        float(
            delta.mean()
        )
    )

    rng = np.random.default_rng(
        seed
    )

    extreme = 0
    done = 0


    while done < B:

        b = min(
            batch,
            B - done,
        )

        bits = rng.integers(
            0,
            2,
            size=(
                b,
                n,
            ),
            dtype=np.int8,
        )

        signs = (
            bits
            *
            2
            -
            1
        )


        null = np.abs(
            (
                signs
                *
                delta[
                    None,
                    :
                ]
            ).mean(
                axis=1
            )
        )


        extreme += int(
            np.count_nonzero(
                null
                >=
                (
                    obs
                    -
                    1e-15
                )
            )
        )

        done += b


    p = (
        extreme + 1
    ) / (
        B + 1
    )


    # Monte-Carlo sampling uncertainty for the
    # unknown randomization tail probability.
    if extreme == 0:

        lo = 0.0

    else:

        lo = float(
            beta.ppf(
                0.025,
                extreme,
                B - extreme + 1,
            )
        )


    if extreme == B:

        hi = 1.0

    else:

        hi = float(
            beta.ppf(
                0.975,
                extreme + 1,
                B - extreme,
            )
        )


    return {
        "observed_abs_mean":
            obs,

        "n_randomizations":
            int(
                B
            ),

        "extreme":
            int(
                extreme
            ),

        "p_mc":
            float(
                p
            ),

        "mc_tail_probability_ci95":
            [
                lo,
                hi,
            ],
    }


def sensitivity_tests(
    delta,
):

    delta = np.asarray(
        delta,
        dtype=float,
    )

    nonzero = delta[
        np.abs(
            delta
        )
        >
        1e-15
    ]

    wins = int(
        (
            nonzero
            >
            0
        ).sum()
    )


    if len(
        nonzero
    ):

        sign_p = float(
            binomtest(
                wins,
                len(
                    nonzero
                ),
                0.5,
                alternative="two-sided",
            ).pvalue
        )

    else:

        sign_p = 1.0


    try:

        wil = float(
            wilcoxon(
                delta,
                zero_method="wilcox",
                alternative="two-sided",
                method="approx",
            ).pvalue
        )

    except Exception:

        wil = np.nan


    return {
        "n_nonzero":
            int(
                len(
                    nonzero
                )
            ),

        "wins":
            wins,

        "exact_binomial_sign_p":
            sign_p,

        "wilcoxon_approx_p":
            wil,
    }


# ============================================================
# PRIMARY
# ============================================================

primary = paired_delta(
    10,
    "late",
    "spearman",
    "relation",
    "relation_perm",
)


primary.to_csv(
    PRIMARY_CSV,
    index=False,
)


delta = primary[
    "delta"
].to_numpy(
    dtype=float
)


mean_delta = float(
    np.mean(
        delta
    )
)

median_delta = float(
    np.median(
        delta
    )
)

wins = int(
    (
        delta > 0
    ).sum()
)

ties = int(
    (
        np.abs(
            delta
        )
        <=
        1e-15
    ).sum()
)

losses = int(
    (
        delta < 0
    ).sum()
)


ci = bootstrap_mean_ci(
    delta,
    B=200000,
    seed=20260928,
)


primary_randomization = (
    mc_signflip(
        delta,
        B=10000000,
        seed=20260927,
    )
)


primary_sensitivity = (
    sensitivity_tests(
        delta
    )
)


# ============================================================
# SECONDARY FAMILY
# ============================================================

secondary_defs = [
    (
        "late_k10_relation_vs_node_spearman",
        10,
        "late",
        "spearman",
        "relation",
        "node",
        False,
    ),

    (
        "late_k10_relation_vs_perm_pearson",
        10,
        "late",
        "pearson",
        "relation",
        "relation_perm",
        False,
    ),

    (
        "late_k10_relation_vs_perm_mae_reduction",
        10,
        "late",
        "mae",
        "relation",
        "relation_perm",
        True,
    ),

    (
        "late_k5_relation_vs_perm_spearman",
        5,
        "late",
        "spearman",
        "relation",
        "relation_perm",
        False,
    ),

    (
        "late_k3_relation_vs_perm_spearman",
        3,
        "late",
        "spearman",
        "relation",
        "relation_perm",
        False,
    ),

    (
        "early_k10_relation_vs_perm_spearman",
        10,
        "early",
        "spearman",
        "relation",
        "relation_perm",
        False,
    ),
]


secondary_rows = []


for ei, (
    name,
    k,
    target,
    metric,
    a,
    b,
    lower,
) in enumerate(
    secondary_defs
):

    z = paired_delta(
        k,
        target,
        metric,
        a,
        b,
        lower_is_better=lower,
    )

    d = z[
        "delta"
    ].to_numpy(
        dtype=float
    )

    rr = mc_signflip(
        d,
        B=2000000,
        seed=(
            20261000
            +
            ei
        ),
    )

    secondary_rows.append({
        "endpoint":
            name,

        "mean_delta":
            float(
                d.mean()
            ),

        "median_delta":
            float(
                np.median(
                    d
                )
            ),

        "wins":
            int(
                (
                    d > 0
                ).sum()
            ),

        "ties":
            int(
                (
                    np.abs(
                        d
                    )
                    <=
                    1e-15
                ).sum()
            ),

        "losses":
            int(
                (
                    d < 0
                ).sum()
            ),

        "p_mc":
            rr[
                "p_mc"
            ],

        "mc_ci_low":
            rr[
                "mc_tail_probability_ci95"
            ][0],

        "mc_ci_high":
            rr[
                "mc_tail_probability_ci95"
            ][1],
    })


secondary = pd.DataFrame(
    secondary_rows
)


# Holm step-down adjusted p-values.
p = secondary[
    "p_mc"
].to_numpy(
    dtype=float
)

order = np.argsort(
    p
)

mtest = len(
    p
)

adj = np.empty(
    mtest,
    dtype=float,
)

running = 0.0


for rank, idx in enumerate(
    order
):

    candidate = (
        (
            mtest
            -
            rank
        )
        *
        p[
            idx
        ]
    )

    running = max(
        running,
        candidate,
    )

    adj[
        idx
    ] = min(
        running,
        1.0,
    )


secondary[
    "holm_adjusted_p"
] = adj


secondary.to_csv(
    SECONDARY_CSV,
    index=False,
)


# ============================================================
# RESULT
# ============================================================

result = {
    "analysis_id":
        "DS004080-primary-external-result-v1.0",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "primary": {
        "endpoint":
            (
                "late k10 patient-level mean "
                "Spearman relation - relation_perm"
            ),

        "n_patients":
            74,

        "mean_delta":
            mean_delta,

        "median_delta":
            median_delta,

        "bootstrap_95ci_mean":
            [
                ci[0],
                ci[1],
            ],

        "wins":
            wins,

        "ties":
            ties,

        "losses":
            losses,

        "randomization":
            primary_randomization,

        "sensitivity":
            primary_sensitivity,

        "support_rule":
            (
                "positive mean delta and "
                "two-sided randomization p < 0.05"
            ),

        "support_rule_met":
            bool(
                mean_delta > 0
                and
                primary_randomization[
                    "p_mc"
                ]
                <
                0.05
            ),
    },

    "secondary_family": (
        secondary.to_dict(
            orient="records"
        )
    ),

    "patient_delta_sha256":
        sha256_file(
            PRIMARY_CSV
        ),

    "secondary_results_sha256":
        sha256_file(
            SECONDARY_CSV
        ),

    "patient_summary_sha256":
        sha256_file(
            PATIENT
        ),

    "statistical_operationalization_sha256":
        sha256_file(
            STAT_PROTOCOL
        ),
}


RESULT_JSON.write_text(
    json.dumps(
        result,
        indent=2
    )
    + "\n"
)


print()
print("=" * 108)
print("DS004080 PRIMARY EXTERNAL LOCKBOX RESULT")
print("=" * 108)

print(
    "N patients:",
    74,
)

print()
print(
    "PRIMARY: late / k=10 / "
    "Spearman(relation - relation_perm)"
)

print(
    "mean delta:",
    mean_delta,
)

print(
    "median delta:",
    median_delta,
)

print(
    "bootstrap 95% CI:",
    ci,
)

print(
    "wins / ties / losses:",
    wins,
    "/",
    ties,
    "/",
    losses,
)

print(
    "Monte-Carlo sign-flip p:",
    primary_randomization[
        "p_mc"
    ],
)

print(
    "MC tail-probability 95% interval:",
    primary_randomization[
        "mc_tail_probability_ci95"
    ],
)

print()
print(
    "Exact sign-test sensitivity p:",
    primary_sensitivity[
        "exact_binomial_sign_p"
    ],
)

print(
    "Wilcoxon sensitivity p:",
    primary_sensitivity[
        "wilcoxon_approx_p"
    ],
)

print()
print(
    "PRIMARY SUPPORT RULE MET:",
    result[
        "primary"
    ][
        "support_rule_met"
    ],
)

print()
print("SECONDARY FAMILY")

print(
    secondary[
        [
            "endpoint",
            "mean_delta",
            "wins",
            "losses",
            "p_mc",
            "holm_adjusted_p",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "PRIMARY RESULT SHA256:",
    sha256_file(
        RESULT_JSON
    ),
)
