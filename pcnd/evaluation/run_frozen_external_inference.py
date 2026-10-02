from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime, timezone

import ast
import hashlib
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import (
    pearsonr,
    spearmanr,
)


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

CORE_SOURCE = (
    ROOT
    / "scripts"
    / "30_train_pcnd_v3f_final_strict.py"
)

FEATURE_DIR = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_v3f_external_features_v1"
)

FEATURE = (
    FEATURE_DIR
    / "DS004080_V3F_EXTERNAL_FEATURES.csv.gz"
)

FEATURE_MANIFEST = (
    FEATURE_DIR
    / "EXTERNAL_V3F_FEATURE_MANIFEST.json"
)

LOCK = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_final_lockbox_v1"
)

PLAN = (
    LOCK
    / "frozen_context_query_plan.csv"
)

CKPT = (
    ROOT
    / "results/clinical_v10/"
    "v3f_all8_deployment/"
    "pcnd_v3f_all8_deployment.pt"
)

OUT = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_frozen_inference_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

REPEAT_METRICS = (
    OUT
    / "FROZEN_REPEAT_METRICS.csv.gz"
)

PATIENT_SUMMARY = (
    OUT
    / "FROZEN_PATIENT_METRIC_SUMMARY.csv"
)

MANIFEST = (
    OUT
    / "FROZEN_INFERENCE_MANIFEST.json"
)


EXPECTED_CKPT_SHA = (
    "399421567b0185383bf8953cdcd6c758"
    "91893b5113fca8a3a41ac190eb68a794"
)

EXPECTED_CORE_SHA = (
    "34750f1cee3a56b24d5049866396ff9d"
    "bbf6df8310f2b2e7f3a81f3cfbb7137c"
)

PERM_SEED_ROOT = 20260926


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
    REPEAT_METRICS.exists()
    or
    PATIENT_SUMMARY.exists()
    or
    MANIFEST.exists()
):

    raise RuntimeError(
        "Frozen external inference outputs "
        "already exist; refusing overwrite."
    )


if sha256_file(CKPT) != EXPECTED_CKPT_SHA:
    raise RuntimeError(
        "Deployment checkpoint hash mismatch."
    )


if sha256_file(CORE_SOURCE) != EXPECTED_CORE_SHA:
    raise RuntimeError(
        "Frozen V3f source hash mismatch."
    )


fm = json.loads(
    FEATURE_MANIFEST.read_text()
)

feature_sha = sha256_file(
    FEATURE
)

if (
    feature_sha
    !=
    fm[
        "feature_table_sha256"
    ]
):

    raise RuntimeError(
        "External feature table hash mismatch."
    )


# ============================================================
# DEVICE / EXACT CORE EXTRACTION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if DEVICE.type == "cuda":

    torch.set_float32_matmul_precision(
        "high"
    )

    torch.backends.cuda.matmul.allow_tf32 = True


CORE_NAMES = {
    "SiteData",
    "MLP",
    "ResidualFFN",
    "CrossBlock",
    "PopulationModel",
    "estimate_hierarchy",
    "RelationalMemory",
}


tree = ast.parse(
    CORE_SOURCE.read_text(),
    filename=str(
        CORE_SOURCE
    ),
)


selected = [
    node
    for node in tree.body
    if isinstance(
        node,
        (
            ast.ClassDef,
            ast.FunctionDef,
        ),
    )
    and node.name in CORE_NAMES
]


found = {
    x.name
    for x in selected
}


if found != CORE_NAMES:

    raise RuntimeError(
        f"Missing exact V3f definitions: "
        f"{sorted(CORE_NAMES-found)}"
    )


module = ast.Module(
    body=selected,
    type_ignores=[],
)

ast.fix_missing_locations(
    module
)


ns = {
    "np":
        np,

    "pd":
        pd,

    "torch":
        torch,

    "nn":
        nn,

    "F":
        F,

    "DEVICE":
        DEVICE,
}


exec(
    compile(
        module,
        filename=str(
            CORE_SOURCE
        ),
        mode="exec",
    ),
    ns,
)


SiteData = ns["SiteData"]
PopulationModel = ns["PopulationModel"]
RelationalMemory = ns["RelationalMemory"]
estimate_hierarchy = ns["estimate_hierarchy"]


# ============================================================
# LOAD FROZEN DEPLOYMENT
# ============================================================

ckpt = torch.load(
    CKPT,
    map_location=DEVICE,
    weights_only=False,
)


PAIR_FEATURES = list(
    ckpt[
        "pair_features"
    ]
)

TARGETS = list(
    ckpt[
        "targets"
    ]
)


if (
    PAIR_FEATURES
    !=
    fm[
        "pair_features"
    ]
):

    raise RuntimeError(
        "External feature order does not match "
        "deployment checkpoint."
    )


x_mean = np.asarray(
    ckpt[
        "x_mean"
    ],
    dtype=np.float32,
)

x_std = np.asarray(
    ckpt[
        "x_std"
    ],
    dtype=np.float32,
)

y_mean = np.asarray(
    ckpt[
        "y_mean"
    ],
    dtype=np.float32,
)

y_std = np.asarray(
    ckpt[
        "y_std"
    ],
    dtype=np.float32,
)

roi_to_idx = dict(
    ckpt[
        "roi_to_idx"
    ]
)

hp = dict(
    ckpt[
        "hyperparameters"
    ]
)


population = PopulationModel(
    pair_dim=len(
        PAIR_FEATURES
    ),

    hidden=hp[
        "hidden"
    ],

    n_roi=len(
        roi_to_idx
    ),

    roi_dim=hp[
        "roi_dim"
    ],

    dropout=hp[
        "dropout"
    ],
).to(
    DEVICE
)


relation = RelationalMemory(
    hidden=hp[
        "hidden"
    ],

    heads=hp[
        "heads"
    ],

    layers=hp[
        "layers"
    ],

    ff_mult=hp[
        "ff_mult"
    ],

    dropout=hp[
        "dropout"
    ],
).to(
    DEVICE
)


population.load_state_dict(
    ckpt[
        "population_state"
    ],
    strict=True,
)

relation.load_state_dict(
    ckpt[
        "relation_state"
    ],
    strict=True,
)


population.eval()
relation.eval()


print(
    "DEVICE:",
    DEVICE
)

print(
    "DEPLOYMENT CHECKPOINT:",
    EXPECTED_CKPT_SHA
)

print(
    "FEATURE TABLE:",
    feature_sha
)


# ============================================================
# BUILD EXTERNAL SiteData
# ============================================================

df = pd.read_csv(
    FEATURE
)


def roi_id(v):

    if pd.isna(v):
        return 0

    return int(
        roi_to_idx.get(
            str(v),
            0,
        )
    )


data = {}


for (
    subject,
    site,
), g in df.groupby(
    [
        "subject",
        "stim_site",
    ],
    sort=True,
):

    subject = str(
        subject
    )

    site = str(
        site
    )

    g = (
        g
        .sort_values(
            "recording_channel"
        )
        .reset_index(
            drop=True
        )
    )


    x_raw = g[
        PAIR_FEATURES
    ].to_numpy(
        dtype=np.float32
    )


    x = (
        x_raw
        -
        x_mean
    ) / x_std


    y_raw = g[
        TARGETS
    ].to_numpy(
        dtype=np.float32
    )


    y_log = np.log1p(
        np.clip(
            y_raw,
            0,
            None,
        )
    )


    y = (
        y_log
        -
        y_mean
    ) / y_std


    obj = SiteData(
        x=torch.tensor(
            x,
            dtype=torch.float32,
            device=DEVICE,
        ),

        y=torch.tensor(
            y,
            dtype=torch.float32,
            device=DEVICE,
        ),

        y_raw=y_raw,

        channels=g[
            "recording_channel"
        ]
        .astype(str)
        .tolist(),

        name=site,

        subject=subject,

        roi_a=torch.tensor(
            [
                roi_id(v)
                for v in
                g[
                    "stim_a_roi"
                ]
            ],
            dtype=torch.long,
            device=DEVICE,
        ),

        roi_b=torch.tensor(
            [
                roi_id(v)
                for v in
                g[
                    "stim_b_roi"
                ]
            ],
            dtype=torch.long,
            device=DEVICE,
        ),

        roi_rec=torch.tensor(
            [
                roi_id(v)
                for v in
                g[
                    "recording_roi"
                ]
            ],
            dtype=torch.long,
            device=DEVICE,
        ),
    )


    data.setdefault(
        subject,
        {}
    )[site] = obj


print(
    "external subjects:",
    len(data)
)

print(
    "external sites:",
    sum(
        len(x)
        for x in data.values()
    )
)


if len(data) != 74:
    raise RuntimeError(
        "Expected 74 external subjects."
    )


if (
    sum(
        len(x)
        for x in data.values()
    )
    != 4135
):

    raise RuntimeError(
        "Expected 4135 external sites."
    )


# ============================================================
# POPULATION PRIOR — ONCE
# ============================================================

with torch.no_grad():

    for si, (
        subject,
        site_dict,
    ) in enumerate(
        sorted(
            data.items()
        ),
        1,
    ):

        for site in (
            site_dict.values()
        ):

            (
                mu,
                lv,
                q,
            ) = (
                population.forward_site(
                    site
                )
            )

            site.pop_mu = (
                mu.detach()
            )

            site.pop_logvar = (
                lv.detach()
            )

            site.pop_q = (
                q.detach()
            )


        if (
            si % 10 == 0
            or
            si == len(data)
        ):

            print(
                f"population encoded "
                f"{si}/{len(data)} subjects"
            )


def raw_prediction(
    mu_norm,
):

    x = (
        mu_norm
        .detach()
        .cpu()
        .numpy()
    )

    log_raw = (
        x
        *
        y_std[
            None,
            :
        ]
        +
        y_mean[
            None,
            :
        ]
    )

    return np.clip(
        np.expm1(
            log_raw
        ),
        0,
        None,
    )


def safe_corr(
    y,
    p,
    kind,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    ok = (
        np.isfinite(y)
        &
        np.isfinite(p)
    )

    y = y[ok]
    p = p[ok]

    if (
        len(y) < 3
        or
        np.std(y) < 1e-12
        or
        np.std(p) < 1e-12
    ):

        return np.nan


    if kind == "pearson":

        return float(
            pearsonr(
                y,
                p,
            ).statistic
        )


    return float(
        spearmanr(
            y,
            p,
        ).statistic
    )


# ============================================================
# FROZEN CONTEXT/QUERY PLAN
# ============================================================

plan = pd.read_csv(
    PLAN
)


required_plan = {
    "subject",
    "context_k",
    "repeat",
    "context_sites",
    "query_sites",
}


missing = (
    required_plan
    -
    set(
        plan.columns
    )
)


if missing:

    raise RuntimeError(
        f"Frozen plan missing columns: "
        f"{sorted(missing)}"
    )


# Only prespecified inferential ks.
plan = plan[
    plan[
        "context_k"
    ].isin(
        [
            3,
            5,
            10,
        ]
    )
].copy()


if len(plan) != 74 * 20 * 3:

    raise RuntimeError(
        f"Expected 4440 frozen inference rows, "
        f"got {len(plan)}"
    )


subjects = sorted(
    data.keys()
)


subject_ordinal = {
    s:
        i + 1

    for i, s
    in enumerate(
        subjects
    )
}


rows = []


# ============================================================
# ONE-SHOT EXTERNAL INFERENCE
# ============================================================

with torch.no_grad():

    for pi, (
        _idx,
        r,
    ) in enumerate(
        plan.iterrows(),
        1,
    ):

        subject = str(
            r[
                "subject"
            ]
        )

        k = int(
            r[
                "context_k"
            ]
        )

        repeat = int(
            r[
                "repeat"
            ]
        )


        context_names = [
            str(x)
            for x in json.loads(
                r[
                    "context_sites"
                ]
            )
        ]

        query_names = [
            str(x)
            for x in json.loads(
                r[
                    "query_sites"
                ]
            )
        ]


        if len(
            context_names
        ) != k:

            raise RuntimeError(
                f"{subject} repeat={repeat} "
                f"k={k}: context length mismatch."
            )


        context = [
            data[
                subject
            ][x]

            for x in context_names
        ]


        condition_y = {
            c:
                [
                    [],
                    [],
                ]

            for c in [
                "population",
                "global",
                "node",
                "relation",
                "relation_perm",
            ]
        }


        condition_p = {
            c:
                [
                    [],
                    [],
                ]

            for c in condition_y
        }


        for q_idx, qname in enumerate(
            query_names
        ):

            q = data[
                subject
            ][
                qname
            ]


            (
                c,
                node_q,
                node_map,
            ) = estimate_hierarchy(
                context,
                q,
            )


            mu_population = (
                q.pop_mu
            )

            mu_global = (
                q.pop_mu
                +
                c
            )

            mu_node = (
                mu_global
                +
                node_q
            )


            delta = relation(
                context,
                q,
                c,
                node_map,
            )


            mu_relation = (
                mu_node
                +
                delta
            )


            perm_seed = (
                PERM_SEED_ROOT
                +
                subject_ordinal[
                    subject
                ]
                *
                100000000
                +
                repeat
                *
                1000000
                +
                k
                *
                10000
                +
                q_idx
            )


            prng = np.random.default_rng(
                perm_seed
            )


            delta_perm = relation(
                context,
                q,
                c,
                node_map,
                permute_within_channel=True,
                rng=prng,
            )


            mu_perm = (
                mu_node
                +
                delta_perm
            )


            predictions = {
                "population":
                    mu_population,

                "global":
                    mu_global,

                "node":
                    mu_node,

                "relation":
                    mu_relation,

                "relation_perm":
                    mu_perm,
            }


            for condition, mu in (
                predictions.items()
            ):

                pred = raw_prediction(
                    mu
                )

                truth = np.asarray(
                    q.y_raw,
                    dtype=float,
                )


                for ti in range(2):

                    condition_y[
                        condition
                    ][ti].append(
                        truth[
                            :,
                            ti
                        ]
                    )

                    condition_p[
                        condition
                    ][ti].append(
                        pred[
                            :,
                            ti
                        ]
                    )


        for condition in (
            condition_y
        ):

            for ti, target in enumerate(
                TARGETS
            ):

                y = np.concatenate(
                    condition_y[
                        condition
                    ][ti]
                )

                p = np.concatenate(
                    condition_p[
                        condition
                    ][ti]
                )


                rows.append({
                    "subject":
                        subject,

                    "context_k":
                        k,

                    "repeat":
                        repeat,

                    "condition":
                        condition,

                    "target":
                        (
                            "early"
                            if target
                            ==
                            "early_rms_z"
                            else
                            "late"
                        ),

                    "n_pairs":
                        int(
                            len(y)
                        ),

                    "pearson":
                        safe_corr(
                            y,
                            p,
                            "pearson",
                        ),

                    "spearman":
                        safe_corr(
                            y,
                            p,
                            "spearman",
                        ),

                    "mae":
                        float(
                            np.mean(
                                np.abs(
                                    y - p
                                )
                            )
                        ),
                })


        if (
            pi % 100 == 0
            or
            pi == len(plan)
        ):

            print(
                f"inference "
                f"{pi}/{len(plan)}"
            )


metrics = pd.DataFrame(
    rows
)


metrics = (
    metrics
    .sort_values(
        [
            "subject",
            "context_k",
            "repeat",
            "condition",
            "target",
        ]
    )
    .reset_index(
        drop=True
    )
)


metrics.to_csv(
    REPEAT_METRICS,
    index=False,
    compression={
        "method":
            "gzip",

        "compresslevel":
            6,

        "mtime":
            0,
    },
)


patient = (
    metrics
    .groupby(
        [
            "subject",
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

        mean_pairs_per_repeat=(
            "n_pairs",
            "mean",
        ),
    )
)


if not (
    patient[
        "n_repeats"
    ]
    ==
    20
).all():

    raise RuntimeError(
        "Patient-level repeat aggregation "
        "is not uniformly 20."
    )


patient.to_csv(
    PATIENT_SUMMARY,
    index=False,
)


repeat_sha = sha256_file(
    REPEAT_METRICS
)

patient_sha = sha256_file(
    PATIENT_SUMMARY
)


manifest = {
    "inference_id":
        "DS004080-frozen-V3f-inference-v1.0",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "device":
        str(
            DEVICE
        ),

    "checkpoint_sha256":
        EXPECTED_CKPT_SHA,

    "frozen_v3f_source_sha256":
        EXPECTED_CORE_SHA,

    "external_feature_table_sha256":
        feature_sha,

    "context_plan":
        str(
            PLAN
        ),

    "evaluated_context_k":
        [
            3,
            5,
            10,
        ],

    "relation_permutation_seed_root":
        PERM_SEED_ROOT,

    "relation_permutation_seed_rule":
        (
            "seed_root + "
            "subject_ordinal*100000000 + "
            "repeat*1000000 + "
            "context_k*10000 + "
            "query_index"
        ),

    "n_repeat_metric_rows":
        int(
            len(metrics)
        ),

    "n_patient_summary_rows":
        int(
            len(patient)
        ),

    "repeat_metrics_sha256":
        repeat_sha,

    "patient_summary_sha256":
        patient_sha,

    "architecture_or_checkpoint_modified":
        False,

    "primary_endpoint_modified":
        False,
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n"
)


print()
print("=" * 100)
print("FROZEN EXTERNAL V3f INFERENCE COMPLETE")
print("=" * 100)

print(
    "repeat metric rows:",
    len(metrics),
)

print(
    "patient summary rows:",
    len(patient),
)

print(
    "REPEAT METRICS SHA256:",
    repeat_sha,
)

print(
    "PATIENT SUMMARY SHA256:",
    patient_sha,
)

print()
print(
    "Primary statistical result has "
    "NOT been printed by this script."
)

print(
    "STATUS: READY FOR PRESPECIFIED "
    "PATIENT-LEVEL STATISTICS"
)
