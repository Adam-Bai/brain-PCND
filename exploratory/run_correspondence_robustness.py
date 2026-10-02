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

from scipy.stats import spearmanr


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
)

CORE = (
    ROOT
    / "scripts"
    / "30_train_pcnd_v3f_final_strict.py"
)

CKPT = (
    ROOT
    / "results/clinical_v10/v3f_all8_deployment/"
    "pcnd_v3f_all8_deployment.pt"
)

FEATURE = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_v3f_external_features_v1/"
    "DS004080_V3F_EXTERNAL_FEATURES.csv.gz"
)

PLAN = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_final_lockbox_v1/"
    "frozen_context_query_plan.csv"
)

PRIMARY = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_frozen_inference_v1/"
    "PRIMARY_PATIENT_DELTAS.csv"
)

OUT = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_exploratory_robustness_v1"
)

PER = OUT / "per_subject"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PER.mkdir(
    parents=True,
    exist_ok=True,
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

COLLISION_LEVELS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

DISTANCE_BINS = [
    (10.0, 20.0, "10–20 mm"),
    (20.0, 40.0, "20–40 mm"),
    (40.0, 80.0, "40–80 mm"),
    (80.0, np.inf, "≥80 mm"),
]

THRESHOLDS = [
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
]


def sha256_file(p):

    h = hashlib.sha256()

    with open(p, "rb") as f:

        for b in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def safe_spearman(
    y,
    p,
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

    return float(
        spearmanr(
            y,
            p,
        ).statistic
    )


assert (
    sha256_file(CKPT)
    ==
    EXPECTED_CKPT_SHA
)

assert (
    sha256_file(CORE)
    ==
    EXPECTED_CORE_SHA
)


# ============================================================
# Frozen model core
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
    CORE.read_text(),
    filename=str(CORE),
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

module = ast.Module(
    body=selected,
    type_ignores=[],
)

ast.fix_missing_locations(
    module
)

ns = {
    "np": np,
    "pd": pd,
    "torch": torch,
    "nn": nn,
    "F": F,
    "DEVICE": DEVICE,
}

exec(
    compile(
        module,
        filename=str(CORE),
        mode="exec",
    ),
    ns,
)

SiteData = ns["SiteData"]
PopulationModel = ns["PopulationModel"]
RelationalMemory = ns["RelationalMemory"]
estimate_hierarchy = ns["estimate_hierarchy"]


# ============================================================
# Checkpoint
# ============================================================

ckpt = torch.load(
    CKPT,
    map_location=DEVICE,
    weights_only=False,
)

PAIR_FEATURES = list(
    ckpt["pair_features"]
)

TARGETS = list(
    ckpt["targets"]
)

x_mean = np.asarray(
    ckpt["x_mean"],
    dtype=np.float32,
)

x_std = np.asarray(
    ckpt["x_std"],
    dtype=np.float32,
)

y_mean = np.asarray(
    ckpt["y_mean"],
    dtype=np.float32,
)

y_std = np.asarray(
    ckpt["y_std"],
    dtype=np.float32,
)

roi_to_idx = dict(
    ckpt["roi_to_idx"]
)

hp = dict(
    ckpt["hyperparameters"]
)


population = PopulationModel(
    pair_dim=len(PAIR_FEATURES),
    hidden=hp["hidden"],
    n_roi=len(roi_to_idx),
    roi_dim=hp["roi_dim"],
    dropout=hp["dropout"],
).to(DEVICE)


relation = RelationalMemory(
    hidden=hp["hidden"],
    heads=hp["heads"],
    layers=hp["layers"],
    ff_mult=hp["ff_mult"],
    dropout=hp["dropout"],
).to(DEVICE)


population.load_state_dict(
    ckpt["population_state"],
    strict=True,
)

relation.load_state_dict(
    ckpt["relation_state"],
    strict=True,
)

population.eval()
relation.eval()


def roi_id(v):

    if pd.isna(v):
        return 0

    return int(
        roi_to_idx.get(
            str(v),
            0,
        )
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


# ============================================================
# Custom relational negative controls
# ============================================================

def donor_map_distance(
    distances,
):

    distances = np.asarray(
        distances,
        dtype=float,
    )

    n = len(distances)

    donor = np.arange(n)

    if n <= 1:
        return donor

    order = np.argsort(
        distances,
        kind="stable",
    )

    for p in range(
        0,
        n - 1,
        2,
    ):

        a = order[p]
        b = order[p + 1]

        donor[a] = b
        donor[b] = a

    return donor


def donor_map_roi(
    roi_pairs,
):

    n = len(roi_pairs)

    donor = np.arange(n)

    groups = {}

    for i, roi in enumerate(
        roi_pairs
    ):

        groups.setdefault(
            roi,
            [],
        ).append(i)

    for idx in groups.values():

        if len(idx) <= 1:
            continue

        rotated = (
            idx[1:]
            +
            idx[:1]
        )

        for a, b in zip(
            idx,
            rotated,
        ):

            donor[a] = b

    return donor


def donor_map_distance_roi(
    distances,
    roi_pairs,
):

    n = len(distances)

    donor = np.arange(n)

    groups = {}

    for i, roi in enumerate(
        roi_pairs
    ):

        groups.setdefault(
            roi,
            [],
        ).append(i)

    for idx in groups.values():

        if len(idx) <= 1:
            continue

        local = sorted(
            idx,
            key=lambda j:
                (
                    distances[j],
                    j,
                ),
        )

        for p in range(
            0,
            len(local) - 1,
            2,
        ):

            a = local[p]
            b = local[p + 1]

            donor[a] = b
            donor[b] = a

    return donor


def custom_relation(
    context_sites,
    query_site,
    global_shift,
    node_map,
    mode,
    rng=None,
    fraction=None,
):

    nq = len(
        query_site.channels
    )

    k = len(
        context_sites
    )

    memory = torch.zeros(
        (
            nq,
            k + 1,
            relation.hidden,
        ),
        device=DEVICE,
    )

    mask = torch.ones(
        (
            nq,
            k + 1,
        ),
        dtype=torch.bool,
        device=DEVICE,
    )

    pair_parts = []
    inter_parts = []

    qpos = []
    cpos = []

    distances = []
    roi_pairs = []


    for j, site in enumerate(
        context_sites
    ):

        residual = (
            site.y
            -
            site.pop_mu
            -
            global_shift
        )

        node_site = torch.stack(
            [
                node_map.get(
                    ch,
                    torch.zeros(
                        2,
                        device=DEVICE,
                    ),
                )
                for ch in site.channels
            ],
            dim=0,
        )

        interaction = (
            residual
            -
            node_site
        )


        for qi, ch in enumerate(
            query_site.channels
        ):

            si = (
                site.channel_to_idx
                .get(
                    ch,
                    None,
                )
            )

            if si is None:
                continue

            pair_parts.append(
                site.pop_q[si]
            )

            inter_parts.append(
                interaction[si]
            )

            qpos.append(qi)
            cpos.append(j)

            distances.append(
                site.meta_distance[
                    ch
                ]
            )

            roi_pairs.append(
                site.meta_roi_pair
            )


    if pair_parts:

        pair_all = torch.stack(
            pair_parts,
            dim=0,
        )

        inter_all = torch.stack(
            inter_parts,
            dim=0,
        )

        q_all = torch.tensor(
            qpos,
            dtype=torch.long,
            device=DEVICE,
        )

        j_all = torch.tensor(
            cpos,
            dtype=torch.long,
            device=DEVICE,
        )

        distances_np = np.asarray(
            distances,
            dtype=float,
        )

        roi_np = list(roi_pairs)

        perm_inter = (
            inter_all.clone()
        )


        for qi in torch.unique(
            q_all
        ).tolist():

            idx = torch.where(
                q_all == qi
            )[0]

            if len(idx) <= 1:
                continue

            local_idx = (
                idx
                .detach()
                .cpu()
                .numpy()
            )

            n = len(local_idx)


            if mode == "ordinary":

                if rng is None:
                    raise ValueError(
                        "rng required"
                    )

                donor = rng.permutation(
                    n
                )


            elif mode == "progressive":

                if fraction is None:
                    raise ValueError(
                        "fraction required"
                    )

                if fraction <= 0:
                    continue

                if rng is None:
                    raise ValueError(
                        "rng required"
                    )

                donor_full = (
                    rng.permutation(
                        n
                    )
                )

                if fraction >= 1:

                    donor = donor_full

                else:

                    donor = np.arange(
                        n
                    )

                    use = (
                        rng.random(n)
                        <
                        fraction
                    )

                    donor[
                        use
                    ] = donor_full[
                        use
                    ]


            elif mode == "distance":

                donor = (
                    donor_map_distance(
                        distances_np[
                            local_idx
                        ]
                    )
                )


            elif mode == "roi":

                donor = (
                    donor_map_roi(
                        [roi_np[int(j)] for j in local_idx]
                    )
                )


            elif mode == "distance_roi":

                donor = (
                    donor_map_distance_roi(
                        distances_np[
                            local_idx
                        ],
                        [roi_np[int(j)] for j in local_idx],
                    )
                )


            else:

                raise ValueError(
                    mode
                )


            donor_t = torch.tensor(
                donor,
                dtype=torch.long,
                device=DEVICE,
            )

            perm_inter[
                idx
            ] = inter_all[
                idx[
                    donor_t
                ]
            ]


        inter_all = perm_inter


        tokens = (
            relation.memory_encoder(
                torch.cat(
                    [
                        pair_all,
                        inter_all,
                    ],
                    dim=-1,
                )
            )
        )


        memory[
            q_all,
            j_all,
            :
        ] = tokens

        mask[
            q_all,
            j_all
        ] = False


    # Dummy token
    mask[:, -1] = False


    q = (
        relation.query_adapter(
            query_site.pop_q
        )
    )


    for block in relation.blocks:

        q = block(
            q,
            memory,
            mask,
        )


    return relation.head(
        relation.norm(q)
    )


# ============================================================
# External data
# ============================================================

features = pd.read_csv(
    FEATURE
)

plans = pd.read_csv(
    PLAN
)

plans = plans[
    plans["context_k"] == 10
].copy()


subjects = sorted(
    features[
        "subject"
    ]
    .astype(str)
    .unique()
)

subject_ordinal = {
    s:
        i + 1
    for i, s
    in enumerate(subjects)
}


def build_subject(
    subject,
):

    f = features[
        features[
            "subject"
        ]
        ==
        subject
    ].copy()

    data = {}


    for site, g in f.groupby(
        "stim_site",
        sort=True,
    ):

        site = str(site)

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
            ].astype(str).tolist(),

            name=site,

            subject=subject,

            roi_a=torch.tensor(
                [
                    roi_id(v)
                    for v in
                    g["stim_a_roi"]
                ],
                dtype=torch.long,
                device=DEVICE,
            ),

            roi_b=torch.tensor(
                [
                    roi_id(v)
                    for v in
                    g["stim_b_roi"]
                ],
                dtype=torch.long,
                device=DEVICE,
            ),

            roi_rec=torch.tensor(
                [
                    roi_id(v)
                    for v in
                    g["recording_roi"]
                ],
                dtype=torch.long,
                device=DEVICE,
            ),
        )


        obj.meta_distance = {
            str(ch):
                float(dist)
            for ch, dist
            in zip(
                g[
                    "recording_channel"
                ].astype(str),
                g[
                    "distance_to_stim"
                ],
            )
        }


        roi_pair = tuple(
            sorted(
                [
                    str(
                        g[
                            "stim_a_roi"
                        ].iloc[0]
                    ),
                    str(
                        g[
                            "stim_b_roi"
                        ].iloc[0]
                    ),
                ]
            )
        )

        obj.meta_roi_pair = (
            roi_pair
        )


        data[
            site
        ] = obj


    with torch.no_grad():

        for site in data.values():

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


    return data


# ============================================================
# Main analysis
# ============================================================

for si, subject in enumerate(
    subjects,
    1,
):

    collision_file = (
        PER
        /
        f"{subject}_collision.csv"
    )

    matched_file = (
        PER
        /
        f"{subject}_matched.csv"
    )

    spatial_file = (
        PER
        /
        f"{subject}_spatial.csv"
    )


    if (
        collision_file.exists()
        and
        matched_file.exists()
        and
        spatial_file.exists()
    ):

        print(
            f"[SKIP] {si}/74 "
            f"{subject}"
        )

        continue


    print()
    print(
        "=" * 100
    )

    print(
        f"{si}/74 {subject}"
    )

    print(
        "=" * 100
    )


    data = build_subject(
        subject
    )


    subject_plans = (
        plans[
            plans[
                "subject"
            ]
            ==
            subject
        ]
        .sort_values(
            "repeat"
        )
    )


    if len(
        subject_plans
    ) != 20:

        raise RuntimeError(
            f"{subject}: expected 20 k10 plans"
        )


    collision_rows = []
    matched_rows = []
    spatial_rows = []


    with torch.no_grad():

        for pii, (
            _,
            pr,
        ) in enumerate(
            subject_plans.iterrows(),
            1,
        ):

            repeat = int(
                pr[
                    "repeat"
                ]
            )

            context_names = [
                str(x)
                for x in json.loads(
                    pr[
                        "context_sites"
                    ]
                )
            ]

            query_names = [
                str(x)
                for x in json.loads(
                    pr[
                        "query_sites"
                    ]
                )
            ]


            context = [
                data[x]
                for x in context_names
            ]


            truth_all = []

            pred_relation = []
            pred_ordinary = []

            pred_collision = {
                0.25: [],
                0.50: [],
                0.75: [],
            }

            pred_distance = []
            pred_roi = []
            pred_distance_roi = []

            distance_all = []


            for q_idx, qname in enumerate(
                query_names
            ):

                qsite = data[
                    qname
                ]

                (
                    c,
                    node_q,
                    node_map,
                ) = estimate_hierarchy(
                    context,
                    qsite,
                )


                mu_node = (
                    qsite.pop_mu
                    +
                    c
                    +
                    node_q
                )


                delta_rel = relation(
                    context,
                    qsite,
                    c,
                    node_map,
                )


                mu_rel = (
                    mu_node
                    +
                    delta_rel
                )


                base_seed = (
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
                    10
                    *
                    10000
                    +
                    q_idx
                )


                # EXACT original ordinary permutation
                prng = np.random.default_rng(
                    base_seed
                )

                delta_ord = relation(
                    context,
                    qsite,
                    c,
                    node_map,
                    permute_within_channel=True,
                    rng=prng,
                )


                mu_ord = (
                    mu_node
                    +
                    delta_ord
                )


                collision_pred = {}

                for fraction in [
                    0.25,
                    0.50,
                    0.75,
                ]:

                    rng = (
                        np.random.default_rng(
                            base_seed
                        )
                    )

                    delta = custom_relation(
                        context,
                        qsite,
                        c,
                        node_map,
                        mode="progressive",
                        fraction=fraction,
                        rng=rng,
                    )

                    collision_pred[
                        fraction
                    ] = raw_prediction(
                        mu_node
                        +
                        delta
                    )[
                        :,
                        1
                    ]


                delta_d = custom_relation(
                    context,
                    qsite,
                    c,
                    node_map,
                    mode="distance",
                )

                delta_r = custom_relation(
                    context,
                    qsite,
                    c,
                    node_map,
                    mode="roi",
                )

                delta_dr = custom_relation(
                    context,
                    qsite,
                    c,
                    node_map,
                    mode="distance_roi",
                )


                truth = np.asarray(
                    qsite.y_raw,
                    dtype=float,
                )[
                    :,
                    1
                ]


                relation_raw = (
                    raw_prediction(
                        mu_rel
                    )[
                        :,
                        1
                    ]
                )

                ordinary_raw = (
                    raw_prediction(
                        mu_ord
                    )[
                        :,
                        1
                    ]
                )

                distance_raw = (
                    raw_prediction(
                        mu_node
                        +
                        delta_d
                    )[
                        :,
                        1
                    ]
                )

                roi_raw = (
                    raw_prediction(
                        mu_node
                        +
                        delta_r
                    )[
                        :,
                        1
                    ]
                )

                distance_roi_raw = (
                    raw_prediction(
                        mu_node
                        +
                        delta_dr
                    )[
                        :,
                        1
                    ]
                )


                distances = np.asarray(
                    [
                        qsite.meta_distance[
                            ch
                        ]
                        for ch in
                        qsite.channels
                    ],
                    dtype=float,
                )


                truth_all.append(
                    truth
                )

                pred_relation.append(
                    relation_raw
                )

                pred_ordinary.append(
                    ordinary_raw
                )

                pred_distance.append(
                    distance_raw
                )

                pred_roi.append(
                    roi_raw
                )

                pred_distance_roi.append(
                    distance_roi_raw
                )

                distance_all.append(
                    distances
                )


                for fraction in [
                    0.25,
                    0.50,
                    0.75,
                ]:

                    pred_collision[
                        fraction
                    ].append(
                        collision_pred[
                            fraction
                        ]
                    )


            y = np.concatenate(
                truth_all
            )

            rel = np.concatenate(
                pred_relation
            )

            ordinary = np.concatenate(
                pred_ordinary
            )

            dist_ctrl = np.concatenate(
                pred_distance
            )

            roi_ctrl = np.concatenate(
                pred_roi
            )

            dist_roi_ctrl = np.concatenate(
                pred_distance_roi
            )

            native_dist = np.concatenate(
                distance_all
            )


            collision_predictions = {
                0.00:
                    rel,

                0.25:
                    np.concatenate(
                        pred_collision[
                            0.25
                        ]
                    ),

                0.50:
                    np.concatenate(
                        pred_collision[
                            0.50
                        ]
                    ),

                0.75:
                    np.concatenate(
                        pred_collision[
                            0.75
                        ]
                    ),

                1.00:
                    ordinary,
            }


            rho_100 = safe_spearman(
                y,
                ordinary,
            )


            for fraction in (
                COLLISION_LEVELS
            ):

                rho = safe_spearman(
                    y,
                    collision_predictions[
                        fraction
                    ],
                )

                collision_rows.append({
                    "subject":
                        subject,

                    "repeat":
                        repeat,

                    "identity_destruction":
                        fraction,

                    "spearman":
                        rho,

                    "delta_vs_100pct":
                        (
                            rho
                            -
                            rho_100
                        ),
                })


            controls = {
                "ordinary":
                    ordinary,

                "distance_matched":
                    dist_ctrl,

                "roi_matched":
                    roi_ctrl,

                "distance_roi_matched":
                    dist_roi_ctrl,
            }


            rho_relation = (
                safe_spearman(
                    y,
                    rel,
                )
            )


            for name, pred in (
                controls.items()
            ):

                rho_control = (
                    safe_spearman(
                        y,
                        pred,
                    )
                )

                matched_rows.append({
                    "subject":
                        subject,

                    "repeat":
                        repeat,

                    "control":
                        name,

                    "relation_spearman":
                        rho_relation,

                    "control_spearman":
                        rho_control,

                    "delta":
                        (
                            rho_relation
                            -
                            rho_control
                        ),
                })


            # ---------------------------------------------
            # Spatial bins
            # ---------------------------------------------

            for (
                lo,
                hi,
                label,
            ) in DISTANCE_BINS:

                use = (
                    native_dist
                    >=
                    lo
                )

                if np.isfinite(
                    hi
                ):

                    use &= (
                        native_dist
                        <
                        hi
                    )


                if use.sum() < 10:

                    delta = np.nan

                else:

                    delta = (
                        safe_spearman(
                            y[use],
                            rel[use],
                        )
                        -
                        safe_spearman(
                            y[use],
                            ordinary[use],
                        )
                    )


                spatial_rows.append({
                    "subject":
                        subject,

                    "repeat":
                        repeat,

                    "kind":
                        "distance_bin",

                    "label":
                        label,

                    "value":
                        lo,

                    "n_pairs":
                        int(
                            use.sum()
                        ),

                    "delta":
                        delta,
                })


            # ---------------------------------------------
            # Stricter near-field exclusion thresholds
            # Primary frozen map already excludes near field;
            # this stress test only moves the threshold upward.
            # ---------------------------------------------

            for threshold in THRESHOLDS:

                use = (
                    native_dist
                    >=
                    threshold
                )


                if use.sum() < 10:

                    delta = np.nan

                else:

                    delta = (
                        safe_spearman(
                            y[use],
                            rel[use],
                        )
                        -
                        safe_spearman(
                            y[use],
                            ordinary[use],
                        )
                    )


                spatial_rows.append({
                    "subject":
                        subject,

                    "repeat":
                        repeat,

                    "kind":
                        "threshold",

                    "label":
                        f"≥{int(threshold)} mm",

                    "value":
                        threshold,

                    "n_pairs":
                        int(
                            use.sum()
                        ),

                    "delta":
                        delta,
                })


            print(
                f"  repeat "
                f"{pii:02d}/20"
            )


    pd.DataFrame(
        collision_rows
    ).to_csv(
        collision_file,
        index=False,
    )

    pd.DataFrame(
        matched_rows
    ).to_csv(
        matched_file,
        index=False,
    )

    pd.DataFrame(
        spatial_rows
    ).to_csv(
        spatial_file,
        index=False,
    )


# ============================================================
# Combine subjects
# ============================================================

collision = pd.concat(
    [
        pd.read_csv(p)
        for p in sorted(
            PER.glob(
                "*_collision.csv"
            )
        )
    ],
    ignore_index=True,
)


matched = pd.concat(
    [
        pd.read_csv(p)
        for p in sorted(
            PER.glob(
                "*_matched.csv"
            )
        )
    ],
    ignore_index=True,
)


spatial = pd.concat(
    [
        pd.read_csv(p)
        for p in sorted(
            PER.glob(
                "*_spatial.csv"
            )
        )
    ],
    ignore_index=True,
)


collision_patient = (
    collision
    .groupby(
        [
            "subject",
            "identity_destruction",
        ],
        as_index=False,
    )[
        "delta_vs_100pct"
    ]
    .mean()
)


matched_patient = (
    matched
    .groupby(
        [
            "subject",
            "control",
        ],
        as_index=False,
    )[
        "delta"
    ]
    .mean()
)


spatial_patient = (
    spatial
    .groupby(
        [
            "subject",
            "kind",
            "label",
            "value",
        ],
        as_index=False,
    )[
        "delta"
    ]
    .mean()
)


collision_patient.to_csv(
    OUT
    / "COLLISION_PATIENT_LEVEL.csv",
    index=False,
)

matched_patient.to_csv(
    OUT
    / "MATCHED_NEGATIVE_PATIENT_LEVEL.csv",
    index=False,
)

spatial_patient.to_csv(
    OUT
    / "SPATIAL_STRESS_PATIENT_LEVEL.csv",
    index=False,
)


# ============================================================
# Primary reproduction audit
# ============================================================

primary = pd.read_csv(
    PRIMARY
).set_index(
    "subject"
)


c0 = (
    collision_patient[
        collision_patient[
            "identity_destruction"
        ]
        ==
        0.0
    ]
    .set_index(
        "subject"
    )[
        "delta_vs_100pct"
    ]
)


common = (
    primary.index
    .intersection(
        c0.index
    )
)


max_primary_error = float(
    np.max(
        np.abs(
            primary.loc[
                common,
                "delta"
            ]
            -
            c0.loc[
                common
            ]
        )
    )
)


print()
print(
    "=" * 100
)

print(
    "EXPLORATORY ROBUSTNESS COMPLETE"
)

print(
    "=" * 100
)

print(
    "subjects:",
    len(
        common
    )
)

print(
    "primary reproduction max error:",
    max_primary_error
)


if max_primary_error > 1e-5:

    raise RuntimeError(
        "Ordinary permutation did not reproduce "
        "the frozen primary patient deltas."
    )


manifest = {
    "analysis_status":
        "POST_LOCKBOX_EXPLORATORY_ROBUSTNESS",

    "confirmatory_primary_modified":
        False,

    "checkpoint_sha256":
        EXPECTED_CKPT_SHA,

    "frozen_source_sha256":
        EXPECTED_CORE_SHA,

    "context_k":
        10,

    "target":
        "late",

    "collision_levels":
        COLLISION_LEVELS,

    "matched_controls": {
        "ordinary":
            (
                "frozen within-recording-channel "
                "stimulation permutation"
            ),

        "distance_matched":
            (
                "within recording channel, "
                "interaction residuals swapped "
                "between adjacent context sites "
                "after sorting by native "
                "distance-to-stimulation"
            ),

        "roi_matched":
            (
                "within recording channel, "
                "interaction residuals cyclically "
                "shifted only among context sites "
                "with identical unordered frozen "
                "stimulation ROI-pair tokens"
            ),

        "distance_roi_matched":
            (
                "within identical frozen stimulation "
                "ROI-pair strata, context sites are "
                "sorted by distance and adjacent "
                "interaction residuals are swapped"
            ),
    },

    "spatial_stress": {
        "distance_bins":
            [
                x[2]
                for x in
                DISTANCE_BINS
            ],

        "near_field_thresholds_mm":
            THRESHOLDS,

        "important":
            (
                "The frozen primary phenotype already "
                "excluded the prespecified near field. "
                "This analysis only applies stricter "
                "post-hoc thresholds; it does not "
                "reintroduce excluded pairs."
            ),
    },

    "primary_reproduction_max_error":
        max_primary_error,

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
}


(
    OUT
    / "EXPLORATORY_ROBUSTNESS_MANIFEST.json"
).write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    +
    "\n"
)


print(
    "STATUS: READY FOR ROBUSTNESS FIGURE"
)
