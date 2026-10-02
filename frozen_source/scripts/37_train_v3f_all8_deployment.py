from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import ast
import copy
import hashlib
import json
import random
import subprocess

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

ROOT = Path(
    "/home/dell/jiaxing/bby/pcnd"
)

CORE_SOURCE = (
    ROOT
    / "scripts"
    / "30_train_pcnd_v3f_final_strict.py"
)

DATA_ROOT = (
    ROOT
    / "results"
    / "clinical_v5"
    / "v3_features"
)

V3F_RESULT_ROOT = (
    ROOT
    / "results"
    / "clinical_v9"
    / "v3f_final_strict"
)

HAP_PROTOCOL = (
    V3F_RESULT_ROOT
    / "protocol_freeze"
    / "V3F_FINAL_PROTOCOL_MANIFEST.json"
)

EXTERNAL_LOCKBOX = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_final_lockbox_v1"
    / "DS004080_FINAL_LOCKBOX_MANIFEST.json"
)

ROI_BRIDGE = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_roi_bridge_frozen_v1"
    / "ROI_BRIDGE_MANIFEST.json"
)

SCHEMA_REFERENCE = (
    V3F_RESULT_ROOT
    / "sub-01"
    / "pcnd_v3f_final_strict.pt"
)

OUT = (
    ROOT
    / "results"
    / "clinical_v10"
    / "v3f_all8_deployment"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_HAP_PROTOCOL_SHA256 = (
    "cb3bd89d6072245ecec7ae608d692f286172fedab517b1b81d821e4e3e710727"
)

EXPECTED_EXTERNAL_LOCKBOX_SHA256 = (
    "8170555317e6a2b510cb7ef1bbb603ba6a4220ec92a0b50cc32c29392f66cf6d"
)

EXPECTED_ROI_BRIDGE_SHA256 = (
    "fb9e3a493bf05e744c65fae165fc1fe3d7734b1a6456685c4d470be9c9efbbca"
)


DEPLOYMENT_VERSION = (
    "PCND-V3f-all8-deployment-v1.0"
)

DEPLOY_SEED = 20260927


SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 9)
]


PAIR_FEATURES = [
    "mni_mid_x",
    "mni_mid_y",
    "mni_mid_z",

    "rec_mni_x",
    "rec_mni_y",
    "rec_mni_z",

    "mni_dx",
    "mni_dy",
    "mni_dz",

    "log_mni_distance",

    "mni_axis_xx",
    "mni_axis_yy",
    "mni_axis_zz",
    "mni_axis_xy",
    "mni_axis_xz",
    "mni_axis_yz",
]


TARGETS = [
    "early_rms_z",
    "late_rms_z",
]


ROI_COLUMNS = [
    "stim_a_roi",
    "stim_b_roi",
    "recording_roi",
]


REL_TRAIN_K = [
    3,
    5,
    10,
]


# ------------------------------------------------------------
# EXACT V3f hyperparameters.
# No tuning is permitted here.
# ------------------------------------------------------------

HIDDEN = 256
HEADS = 8
LAYERS = 3
ROI_DIM = 48
FF_MULT = 4
DROPOUT = 0.10

LR_POP = 2e-4
LR_REL = 2e-4

POP_STEPS = 4000
REL_STEPS = 5000

TASK_BATCH = 4
EVAL_EVERY = 250

W_PRED = 0.25
W_CORR = 0.25
W_RANK = 0.10
W_NEG = 0.25

NEG_MARGIN = 0.01


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def git_cmd(*args):

    try:

        return subprocess.check_output(
            [
                "git",
                *args,
            ],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    except Exception:

        return None


def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def subnum(sub):

    return int(
        sub.split("-")[-1]
    )


# ============================================================
# VERIFY PRE-EXTERNAL FREEZES
# ============================================================

required_hashes = [
    (
        HAP_PROTOCOL,
        EXPECTED_HAP_PROTOCOL_SHA256,
        "HAPwave development protocol",
    ),
    (
        EXTERNAL_LOCKBOX,
        EXPECTED_EXTERNAL_LOCKBOX_SHA256,
        "ds004080 external lockbox",
    ),
    (
        ROI_BRIDGE,
        EXPECTED_ROI_BRIDGE_SHA256,
        "ds004080 ROI bridge",
    ),
]


for path, expected, label in required_hashes:

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    actual = sha256_file(
        path
    )

    if actual != expected:

        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )


if not CORE_SOURCE.exists():

    raise FileNotFoundError(
        CORE_SOURCE
    )


if not SCHEMA_REFERENCE.exists():

    raise FileNotFoundError(
        SCHEMA_REFERENCE
    )


# ============================================================
# REQUIRE COMMITTED TRACKED CODE
# ============================================================

git_commit = git_cmd(
    "rev-parse",
    "HEAD",
)


tracked_status = git_cmd(
    "status",
    "--porcelain",
    "--untracked-files=no",
)


if git_commit is None:

    raise RuntimeError(
        "No PCND Git commit."
    )


if tracked_status != "":

    raise RuntimeError(
        "Tracked PCND code is dirty.\n\n"
        + str(
            tracked_status
        )
    )


# ============================================================
# EXACTLY REUSE V3f CORE IMPLEMENTATION
#
# Do NOT reimplement the neural architecture here.
# Extract the already-frozen definitions directly from script30.
# ============================================================

CORE_NAMES = {
    "SiteData",
    "MLP",
    "ResidualFFN",
    "CrossBlock",
    "PopulationModel",
    "RelationalMemory",
    "estimate_hierarchy",
    "gaussian_nll",
    "differentiable_corr_loss",
    "pairwise_rank_loss",
}


source_text = CORE_SOURCE.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source_text,
    filename=str(
        CORE_SOURCE
    ),
)


selected_nodes = []


for node in tree.body:

    if isinstance(
        node,
        (
            ast.ClassDef,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):

        if node.name in CORE_NAMES:

            selected_nodes.append(
                node
            )


selected_names = {
    node.name
    for node in selected_nodes
}


missing_core = (
    CORE_NAMES
    -
    selected_names
)


if missing_core:

    raise RuntimeError(
        "Could not extract frozen V3f definitions: "
        + repr(
            sorted(
                missing_core
            )
        )
    )


core_module = ast.Module(
    body=selected_nodes,
    type_ignores=[],
)


ast.fix_missing_locations(
    core_module
)


core_namespace = {
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
        core_module,
        filename=str(
            CORE_SOURCE
        ),
        mode="exec",
    ),
    core_namespace,
)


SiteData = core_namespace[
    "SiteData"
]

PopulationModel = core_namespace[
    "PopulationModel"
]

RelationalMemory = core_namespace[
    "RelationalMemory"
]

estimate_hierarchy = core_namespace[
    "estimate_hierarchy"
]

gaussian_nll = core_namespace[
    "gaussian_nll"
]

differentiable_corr_loss = (
    core_namespace[
        "differentiable_corr_loss"
    ]
)

pairwise_rank_loss = (
    core_namespace[
        "pairwise_rank_loss"
    ]
)


core_source_sha = sha256_file(
    CORE_SOURCE
)


print()
print(
    "=" * 115
)

print(
    "PCND V3f ALL-8 DEVELOPMENT DEPLOYMENT TRAINING"
)

print(
    "=" * 115
)

print(
    "device:",
    DEVICE
)

print(
    "Git commit:",
    git_commit
)

print(
    "frozen V3f source SHA256:",
    core_source_sha
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

seed_everything(
    DEPLOY_SEED
)


if DEVICE.type == "cuda":

    torch.set_float32_matmul_precision(
        "high"
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True


# ============================================================
# LOAD HAPWAVE DEVELOPMENT FEATURE TABLES ONLY
# ============================================================

RAW = {}

feature_hashes = {}


for sub in SUBJECTS:

    path = (
        DATA_ROOT
        / f"{sub}_pcnd_v3_pairs.csv.gz"
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


    feature_hashes[
        path.name
    ] = sha256_file(
        path
    )


    df = pd.read_csv(
        path
    )


    needed = (
        PAIR_FEATURES
        +
        TARGETS
        +
        ROI_COLUMNS
        +
        [
            "stim_site",
            "recording_channel",
        ]
    )


    missing = [
        c
        for c in needed
        if c not in df.columns
    ]


    if missing:

        raise RuntimeError(
            f"{sub}: missing columns "
            f"{missing}"
        )


    for c in (
        PAIR_FEATURES
        +
        TARGETS
    ):

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )


    df = df.dropna(
        subset=(
            PAIR_FEATURES
            +
            TARGETS
        )
    ).copy()


    RAW[sub] = df


    print(
        sub,
        "pairs=",
        len(df),
        "sites=",
        df[
            "stim_site"
        ].nunique(),
    )


# ============================================================
# ALL-8 INTERNAL TRAIN / VALIDATION SPLIT
#
# Fixed BEFORE any normalization.
# ============================================================

site_split = {}


for sub in SUBJECTS:

    names = np.asarray(
        sorted(
            RAW[sub][
                "stim_site"
            ]
            .unique()
            .tolist()
        ),
        dtype=object,
    )


    rng = np.random.default_rng(
        DEPLOY_SEED
        +
        subnum(sub)
        *
        100
    )


    rng.shuffle(
        names
    )


    n_val = max(
        1,
        int(
            round(
                0.15
                *
                len(names)
            )
        ),
    )


    # Preserve enough training sites for k=10.
    n_val = min(
        n_val,
        len(names) - 11,
    )


    val = set(
        names[
            :n_val
        ].tolist()
    )


    train = set(
        names[
            n_val:
        ].tolist()
    )


    if len(train) < 11:

        raise RuntimeError(
            f"{sub}: fewer than 11 "
            f"internal-training sites."
        )


    site_split[
        sub
    ] = {
        "train":
            train,

        "val":
            val,
    }


    print(
        sub,
        "train sites=",
        len(train),
        "val sites=",
        len(val),
    )


# ============================================================
# STRICT TRAIN-SITE-ONLY NORMALIZATION
# ============================================================

norm_frames = []


for sub in SUBJECTS:

    norm_frames.append(
        RAW[sub][
            RAW[sub][
                "stim_site"
            ].isin(
                site_split[sub][
                    "train"
                ]
            )
        ]
    )


norm_df = pd.concat(
    norm_frames,
    ignore_index=True,
)


x_mean = (
    norm_df[
        PAIR_FEATURES
    ]
    .mean()
    .to_numpy(
        dtype=np.float32
    )
)


x_std = (
    norm_df[
        PAIR_FEATURES
    ]
    .std()
    .replace(
        0,
        1,
    )
    .fillna(1)
    .to_numpy(
        dtype=np.float32
    )
)


y_log = np.log1p(
    np.clip(
        norm_df[
            TARGETS
        ]
        .to_numpy(
            dtype=np.float32
        ),
        0,
        None,
    )
)


y_mean = y_log.mean(
    axis=0
)


y_std = y_log.std(
    axis=0
)


y_std[
    y_std
    <
    1e-6
] = 1.0


# ============================================================
# STRICT TRAIN-SITE-ONLY ROI VOCABULARY
# ============================================================

roi_values = set()


for c in ROI_COLUMNS:

    roi_values.update(
        norm_df[c]
        .fillna(
            "UNK"
        )
        .astype(str)
        .tolist()
    )


roi_values.discard(
    "UNK"
)

roi_values.discard(
    "nan"
)


roi_to_idx = {
    "UNK":
        0
}


for roi in sorted(
    roi_values
):

    roi_to_idx[
        roi
    ] = len(
        roi_to_idx
    )


def roi_id(v):

    if pd.isna(v):

        return 0

    return roi_to_idx.get(
        str(v),
        0,
    )


print(
    "deployment ROI vocabulary:",
    len(
        roi_to_idx
    )
)


# ============================================================
# CONSTRUCT SITE OBJECTS
# ============================================================

data = {}


for sub in SUBJECTS:

    data[sub] = {}


    for site, g in (
        RAW[sub]
        .groupby(
            "stim_site"
        )
    ):

        g = (
            g
            .sort_values(
                "recording_channel"
            )
            .reset_index(
                drop=True
            )
        )


        x = (
            g[
                PAIR_FEATURES
            ]
            .to_numpy(
                dtype=np.float32
            )
            -
            x_mean
        ) / x_std


        y_raw = (
            g[
                TARGETS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        y = (
            np.log1p(
                np.clip(
                    y_raw,
                    0,
                    None,
                )
            )
            -
            y_mean
        ) / y_std


        data[sub][site] = SiteData(
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
            ].tolist(),

            name=site,

            subject=sub,

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


# ============================================================
# BUILD MODELS USING FROZEN V3f CLASSES
# ============================================================

population = PopulationModel(
    pair_dim=len(
        PAIR_FEATURES
    ),

    hidden=HIDDEN,

    n_roi=len(
        roi_to_idx
    ),

    roi_dim=ROI_DIM,

    dropout=DROPOUT,
).to(
    DEVICE
)


relation = RelationalMemory(
    hidden=HIDDEN,
    heads=HEADS,
    layers=LAYERS,
    ff_mult=FF_MULT,
    dropout=DROPOUT,
).to(
    DEVICE
)


# ============================================================
# IMPLEMENTATION REGRESSION AGAINST EXISTING V3f CHECKPOINT
# ============================================================

reference = torch.load(
    SCHEMA_REFERENCE,
    map_location="cpu",
    weights_only=False,
)


def assert_state_schema(
    new_state,
    old_state,
    allow_roi_resize=False,
):

    new_keys = set(
        new_state.keys()
    )

    old_keys = set(
        old_state.keys()
    )


    if new_keys != old_keys:

        raise RuntimeError(
            "State-dict key mismatch.\n"
            f"new-only: "
            f"{sorted(new_keys-old_keys)}\n"
            f"old-only: "
            f"{sorted(old_keys-new_keys)}"
        )


    for key in sorted(
        new_keys
    ):

        a = tuple(
            new_state[key].shape
        )

        b = tuple(
            old_state[key].shape
        )


        if (
            allow_roi_resize
            and
            key
            ==
            "roi.weight"
        ):

            if (
                len(a)
                !=
                2
                or
                len(b)
                !=
                2
                or
                a[1]
                !=
                b[1]
            ):

                raise RuntimeError(
                    f"ROI embedding schema mismatch: "
                    f"{a} vs {b}"
                )

            continue


        if a != b:

            raise RuntimeError(
                f"Shape mismatch for {key}: "
                f"{a} vs {b}"
            )


assert_state_schema(
    population.state_dict(),
    reference[
        "population_state"
    ],
    allow_roi_resize=True,
)


assert_state_schema(
    relation.state_dict(),
    reference[
        "relation_state"
    ],
    allow_roi_resize=False,
)


print(
    "V3f architecture schema regression: PASS"
)


# ============================================================
# STAGE A — POPULATION PRIOR
# ============================================================

opt_pop = torch.optim.AdamW(
    population.parameters(),
    lr=LR_POP,
    weight_decay=1e-4,
)


rng = np.random.default_rng(
    DEPLOY_SEED
)


@torch.no_grad()
def population_validation():

    population.eval()

    losses = []


    for sub in SUBJECTS:

        for qname in sorted(
            site_split[sub][
                "val"
            ]
        ):

            q = data[sub][qname]

            mu, lv, _ = (
                population.forward_site(
                    q
                )
            )

            losses.append(
                float(
                    gaussian_nll(
                        mu,
                        lv,
                        q.y,
                    ).item()
                )
            )


    return float(
        np.mean(
            losses
        )
    )


best_pop_val = np.inf

best_pop_step = None

best_pop_state = None


print()
print(
    "=" * 115
)

print(
    "STAGE A — ALL-8 STRICT POPULATION TRAINING"
)

print(
    "=" * 115
)


for step in range(
    1,
    POP_STEPS + 1
):

    population.train()

    opt_pop.zero_grad(
        set_to_none=True
    )


    total = 0.0


    for _ in range(
        TASK_BATCH
    ):

        sub = str(
            rng.choice(
                SUBJECTS
            )
        )


        qname = str(
            rng.choice(
                sorted(
                    site_split[sub][
                        "train"
                    ]
                )
            )
        )


        q = data[sub][qname]


        mu, lv, _ = (
            population.forward_site(
                q
            )
        )


        total = (
            total
            +
            gaussian_nll(
                mu,
                lv,
                q.y,
            )
        )


    total = (
        total
        /
        TASK_BATCH
    )


    total.backward()


    torch.nn.utils.clip_grad_norm_(
        population.parameters(),
        5.0,
    )


    opt_pop.step()


    if (
        step == 1
        or
        step % EVAL_EVERY == 0
    ):

        val = (
            population_validation()
        )


        print(
            f"POP {step:05d} "
            f"train={total.item():.6f} "
            f"val={val:.6f}"
        )


        if val < best_pop_val:

            best_pop_val = val

            best_pop_step = step

            best_pop_state = (
                copy.deepcopy(
                    population.state_dict()
                )
            )


if best_pop_state is None:

    raise RuntimeError(
        "No population checkpoint selected."
    )


population.load_state_dict(
    best_pop_state
)


population.eval()


for p in (
    population.parameters()
):

    p.requires_grad = False


print()
print(
    "BEST POPULATION STEP:",
    best_pop_step
)

print(
    "BEST POPULATION VAL:",
    best_pop_val
)


# ============================================================
# CACHE FROZEN POPULATION OUTPUTS
# ============================================================

with torch.no_grad():

    for sub in SUBJECTS:

        for site in (
            data[sub]
            .values()
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


# ============================================================
# STAGE B — RELATIONAL MEMORY
# ============================================================

opt_rel = torch.optim.AdamW(
    relation.parameters(),
    lr=LR_REL,
    weight_decay=1e-4,
)


rng_rel = np.random.default_rng(
    DEPLOY_SEED
    +
    999
)


def sample_relation_task():

    sub = str(
        rng_rel.choice(
            SUBJECTS
        )
    )


    names = sorted(
        site_split[sub][
            "train"
        ]
    )


    qname = str(
        rng_rel.choice(
            names
        )
    )


    candidates = [
        x
        for x in names
        if x != qname
    ]


    k = int(
        rng_rel.choice(
            REL_TRAIN_K
        )
    )


    if len(
        candidates
    ) < k:

        raise RuntimeError(
            f"{sub}: insufficient "
            f"relation context candidates."
        )


    chosen = (
        rng_rel.choice(
            candidates,
            size=k,
            replace=False,
        )
        .tolist()
    )


    context = [
        data[sub][x]
        for x in chosen
    ]


    return (
        sub,
        context,
        data[sub][qname],
        k,
    )


@torch.no_grad()
def relation_validation():

    relation.eval()

    scores = []


    for sub in SUBJECTS:

        train_names = sorted(
            site_split[sub][
                "train"
            ]
        )


        val_names = sorted(
            site_split[sub][
                "val"
            ]
        )


        for k in REL_TRAIN_K:

            vrng = np.random.default_rng(
                DEPLOY_SEED
                +
                subnum(sub)
                *
                1000
                +
                k
            )


            context_names = (
                vrng.choice(
                    train_names,
                    size=k,
                    replace=False,
                )
                .tolist()
            )


            context = [
                data[sub][x]
                for x in context_names
            ]


            for qi, qname in enumerate(
                val_names
            ):

                q = data[sub][qname]


                (
                    c,
                    node_q,
                    node_map,
                ) = estimate_hierarchy(
                    context,
                    q,
                )


                hierarchy_mu = (
                    q.pop_mu
                    +
                    c
                    +
                    node_q
                )


                target_t = (
                    q.y
                    -
                    hierarchy_mu
                )


                delta = relation(
                    context,
                    q,
                    c,
                    node_map,
                )


                prng = np.random.default_rng(
                    DEPLOY_SEED
                    +
                    subnum(sub)
                    *
                    100000
                    +
                    k
                    *
                    100
                    +
                    qi
                )


                delta_perm = relation(
                    context,
                    q,
                    c,
                    node_map,
                    permute_within_channel=True,
                    rng=prng,
                )


                hub = F.smooth_l1_loss(
                    delta,
                    target_t,
                )


                corr = (
                    differentiable_corr_loss(
                        delta,
                        target_t,
                    )
                )


                hub_perm = (
                    F.smooth_l1_loss(
                        delta_perm,
                        target_t,
                    )
                )


                neg = F.relu(
                    NEG_MARGIN
                    +
                    hub
                    -
                    hub_perm
                )


                score = (
                    hub
                    +
                    W_CORR
                    *
                    corr
                    +
                    W_NEG
                    *
                    neg
                )


                scores.append(
                    float(
                        score.item()
                    )
                )


    return float(
        np.mean(
            scores
        )
    )


best_rel_val = np.inf

best_rel_step = None

best_rel_state = None


print()
print(
    "=" * 115
)

print(
    "STAGE B — ALL-8 STRICT RELATIONAL TRAINING"
)

print(
    "=" * 115
)


for step in range(
    1,
    REL_STEPS + 1
):

    relation.train()

    opt_rel.zero_grad(
        set_to_none=True
    )


    total = 0.0


    log_hub = 0.0
    log_corr = 0.0
    log_rank = 0.0
    log_neg = 0.0
    log_pred = 0.0


    for task_idx in range(
        TASK_BATCH
    ):

        (
            sub,
            context,
            q,
            k,
        ) = sample_relation_task()


        with torch.no_grad():

            (
                c,
                node_q,
                node_map,
            ) = estimate_hierarchy(
                context,
                q,
            )


            hierarchy_mu = (
                q.pop_mu
                +
                c
                +
                node_q
            )


            target_t = (
                q.y
                -
                hierarchy_mu
            ).detach()


        delta = relation(
            context,
            q,
            c,
            node_map,
        )


        prng = np.random.default_rng(
            DEPLOY_SEED
            +
            step
            *
            1000
            +
            task_idx
        )


        delta_perm = relation(
            context,
            q,
            c,
            node_map,
            permute_within_channel=True,
            rng=prng,
        )


        hub = F.smooth_l1_loss(
            delta,
            target_t,
        )


        corr = (
            differentiable_corr_loss(
                delta,
                target_t,
            )
        )


        rank = (
            pairwise_rank_loss(
                delta,
                target_t,
            )
        )


        hub_perm = (
            F.smooth_l1_loss(
                delta_perm,
                target_t,
            )
        )


        neg = F.relu(
            NEG_MARGIN
            +
            hub
            -
            hub_perm
        )


        final_mu = (
            hierarchy_mu
            +
            delta
        )


        pred = gaussian_nll(
            final_mu,
            q.pop_logvar,
            q.y,
        )


        loss = (
            hub
            +
            W_PRED
            *
            pred
            +
            W_CORR
            *
            corr
            +
            W_RANK
            *
            rank
            +
            W_NEG
            *
            neg
        )


        total = (
            total
            +
            loss
        )


        log_hub += float(
            hub.item()
        )

        log_corr += float(
            corr.item()
        )

        log_rank += float(
            rank.item()
        )

        log_neg += float(
            neg.item()
        )

        log_pred += float(
            pred.item()
        )


    total = (
        total
        /
        TASK_BATCH
    )


    total.backward()


    torch.nn.utils.clip_grad_norm_(
        relation.parameters(),
        5.0,
    )


    opt_rel.step()


    if (
        step == 1
        or
        step % EVAL_EVERY == 0
    ):

        val = (
            relation_validation()
        )


        print(
            f"REL {step:05d} "
            f"loss={total.item():.5f} "
            f"hub={log_hub/TASK_BATCH:.5f} "
            f"corr={log_corr/TASK_BATCH:.5f} "
            f"rank={log_rank/TASK_BATCH:.5f} "
            f"neg={log_neg/TASK_BATCH:.5f} "
            f"pred={log_pred/TASK_BATCH:.5f} "
            f"val={val:.5f}"
        )


        if val < best_rel_val:

            best_rel_val = val

            best_rel_step = step

            best_rel_state = (
                copy.deepcopy(
                    relation.state_dict()
                )
            )


if best_rel_state is None:

    raise RuntimeError(
        "No relation checkpoint selected."
    )


relation.load_state_dict(
    best_rel_state
)


relation.eval()


print()
print(
    "BEST RELATION STEP:",
    best_rel_step
)

print(
    "BEST RELATION VAL:",
    best_rel_val
)


# ============================================================
# FROZEN SPLIT SERIALIZATION
# ============================================================

split_json = {
    sub: {
        "train":
            sorted(
                site_split[sub][
                    "train"
                ]
            ),

        "val":
            sorted(
                site_split[sub][
                    "val"
                ]
            ),
    }

    for sub in SUBJECTS
}


split_path = (
    OUT
    / "all8_internal_site_split.json"
)


split_path.write_text(
    json.dumps(
        split_json,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# SAVE UNIQUE DEPLOYMENT CHECKPOINT
# ============================================================

checkpoint_path = (
    OUT
    / "pcnd_v3f_all8_deployment.pt"
)


torch.save(
    {
        "deployment_version":
            DEPLOYMENT_VERSION,

        "deployment_seed":
            DEPLOY_SEED,

        "subjects":
            SUBJECTS,

        "population_state":
            population.state_dict(),

        "relation_state":
            relation.state_dict(),

        "best_population_step":
            best_pop_step,

        "best_population_val":
            best_pop_val,

        "best_relation_step":
            best_rel_step,

        "best_relation_val":
            best_rel_val,

        "x_mean":
            x_mean,

        "x_std":
            x_std,

        "y_mean":
            y_mean,

        "y_std":
            y_std,

        "roi_to_idx":
            roi_to_idx,

        "pair_features":
            PAIR_FEATURES,

        "targets":
            TARGETS,

        "site_split":
            split_json,

        "hyperparameters": {
            "hidden":
                HIDDEN,

            "heads":
                HEADS,

            "layers":
                LAYERS,

            "roi_dim":
                ROI_DIM,

            "ff_mult":
                FF_MULT,

            "dropout":
                DROPOUT,

            "lr_pop":
                LR_POP,

            "lr_rel":
                LR_REL,

            "pop_steps":
                POP_STEPS,

            "rel_steps":
                REL_STEPS,

            "task_batch":
                TASK_BATCH,

            "w_pred":
                W_PRED,

            "w_corr":
                W_CORR,

            "w_rank":
                W_RANK,

            "w_neg":
                W_NEG,

            "neg_margin":
                NEG_MARGIN,
        },

        "provenance": {
            "pcnd_git_commit":
                git_commit,

            "v3f_source_script":
                str(
                    CORE_SOURCE
                ),

            "v3f_source_sha256":
                core_source_sha,

            "hapwave_protocol_sha256":
                EXPECTED_HAP_PROTOCOL_SHA256,

            "external_lockbox_sha256":
                EXPECTED_EXTERNAL_LOCKBOX_SHA256,

            "roi_bridge_sha256":
                EXPECTED_ROI_BRIDGE_SHA256,
        },
    },

    checkpoint_path,
)


checkpoint_sha = sha256_file(
    checkpoint_path
)


# ============================================================
# DEPLOYMENT MANIFEST
# ============================================================

feature_table_hashes = {
    str(
        DATA_ROOT
        /
        filename
    ):
        digest

    for filename, digest in (
        feature_hashes.items()
    )
}


manifest = {
    "deployment_version":
        DEPLOYMENT_VERSION,

    "freeze_time_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "purpose":
        (
            "Single frozen all-8 HAPwave "
            "development deployment model for "
            "one-shot evaluation on the previously "
            "sealed ds004080 external lockbox."
        ),

    "development_subjects":
        SUBJECTS,

    "external_data_used_for_training":
        False,

    "external_signals_opened":
        False,

    "external_response_features_used":
        False,

    "external_model_performance_observed":
        False,

    "architecture_source": {
        "path":
            str(
                CORE_SOURCE
            ),

        "sha256":
            core_source_sha,

        "reuse_method":
            (
                "AST extraction of the exact frozen "
                "V3f architecture/loss definitions "
                "from script30."
            ),

        "state_schema_regression":
            "PASS",
    },

    "training": {
        "seed":
            DEPLOY_SEED,

        "normalization":
            (
                "Fit only on internal-training "
                "stimulation sites from all 8 "
                "HAPwave development patients."
            ),

        "checkpoint_selection":
            (
                "Internal HAPwave validation "
                "stimulation sites only."
            ),

        "best_population_step":
            best_pop_step,

        "best_population_val":
            best_pop_val,

        "best_relation_step":
            best_rel_step,

        "best_relation_val":
            best_rel_val,

        "hyperparameters":
            {
                "hidden":
                    HIDDEN,

                "heads":
                    HEADS,

                "layers":
                    LAYERS,

                "roi_dim":
                    ROI_DIM,

                "ff_mult":
                    FF_MULT,

                "dropout":
                    DROPOUT,

                "population_steps":
                    POP_STEPS,

                "relation_steps":
                    REL_STEPS,

                "lr_pop":
                    LR_POP,

                "lr_rel":
                    LR_REL,

                "w_pred":
                    W_PRED,

                "w_corr":
                    W_CORR,

                "w_rank":
                    W_RANK,

                "w_neg":
                    W_NEG,

                "negative_margin":
                    NEG_MARGIN,
            },
    },

    "normalization": {
        "n_internal_training_rows":
            int(
                len(
                    norm_df
                )
            ),

        "roi_vocab_size":
            int(
                len(
                    roi_to_idx
                )
            ),

        "unseen_external_roi_rule":
            "UNK",
    },

    "frozen_dependencies": {
        "hapwave_protocol_sha256":
            EXPECTED_HAP_PROTOCOL_SHA256,

        "ds004080_lockbox_sha256":
            EXPECTED_EXTERNAL_LOCKBOX_SHA256,

        "ds004080_roi_bridge_sha256":
            EXPECTED_ROI_BRIDGE_SHA256,
    },

    "code": {
        "pcnd_git_commit":
            git_commit,

        "tracked_worktree_clean_at_training_start":
            True,
    },

    "development_feature_tables":
        feature_table_hashes,

    "frozen_artifacts": {
        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_sha256":
            checkpoint_sha,

        "internal_split":
            str(
                split_path
            ),

        "internal_split_sha256":
            sha256_file(
                split_path
            ),
    },

    "post_freeze_rule":
        (
            "This checkpoint may not be replaced, "
            "retrained, seed-selected, or modified "
            "after ds004080 raw signal opening."
        ),
}


manifest_path = (
    OUT
    / "ALL8_DEPLOYMENT_MANIFEST.json"
)


manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)


manifest_sha = sha256_file(
    manifest_path
)


seal = {
    "deployment_version":
        DEPLOYMENT_VERSION,

    "checkpoint_sha256":
        checkpoint_sha,

    "deployment_manifest_sha256":
        manifest_sha,

    "development_protocol_sha256":
        EXPECTED_HAP_PROTOCOL_SHA256,

    "external_lockbox_sha256":
        EXPECTED_EXTERNAL_LOCKBOX_SHA256,

    "roi_bridge_sha256":
        EXPECTED_ROI_BRIDGE_SHA256,

    "pcnd_git_commit":
        git_commit,

    "external_raw_opened_before_freeze":
        False,
}


seal_path = (
    OUT
    / "ALL8_DEPLOYMENT_FREEZE_SEAL.json"
)


seal_path.write_text(
    json.dumps(
        seal,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print(
    "=" * 115
)

print(
    "PCND V3f ALL-8 DEPLOYMENT MODEL FROZEN"
)

print(
    "=" * 115
)

print(
    "deployment seed:",
    DEPLOY_SEED
)

print(
    "best population step:",
    best_pop_step
)

print(
    "best population val:",
    best_pop_val
)

print(
    "best relation step:",
    best_rel_step
)

print(
    "best relation val:",
    best_rel_val
)

print(
    "ROI vocabulary size:",
    len(
        roi_to_idx
    )
)

print(
    "PCND Git commit:",
    git_commit
)

print()
print(
    "CHECKPOINT SHA256:",
    checkpoint_sha
)

print(
    "DEPLOYMENT MANIFEST SHA256:",
    manifest_sha
)

print()
print(
    "External lockbox SHA256:",
    EXPECTED_EXTERNAL_LOCKBOX_SHA256
)

print(
    "HAPwave protocol SHA256:",
    EXPECTED_HAP_PROTOCOL_SHA256
)

print(
    "ROI bridge SHA256:",
    EXPECTED_ROI_BRIDGE_SHA256
)

print()
print(
    "DS004080 RAW SIGNALS OPENED BEFORE FREEZE: NO"
)

print(
    "EXTERNAL MODEL PERFORMANCE OBSERVED: NO"
)

print()
print(
    "MODEL STATUS: IMMUTABLE FOR EXTERNAL EVALUATION"
)
