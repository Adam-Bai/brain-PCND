from __future__ import annotations

from pathlib import Path
import argparse
import copy
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import pearsonr, spearmanr


ROOT = Path("/home/dell/jiaxing/bby/pcnd")

DATA_ROOT = (
    ROOT
    / "results"
    / "clinical_v5"
    / "v3_features"
)

OUT_ROOT = (
    ROOT
    / "results"
    / "clinical_v9"
    / "v3f_final_strict"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--heldout",
    default="all",
)

parser.add_argument(
    "--pop-steps",
    type=int,
    default=4000,
)

parser.add_argument(
    "--rel-steps",
    type=int,
    default=5000,
)

parser.add_argument(
    "--task-batch",
    type=int,
    default=4,
)

parser.add_argument(
    "--hidden",
    type=int,
    default=256,
)

parser.add_argument(
    "--heads",
    type=int,
    default=8,
)

parser.add_argument(
    "--layers",
    type=int,
    default=3,
)

parser.add_argument(
    "--roi-dim",
    type=int,
    default=48,
)

parser.add_argument(
    "--ff-mult",
    type=int,
    default=4,
)

parser.add_argument(
    "--dropout",
    type=float,
    default=0.10,
)

parser.add_argument(
    "--lr-pop",
    type=float,
    default=2e-4,
)

parser.add_argument(
    "--lr-rel",
    type=float,
    default=2e-4,
)

parser.add_argument(
    "--eval-every",
    type=int,
    default=250,
)

parser.add_argument(
    "--test-repeats",
    type=int,
    default=20,
)

parser.add_argument(
    "--w-pred",
    type=float,
    default=0.25,
)

parser.add_argument(
    "--w-corr",
    type=float,
    default=0.25,
)

parser.add_argument(
    "--w-rank",
    type=float,
    default=0.10,
)

parser.add_argument(
    "--w-neg",
    type=float,
    default=0.25,
)

parser.add_argument(
    "--neg-margin",
    type=float,
    default=0.01,
)

parser.add_argument(
    "--seed",
    type=int,
    default=20260925,
)

args = parser.parse_args()


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "device:",
    DEVICE
)

if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision(
        "high"
    )

    torch.backends.cuda.matmul.allow_tf32 = True


SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 9)
]

if args.heldout == "all":
    HELDOUT_SUBJECTS = SUBJECTS
else:
    if args.heldout not in SUBJECTS:
        raise ValueError(
            args.heldout
        )

    HELDOUT_SUBJECTS = [
        args.heldout
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


# Relation is not identifiable after node decomposition with k=1.
REL_TRAIN_K = [
    3,
    5,
    10,
]

TEST_K = [
    0,
    1,
    3,
    5,
    10,
]


# ============================================================
# LOAD RAW FEATURE TABLES
# ============================================================

RAW = {}

for sub in SUBJECTS:

    f = (
        DATA_ROOT
        / f"{sub}_pcnd_v3_pairs.csv.gz"
    )

    df = pd.read_csv(f)

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
            f"{sub}: missing {missing}"
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
        df["stim_site"].nunique(),
    )


# ============================================================
# UTILITIES
# ============================================================

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


def safe_corr(y, p, kind):

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
        or np.std(y) < 1e-12
        or np.std(p) < 1e-12
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


def top_recall(y, p, frac=0.10):

    y = np.asarray(y)
    p = np.asarray(p)

    k = max(
        1,
        int(
            round(
                frac * len(y)
            )
        )
    )

    a = set(
        np.argpartition(
            y,
            -k,
        )[-k:]
    )

    b = set(
        np.argpartition(
            p,
            -k,
        )[-k:]
    )

    return len(
        a & b
    ) / k


# ============================================================
# STRICT INTERNAL SPLIT — BEFORE NORMALIZATION
# ============================================================

def make_internal_split(
    sub,
    heldout,
):

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
        args.seed
        +
        subnum(heldout)
        * 10000
        +
        subnum(sub)
        * 100
    )

    rng.shuffle(names)

    n_val = max(
        1,
        int(
            round(
                0.15
                * len(names)
            )
        )
    )

    # Need at least query + k=10.
    n_val = min(
        n_val,
        len(names) - 11,
    )

    val = set(
        names[:n_val]
        .tolist()
    )

    train = set(
        names[n_val:]
        .tolist()
    )

    return train, val


# ============================================================
# SITE DATA
# ============================================================

class SiteData:

    def __init__(
        self,
        x,
        y,
        y_raw,
        channels,
        name,
        subject,
        roi_a,
        roi_b,
        roi_rec,
    ):

        self.x = x
        self.y = y
        self.y_raw = y_raw

        self.channels = list(
            channels
        )

        self.name = name
        self.subject = subject

        self.roi_a = roi_a
        self.roi_b = roi_b
        self.roi_rec = roi_rec

        self.channel_to_idx = {
            ch: i
            for i, ch
            in enumerate(
                self.channels
            )
        }


# ============================================================
# BUILD FOLD — STRICT NORMALIZATION
# ============================================================

def build_fold_data(
    heldout,
):

    train_subjects = [
        s
        for s in SUBJECTS
        if s != heldout
    ]

    # FIRST split.
    site_split = {}

    for sub in train_subjects:

        tr, va = make_internal_split(
            sub,
            heldout,
        )

        site_split[sub] = {
            "train": tr,
            "val": va,
        }

    # THEN normalization from INTERNAL TRAIN SITES ONLY.
    norm_frames = []

    for sub in train_subjects:

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
            ].to_numpy(
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
        y_std < 1e-6
    ] = 1.0

    # ROI vocabulary also INTERNAL TRAIN ONLY.
    roi_values = set()

    for c in ROI_COLUMNS:

        roi_values.update(
            norm_df[c]
            .fillna("UNK")
            .astype(str)
            .tolist()
        )

    roi_values.discard("UNK")
    roi_values.discard("nan")

    roi_to_idx = {
        "UNK": 0
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

    data = {}

    for sub in SUBJECTS:

        data[sub] = {}

        for site, g in RAW[
            sub
        ].groupby(
            "stim_site"
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

    return (
        train_subjects,
        site_split,
        data,
        x_mean,
        x_std,
        y_mean,
        y_std,
        roi_to_idx,
    )


# ============================================================
# NETWORK BLOCKS
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        dims,
        dropout=0.0,
    ):

        super().__init__()

        layers = []

        for i in range(
            len(dims) - 1
        ):

            layers.append(
                nn.Linear(
                    dims[i],
                    dims[i + 1],
                )
            )

            if i < len(dims) - 2:

                layers.append(
                    nn.GELU()
                )

                if dropout > 0:
                    layers.append(
                        nn.Dropout(
                            dropout
                        )
                    )

        self.net = nn.Sequential(
            *layers
        )


    def forward(self, x):

        return self.net(x)


class ResidualFFN(nn.Module):

    def __init__(
        self,
        hidden,
        dropout,
    ):

        super().__init__()

        self.norm = nn.LayerNorm(
            hidden
        )

        self.ff = nn.Sequential(
            nn.Linear(
                hidden,
                hidden * 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden * 2,
                hidden,
            ),
            nn.Dropout(dropout),
        )


    def forward(self, x):

        return (
            x
            +
            self.ff(
                self.norm(x)
            )
        )


class CrossBlock(nn.Module):

    def __init__(
        self,
        hidden,
        heads,
        ff_mult,
        dropout,
    ):

        super().__init__()

        self.qnorm = nn.LayerNorm(
            hidden
        )

        self.mnorm = nn.LayerNorm(
            hidden
        )

        self.attn = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffnorm = nn.LayerNorm(
            hidden
        )

        self.ff = nn.Sequential(
            nn.Linear(
                hidden,
                hidden * ff_mult,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden * ff_mult,
                hidden,
            ),
            nn.Dropout(dropout),
        )


    def forward(
        self,
        q,
        memory,
        mask,
    ):

        out, _ = self.attn(
            query=self.qnorm(
                q
            ).unsqueeze(1),

            key=self.mnorm(
                memory
            ),

            value=self.mnorm(
                memory
            ),

            key_padding_mask=mask,

            need_weights=False,
        )

        q = (
            q
            +
            out.squeeze(1)
        )

        q = (
            q
            +
            self.ff(
                self.ffnorm(q)
            )
        )

        return q


# ============================================================
# POPULATION PRIOR
# ============================================================

class PopulationModel(nn.Module):

    def __init__(
        self,
        pair_dim,
        hidden,
        n_roi,
        roi_dim,
        dropout,
    ):

        super().__init__()

        self.hidden = hidden

        self.roi = nn.Embedding(
            n_roi,
            roi_dim,
        )

        self.geo = MLP(
            [
                pair_dim,
                hidden,
                hidden * 2,
                hidden,
            ],
            dropout,
        )

        self.anat = MLP(
            [
                roi_dim * 3,
                hidden * 2,
                hidden,
            ],
            dropout,
        )

        self.fuse = MLP(
            [
                hidden * 2,
                hidden * 2,
                hidden,
            ],
            dropout,
        )

        self.refine = nn.ModuleList(
            [
                ResidualFFN(
                    hidden,
                    dropout,
                )
                for _ in range(2)
            ]
        )

        self.head_e = MLP(
            [
                hidden,
                hidden * 2,
                hidden,
                2,
            ],
            dropout,
        )

        self.head_l = MLP(
            [
                hidden,
                hidden * 2,
                hidden,
                2,
            ],
            dropout,
        )


    def encode(
        self,
        x,
        roi_a,
        roi_b,
        roi_rec,
    ):

        g = self.geo(x)

        ea = self.roi(roi_a)
        eb = self.roi(roi_b)
        er = self.roi(roi_rec)

        anat = torch.cat(
            [
                (
                    ea + eb
                ) / 2.0,

                torch.abs(
                    ea - eb
                ),

                er,
            ],
            dim=-1,
        )

        a = self.anat(anat)

        q = self.fuse(
            torch.cat(
                [
                    g,
                    a,
                ],
                dim=-1,
            )
        )

        for block in self.refine:
            q = block(q)

        return q


    def forward_site(
        self,
        site,
    ):

        q = self.encode(
            site.x,
            site.roi_a,
            site.roi_b,
            site.roi_rec,
        )

        e = self.head_e(q)
        l = self.head_l(q)

        mu = torch.stack(
            [
                e[:, 0],
                l[:, 0],
            ],
            dim=-1,
        )

        logvar = torch.stack(
            [
                e[:, 1],
                l[:, 1],
            ],
            dim=-1,
        )

        logvar = torch.clamp(
            logvar,
            -6.0,
            3.0,
        )

        return (
            mu,
            logvar,
            q,
        )


# ============================================================
# ANALYTIC HIERARCHY
# ============================================================

@torch.no_grad()
def estimate_hierarchy(
    context_sites,
    query_site,
):

    residuals = []

    channel_values = {}

    for site in context_sites:

        r = (
            site.y
            -
            site.pop_mu
        )

        residuals.append(r)

    # C_s
    global_shift = torch.cat(
        residuals,
        dim=0,
    ).median(
        dim=0
    ).values

    # A_{s,i}
    for site in context_sites:

        centered = (
            site.y
            -
            site.pop_mu
            -
            global_shift
        )

        for idx, ch in enumerate(
            site.channels
        ):

            channel_values.setdefault(
                ch,
                []
            ).append(
                centered[idx]
            )

    node_map = {}

    for ch, vals in (
        channel_values.items()
    ):

        node_map[ch] = torch.stack(
            vals,
            dim=0,
        ).median(
            dim=0
        ).values

    node_query = torch.stack(
        [
            node_map.get(
                ch,
                torch.zeros(
                    2,
                    device=DEVICE,
                ),
            )
            for ch in query_site.channels
        ],
        dim=0,
    )

    return (
        global_shift,
        node_query,
        node_map,
    )


# ============================================================
# RELATIONAL MEMORY T_{s,u,i}
# ============================================================

class RelationalMemory(nn.Module):

    def __init__(
        self,
        hidden,
        heads,
        layers,
        ff_mult,
        dropout,
    ):

        super().__init__()

        self.hidden = hidden

        self.memory_encoder = MLP(
            [
                hidden + 2,
                hidden * 2,
                hidden,
            ],
            dropout,
        )

        self.query_adapter = MLP(
            [
                hidden,
                hidden * 2,
                hidden,
            ],
            dropout,
        )

        self.blocks = nn.ModuleList(
            [
                CrossBlock(
                    hidden,
                    heads,
                    ff_mult,
                    dropout,
                )
                for _ in range(
                    layers
                )
            ]
        )

        self.norm = nn.LayerNorm(
            hidden
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden,
                2,
            ),
        )

        # Start with exactly M2 = population + calibration + node.
        nn.init.zeros_(
            self.head[-1].weight
        )

        nn.init.zeros_(
            self.head[-1].bias
        )


    def build_memory(
        self,
        context_sites,
        query_site,
        global_shift,
        node_map,
        permute_within_channel=False,
        rng=None,
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
                self.hidden,
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

                si = site.channel_to_idx.get(
                    ch,
                    None,
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

            # ------------------------------------------------
            # STRICT NEGATIVE:
            # keep recording channel fixed,
            # only permute stimulation identity.
            # ------------------------------------------------
            if permute_within_channel:

                if rng is None:
                    raise ValueError(
                        "rng required"
                    )

                perm_inter = inter_all.clone()

                for qi in torch.unique(
                    q_all
                ).tolist():

                    idx = torch.where(
                        q_all == qi
                    )[0]

                    if len(idx) <= 1:
                        continue

                    perm = rng.permutation(
                        len(idx)
                    )

                    perm = torch.tensor(
                        perm,
                        dtype=torch.long,
                        device=DEVICE,
                    )

                    perm_inter[idx] = (
                        inter_all[
                            idx[perm]
                        ]
                    )

                inter_all = perm_inter

            tokens = self.memory_encoder(
                torch.cat(
                    [
                        pair_all,
                        inter_all,
                    ],
                    dim=-1,
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

        # Dummy token.
        mask[:, -1] = False

        return memory, mask


    def forward(
        self,
        context_sites,
        query_site,
        global_shift,
        node_map,
        permute_within_channel=False,
        rng=None,
    ):

        # After decomposition, k=1 contains no identifiable
        # stimulation-specific interaction.
        if len(context_sites) < 2:

            return torch.zeros(
                (
                    len(
                        query_site.channels
                    ),
                    2,
                ),
                device=DEVICE,
            )

        memory, mask = self.build_memory(
            context_sites,
            query_site,
            global_shift,
            node_map,
            permute_within_channel,
            rng,
        )

        q = self.query_adapter(
            query_site.pop_q
        )

        for block in self.blocks:

            q = block(
                q,
                memory,
                mask,
            )

        return self.head(
            self.norm(q)
        )


# ============================================================
# LOSSES
# ============================================================

def gaussian_nll(
    mu,
    logvar,
    y,
):

    return (
        0.5
        *
        (
            torch.exp(
                -logvar
            )
            *
            (
                y - mu
            ) ** 2
            +
            logvar
        )
    ).mean()


def differentiable_corr_loss(
    pred,
    target,
):

    losses = []

    for d in range(2):

        p = pred[:, d]
        t = target[:, d]

        p = p - p.mean()
        t = t - t.mean()

        denom = torch.sqrt(
            (
                p.pow(2).sum()
                *
                t.pow(2).sum()
            )
            +
            1e-8
        )

        corr = (
            p * t
        ).sum() / denom

        losses.append(
            1.0 - corr
        )

    return torch.stack(
        losses
    ).mean()


def pairwise_rank_loss(
    pred,
    target,
    n_pairs=256,
):

    n = pred.shape[0]

    if n < 2:
        return pred.sum() * 0.0

    losses = []

    for d in range(2):

        i = torch.randint(
            0,
            n,
            (
                n_pairs,
            ),
            device=pred.device,
        )

        j = torch.randint(
            0,
            n,
            (
                n_pairs,
            ),
            device=pred.device,
        )

        dt = (
            target[i, d]
            -
            target[j, d]
        )

        sign = torch.sign(dt)

        valid = sign != 0

        if valid.sum() == 0:
            continue

        dp = (
            pred[i, d]
            -
            pred[j, d]
        )

        losses.append(
            F.softplus(
                -sign[valid]
                *
                dp[valid]
            ).mean()
        )

    if not losses:
        return pred.sum() * 0.0

    return torch.stack(
        losses
    ).mean()


# ============================================================
# TRAIN FOLD
# ============================================================

def train_fold(
    heldout,
):

    fold_seed = (
        args.seed
        +
        subnum(heldout)
        * 100000
    )

    seed_everything(
        fold_seed
    )

    fold_out = (
        OUT_ROOT
        / heldout
    )

    fold_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_subjects,
        site_split,
        data,
        x_mean,
        x_std,
        y_mean,
        y_std,
        roi_to_idx,
    ) = build_fold_data(
        heldout
    )

    print()
    print("=" * 110)
    print(
        "STRICT LOPO:",
        heldout
    )
    print("=" * 110)

    print(
        "ROI vocabulary:",
        len(
            roi_to_idx
        )
    )


    # ========================================================
    # STAGE A — CLEAN POPULATION PRIOR
    # ========================================================

    population = PopulationModel(
        pair_dim=len(
            PAIR_FEATURES
        ),

        hidden=args.hidden,

        n_roi=len(
            roi_to_idx
        ),

        roi_dim=args.roi_dim,

        dropout=args.dropout,
    ).to(
        DEVICE
    )

    opt_pop = torch.optim.AdamW(
        population.parameters(),
        lr=args.lr_pop,
        weight_decay=1e-4,
    )

    rng = np.random.default_rng(
        fold_seed
    )

    best_pop = np.inf
    best_pop_state = None
    best_pop_step = None


    @torch.no_grad()
    def pop_val():

        population.eval()

        losses = []

        for sub in train_subjects:

            for qname in (
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
            np.mean(losses)
        )


    print()
    print(
        "STAGE A — STRICT POPULATION"
    )

    for step in range(
        1,
        args.pop_steps + 1
    ):

        population.train()

        opt_pop.zero_grad(
            set_to_none=True
        )

        loss = 0.0

        for _ in range(
            args.task_batch
        ):

            sub = str(
                rng.choice(
                    train_subjects
                )
            )

            qname = str(
                rng.choice(
                    list(
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

            loss = (
                loss
                +
                gaussian_nll(
                    mu,
                    lv,
                    q.y,
                )
            )

        loss = (
            loss
            /
            args.task_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            population.parameters(),
            5.0,
        )

        opt_pop.step()

        if (
            step == 1
            or
            step
            % args.eval_every
            == 0
        ):

            val = pop_val()

            print(
                f"{heldout} "
                f"POP {step:05d} "
                f"train={loss.item():.5f} "
                f"val={val:.5f}"
            )

            if val < best_pop:

                best_pop = val
                best_pop_step = step

                best_pop_state = copy.deepcopy(
                    population.state_dict()
                )

    population.load_state_dict(
        best_pop_state
    )

    population.eval()

    for p in population.parameters():
        p.requires_grad = False

    print(
        "best population:",
        best_pop_step,
        best_pop
    )


    # ========================================================
    # CACHE POPULATION FEATURES FOR ALL PATIENTS
    # ========================================================

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
                ) = population.forward_site(
                    site
                )

                site.pop_mu = mu.detach()
                site.pop_logvar = lv.detach()
                site.pop_q = q.detach()


    # ========================================================
    # STAGE B — RELATIONAL T
    # ========================================================

    relation = RelationalMemory(
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
    ).to(
        DEVICE
    )

    print(
        "relation trainable params:",
        f"{sum(p.numel() for p in relation.parameters()):,}"
    )

    opt_rel = torch.optim.AdamW(
        relation.parameters(),
        lr=args.lr_rel,
        weight_decay=1e-4,
    )

    rng = np.random.default_rng(
        fold_seed + 999
    )


    def sample_relation_task():

        sub = str(
            rng.choice(
                train_subjects
            )
        )

        names = list(
            site_split[sub][
                "train"
            ]
        )

        qname = str(
            rng.choice(names)
        )

        candidates = [
            x
            for x in names
            if x != qname
        ]

        k = int(
            rng.choice(
                REL_TRAIN_K
            )
        )

        chosen = (
            rng.choice(
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
    def relation_val():

        relation.eval()

        scores = []

        for sub in train_subjects:

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
                    fold_seed
                    +
                    subnum(sub) * 1000
                    +
                    k
                )

                chosen = (
                    vrng.choice(
                        train_names,
                        size=k,
                        replace=False,
                    )
                    .tolist()
                )

                context = [
                    data[sub][x]
                    for x in chosen
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
                        fold_seed
                        +
                        subnum(sub)
                        * 100000
                        +
                        k * 100
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

                    cor = (
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
                        args.neg_margin
                        +
                        hub
                        -
                        hub_perm
                    )

                    scores.append(
                        float(
                            (
                                hub
                                +
                                args.w_corr
                                * cor
                                +
                                args.w_neg
                                * neg
                            )
                            .item()
                        )
                    )

        return float(
            np.mean(scores)
        )


    best_rel = np.inf
    best_rel_state = None
    best_rel_step = None


    print()
    print(
        "STAGE B — RELATIONAL INTERACTION"
    )

    for step in range(
        1,
        args.rel_steps + 1
    ):

        relation.train()

        opt_rel.zero_grad(
            set_to_none=True
        )

        total = 0.0

        logs = {
            "hub": 0.0,
            "corr": 0.0,
            "rank": 0.0,
            "neg": 0.0,
            "pred": 0.0,
        }

        for task_idx in range(
            args.task_batch
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
                fold_seed
                +
                step * 1000
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

            corr = differentiable_corr_loss(
                delta,
                target_t,
            )

            rank = pairwise_rank_loss(
                delta,
                target_t,
            )

            hub_perm = (
                F.smooth_l1_loss(
                    delta_perm,
                    target_t,
                )
            )

            neg = F.relu(
                args.neg_margin
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
                args.w_pred
                * pred
                +
                args.w_corr
                * corr
                +
                args.w_rank
                * rank
                +
                args.w_neg
                * neg
            )

            total = (
                total
                +
                loss
            )

            logs["hub"] += hub.item()
            logs["corr"] += corr.item()
            logs["rank"] += rank.item()
            logs["neg"] += neg.item()
            logs["pred"] += pred.item()

        total = (
            total
            /
            args.task_batch
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
            step
            % args.eval_every
            == 0
        ):

            val = relation_val()

            print(
                f"{heldout} "
                f"REL {step:05d} "
                f"loss={total.item():.4f} "
                f"hub={logs['hub']/args.task_batch:.4f} "
                f"corr={logs['corr']/args.task_batch:.4f} "
                f"rank={logs['rank']/args.task_batch:.4f} "
                f"neg={logs['neg']/args.task_batch:.4f} "
                f"val={val:.4f}"
            )

            if val < best_rel:

                best_rel = val
                best_rel_step = step

                best_rel_state = copy.deepcopy(
                    relation.state_dict()
                )

    relation.load_state_dict(
        best_rel_state
    )

    relation.eval()

    print(
        "best relation:",
        best_rel_step,
        best_rel
    )


    # ========================================================
    # HELD-OUT PATIENT EVALUATION
    # ========================================================

    heldout_names = sorted(
        data[heldout].keys()
    )

    pair_rows = []


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


    with torch.no_grad():

        for k in TEST_K:

            repeats = (
                1
                if k == 0
                else
                args.test_repeats
            )

            for repeat in range(
                repeats
            ):

                if k == 0:

                    chosen = []

                else:

                    trng = np.random.default_rng(
                        fold_seed
                        +
                        k * 10000
                        +
                        repeat
                    )

                    chosen = (
                        trng.choice(
                            heldout_names,
                            size=k,
                            replace=False,
                        )
                        .tolist()
                    )

                chosen_set = set(
                    chosen
                )

                query_names = [
                    x
                    for x in heldout_names
                    if x not in chosen_set
                ]

                context = [
                    data[heldout][x]
                    for x in chosen
                ]


                for q_idx, qname in enumerate(
                    query_names
                ):

                    q = data[
                        heldout
                    ][
                        qname
                    ]

                    predictions = {
                        "population":
                            q.pop_mu
                    }

                    if k > 0:

                        (
                            c,
                            node_q,
                            node_map,
                        ) = estimate_hierarchy(
                            context,
                            q,
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

                        predictions[
                            "global"
                        ] = mu_global

                        predictions[
                            "node"
                        ] = mu_node

                        # k=1 relation is explicitly zero.
                        delta = relation(
                            context,
                            q,
                            c,
                            node_map,
                        )

                        predictions[
                            "relation"
                        ] = (
                            mu_node
                            +
                            delta
                        )

                        prng = np.random.default_rng(
                            fold_seed
                            +
                            repeat * 1000000
                            +
                            k * 10000
                            +
                            q_idx
                        )

                        delta_perm = relation(
                            context,
                            q,
                            c,
                            node_map,
                            permute_within_channel=True,
                            rng=prng,
                        )

                        predictions[
                            "relation_perm"
                        ] = (
                            mu_node
                            +
                            delta_perm
                        )


                    for condition, mu in (
                        predictions.items()
                    ):

                        pred_raw = raw_prediction(
                            mu
                        )

                        for i, ch in enumerate(
                            q.channels
                        ):

                            pair_rows.append({
                                "heldout":
                                    heldout,

                                "context_k":
                                    k,

                                "repeat":
                                    repeat,

                                "condition":
                                    condition,

                                "query_site":
                                    qname,

                                "recording_channel":
                                    ch,

                                "true_early":
                                    float(
                                        q.y_raw[
                                            i,
                                            0
                                        ]
                                    ),

                                "pred_early":
                                    float(
                                        pred_raw[
                                            i,
                                            0
                                        ]
                                    ),

                                "true_late":
                                    float(
                                        q.y_raw[
                                            i,
                                            1
                                        ]
                                    ),

                                "pred_late":
                                    float(
                                        pred_raw[
                                            i,
                                            1
                                        ]
                                    ),
                            })


    pairs = pd.DataFrame(
        pair_rows
    )

    pairs.to_csv(
        fold_out
        / "test_pair_predictions.csv.gz",
        index=False,
        compression="gzip",
    )


    # ========================================================
    # METRICS
    # ========================================================

    metric_rows = []

    for (
        k,
        repeat,
        condition
    ), g in pairs.groupby(
        [
            "context_k",
            "repeat",
            "condition",
        ]
    ):

        for target in [
            "early",
            "late",
        ]:

            y = g[
                f"true_{target}"
            ].to_numpy()

            p = g[
                f"pred_{target}"
            ].to_numpy()

            metric_rows.append({
                "heldout":
                    heldout,

                "context_k":
                    k,

                "repeat":
                    repeat,

                "condition":
                    condition,

                "target":
                    target,

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

                "top10_recall":
                    top_recall(
                        y,
                        p,
                    ),
            })


    metrics = pd.DataFrame(
        metric_rows
    )

    metrics.to_csv(
        fold_out
        / "test_metrics_raw.csv",
        index=False,
    )


    fold_summary = (
        metrics
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

    fold_summary.to_csv(
        fold_out
        / "fold_summary.csv",
        index=False,
    )


    torch.save(
        {
            "heldout":
                heldout,

            "population_state":
                population.state_dict(),

            "relation_state":
                relation.state_dict(),

            "best_population_step":
                best_pop_step,

            "best_population_val":
                best_pop,

            "best_relation_step":
                best_rel_step,

            "best_relation_val":
                best_rel,

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

            "site_split": {
                s: {
                    "train":
                        sorted(
                            site_split[s][
                                "train"
                            ]
                        ),

                    "val":
                        sorted(
                            site_split[s][
                                "val"
                            ]
                        ),
                }
                for s in train_subjects
            },

            "args":
                vars(args),
        },

        fold_out
        / "pcnd_v3f_final_strict.pt",
    )


    print()
    print(
        fold_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    return fold_summary


# ============================================================
# RUN
# ============================================================

summaries = []

for heldout in HELDOUT_SUBJECTS:

    summaries.append(
        train_fold(
            heldout
        )
    )


# Parallel safe.
if len(
    HELDOUT_SUBJECTS
) == 1:

    print()
    print(
        "SINGLE FOLD COMPLETE:",
        HELDOUT_SUBJECTS[0]
    )

    raise SystemExit(0)


all_summary = pd.concat(
    summaries,
    ignore_index=True,
)

all_summary.to_csv(
    OUT_ROOT
    / "all_available_fold_summaries.csv",
    index=False,
)

print(
    "Saved:",
    OUT_ROOT
)
