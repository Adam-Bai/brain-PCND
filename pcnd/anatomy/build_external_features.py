from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

LOCK = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_final_lockbox_v1"
)

DRY = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_lockbox_freeze_dryrun_v1"
)

PHENO = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_primary_phenotypes_v1"
)

OUT = (
    ROOT
    / "results/external_lockbox/"
    "ds004080_v3f_external_features_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE = (
    OUT
    / "DS004080_V3F_EXTERNAL_FEATURES.csv.gz"
)

MANIFEST = (
    OUT
    / "EXTERNAL_V3F_FEATURE_MANIFEST.json"
)


EXPECTED_MAP_SHA = (
    "7956a52b168c64266c656ffdfd6576dd"
    "23e6a1cc5857297b677e33609b87094e"
)

EXPECTED_LOCKBOX_SHA = (
    "8170555317e6a2b510cb7ef1bbb603ba"
    "6a4220ec92a0b50cc32c29392f66cf6d"
)


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


def sha256_file(p):

    h = hashlib.sha256()

    with open(p, "rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def cf(x):

    return str(x).strip().casefold()


def require_unique_column(
    df,
    candidates,
    label,
):

    hits = [
        c
        for c in candidates
        if c in df.columns
    ]

    if len(hits) != 1:

        raise RuntimeError(
            f"{label}: expected exactly one "
            f"of {candidates}, got {hits}"
        )

    return hits[0]


if FEATURE.exists() or MANIFEST.exists():
    raise RuntimeError(
        "External V3f feature artifacts already exist; "
        "refusing overwrite."
    )


map_path = (
    PHENO
    / "ds004080_primary_perturbation_map.csv.gz"
)

lockbox_manifest = (
    LOCK
    / "DS004080_FINAL_LOCKBOX_MANIFEST.json"
)

assert sha256_file(map_path) == EXPECTED_MAP_SHA
assert sha256_file(lockbox_manifest) == EXPECTED_LOCKBOX_SHA


mp = pd.read_csv(map_path)

sites = pd.read_csv(
    LOCK
    / "strict_usable_stimulation_sites.csv"
)

rec = pd.read_csv(
    LOCK
    / "recording_channel_inventory.csv"
)

bridge = pd.read_csv(
    LOCK
    / "electrode_roi_bridge.csv"
)

adapter_path = (
    DRY
    / "electrodes_anatomical_adapter.csv"
)

if not adapter_path.exists():
    raise FileNotFoundError(adapter_path)

adapter = pd.read_csv(adapter_path)


# ------------------------------------------------------------
# Adapter electrode schema.
# Fail rather than guess ambiguously.
# ------------------------------------------------------------

acol = require_unique_column(
    adapter,
    [
        "electrode",
        "name",
        "contact",
    ],
    "adapter electrode name",
)

for c in [
    "subject",
    "adapter_mni_x",
    "adapter_mni_y",
    "adapter_mni_z",
]:
    if c not in adapter.columns:
        raise RuntimeError(
            f"adapter table missing {c}; "
            f"columns={adapter.columns.tolist()}"
        )


adapter = adapter.copy()

adapter["_key"] = list(
    zip(
        adapter["subject"].astype(str),
        adapter[acol].map(cf),
    )
)

if adapter["_key"].duplicated().any():
    dup = adapter[
        adapter["_key"].duplicated(
            keep=False
        )
    ]

    raise RuntimeError(
        "Duplicate adapter electrode keys:\n"
        +
        dup.head(30).to_string(index=False)
    )


adapter_lookup = {
    k:
        np.asarray(
            [
                r["adapter_mni_x"],
                r["adapter_mni_y"],
                r["adapter_mni_z"],
            ],
            dtype=float,
        )

    for k, (_, r)
    in zip(
        adapter["_key"],
        adapter.iterrows(),
    )
}


# ------------------------------------------------------------
# Frozen ROI bridge
# ------------------------------------------------------------

bridge = bridge.copy()

bridge["_key"] = list(
    zip(
        bridge["subject"].astype(str),
        bridge["electrode"].map(cf),
    )
)

if bridge["_key"].duplicated().any():

    raise RuntimeError(
        "Duplicate frozen ROI bridge keys."
    )


roi_lookup = {
    k:
        (
            "UNK"
            if pd.isna(r["frozen_hap_roi_token"])
            else str(
                r["frozen_hap_roi_token"]
            )
        )

    for k, (_, r)
    in zip(
        bridge["_key"],
        bridge.iterrows(),
    )
}


# ------------------------------------------------------------
# Build stimulation-site features
# ------------------------------------------------------------

site_rows = []

mid_errors = []
dist_errors = []


for _, r in sites.iterrows():

    sub = str(r["subject"])
    site = str(r["stim_site"])

    a = str(r["stim_contact_a"])
    b = str(r["stim_contact_b"])

    ka = (
        sub,
        cf(a),
    )

    kb = (
        sub,
        cf(b),
    )

    if ka not in adapter_lookup:
        raise RuntimeError(
            f"Missing adapter coordinate: "
            f"{sub} {a}"
        )

    if kb not in adapter_lookup:
        raise RuntimeError(
            f"Missing adapter coordinate: "
            f"{sub} {b}"
        )

    pa = adapter_lookup[ka]
    pb = adapter_lookup[kb]

    if (
        not np.isfinite(pa).all()
        or
        not np.isfinite(pb).all()
    ):
        raise RuntimeError(
            f"Nonfinite stimulation contact MNI: "
            f"{sub} {site}"
        )

    midpoint = (
        pa + pb
    ) / 2.0

    frozen_mid = np.asarray(
        [
            r["stim_mid_adapter_mni_x"],
            r["stim_mid_adapter_mni_y"],
            r["stim_mid_adapter_mni_z"],
        ],
        dtype=float,
    )

    mid_error = float(
        np.linalg.norm(
            midpoint
            -
            frozen_mid
        )
    )

    mid_errors.append(
        mid_error
    )

    vec = (
        pb - pa
    )

    norm = float(
        np.linalg.norm(
            vec
        )
    )

    if (
        not np.isfinite(norm)
        or norm <= 0
    ):
        raise RuntimeError(
            f"Invalid bipolar orientation: "
            f"{sub} {site}"
        )

    frozen_dist = float(
        r[
            "stim_pair_distance_adapter_mni"
        ]
    )

    dist_errors.append(
        abs(
            norm
            -
            frozen_dist
        )
    )

    o = (
        vec / norm
    )

    ox, oy, oz = o

    site_rows.append({
        "subject":
            sub,

        "stim_site":
            site,

        "stim_a":
            a,

        "stim_b":
            b,

        "mni_mid_x":
            float(
                frozen_mid[0]
            ),

        "mni_mid_y":
            float(
                frozen_mid[1]
            ),

        "mni_mid_z":
            float(
                frozen_mid[2]
            ),

        "mni_axis_xx":
            ox * ox,

        "mni_axis_yy":
            oy * oy,

        "mni_axis_zz":
            oz * oz,

        "mni_axis_xy":
            ox * oy,

        "mni_axis_xz":
            ox * oz,

        "mni_axis_yz":
            oy * oz,

        "stim_a_roi":
            roi_lookup.get(
                ka,
                "UNK",
            ),

        "stim_b_roi":
            roi_lookup.get(
                kb,
                "UNK",
            ),
    })


max_mid_error = float(
    np.max(
        mid_errors
    )
)

max_dist_error = float(
    np.max(
        dist_errors
    )
)


print(
    "max midpoint reconstruction error:",
    max_mid_error,
)

print(
    "max bipolar-distance reconstruction error:",
    max_dist_error,
)


if max_mid_error > 1e-5:

    raise RuntimeError(
        "Adapter midpoint does not reproduce "
        "frozen stimulation midpoint."
    )


if max_dist_error > 1e-5:

    raise RuntimeError(
        "Adapter bipolar distance does not reproduce "
        "frozen site geometry."
    )


site_df = pd.DataFrame(
    site_rows
)


# ------------------------------------------------------------
# Recording metadata
# ------------------------------------------------------------

rec_rows = []


for _, r in rec.iterrows():

    sub = str(
        r[
            "subject"
        ]
    )

    ch = str(
        r[
            "recording_channel"
        ]
    )

    key = (
        sub,
        cf(ch),
    )

    rec_rows.append({
        "subject":
            sub,

        "recording_channel":
            ch,

        "rec_mni_x":
            float(
                r[
                    "adapter_mni_x"
                ]
            ),

        "rec_mni_y":
            float(
                r[
                    "adapter_mni_y"
                ]
            ),

        "rec_mni_z":
            float(
                r[
                    "adapter_mni_z"
                ]
            ),

        "recording_roi":
            roi_lookup.get(
                key,
                "UNK",
            ),
    })


rec_df = pd.DataFrame(
    rec_rows
)


# ------------------------------------------------------------
# Join phenotype + exact frozen anatomy
# ------------------------------------------------------------

df = (
    mp
    .merge(
        site_df,
        on=[
            "subject",
            "stim_site",
        ],
        how="left",
        validate="many_to_one",
    )
    .merge(
        rec_df,
        on=[
            "subject",
            "recording_channel",
        ],
        how="left",
        validate="many_to_one",
    )
)


# EXACT development convention:
# recording coordinate MINUS stimulation midpoint.

df["mni_dx"] = (
    df["rec_mni_x"]
    -
    df["mni_mid_x"]
)

df["mni_dy"] = (
    df["rec_mni_y"]
    -
    df["mni_mid_y"]
)

df["mni_dz"] = (
    df["rec_mni_z"]
    -
    df["mni_mid_z"]
)


mni_distance = np.sqrt(
    df["mni_dx"] ** 2
    +
    df["mni_dy"] ** 2
    +
    df["mni_dz"] ** 2
)


df["log_mni_distance"] = np.log1p(
    np.clip(
        mni_distance,
        0,
        None,
    )
)


needed = (
    [
        "early_rms_z",
        "late_rms_z",
    ]
    +
    PAIR_FEATURES
)


for c in needed:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )


finite = np.isfinite(
    df[
        needed
    ].to_numpy(
        dtype=float
    )
).all(
    axis=1
)


if not finite.all():

    bad = df[
        ~finite
    ]

    raise RuntimeError(
        f"{len(bad)} external V3f rows have "
        f"nonfinite primary/model features."
    )


if len(df) != 290758:

    raise RuntimeError(
        f"Expected 290758 map rows, "
        f"got {len(df)}"
    )


if (
    df[
        [
            "subject",
            "stim_site",
        ]
    ]
    .drop_duplicates()
    .shape[0]
    != 4135
):

    raise RuntimeError(
        "External V3f features lost frozen sites."
    )


columns = (
    [
        "subject",
        "stim_site",
        "recording_channel",
        "n_trials",
        "n_runs",
        "distance_to_stim",
        "early_rms_z",
        "late_rms_z",
    ]
    +
    PAIR_FEATURES
    +
    [
        "stim_a_roi",
        "stim_b_roi",
        "recording_roi",
    ]
)


df = (
    df[
        columns
    ]
    .sort_values(
        [
            "subject",
            "stim_site",
            "recording_channel",
        ]
    )
    .reset_index(
        drop=True
    )
)


df.to_csv(
    FEATURE,
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


feature_sha = sha256_file(
    FEATURE
)


roi_cols = [
    "stim_a_roi",
    "stim_b_roi",
    "recording_roi",
]


manifest = {
    "feature_protocol_id":
        "DS004080-V3f-external-features-v1.0",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "phenotype_map_sha256":
        EXPECTED_MAP_SHA,

    "external_lockbox_sha256":
        EXPECTED_LOCKBOX_SHA,

    "electrode_adapter_source":
        str(
            adapter_path
        ),

    "electrode_adapter_sha256":
        sha256_file(
            adapter_path
        ),

    "pair_features":
        PAIR_FEATURES,

    "n_rows":
        int(
            len(df)
        ),

    "n_subjects":
        int(
            df[
                "subject"
            ].nunique()
        ),

    "n_sites":
        int(
            df[
                [
                    "subject",
                    "stim_site",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),

    "max_midpoint_reconstruction_error":
        max_mid_error,

    "max_bipolar_distance_reconstruction_error":
        max_dist_error,

    "roi_known_fraction": {
        c:
            float(
                (
                    df[c]
                    !=
                    "UNK"
                ).mean()
            )

        for c in roi_cols
    },

    "feature_table_sha256":
        feature_sha,

    "model_checkpoint_loaded":
        False,

    "external_model_performance_observed":
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
print("DS004080 FROZEN V3f FEATURE TABLE COMPLETE")
print("=" * 100)

print("rows:", len(df))
print("subjects:", df["subject"].nunique())

print(
    "sites:",
    df[
        [
            "subject",
            "stim_site",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)

print(
    "ROI known fractions:",
    manifest[
        "roi_known_fraction"
    ],
)

print(
    "FEATURE TABLE SHA256:",
    feature_sha,
)

print()
print("MODEL CHECKPOINT LOADED: NO")
print("EXTERNAL MODEL PERFORMANCE OBSERVED: NO")
print("STATUS: READY FOR FROZEN V3f INFERENCE")
