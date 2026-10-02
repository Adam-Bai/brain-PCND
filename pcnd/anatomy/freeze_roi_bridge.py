from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime, timezone
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

AUDIT = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_roi_bridge_audit_v1"
)

SRC = (
    AUDIT
    / "electrode_level_metadata_augmented_mapping.csv"
)

CANDIDATES = (
    AUDIT
    / "canonical_candidate_mapping.csv"
)

HAP_FREEZE = (
    ROOT
    / "results"
    / "clinical_v9"
    / "v3f_final_strict"
    / "protocol_freeze"
    / "V3F_FINAL_PROTOCOL_MANIFEST.json"
)

DATA = (
    ROOT
    / "data_external_lockbox"
    / "ds004080_metadata"
)

OUT = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_roi_bridge_frozen_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_HAP_SHA256 = (
    "cb3bd89d6072245ecec7ae608d692f286172fedab517b1b81d821e4e3e710727"
)


BRIDGE_VERSION = (
    "ds004080-to-hapwave-destrieux-bridge-v1.0"
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


def git_cmd(
    cwd,
    *args,
):

    try:

        return subprocess.check_output(
            [
                "git",
                *args,
            ],
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    except Exception:

        return None


def valid_string(x):

    if pd.isna(x):
        return False

    s = str(x).strip()

    return (
        s != ""
        and
        s.lower()
        not in {
            "nan",
            "none",
            "null",
            "na",
            "n/a",
            "unk",
            "unknown",
        }
    )


# ============================================================
# VERIFY SOURCES
# ============================================================

for f in [
    SRC,
    CANDIDATES,
    HAP_FREEZE,
]:

    if not f.exists():

        raise FileNotFoundError(f)


hap_sha = sha256_file(
    HAP_FREEZE
)


if hap_sha != EXPECTED_HAP_SHA256:

    raise RuntimeError(
        "Frozen HAPwave protocol hash changed.\n"
        f"Expected: {EXPECTED_HAP_SHA256}\n"
        f"Actual:   {hap_sha}"
    )


source_sha = sha256_file(
    SRC
)

candidate_sha = sha256_file(
    CANDIDATES
)


# ============================================================
# LOAD OUTCOME-BLIND AUDIT
# ============================================================

df = pd.read_csv(
    SRC
)


required = {
    "subject",
    "electrode",
    "metadata_hemi",
    "roi_text",
    "metadata_augmented_target",
}


missing = (
    required
    -
    set(df.columns)
)


if missing:

    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# SAFETY
#
# Only anatomy strings are permitted here.
# No CCEP response, SOZ, resection, or model output.
# ============================================================

for forbidden in [
    "soz",
    "resect",
    "response",
    "prediction",
    "true_",
    "pred_",
    "early",
    "late",
]:

    bad = [
        c
        for c in df.columns
        if forbidden
        in c.lower()
    ]

    if bad:

        raise RuntimeError(
            "Forbidden column(s) in bridge source: "
            + repr(bad)
        )


# ============================================================
# BUILD ELECTRODE-LEVEL FROZEN TARGET
# ============================================================

df[
    "roi_text"
] = (
    df[
        "roi_text"
    ]
    .astype("string")
)


df[
    "metadata_hemi"
] = (
    df[
        "metadata_hemi"
    ]
    .astype("string")
)


target = []

status = []


for _, r in df.iterrows():

    candidate = (
        r[
            "metadata_augmented_target"
        ]
    )

    if valid_string(
        candidate
    ):

        target.append(
            str(candidate).strip()
        )

        status.append(
            "unique_hemi_stem_match"
        )

    else:

        target.append(
            "UNK"
        )

        status.append(
            "unseen_or_ambiguous_to_UNK"
        )


df[
    "frozen_hap_roi_token"
] = target

df[
    "bridge_status"
] = status


# ============================================================
# CONSISTENCY:
#
# Every (DS ROI text, hemisphere) pair must map to at most
# one HAP token.
# ============================================================

key_cols = [
    "roi_text",
    "metadata_hemi",
]


consistency = (
    df
    .groupby(
        key_cols,
        dropna=False,
    )[
        "frozen_hap_roi_token"
    ]
    .nunique(
        dropna=False
    )
)


bad = consistency[
    consistency > 1
]


if len(bad):

    raise RuntimeError(
        "Non-deterministic ROI bridge detected:\n"
        + bad.to_string()
    )


# ============================================================
# CROSSWALK
# ============================================================

crosswalk = (
    df
    .groupby(
        key_cols
        +
        [
            "frozen_hap_roi_token",
            "bridge_status",
        ],
        dropna=False,
        as_index=False,
    )
    .agg(
        n_electrodes=(
            "electrode",
            "size",
        ),

        n_subjects=(
            "subject",
            "nunique",
        ),
    )
    .sort_values(
        [
            "n_electrodes",
            "roi_text",
        ],
        ascending=[
            False,
            True,
        ],
    )
)


# ============================================================
# ELECTRODE COVERAGE
# ============================================================

df[
    "bridge_matched"
] = (
    df[
        "frozen_hap_roi_token"
    ]
    !=
    "UNK"
)


overall_coverage = float(
    df[
        "bridge_matched"
    ].mean()
)


subject = (
    df
    .groupby(
        "subject",
        as_index=False,
    )
    .agg(
        n_electrodes=(
            "electrode",
            "nunique",
        ),

        n_roi_rows=(
            "electrode",
            "size",
        ),

        n_bridge_matched=(
            "bridge_matched",
            "sum",
        ),

        bridge_coverage=(
            "bridge_matched",
            "mean",
        ),
    )
    .sort_values(
        "subject"
    )
)


# ============================================================
# OUTPUT
# ============================================================

electrode_out = df[
    [
        "subject",
        "electrode",
        "metadata_hemi",
        "roi_text",
        "frozen_hap_roi_token",
        "bridge_status",
        "bridge_matched",
    ]
].copy()


electrode_path = (
    OUT
    / "electrode_roi_bridge.csv"
)

crosswalk_path = (
    OUT
    / "roi_crosswalk.csv"
)

subject_path = (
    OUT
    / "subject_roi_bridge_coverage.csv"
)


electrode_out.to_csv(
    electrode_path,
    index=False,
)

crosswalk.to_csv(
    crosswalk_path,
    index=False,
)

subject.to_csv(
    subject_path,
    index=False,
)


# ============================================================
# PROVENANCE
# ============================================================

dataset_commit = git_cmd(
    DATA,
    "rev-parse",
    "HEAD",
)

pcnd_commit = git_cmd(
    ROOT,
    "rev-parse",
    "HEAD",
)


manifest = {
    "bridge_version":
        BRIDGE_VERSION,

    "freeze_time_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "development_protocol_sha256":
        hap_sha,

    "source_outcome_blind_audit":
        str(
            SRC
        ),

    "source_audit_sha256":
        source_sha,

    "candidate_mapping_sha256":
        candidate_sha,

    "dataset_git_commit":
        dataset_commit,

    "pcnd_git_commit":
        pcnd_commit,

    "rule": (
        "For each ds004080 Destrieux_label_text and "
        "metadata hemisphere pair, map to the unique "
        "HAPwave ROI token with the same canonical "
        "Destrieux stem and hemisphere. "
        "If no unique HAPwave token exists, map to UNK."
    ),

    "prohibited_mapping_rules": [
        "no fuzzy string matching",
        "no contralateral fallback",
        "no semantic nearest-neighbor mapping",
        "no outcome-dependent remapping",
        "no patient exclusion based on ROI bridge coverage",
    ],

    "overall_electrode_weighted_coverage":
        overall_coverage,

    "n_electrode_rows":
        int(
            len(df)
        ),

    "n_matched":
        int(
            df[
                "bridge_matched"
            ].sum()
        ),

    "n_UNK":
        int(
            (
                ~df[
                    "bridge_matched"
                ]
            ).sum()
        ),

    "coverage_by_subject": {
        "min":
            float(
                subject[
                    "bridge_coverage"
                ].min()
            ),

        "median":
            float(
                subject[
                    "bridge_coverage"
                ].median()
            ),

        "max":
            float(
                subject[
                    "bridge_coverage"
                ].max()
            ),
    },

    "blindness": {
        "signals_read":
            False,

        "response_derivatives_read":
            False,

        "clinical_outcome_labels_used":
            False,

        "model_performance_used":
            False,
    },

    "files": {
        "electrode_roi_bridge.csv":
            sha256_file(
                electrode_path
            ),

        "roi_crosswalk.csv":
            sha256_file(
                crosswalk_path
            ),

        "subject_roi_bridge_coverage.csv":
            sha256_file(
                subject_path
            ),
    },
}


manifest_path = (
    OUT
    / "ROI_BRIDGE_MANIFEST.json"
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
    "bridge_version":
        BRIDGE_VERSION,

    "manifest_sha256":
        manifest_sha,

    "development_protocol_sha256":
        hap_sha,

    "dataset_git_commit":
        dataset_commit,

    "pcnd_git_commit":
        pcnd_commit,
}


(
    OUT
    / "ROI_BRIDGE_FREEZE_SEAL.json"
).write_text(
    json.dumps(
        seal,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print(
    "=" * 110
)
print(
    "DS004080 ROI BRIDGE FROZEN"
)
print(
    "=" * 110
)

print(
    "overall electrode-weighted coverage:",
    round(
        overall_coverage,
        4,
    )
)

print(
    "matched / total:",
    int(
        df[
            "bridge_matched"
        ].sum()
    ),
    "/",
    len(df),
)

print(
    "subject coverage min / median / max:",
    round(
        float(
            subject[
                "bridge_coverage"
            ].min()
        ),
        4,
    ),
    "/",
    round(
        float(
            subject[
                "bridge_coverage"
            ].median()
        ),
        4,
    ),
    "/",
    round(
        float(
            subject[
                "bridge_coverage"
            ].max()
        ),
        4,
    ),
)

print()
print(
    "TOP FROZEN CROSSWALK"
)

print(
    crosswalk
    .head(30)
    .to_string(
        index=False
    )
)

print()
print(
    "LOWEST-COVERAGE SUBJECTS"
)

print(
    subject
    .sort_values(
        "bridge_coverage"
    )
    .head(15)
    .round(4)
    .to_string(
        index=False
    )
)

print()
print(
    "Manifest SHA256:",
    manifest_sha
)

print(
    "NO SIGNALS READ."
)

print(
    "NO RESPONSE DERIVATIVES READ."
)

print(
    "NO CLINICAL OUTCOME LABELS USED."
)

print(
    "NO MODEL PERFORMANCE USED."
)

print()
print(
    "Saved:",
    OUT
)
