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

BRIDGE_ROOT = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_roi_bridge_frozen_v1"
)

DRY_ROOT = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_lockbox_freeze_dryrun_v1"
)

OUT = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_operational_roi_audit_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


BRIDGE_FILE = (
    BRIDGE_ROOT
    / "electrode_roi_bridge.csv"
)

BRIDGE_MANIFEST = (
    BRIDGE_ROOT
    / "ROI_BRIDGE_MANIFEST.json"
)

REC_FILE = (
    DRY_ROOT
    / "recording_channel_inventory.csv"
)

SITE_FILE = (
    DRY_ROOT
    / "strict_usable_stimulation_sites.csv"
)

SUBJECT_FILE = (
    DRY_ROOT
    / "subject_strict_eligibility.csv"
)


NEAR_FIELD_MM = 10.0


# ============================================================
# HELPERS
# ============================================================

def sha256_file(
    path: Path,
):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(block)

    return h.hexdigest()


def normalize_name(x):

    return (
        str(x)
        .strip()
        .replace(
            " ",
            "",
        )
    )


def to_bool_series(s):

    if s.dtype == bool:
        return s

    return (
        s
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


def valid_roi_text(x):

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
            "n/a",
            "na",
            "<na>",
        }
    )


# ============================================================
# VERIFY INPUTS
# ============================================================

for f in [
    BRIDGE_FILE,
    BRIDGE_MANIFEST,
    REC_FILE,
    SITE_FILE,
    SUBJECT_FILE,
]:

    if not f.exists():

        raise FileNotFoundError(f)


bridge_manifest_sha = (
    sha256_file(
        BRIDGE_MANIFEST
    )
)


# ============================================================
# LOAD FROZEN BRIDGE
# ============================================================

bridge = pd.read_csv(
    BRIDGE_FILE
)


bridge[
    "electrode"
] = (
    bridge[
        "electrode"
    ]
    .astype(str)
    .map(
        normalize_name
    )
)


bridge[
    "bridge_matched"
] = to_bool_series(
    bridge[
        "bridge_matched"
    ]
)


bridge[
    "roi_text_present"
] = (
    bridge[
        "roi_text"
    ]
    .map(
        valid_roi_text
    )
)


# ============================================================
# LOAD OPERATIONAL TABLES
# ============================================================

rec = pd.read_csv(
    REC_FILE
)


rec[
    "recording_channel"
] = (
    rec[
        "recording_channel"
    ]
    .astype(str)
    .map(
        normalize_name
    )
)


sites = pd.read_csv(
    SITE_FILE
)


sites[
    "stim_contact_a"
] = (
    sites[
        "stim_contact_a"
    ]
    .astype(str)
    .map(
        normalize_name
    )
)


sites[
    "stim_contact_b"
] = (
    sites[
        "stim_contact_b"
    ]
    .astype(str)
    .map(
        normalize_name
    )
)


sites[
    "site_usable"
] = to_bool_series(
    sites[
        "site_usable"
    ]
)


sites = sites[
    sites[
        "site_usable"
    ]
].copy()


subjects = pd.read_csv(
    SUBJECT_FILE
)


subjects[
    "strict_k10_lockbox_eligible"
] = to_bool_series(
    subjects[
        "strict_k10_lockbox_eligible"
    ]
)


eligible_subjects = set(
    subjects.loc[
        subjects[
            "strict_k10_lockbox_eligible"
        ],
        "subject",
    ]
    .astype(str)
    .tolist()
)


bridge = bridge[
    bridge[
        "subject"
    ].isin(
        eligible_subjects
    )
].copy()


rec = rec[
    rec[
        "subject"
    ].isin(
        eligible_subjects
    )
].copy()


sites = sites[
    sites[
        "subject"
    ].isin(
        eligible_subjects
    )
].copy()


# ============================================================
# LOOKUP
# ============================================================

bridge_lookup = {}


for _, r in (
    bridge.iterrows()
):

    key = (
        str(
            r[
                "subject"
            ]
        ),
        normalize_name(
            r[
                "electrode"
            ]
        ),
    )

    bridge_lookup[
        key
    ] = {
        "matched":
            bool(
                r[
                    "bridge_matched"
                ]
            ),

        "roi_present":
            bool(
                r[
                    "roi_text_present"
                ]
            ),

        "token":
            str(
                r[
                    "frozen_hap_roi_token"
                ]
            ),
    }


# ============================================================
# ALL ELECTRODE DENOMINATOR
# ============================================================

all_total = len(
    bridge
)

all_labeled = int(
    bridge[
        "roi_text_present"
    ].sum()
)

all_matched = int(
    bridge[
        "bridge_matched"
    ].sum()
)


all_coverage = (
    all_matched
    /
    all_total
    if all_total
    else np.nan
)


labeled_coverage = (
    all_matched
    /
    all_labeled
    if all_labeled
    else np.nan
)


# ============================================================
# OPERATIONAL RECORDING CHANNEL COVERAGE
# ============================================================

rec_keys = (
    rec[
        [
            "subject",
            "recording_channel",
        ]
    ]
    .drop_duplicates()
    .copy()
)


rec_rows = []


for _, r in (
    rec_keys.iterrows()
):

    key = (
        str(
            r[
                "subject"
            ]
        ),
        normalize_name(
            r[
                "recording_channel"
            ]
        ),
    )

    info = bridge_lookup.get(
        key,
        None,
    )

    rec_rows.append({
        "subject":
            key[0],

        "electrode":
            key[1],

        "bridge_row_found":
            info is not None,

        "roi_text_present":
            (
                info[
                    "roi_present"
                ]
                if info
                else False
            ),

        "roi_known":
            (
                info[
                    "matched"
                ]
                if info
                else False
            ),

        "token":
            (
                info[
                    "token"
                ]
                if info
                else "UNK"
            ),
    })


rec_cov = pd.DataFrame(
    rec_rows
)


# ============================================================
# OPERATIONAL STIMULATION-CONTACT COVERAGE
# ============================================================

stim_a = sites[
    [
        "subject",
        "stim_contact_a",
    ]
].rename(
    columns={
        "stim_contact_a":
            "electrode"
    }
)


stim_b = sites[
    [
        "subject",
        "stim_contact_b",
    ]
].rename(
    columns={
        "stim_contact_b":
            "electrode"
    }
)


stim_contacts = (
    pd.concat(
        [
            stim_a,
            stim_b,
        ],
        ignore_index=True,
    )
    .drop_duplicates()
)


stim_rows = []


for _, r in (
    stim_contacts.iterrows()
):

    key = (
        str(
            r[
                "subject"
            ]
        ),
        normalize_name(
            r[
                "electrode"
            ]
        ),
    )

    info = bridge_lookup.get(
        key,
        None,
    )

    stim_rows.append({
        "subject":
            key[0],

        "electrode":
            key[1],

        "bridge_row_found":
            info is not None,

        "roi_text_present":
            (
                info[
                    "roi_present"
                ]
                if info
                else False
            ),

        "roi_known":
            (
                info[
                    "matched"
                ]
                if info
                else False
            ),

        "token":
            (
                info[
                    "token"
                ]
                if info
                else "UNK"
            ),
    })


stim_cov = pd.DataFrame(
    stim_rows
)


# ============================================================
# UNION OF ALL MODEL-USED CONTACTS
# ============================================================

operational_union = (
    pd.concat(
        [
            rec_cov[
                [
                    "subject",
                    "electrode",
                ]
            ],

            stim_cov[
                [
                    "subject",
                    "electrode",
                ]
            ],
        ],
        ignore_index=True,
    )
    .drop_duplicates()
)


union_rows = []


for _, r in (
    operational_union.iterrows()
):

    key = (
        str(
            r[
                "subject"
            ]
        ),
        normalize_name(
            r[
                "electrode"
            ]
        ),
    )

    info = bridge_lookup.get(
        key,
        None,
    )

    union_rows.append({
        "subject":
            key[0],

        "electrode":
            key[1],

        "roi_text_present":
            (
                info[
                    "roi_present"
                ]
                if info
                else False
            ),

        "roi_known":
            (
                info[
                    "matched"
                ]
                if info
                else False
            ),

        "token":
            (
                info[
                    "token"
                ]
                if info
                else "UNK"
            ),
    })


union_cov = pd.DataFrame(
    union_rows
)


# ============================================================
# ACTUAL PERTURBATION-PAIR COVERAGE
#
# Reconstruct the frozen 10-mm operational pair universe.
# No signal data involved.
# ============================================================

rec_geometry = (
    rec[
        [
            "subject",
            "recording_channel",
            "fsaverage_x",
            "fsaverage_y",
            "fsaverage_z",
        ]
    ]
    .drop_duplicates(
        subset=[
            "subject",
            "recording_channel",
        ]
    )
)


pair_rows = []


for subject in sorted(
    eligible_subjects
):

    rs = rec_geometry[
        rec_geometry[
            "subject"
        ]
        ==
        subject
    ].copy()


    ss = sites[
        sites[
            "subject"
        ]
        ==
        subject
    ].copy()


    for _, site in (
        ss.iterrows()
    ):

        a = normalize_name(
            site[
                "stim_contact_a"
            ]
        )

        b = normalize_name(
            site[
                "stim_contact_b"
            ]
        )


        a_info = bridge_lookup.get(
            (
                subject,
                a,
            ),
            None,
        )


        b_info = bridge_lookup.get(
            (
                subject,
                b,
            ),
            None,
        )


        a_known = (
            bool(
                a_info[
                    "matched"
                ]
            )
            if a_info
            else False
        )


        b_known = (
            bool(
                b_info[
                    "matched"
                ]
            )
            if b_info
            else False
        )


        stim_mid = np.array(
            [
                site[
                    "stim_mid_fsaverage_x"
                ],
                site[
                    "stim_mid_fsaverage_y"
                ],
                site[
                    "stim_mid_fsaverage_z"
                ],
            ],
            dtype=float,
        )


        for _, rr in (
            rs.iterrows()
        ):

            ch = normalize_name(
                rr[
                    "recording_channel"
                ]
            )


            # Exact stimulation contacts excluded.
            if ch in {
                a,
                b,
            }:

                continue


            rec_xyz = np.array(
                [
                    rr[
                        "fsaverage_x"
                    ],
                    rr[
                        "fsaverage_y"
                    ],
                    rr[
                        "fsaverage_z"
                    ],
                ],
                dtype=float,
            )


            dist = float(
                np.linalg.norm(
                    rec_xyz
                    -
                    stim_mid
                )
            )


            if dist < NEAR_FIELD_MM:

                continue


            r_info = bridge_lookup.get(
                (
                    subject,
                    ch,
                ),
                None,
            )


            rec_known = (
                bool(
                    r_info[
                        "matched"
                    ]
                )
                if r_info
                else False
            )


            pair_rows.append({
                "subject":
                    subject,

                "stim_site":
                    site[
                        "stim_site"
                    ],

                "recording_channel":
                    ch,

                "stim_a_roi_known":
                    a_known,

                "stim_b_roi_known":
                    b_known,

                "recording_roi_known":
                    rec_known,

                "both_stim_roi_known":
                    (
                        a_known
                        and
                        b_known
                    ),

                "all_three_roi_known":
                    (
                        a_known
                        and
                        b_known
                        and
                        rec_known
                    ),
            })


pair_df = pd.DataFrame(
    pair_rows
)


# ============================================================
# SUBJECT-LEVEL OPERATIONAL COVERAGE
# ============================================================

def subject_coverage(
    frame,
    value_col,
    prefix,
):

    return (
        frame
        .groupby(
            "subject",
            as_index=False,
        )
        .agg(
            **{
                f"{prefix}_n":
                    (
                        value_col,
                        "size",
                    ),

                f"{prefix}_known":
                    (
                        value_col,
                        "sum",
                    ),

                f"{prefix}_coverage":
                    (
                        value_col,
                        "mean",
                    ),
            }
        )
    )


s_rec = subject_coverage(
    rec_cov,
    "roi_known",
    "recording",
)


s_stim = subject_coverage(
    stim_cov,
    "roi_known",
    "stim_contact",
)


s_union = subject_coverage(
    union_cov,
    "roi_known",
    "operational_contact",
)


s_pair = (
    pair_df
    .groupby(
        "subject",
        as_index=False,
    )
    .agg(
        n_pairs=(
            "all_three_roi_known",
            "size",
        ),

        recording_roi_known_fraction=(
            "recording_roi_known",
            "mean",
        ),

        both_stim_roi_known_fraction=(
            "both_stim_roi_known",
            "mean",
        ),

        all_three_roi_known_fraction=(
            "all_three_roi_known",
            "mean",
        ),
    )
)


subject_summary = (
    s_rec
    .merge(
        s_stim,
        on="subject",
        how="outer",
    )
    .merge(
        s_union,
        on="subject",
        how="outer",
    )
    .merge(
        s_pair,
        on="subject",
        how="outer",
    )
    .sort_values(
        "subject"
    )
)


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def mean_bool(
    frame,
    col,
):

    if len(frame) == 0:
        return np.nan

    return float(
        frame[
            col
        ].mean()
    )


summary = {
    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "frozen_bridge_manifest_sha256":
        bridge_manifest_sha,

    "n_eligible_subjects":
        len(
            eligible_subjects
        ),

    "all_electrode_rows": {
        "n_total":
            all_total,

        "n_with_roi_text":
            all_labeled,

        "n_known_bridge":
            all_matched,

        "coverage_all_rows":
            all_coverage,

        "coverage_conditional_on_roi_text":
            labeled_coverage,
    },

    "operational_recording_channels": {
        "n":
            int(
                len(
                    rec_cov
                )
            ),

        "known_fraction":
            mean_bool(
                rec_cov,
                "roi_known",
            ),

        "roi_text_present_fraction":
            mean_bool(
                rec_cov,
                "roi_text_present",
            ),
    },

    "operational_stimulation_contacts": {
        "n":
            int(
                len(
                    stim_cov
                )
            ),

        "known_fraction":
            mean_bool(
                stim_cov,
                "roi_known",
            ),

        "roi_text_present_fraction":
            mean_bool(
                stim_cov,
                "roi_text_present",
            ),
    },

    "operational_union_contacts": {
        "n":
            int(
                len(
                    union_cov
                )
            ),

        "known_fraction":
            mean_bool(
                union_cov,
                "roi_known",
            ),

        "roi_text_present_fraction":
            mean_bool(
                union_cov,
                "roi_text_present",
            ),
    },

    "perturbation_pair_features": {
        "n_pairs":
            int(
                len(
                    pair_df
                )
            ),

        "recording_roi_known_fraction":
            mean_bool(
                pair_df,
                "recording_roi_known",
            ),

        "both_stim_roi_known_fraction":
            mean_bool(
                pair_df,
                "both_stim_roi_known",
            ),

        "all_three_roi_known_fraction":
            mean_bool(
                pair_df,
                "all_three_roi_known",
            ),
    },

    "interpretation": (
        "The previously reported ~0.748 outcome-blind "
        "coverage was conditional on electrodes with a "
        "non-missing Destrieux label. "
        "The frozen ~0.477 coverage uses all electrode-table "
        "rows as denominator. This audit reports the "
        "operational model-used denominator without changing "
        "any frozen ROI mapping."
    ),

    "mapping_changed":
        False,

    "signals_read":
        False,

    "response_derivatives_read":
        False,

    "clinical_outcome_labels_used":
        False,

    "model_performance_used":
        False,
}


summary_path = (
    OUT
    / "OPERATIONAL_ROI_COVERAGE_SUMMARY.json"
)


summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# SAVE TABLES
# ============================================================

rec_cov.to_csv(
    OUT
    / "operational_recording_roi_coverage.csv",
    index=False,
)


stim_cov.to_csv(
    OUT
    / "operational_stimulation_roi_coverage.csv",
    index=False,
)


union_cov.to_csv(
    OUT
    / "operational_union_roi_coverage.csv",
    index=False,
)


subject_summary.to_csv(
    OUT
    / "subject_operational_roi_coverage.csv",
    index=False,
)


# Pair-level table can be moderately large;
# save compressed.
pair_df.to_csv(
    OUT
    / "operational_pair_roi_coverage.csv.gz",
    index=False,
    compression="gzip",
)


# ============================================================
# NON-DESTRUCTIVE CLARIFICATION RECORD
# ============================================================

clarification = {
    "record_type":
        "post-freeze coverage clarification",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "original_bridge_manifest_sha256":
        bridge_manifest_sha,

    "original_mapping_modified":
        False,

    "all_electrode_coverage":
        all_coverage,

    "coverage_conditional_on_nonmissing_roi_text":
        labeled_coverage,

    "reason_for_difference":
        (
            "The all-electrode denominator includes "
            "electrode-table rows with missing "
            "Destrieux_label_text, all of which were "
            "correctly frozen to UNK. "
            "The earlier audit conditioned on non-missing "
            "ROI text."
        ),

    "operational_coverage_is_descriptive_only":
        True,

    "no_patient_exclusion_based_on_bridge_coverage":
        True,
}


(
    OUT
    / "ROI_BRIDGE_COVERAGE_CLARIFICATION.json"
).write_text(
    json.dumps(
        clarification,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print(
    "=" * 120
)
print(
    "DS004080 OPERATIONAL ROI COVERAGE AUDIT"
)
print(
    "=" * 120
)


print()
print(
    "ALL ELECTRODE TABLE ROWS"
)

print(
    "total:",
    all_total
)

print(
    "ROI text present:",
    all_labeled
)

print(
    "known bridge:",
    all_matched
)

print(
    "coverage over all rows:",
    round(
        all_coverage,
        4,
    )
)

print(
    "coverage conditional on ROI text:",
    round(
        labeled_coverage,
        4,
    )
)


print()
print(
    "MODEL-USED CONTACTS"
)

print(
    "recording channel known fraction:",
    round(
        mean_bool(
            rec_cov,
            "roi_known",
        ),
        4,
    )
)

print(
    "stimulation contact known fraction:",
    round(
        mean_bool(
            stim_cov,
            "roi_known",
        ),
        4,
    )
)

print(
    "operational union known fraction:",
    round(
        mean_bool(
            union_cov,
            "roi_known",
        ),
        4,
    )
)


print()
print(
    "ACTUAL PERTURBATION PAIRS"
)

print(
    "n pairs:",
    len(
        pair_df
    )
)

print(
    "recording ROI known:",
    round(
        mean_bool(
            pair_df,
            "recording_roi_known",
        ),
        4,
    )
)

print(
    "both stim ROIs known:",
    round(
        mean_bool(
            pair_df,
            "both_stim_roi_known",
        ),
        4,
    )
)

print(
    "ALL THREE ROI TOKENS KNOWN:",
    round(
        mean_bool(
            pair_df,
            "all_three_roi_known",
        ),
        4,
    )
)


print()
print(
    "SUBJECT-LEVEL OPERATIONAL COVERAGE"
)

for col in [
    "recording_coverage",
    "stim_contact_coverage",
    "operational_contact_coverage",
    "all_three_roi_known_fraction",
]:

    values = (
        subject_summary[
            col
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    print(
        col,
        "min/median/max =",
        round(
            float(
                np.min(
                    values
                )
            ),
            4,
        ),
        "/",
        round(
            float(
                np.median(
                    values
                )
            ),
            4,
        ),
        "/",
        round(
            float(
                np.max(
                    values
                )
            ),
            4,
        ),
    )


print()
print(
    "FROZEN BRIDGE MANIFEST SHA256:",
    bridge_manifest_sha
)

print(
    "MAPPING CHANGED: NO"
)

print(
    "PATIENT EXCLUSION BASED ON ROI COVERAGE: NO"
)

print()
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
