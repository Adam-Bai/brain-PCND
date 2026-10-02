from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

import pandas as pd


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

DRY = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_lockbox_freeze_dryrun_v1"
)

BRIDGE = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_roi_bridge_frozen_v1"
)

OPROI = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_operational_roi_audit_v1"
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
    / "ds004080_final_lockbox_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_HAP_SHA = (
    "cb3bd89d6072245ecec7ae608d692f286172fedab517b1b81d821e4e3e710727"
)

EXPECTED_ROI_BRIDGE_SHA = (
    "fb9e3a493bf05e744c65fae165fc1fe3d7734b1a6456685c4d470be9c9efbbca"
)


FREEZE_VERSION = (
    "ds004080-external-lockbox-v1.0"
)


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


def git_cmd(cwd, *args):

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


def copy_frozen(
    source: Path,
    destination: Path,
):

    if not source.exists():
        raise FileNotFoundError(source)

    shutil.copy2(
        source,
        destination,
    )

    if (
        sha256_file(source)
        !=
        sha256_file(destination)
    ):
        raise RuntimeError(
            f"Copy hash mismatch: {source}"
        )


# ============================================================
# VERIFY UPSTREAM SEALS
# ============================================================

if (
    sha256_file(HAP_FREEZE)
    !=
    EXPECTED_HAP_SHA
):

    raise RuntimeError(
        "HAPwave development freeze changed."
    )


bridge_manifest = (
    BRIDGE
    / "ROI_BRIDGE_MANIFEST.json"
)


if (
    sha256_file(bridge_manifest)
    !=
    EXPECTED_ROI_BRIDGE_SHA
):

    raise RuntimeError(
        "Frozen ROI bridge changed."
    )


pcnd_commit = git_cmd(
    ROOT,
    "rev-parse",
    "HEAD",
)

# Lockbox integrity depends on committed/tracked code
# being unchanged.
#
# Untracked historical scripts or nested external-data
# repositories do not modify the frozen implementation
# and therefore do not invalidate the seal.
pcnd_status = git_cmd(
    ROOT,
    "status",
    "--porcelain",
    "--untracked-files=no",
)

dataset_commit = git_cmd(
    DATA,
    "rev-parse",
    "HEAD",
)


if pcnd_commit is None:

    raise RuntimeError(
        "PCND repository has no Git commit."
    )


if pcnd_status != "":

    raise RuntimeError(
        "Refusing final lockbox seal: "
        "PCND Git worktree is dirty.\n\n"
        + str(pcnd_status)
    )


# ============================================================
# REQUIRED OUTCOME-BLIND ARTIFACTS
# ============================================================

source_files = {
    "lockbox_cohort.csv":
        DRY
        / "lockbox_cohort.csv",

    "subject_strict_eligibility.csv":
        DRY
        / "subject_strict_eligibility.csv",

    "strict_usable_stimulation_sites.csv":
        DRY
        / "strict_usable_stimulation_sites.csv",

    "recording_channel_inventory.csv":
        DRY
        / "recording_channel_inventory.csv",

    "frozen_context_query_plan.csv":
        DRY
        / "frozen_context_query_plan.csv",

    "ANATOMICAL_ADAPTER.json":
        DRY
        / "ANATOMICAL_ADAPTER.json",

    "ROI_BRIDGE_MANIFEST.json":
        BRIDGE
        / "ROI_BRIDGE_MANIFEST.json",

    "electrode_roi_bridge.csv":
        BRIDGE
        / "electrode_roi_bridge.csv",

    "roi_crosswalk.csv":
        BRIDGE
        / "roi_crosswalk.csv",

    "OPERATIONAL_ROI_COVERAGE_SUMMARY.json":
        OPROI
        / "OPERATIONAL_ROI_COVERAGE_SUMMARY.json",

    "ROI_BRIDGE_COVERAGE_CLARIFICATION.json":
        OPROI
        / "ROI_BRIDGE_COVERAGE_CLARIFICATION.json",
}


for name, source in (
    source_files.items()
):

    copy_frozen(
        source,
        OUT / name,
    )


# ============================================================
# VERIFY COHORT
# ============================================================

cohort = pd.read_csv(
    OUT
    / "lockbox_cohort.csv"
)


if (
    cohort[
        "subject"
    ].nunique()
    !=
    74
):

    raise RuntimeError(
        "Expected all 74 subjects "
        "in frozen external cohort."
    )


if len(cohort) != 74:

    raise RuntimeError(
        "Duplicate or missing cohort rows."
    )


contexts = pd.read_csv(
    OUT
    / "frozen_context_query_plan.csv"
)


expected_rows = (
    74
    *
    4
    *
    20
)


if len(contexts) != expected_rows:

    raise RuntimeError(
        f"Expected {expected_rows} "
        f"context rows, found {len(contexts)}"
    )


for k in [
    1,
    3,
    5,
    10,
]:

    q = contexts[
        contexts[
            "context_k"
        ]
        ==
        k
    ]

    expected = (
        74
        *
        20
    )

    if len(q) != expected:

        raise RuntimeError(
            f"k={k}: expected "
            f"{expected}, got {len(q)}"
        )


# ============================================================
# HASH EVERYTHING BEFORE MANIFEST
# ============================================================

artifact_hashes = {
    name:
        sha256_file(
            OUT / name
        )

    for name in source_files
}


# ============================================================
# FINAL MANIFEST
# ============================================================

manifest = {
    "freeze_version":
        FREEZE_VERSION,

    "freeze_time_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "target_journal":
        "Brain Stimulation",

    "external_dataset":
        "OpenNeuro ds004080 / RESPect",

    "dataset_metadata_git_commit":
        dataset_commit,

    "pcnd_code_git_commit":
        pcnd_commit,

    "development_protocol": {
        "cohort":
            "HAPwave / ds004696",

        "status":
            "development closed",

        "manifest_sha256":
            EXPECTED_HAP_SHA,
    },

    "external_cohort": {
        "n_subjects":
            74,

        "selection_rule":
            (
                "All subjects satisfying the "
                "pre-outcome metadata, stimulation-site, "
                "recording-channel, coordinate, and "
                "dominant-protocol eligibility rules."
            ),

        "performance_based_exclusion":
            False,

        "roi_coverage_based_exclusion":
            False,
    },

    "stimulation_site_rule": {
        "dominant_protocol_per_subject":
            True,

        "minimum_repeats_per_site":
            5,

        "minimum_usable_sites":
            11,

        "primary_near_field_exclusion_mm":
            10.0,

        "exact_stimulation_contacts_excluded":
            True,

        "minimum_recording_channels_per_site":
            20,
    },

    "anatomical_interface": {
        "source_space":
            "fsaverage / MNI305-like",

        "source_units":
            "mm",

        "fixed_affine":
            (
                "FieldTrip fsaverage/MNI305 "
                "to MNI152/SPM affine"
            ),

        "important_limitation":
            (
                "This is an a-priori approximate "
                "template bridge and is not claimed "
                "to be an exact nonlinear transform "
                "to MNI152NLin6Sym."
            ),

        "roi_bridge_manifest_sha256":
            EXPECTED_ROI_BRIDGE_SHA,

        "unmatched_roi_rule":
            "UNK",

        "fuzzy_mapping":
            False,

        "contralateral_fallback":
            False,

        "outcome_dependent_mapping":
            False,
    },

    "operational_roi_domain_shift": {
        "recording_channel_known_fraction":
            0.7465,

        "stimulation_contact_known_fraction":
            0.7462,

        "operational_union_known_fraction":
            0.7448,

        "all_three_pair_roi_known_fraction":
            0.5128,

        "patient_all_three_fraction_min":
            0.0898,

        "patient_all_three_fraction_median":
            0.5490,

        "patient_all_three_fraction_max":
            0.8793,

        "interpretation":
            (
                "ROI coverage is a frozen descriptor "
                "of external anatomical domain shift "
                "and is not an exclusion criterion."
            ),
    },

    "context_plan": {
        "context_k":
            [
                1,
                3,
                5,
                10,
            ],

        "n_repeats_per_k":
            20,

        "actual_context_lists_frozen":
            True,

        "actual_query_lists_frozen":
            True,

        "selection_based_on_response":
            False,
    },

    "primary_endpoint": {
        "response":
            "late",

        "context_k":
            10,

        "metric":
            "patient-level Spearman map correlation",

        "contrast":
            (
                "relation minus "
                "within-recording-channel "
                "stimulation permutation"
            ),

        "inference_unit":
            "patient",

        "repeat_aggregation":
            (
                "20 context repeats averaged "
                "within patient before inference"
            ),

        "test":
            (
                "two-sided exact patient-level "
                "sign-flip/randomization test"
            ),

        "alpha":
            0.05,

        "confidence_interval":
            (
                "95% nonparametric patient bootstrap "
                "CI for mean paired difference"
            ),
    },

    "lockbox_blindness_at_freeze": {
        "raw_signals_opened":
            False,

        "response_features_computed":
            False,

        "response_derivatives_read":
            False,

        "clinical_outcome_labels_used":
            False,

        "external_model_performance_observed":
            False,
    },

    "post_opening_prohibitions": [
        "no patient exclusion based on model performance",
        "no patient exclusion based on ROI coverage",
        "no architecture changes",
        "no ROI remapping",
        "no anatomical-adapter tuning",
        "no context reselection",
        "no primary-endpoint changes",
        "no metric changes",
        "no loss-weight changes",
        "no preferred-seed selection",
    ],

    "frozen_artifacts":
        artifact_hashes,
}


manifest_path = (
    OUT
    / "DS004080_FINAL_LOCKBOX_MANIFEST.json"
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
    "freeze_version":
        FREEZE_VERSION,

    "manifest_sha256":
        manifest_sha,

    "development_protocol_sha256":
        EXPECTED_HAP_SHA,

    "roi_bridge_manifest_sha256":
        EXPECTED_ROI_BRIDGE_SHA,

    "dataset_metadata_git_commit":
        dataset_commit,

    "pcnd_code_git_commit":
        pcnd_commit,

    "n_external_subjects":
        74,

    "raw_signals_opened_at_freeze":
        False,
}


(
    OUT
    / "FREEZE_SEAL.json"
).write_text(
    json.dumps(
        seal,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print(
    "=" * 120
)
print(
    "DS004080 FINAL EXTERNAL LOCKBOX FROZEN"
)
print(
    "=" * 120
)

print(
    "subjects:",
    74
)

print(
    "context-plan rows:",
    len(
        contexts
    )
)

print(
    "PCND Git commit:",
    pcnd_commit
)

print(
    "dataset metadata commit:",
    dataset_commit
)

print(
    "development SHA256:",
    EXPECTED_HAP_SHA
)

print(
    "ROI bridge SHA256:",
    EXPECTED_ROI_BRIDGE_SHA
)

print(
    "FINAL LOCKBOX MANIFEST SHA256:",
    manifest_sha
)

print()
print(
    "RAW SIGNALS OPENED: NO"
)

print(
    "EXTERNAL MODEL PERFORMANCE OBSERVED: NO"
)

print()
print(
    "Next required step:"
)

print(
    "freeze one all-8 HAPwave "
    "deployment checkpoint before raw lockbox opening."
)
