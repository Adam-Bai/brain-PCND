from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime, timezone

import hashlib
import json
import platform
import subprocess

import pandas as pd


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

DATASET = (
    ROOT
    / "data_external_lockbox"
    / "ds004080_metadata"
)

LOCKBOX = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_final_lockbox_v1"
)

DEPLOY = (
    ROOT
    / "results"
    / "clinical_v10"
    / "v3f_all8_deployment"
)

RESPONSE = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_response_protocol_v1"
)

ACQ = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_raw_acquisition_v1"
)

OUT = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_formal_open_v1"
)

OPEN_RECORD = (
    OUT
    / "DS004080_LOCKBOX_OPEN_RECORD.json"
)

OPEN_SEAL = (
    OUT
    / "DS004080_LOCKBOX_OPEN_SEAL.json"
)


EXPECTED = {

    LOCKBOX
    / "DS004080_FINAL_LOCKBOX_MANIFEST.json":
        "8170555317e6a2b510cb7ef1bbb603ba"
        "6a4220ec92a0b50cc32c29392f66cf6d",

    DEPLOY
    / "pcnd_v3f_all8_deployment.pt":
        "399421567b0185383bf8953cdcd6c758"
        "91893b5113fca8a3a41ac190eb68a794",

    DEPLOY
    / "ALL8_DEPLOYMENT_MANIFEST.json":
        "dd69bb8963a678a9459987ebc5e37115"
        "49b42679ed27b1ed57bbbf0628b593b0",

    RESPONSE
    / "DS004080_RESPONSE_PROTOCOL.json":
        "df944bdf79b50f8609320cad46e15c3d"
        "f4bcf1d207335b83a6d18d85e883a423",

    RESPONSE
    / "RESPONSE_PROTOCOL_FREEZE_SEAL.json":
        "dd147ce605faf330d5bfcdfbd2ccc92a"
        "da80e2dc3bd60133ce27803f30347061",

    ACQ
    / "FROZEN_EVENT_ALLOWLIST.csv":
        "6f6be3335da9a45fcd946d3fd6832bab"
        "0c8099e0443aadec1cf69eb64c5e39b8",

    ACQ
    / "FROZEN_SITE_EVENT_COUNT_AUDIT.csv":
        "b253d463d95d58f70a95be0bd5705f2b"
        "7f51452665b6e8014419b18ff719b3ea",

    ACQ
    / "RAW_FILE_ALLOWLIST.csv":
        "681b896a202edbef69a46d991fb65dea9"
        "57c7382ce73ddc9c3120026e9cfa29c",

    ACQ
    / "RAW_ACQUISITION_MANIFEST.json":
        "11d09494db152e973becf682ca473ee7d"
        "dc22209c243c6b4e46e893d440f98df",

    ACQ
    / "RAW_ACQUISITION_FREEZE_SEAL.json":
        "141dea4521a2618399037b4c014072d3"
        "fb2ce13fd2d66566b1f5bfb17db5d1a2",
}


EXPECTED_DATASET_COMMIT = (
    "c4fd7418883e33b024292468eb14da1649f51aae"
)


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def git(repo: Path, *args: str) -> str:

    return subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            *args,
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


# ============================================================
# REFUSE DUPLICATE OPEN
# ============================================================

if OPEN_RECORD.exists() or OPEN_SEAL.exists():

    raise RuntimeError(
        "Formal lockbox OPEN record already exists. "
        "Refusing to create a second opening event."
    )


# ============================================================
# VERIFY ALL PRE-OPENING ARTIFACTS BYTE-FOR-BYTE
# ============================================================

for path, expected in EXPECTED.items():

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    actual = sha256_file(
        path
    )

    if actual != expected:

        raise RuntimeError(
            f"Pre-opening artifact changed:\n"
            f"{path}\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )


# ============================================================
# VERIFY DATASET METADATA VERSION
# ============================================================

dataset_commit = git(
    DATASET,
    "rev-parse",
    "HEAD",
)

if dataset_commit != EXPECTED_DATASET_COMMIT:

    raise RuntimeError(
        "Dataset metadata commit mismatch.\n"
        f"expected: {EXPECTED_DATASET_COMMIT}\n"
        f"actual:   {dataset_commit}"
    )


dataset_dirty = git(
    DATASET,
    "status",
    "--porcelain",
    "--untracked-files=no",
)

if dataset_dirty:

    raise RuntimeError(
        "Tracked dataset metadata repo is dirty:\n"
        + dataset_dirty
    )


pcnd_commit = git(
    ROOT,
    "rev-parse",
    "HEAD",
)

pcnd_dirty = git(
    ROOT,
    "status",
    "--porcelain",
    "--untracked-files=no",
)

if pcnd_dirty:

    raise RuntimeError(
        "Tracked PCND worktree is dirty:\n"
        + pcnd_dirty
    )


# ============================================================
# VERIFY ALLOWLIST STRUCTURE
# ============================================================

raw = pd.read_csv(
    ACQ
    / "RAW_FILE_ALLOWLIST.csv"
)

events = pd.read_csv(
    ACQ
    / "FROZEN_EVENT_ALLOWLIST.csv"
)

audit = pd.read_csv(
    ACQ
    / "FROZEN_SITE_EVENT_COUNT_AUDIT.csv"
)


if len(audit) != 4135:

    raise RuntimeError(
        f"Expected 4135 frozen sites, found {len(audit)}"
    )


if not audit[
    "exact_match"
].astype(bool).all():

    raise RuntimeError(
        "Site pulse-count reproduction is no longer exact."
    )


if len(events) != 46458:

    raise RuntimeError(
        f"Expected 46458 frozen events, found {len(events)}"
    )


eeg = raw[
    raw[
        "component"
    ].astype(str)
    ==
    "eeg"
].copy()


if len(eeg) != 113:

    raise RuntimeError(
        f"Expected 113 EEG files, found {len(eeg)}"
    )


if eeg[
    "subject"
].nunique() != 74:

    raise RuntimeError(
        "Frozen EEG allowlist does not cover 74 subjects."
    )


# ============================================================
# FINAL REAL-TIME CHECK:
# NONE OF THE FROZEN EEG CONTENT IS PRESENT LOCALLY
#
# This uses git-annex metadata only and reads no signal sample.
# ============================================================

try:

    present_text = git(
        DATASET,
        "annex",
        "find",
        "--in=here",
    )

except subprocess.CalledProcessError:

    present_text = ""


present_here = {
    x.strip()
    for x in present_text.splitlines()
    if x.strip()
}


frozen_eeg_paths = set(
    eeg[
        "relative_path"
    ]
    .astype(str)
    .tolist()
)


preopen_overlap = sorted(
    frozen_eeg_paths
    &
    present_here
)


if preopen_overlap:

    raise RuntimeError(
        "At least one frozen EEG file is already locally "
        "available before formal OPEN:\n"
        +
        "\n".join(
            preopen_overlap[:20]
        )
    )


# ============================================================
# FORMAL OPEN
# ============================================================

opened_at = datetime.now(
    timezone.utc
).isoformat()


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


record = {

    "record_id":
        "DS004080-FORMAL-LOCKBOX-OPEN-v1.0",

    "lockbox_status":
        "OPEN",

    "opened_at_utc":
        opened_at,

    "meaning": (
        "From this timestamp onward, acquisition and "
        "reading of the previously frozen ds004080 raw "
        "electrophysiological signal files is permitted. "
        "All primary scientific decisions listed below "
        "remain immutable."
    ),

    "pcnd_git_commit_immediately_before_open":
        pcnd_commit,

    "dataset_metadata_commit":
        dataset_commit,

    "machine": {
        "hostname":
            platform.node(),

        "python":
            platform.python_version(),
    },

    "preopening_integrity": {

        "external_lockbox_sha256":
            EXPECTED[
                LOCKBOX
                / "DS004080_FINAL_LOCKBOX_MANIFEST.json"
            ],

        "deployment_checkpoint_sha256":
            EXPECTED[
                DEPLOY
                / "pcnd_v3f_all8_deployment.pt"
            ],

        "deployment_manifest_sha256":
            EXPECTED[
                DEPLOY
                / "ALL8_DEPLOYMENT_MANIFEST.json"
            ],

        "response_protocol_sha256":
            EXPECTED[
                RESPONSE
                / "DS004080_RESPONSE_PROTOCOL.json"
            ],

        "response_freeze_seal_sha256":
            EXPECTED[
                RESPONSE
                / "RESPONSE_PROTOCOL_FREEZE_SEAL.json"
            ],

        "event_allowlist_sha256":
            EXPECTED[
                ACQ
                / "FROZEN_EVENT_ALLOWLIST.csv"
            ],

        "site_audit_sha256":
            EXPECTED[
                ACQ
                / "FROZEN_SITE_EVENT_COUNT_AUDIT.csv"
            ],

        "raw_file_allowlist_sha256":
            EXPECTED[
                ACQ
                / "RAW_FILE_ALLOWLIST.csv"
            ],

        "raw_acquisition_manifest_sha256":
            EXPECTED[
                ACQ
                / "RAW_ACQUISITION_MANIFEST.json"
            ],

        "raw_acquisition_freeze_seal_sha256":
            EXPECTED[
                ACQ
                / "RAW_ACQUISITION_FREEZE_SEAL.json"
            ],
    },

    "frozen_population": {
        "subjects":
            74,

        "stimulation_sites":
            4135,

        "spesclin_runs":
            113,

        "stimulation_events":
            46458,

        "eeg_files":
            113,
    },

    "preopening_raw_state": {

        "allowlisted_eeg_contents_present_locally":
            0,

        "raw_signal_samples_read":
            False,

        "external_response_features_computed":
            False,

        "external_model_inference_run":
            False,

        "external_model_performance_observed":
            False,
    },

    "immutable_primary_analysis": {

        "response":
            "late_rms_z",

        "context_k":
            10,

        "contrast":
            "relation - relation_perm",

        "patient_metric":
            "Spearman map correlation",

        "patient_is_inference_unit":
            True,

        "context_repeats":
            20,

        "test":
            (
                "two-sided exact patient-level "
                "sign-flip/randomization test"
            ),

        "alpha":
            0.05,

        "response_protocol_change_after_open":
            False,

        "cohort_change_after_open":
            False,

        "context_plan_change_after_open":
            False,

        "roi_bridge_change_after_open":
            False,

        "deployment_checkpoint_change_after_open":
            False,

        "endpoint_change_after_open":
            False,
    },

    "failure_policy": (
        "A negative or nonsignificant primary result is "
        "reported as such. No outcome-motivated change to "
        "the frozen primary pipeline can restore "
        "confirmatory lockbox status for this cohort."
    ),
}


OPEN_RECORD.write_text(
    json.dumps(
        record,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


record_sha = sha256_file(
    OPEN_RECORD
)


seal = {

    "record_id":
        record[
            "record_id"
        ],

    "opened_at_utc":
        opened_at,

    "open_record_sha256":
        record_sha,

    "raw_acquisition_manifest_sha256":
        EXPECTED[
            ACQ
            / "RAW_ACQUISITION_MANIFEST.json"
        ],

    "raw_file_allowlist_sha256":
        EXPECTED[
            ACQ
            / "RAW_FILE_ALLOWLIST.csv"
        ],

    "response_protocol_sha256":
        EXPECTED[
            RESPONSE
            / "DS004080_RESPONSE_PROTOCOL.json"
        ],

    "deployment_checkpoint_sha256":
        EXPECTED[
            DEPLOY
            / "pcnd_v3f_all8_deployment.pt"
        ],

    "external_lockbox_sha256":
        EXPECTED[
            LOCKBOX
            / "DS004080_FINAL_LOCKBOX_MANIFEST.json"
        ],

    "dataset_metadata_commit":
        dataset_commit,

    "pcnd_git_commit_immediately_before_open":
        pcnd_commit,
}


OPEN_SEAL.write_text(
    json.dumps(
        seal,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


seal_sha = sha256_file(
    OPEN_SEAL
)


print()
print("=" * 108)
print("DS004080 FORMAL EXTERNAL LOCKBOX OPEN")
print("=" * 108)

print(
    "OPENED AT UTC:",
    opened_at,
)

print(
    "PCND pre-open Git commit:",
    pcnd_commit,
)

print(
    "dataset metadata commit:",
    dataset_commit,
)

print()
print(
    "subjects: 74"
)

print(
    "frozen stimulation sites: 4135"
)

print(
    "frozen stimulation events: 46458"
)

print(
    "frozen SPESclin runs: 113"
)

print(
    "frozen EEG files: 113"
)

print()
print(
    "PRE-OPEN EEG CONTENT PRESENT: 0"
)

print(
    "PRE-OPEN EXTERNAL PERFORMANCE OBSERVED: NO"
)

print()
print(
    "LOCKBOX OPEN RECORD SHA256:",
    record_sha,
)

print(
    "LOCKBOX OPEN SEAL SHA256:",
    seal_sha,
)

print()
print(
    "LOCKBOX STATUS: OPEN"
)

print()
print(
    "PRIMARY ANALYSIS IS NOW IMMUTABLE."
)
