from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import os
import mne
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

RAW_ROOT = (
    ROOT
    / "data_external_lockbox"
    / "ds004080_raw_frozen_v1"
)

LOCKBOX = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_final_lockbox_v1"
)

ACQ = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_raw_acquisition_v1"
)

RESPONSE = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_response_protocol_v1"
)

OPEN_DIR = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_formal_open_v1"
)

OUT = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_primary_phenotypes_v1"
)

PER_RUN = (
    OUT
    / "per_run"
)

PROGRESS = (
    OUT
    / "EXTRACTION_PROGRESS.json"
)

FINAL_TRIAL = (
    OUT
    / "ds004080_primary_trial_response_features.csv.gz"
)

FINAL_MAP = (
    OUT
    / "ds004080_primary_perturbation_map.csv.gz"
)

FINAL_QC = (
    OUT
    / "PHENOTYPE_EXTRACTION_QC.json"
)

FINAL_SEAL = (
    OUT
    / "PHENOTYPE_EXTRACTION_SEAL.json"
)


# ============================================================
# FROZEN HASHES
# ============================================================

EXPECTED_EXTERNAL_LOCKBOX_SHA256 = (
    "8170555317e6a2b510cb7ef1bbb603ba"
    "6a4220ec92a0b50cc32c29392f66cf6d"
)

EXPECTED_EVENT_ALLOWLIST_SHA256 = (
    "6f6be3335da9a45fcd946d3fd6832bab"
    "0c8099e0443aadec1cf69eb64c5e39b8"
)

EXPECTED_RAW_ALLOWLIST_SHA256 = (
    "681b896a202edbef69a46d991fb65dea9"
    "57c7382ce73ddc9c3120026e9cfa29c"
)

EXPECTED_ACQ_MANIFEST_SHA256 = (
    "11d09494db152e973becf682ca473ee7d"
    "dc22209c243c6b4e46e893d440f98df"
)

EXPECTED_RESPONSE_PROTOCOL_SHA256 = (
    "df944bdf79b50f8609320cad46e15c3d"
    "f4bcf1d207335b83a6d18d85e883a423"
)

EXPECTED_DEPLOYMENT_SHA256 = (
    "399421567b0185383bf8953cdcd6c758"
    "91893b5113fca8a3a41ac190eb68a794"
)

EXPECTED_N_SUBJECTS = 74
EXPECTED_N_RUNS = 113
EXPECTED_N_EVENTS = 46458
EXPECTED_N_SITES = 4135


# ============================================================
# FROZEN PHENOTYPE
# ============================================================

PRE = (
    -0.500,
    -0.020,
)

EARLY = (
    0.010,
    0.100,
)

LATE = (
    0.100,
    0.500,
)

NEAR_FIELD_MM = 10.0


# ============================================================
# HELPERS
# ============================================================

def sha256_file(
    path: Path,
) -> str:

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


def assert_hash(
    path: Path,
    expected: str,
    label: str,
):

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    actual = sha256_file(
        path
    )

    if actual != expected:

        raise RuntimeError(
            f"{label} hash mismatch\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )


def cf(x) -> str:

    return str(x).strip().casefold()


def iso_now() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def robust_scale_rows(
    baseline: np.ndarray,
):

    center = np.nanmedian(
        baseline,
        axis=1,
    )

    mad = np.nanmedian(
        np.abs(
            baseline
            -
            center[:, None]
        ),
        axis=1,
    )

    scale = (
        1.4826
        *
        mad
    )

    bad = (
        ~np.isfinite(scale)
        |
        (scale < 1e-6)
    )

    if np.any(bad):

        std = np.nanstd(
            baseline,
            axis=1,
        )

        scale[bad] = std[bad]


    bad = (
        ~np.isfinite(scale)
        |
        (scale < 1e-6)
    )

    scale[bad] = 1.0

    return (
        center,
        scale,
    )


def rms_rows(
    x: np.ndarray,
):

    return np.sqrt(
        np.nanmean(
            x ** 2,
            axis=1,
        )
    )


# ============================================================
# VERIFY FROZEN STATE
# ============================================================

assert_hash(
    LOCKBOX
    / "DS004080_FINAL_LOCKBOX_MANIFEST.json",
    EXPECTED_EXTERNAL_LOCKBOX_SHA256,
    "external lockbox",
)

assert_hash(
    ACQ
    / "FROZEN_EVENT_ALLOWLIST.csv",
    EXPECTED_EVENT_ALLOWLIST_SHA256,
    "event allowlist",
)

assert_hash(
    ACQ
    / "RAW_FILE_ALLOWLIST.csv",
    EXPECTED_RAW_ALLOWLIST_SHA256,
    "raw file allowlist",
)

assert_hash(
    ACQ
    / "RAW_ACQUISITION_MANIFEST.json",
    EXPECTED_ACQ_MANIFEST_SHA256,
    "raw acquisition manifest",
)

assert_hash(
    RESPONSE
    / "DS004080_RESPONSE_PROTOCOL.json",
    EXPECTED_RESPONSE_PROTOCOL_SHA256,
    "response protocol",
)

deployment = (
    ROOT
    / "results"
    / "clinical_v10"
    / "v3f_all8_deployment"
    / "pcnd_v3f_all8_deployment.pt"
)

assert_hash(
    deployment,
    EXPECTED_DEPLOYMENT_SHA256,
    "deployment checkpoint",
)


open_record_path = (
    OPEN_DIR
    / "DS004080_LOCKBOX_OPEN_RECORD.json"
)

if not open_record_path.exists():

    raise RuntimeError(
        "Formal lockbox OPEN record not found."
    )

open_record = json.loads(
    open_record_path.read_text()
)

if (
    open_record.get(
        "lockbox_status"
    )
    !=
    "OPEN"
):

    raise RuntimeError(
        "External lockbox is not formally OPEN."
    )


# ============================================================
# LOAD FROZEN TABLES
# ============================================================

events = pd.read_csv(
    ACQ
    / "FROZEN_EVENT_ALLOWLIST.csv"
)

raw_allow = pd.read_csv(
    ACQ
    / "RAW_FILE_ALLOWLIST.csv"
)

inventory = pd.read_csv(
    LOCKBOX
    / "recording_channel_inventory.csv"
)

sites = pd.read_csv(
    LOCKBOX
    / "strict_usable_stimulation_sites.csv"
)

cohort = pd.read_csv(
    LOCKBOX
    / "lockbox_cohort.csv"
)


if len(events) != EXPECTED_N_EVENTS:

    raise RuntimeError(
        f"Frozen event count changed: "
        f"{len(events)} != {EXPECTED_N_EVENTS}"
    )


subjects = (
    cohort[
        "subject"
    ]
    .astype(str)
    .tolist()
)


if len(subjects) != EXPECTED_N_SUBJECTS:

    raise RuntimeError(
        "Frozen cohort count mismatch."
    )


expected_runs = sorted(
    events[
        "run_base"
    ]
    .astype(str)
    .unique()
    .tolist()
)


if len(expected_runs) != EXPECTED_N_RUNS:

    raise RuntimeError(
        f"Frozen run count changed: "
        f"{len(expected_runs)} != "
        f"{EXPECTED_N_RUNS}"
    )


# ============================================================
# RAW COMPONENT LOOKUP
# ============================================================

raw_component = {}

for _, r in raw_allow.iterrows():

    key = (
        str(
            r[
                "run_base"
            ]
        ),
        str(
            r[
                "component"
            ]
        ),
    )

    raw_component[key] = str(
        r[
            "relative_path"
        ]
    )


def component_path(
    run_base: str,
    component: str,
) -> Path:

    key = (
        run_base,
        component,
    )

    if key not in raw_component:

        raise RuntimeError(
            f"No frozen {component} path "
            f"for {run_base}"
        )

    return (
        RAW_ROOT
        /
        raw_component[key]
    )


def run_raw_available(
    run_base: str,
) -> bool:

    return all(
        component_path(
            run_base,
            c,
        ).exists()

        for c in (
            "eeg",
            "vhdr",
            "vmrk",
        )
    )


# ============================================================
# FROZEN ANATOMICAL LOOKUPS
# ============================================================

subject_inventory = {}

for sub, g in inventory.groupby(
    "subject"
):

    subject_inventory[
        str(sub)
    ] = (
        g.copy()
        .reset_index(
            drop=True
        )
    )


site_lookup = {}

for _, r in sites.iterrows():

    sub = str(
        r[
            "subject"
        ]
    )

    site = str(
        r[
            "stim_site"
        ]
    )

    site_lookup[
        (
            sub,
            site,
        )
    ] = {
        "stim_a":
            str(
                r[
                    "stim_contact_a"
                ]
            ),

        "stim_b":
            str(
                r[
                    "stim_contact_b"
                ]
            ),

        "mid":
            np.asarray(
                [
                    r[
                        "stim_mid_fsaverage_x"
                    ],
                    r[
                        "stim_mid_fsaverage_y"
                    ],
                    r[
                        "stim_mid_fsaverage_z"
                    ],
                ],
                dtype=np.float64,
            ),
    }


# ============================================================
# OUTPUT
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PER_RUN.mkdir(
    parents=True,
    exist_ok=True,
)


def per_run_trial_path(
    run_base: str,
):

    return (
        PER_RUN
        /
        f"{run_base}_trial_primary.csv.gz"
    )


def per_run_summary_path(
    run_base: str,
):

    return (
        PER_RUN
        /
        f"{run_base}_summary.json"
    )


# ============================================================
# RUN EXTRACTION
# ============================================================

def extract_run(
    run_base: str,
    force: bool = False,
):

    out_trial = per_run_trial_path(
        run_base
    )

    out_summary = per_run_summary_path(
        run_base
    )


    if (
        out_trial.exists()
        and
        out_summary.exists()
        and
        not force
    ):

        summary = json.loads(
            out_summary.read_text()
        )

        if (
            summary.get(
                "status"
            )
            ==
            "PASS"
        ):

            print(
                f"[SKIP] {run_base}: "
                f"already extracted"
            )

            return summary


    if not run_raw_available(
        run_base
    ):

        print(
            f"[WAIT] {run_base}: raw trio incomplete"
        )

        return {
            "run_base":
                run_base,

            "status":
                "RAW_NOT_AVAILABLE",
        }


    ev = (
        events[
            events[
                "run_base"
            ].astype(str)
            ==
            run_base
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    if ev.empty:

        raise RuntimeError(
            f"{run_base}: no frozen events."
        )


    run_subjects = (
        ev[
            "subject"
        ]
        .astype(str)
        .unique()
    )


    if len(run_subjects) != 1:

        raise RuntimeError(
            f"{run_base}: expected one subject."
        )


    subject = str(
        run_subjects[0]
    )


    if subject not in subject_inventory:

        raise RuntimeError(
            f"{subject}: no frozen recording inventory."
        )


    inv = (
        subject_inventory[
            subject
        ]
        .copy()
    )


    vhdr = component_path(
        run_base,
        "vhdr",
    )


    t0 = time.time()


    raw = mne.io.read_raw_brainvision(
        vhdr,
        preload=False,
        verbose="ERROR",
    )


    fs = float(
        raw.info[
            "sfreq"
        ]
    )


    # --------------------------------------------------------
    # Case-insensitive raw channel mapping.
    # Refuse ambiguity.
    # --------------------------------------------------------

    raw_casefold = {}

    for idx, name in enumerate(
        raw.ch_names
    ):

        key = cf(
            name
        )

        if key in raw_casefold:

            raise RuntimeError(
                f"{run_base}: duplicate raw channel "
                f"after casefold: {name}"
            )

        raw_casefold[
            key
        ] = (
            idx,
            name,
        )


    rec_names = []
    rec_raw_indices = []
    rec_xyz = []


    for _, r in inv.iterrows():

        name = str(
            r[
                "recording_channel"
            ]
        )

        key = cf(
            name
        )

        if key not in raw_casefold:

            continue


        xyz = np.asarray(
            [
                r[
                    "fsaverage_x"
                ],
                r[
                    "fsaverage_y"
                ],
                r[
                    "fsaverage_z"
                ],
            ],
            dtype=np.float64,
        )


        if not np.isfinite(
            xyz
        ).all():

            continue


        idx, _raw_name = (
            raw_casefold[
                key
            ]
        )


        rec_names.append(
            name
        )

        rec_raw_indices.append(
            int(idx)
        )

        rec_xyz.append(
            xyz
        )


    if not rec_names:

        raise RuntimeError(
            f"{run_base}: zero frozen recording "
            f"channels found in raw."
        )


    rec_xyz = np.stack(
        rec_xyz,
        axis=0,
    )


    rec_name_cf = np.asarray(
        [
            cf(x)
            for x in rec_names
        ],
        dtype=object,
    )


    rows = []

    n_epoch_out_of_bounds = 0
    n_raw_read_failures = 0
    n_events_with_response = 0
    n_nonfinite_features = 0
    n_scale_fallback_std = 0
    n_scale_fallback_one = 0


    for _, e in ev.iterrows():

        site = str(
            e[
                "frozen_stim_site"
            ]
        )

        site_key = (
            subject,
            site,
        )


        if site_key not in site_lookup:

            raise RuntimeError(
                f"{run_base}: frozen site missing "
                f"from anatomical table: {site}"
            )


        sinfo = site_lookup[
            site_key
        ]


        s0 = int(
            e[
                "sample_start"
            ]
        )


        pre0 = (
            s0
            +
            int(
                round(
                    PRE[0]
                    *
                    fs
                )
            )
        )

        pre1 = (
            s0
            +
            int(
                round(
                    PRE[1]
                    *
                    fs
                )
            )
        )

        early0 = (
            s0
            +
            int(
                round(
                    EARLY[0]
                    *
                    fs
                )
            )
        )

        early1 = (
            s0
            +
            int(
                round(
                    EARLY[1]
                    *
                    fs
                )
            )
        )

        late0 = (
            s0
            +
            int(
                round(
                    LATE[0]
                    *
                    fs
                )
            )
        )

        late1 = (
            s0
            +
            int(
                round(
                    LATE[1]
                    *
                    fs
                )
            )
        )


        if (
            pre0 < 0
            or
            late1 > raw.n_times
        ):

            n_epoch_out_of_bounds += 1

            continue


        try:

            # MNE returns SI volts.
            # Frozen development phenotype uses
            # microvolt-scale semantics for the 1e-6
            # robust-scale fallback threshold.
            x_uV = (
                raw.get_data(
                    picks=rec_raw_indices,
                    start=pre0,
                    stop=late1,
                )
                *
                1e6
            )

        except Exception:

            n_raw_read_failures += 1

            continue


        p1 = (
            pre1
            -
            pre0
        )

        e0 = (
            early0
            -
            pre0
        )

        e1 = (
            early1
            -
            pre0
        )

        l0 = (
            late0
            -
            pre0
        )

        l1 = (
            late1
            -
            pre0
        )


        baseline = x_uV[
            :,
            0:p1,
        ]

        early = x_uV[
            :,
            e0:e1,
        ]

        late = x_uV[
            :,
            l0:l1,
        ]


        if (
            baseline.shape[1] == 0
            or
            early.shape[1] == 0
            or
            late.shape[1] == 0
        ):

            n_epoch_out_of_bounds += 1

            continue


        # ----------------------------------------------------
        # Frozen recording-channel exclusions
        # ----------------------------------------------------

        mid = sinfo[
            "mid"
        ]


        if not np.isfinite(
            mid
        ).all():

            raise RuntimeError(
                f"{run_base}: nonfinite frozen midpoint "
                f"for {site}"
            )


        distance = np.linalg.norm(
            rec_xyz
            -
            mid[None, :],
            axis=1,
        )


        stim_a_cf = cf(
            sinfo[
                "stim_a"
            ]
        )

        stim_b_cf = cf(
            sinfo[
                "stim_b"
            ]
        )


        eligible = (
            np.isfinite(
                distance
            )
            &
            (
                distance
                >
                NEAR_FIELD_MM
            )
            &
            (
                rec_name_cf
                !=
                stim_a_cf
            )
            &
            (
                rec_name_cf
                !=
                stim_b_cf
            )
        )


        if not np.any(
            eligible
        ):

            continue


        # ----------------------------------------------------
        # EXACT frozen robust scale semantics
        # ----------------------------------------------------

        center = np.nanmedian(
            baseline,
            axis=1,
        )

        mad = np.nanmedian(
            np.abs(
                baseline
                -
                center[:, None]
            ),
            axis=1,
        )

        primary_scale = (
            1.4826
            *
            mad
        )


        fallback_std = (
            ~np.isfinite(
                primary_scale
            )
            |
            (
                primary_scale
                <
                1e-6
            )
        )

        n_scale_fallback_std += int(
            fallback_std.sum()
        )


        scale = (
            primary_scale.copy()
        )


        if np.any(
            fallback_std
        ):

            std = np.nanstd(
                baseline,
                axis=1,
            )

            scale[
                fallback_std
            ] = std[
                fallback_std
            ]


        fallback_one = (
            ~np.isfinite(
                scale
            )
            |
            (
                scale
                <
                1e-6
            )
        )

        n_scale_fallback_one += int(
            fallback_one.sum()
        )


        scale[
            fallback_one
        ] = 1.0


        early_z = (
            early
            -
            center[:, None]
        ) / scale[:, None]


        late_z = (
            late
            -
            center[:, None]
        ) / scale[:, None]


        early_rms = rms_rows(
            early_z
        )

        late_rms = rms_rows(
            late_z
        )


        finite_feature = (
            np.isfinite(
                early_rms
            )
            &
            np.isfinite(
                late_rms
            )
        )


        n_nonfinite_features += int(
            (
                eligible
                &
                ~finite_feature
            ).sum()
        )


        keep = (
            eligible
            &
            finite_feature
        )


        idx_keep = np.flatnonzero(
            keep
        )


        if len(
            idx_keep
        ) == 0:

            continue


        n_events_with_response += 1


        event_rows = pd.DataFrame({
            "subject":
                subject,

            "run_base":
                run_base,

            "event_row_index":
                int(
                    e[
                        "event_row_index"
                    ]
                ),

            "sample_start":
                s0,

            "stim_site":
                site,

            "recording_channel":
                [
                    rec_names[i]
                    for i in idx_keep
                ],

            "distance_to_stim":
                distance[
                    idx_keep
                ],

            "baseline_center_uV":
                center[
                    idx_keep
                ],

            "baseline_scale_uV":
                scale[
                    idx_keep
                ],

            "early_rms_z":
                early_rms[
                    idx_keep
                ],

            "late_rms_z":
                late_rms[
                    idx_keep
                ],
        })


        rows.append(
            event_rows
        )


    raw.close()


    if rows:

        trial = pd.concat(
            rows,
            ignore_index=True,
        )

    else:

        trial = pd.DataFrame(
            columns=[
                "subject",
                "run_base",
                "event_row_index",
                "sample_start",
                "stim_site",
                "recording_channel",
                "distance_to_stim",
                "baseline_center_uV",
                "baseline_scale_uV",
                "early_rms_z",
                "late_rms_z",
            ]
        )


    # Deterministic order.
    trial = trial.sort_values(
        [
            "subject",
            "run_base",
            "event_row_index",
            "stim_site",
            "recording_channel",
        ]
    ).reset_index(
        drop=True
    )


    trial.to_csv(
        out_trial,
        index=False,
        compression="gzip",
    )


    # Per-run aggregation is QC only.
    # It is NOT the final patient map because the same
    # stimulation site may occur in multiple runs.
    if len(trial):

        run_map = (
            trial
            .assign(
                event_uid=(
                    trial[
                        "run_base"
                    ].astype(str)
                    +
                    ":"
                    +
                    trial[
                        "event_row_index"
                    ].astype(str)
                )
            )
            .groupby(
                [
                    "subject",
                    "stim_site",
                    "recording_channel",
                ],
                as_index=False,
            )
            .agg(
                n_trials=(
                    "event_uid",
                    "nunique",
                ),

                early_rms_z=(
                    "early_rms_z",
                    "median",
                ),

                late_rms_z=(
                    "late_rms_z",
                    "median",
                ),
            )
        )

        run_map_rows = len(
            run_map
        )

        n_trials_min = int(
            run_map[
                "n_trials"
            ].min()
        )

        n_trials_median = float(
            run_map[
                "n_trials"
            ].median()
        )

        n_trials_max = int(
            run_map[
                "n_trials"
            ].max()
        )

    else:

        run_map_rows = 0
        n_trials_min = None
        n_trials_median = None
        n_trials_max = None


    elapsed = (
        time.time()
        -
        t0
    )


    summary = {
        "status":
            "PASS",

        "created_at_utc":
            iso_now(),

        "subject":
            subject,

        "run_base":
            run_base,

        "sampling_frequency_hz":
            fs,

        "raw_n_channels":
            len(
                raw.ch_names
            ),

        "raw_n_times":
            int(
                raw.n_times
            ),

        "frozen_events_expected":
            int(
                len(ev)
            ),

        "events_with_valid_response":
            int(
                n_events_with_response
            ),

        "epoch_out_of_bounds":
            int(
                n_epoch_out_of_bounds
            ),

        "raw_read_failures":
            int(
                n_raw_read_failures
            ),

        "frozen_recording_channels":
            int(
                len(inv)
            ),

        "frozen_recording_channels_present_in_raw":
            int(
                len(rec_names)
            ),

        "trial_response_rows":
            int(
                len(trial)
            ),

        "unique_stimulation_sites":
            int(
                trial[
                    "stim_site"
                ].nunique()
                if len(trial)
                else 0
            ),

        "unique_recording_channels":
            int(
                trial[
                    "recording_channel"
                ].nunique()
                if len(trial)
                else 0
            ),

        "nonfinite_primary_features_excluded":
            int(
                n_nonfinite_features
            ),

        "baseline_scale_std_fallback_count":
            int(
                n_scale_fallback_std
            ),

        "baseline_scale_one_fallback_count":
            int(
                n_scale_fallback_one
            ),

        "run_level_map_rows_qc_only":
            int(
                run_map_rows
            ),

        "run_level_n_trials_min":
            n_trials_min,

        "run_level_n_trials_median":
            n_trials_median,

        "run_level_n_trials_max":
            n_trials_max,

        "trial_output":
            str(
                out_trial
            ),

        "trial_output_sha256":
            sha256_file(
                out_trial
            ),

        "elapsed_seconds":
            elapsed,

        "model_inference_run":
            False,

        "external_model_performance_observed":
            False,
    }


    out_summary.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )


    print()
    print(
        f"[PASS] {run_base}"
    )

    print(
        "  subject:",
        subject
    )

    print(
        "  frozen events:",
        len(ev)
    )

    print(
        "  valid-response events:",
        n_events_with_response
    )

    print(
        "  trial rows:",
        len(trial)
    )

    print(
        "  run-level map rows (QC only):",
        run_map_rows
    )

    print(
        "  channels present:",
        f"{len(rec_names)}/{len(inv)}"
    )

    print(
        "  scale fallback std:",
        n_scale_fallback_std
    )

    print(
        "  scale fallback 1.0:",
        n_scale_fallback_one
    )

    print(
        "  elapsed:",
        f"{elapsed:.1f}s"
    )


    return summary


# ============================================================
# PROGRESS
# ============================================================

def collect_progress():

    completed = []

    waiting = []

    failed = []


    for run_base in expected_runs:

        summary_path = (
            per_run_summary_path(
                run_base
            )
        )

        trial_path = (
            per_run_trial_path(
                run_base
            )
        )


        if (
            summary_path.exists()
            and
            trial_path.exists()
        ):

            try:

                s = json.loads(
                    summary_path.read_text()
                )

                if (
                    s.get(
                        "status"
                    )
                    ==
                    "PASS"
                ):

                    completed.append(
                        run_base
                    )

                    continue

            except Exception:

                failed.append(
                    run_base
                )

                continue


        if run_raw_available(
            run_base
        ):

            failed.append(
                run_base
            )

        else:

            waiting.append(
                run_base
            )


    progress = {
        "updated_at_utc":
            iso_now(),

        "expected_runs":
            EXPECTED_N_RUNS,

        "completed_runs":
            len(
                completed
            ),

        "waiting_for_raw":
            len(
                waiting
            ),

        "incomplete_or_failed":
            len(
                failed
            ),

        "completed_run_bases":
            completed,

        "waiting_run_bases":
            waiting,

        "incomplete_run_bases":
            failed,

        "model_inference_run":
            False,

        "external_model_performance_observed":
            False,
    }


    PROGRESS.write_text(
        json.dumps(
            progress,
            indent=2,
        )
        + "\n"
    )


    return progress


# ============================================================
# FINALIZE ONLY WHEN 113 / 113 COMPLETE
# ============================================================

def finalize():

    progress = collect_progress()


    if (
        progress[
            "completed_runs"
        ]
        !=
        EXPECTED_N_RUNS
    ):

        print()
        print(
            "FINALIZATION BLOCKED"
        )

        print(
            "completed:",
            progress[
                "completed_runs"
            ],
            "/",
            EXPECTED_N_RUNS,
        )

        print(
            "waiting for raw:",
            progress[
                "waiting_for_raw"
            ],
        )

        print(
            "incomplete/failed:",
            progress[
                "incomplete_or_failed"
            ],
        )

        print()
        print(
            "No final perturbation map was created."
        )

        return False


    if (
        FINAL_TRIAL.exists()
        or
        FINAL_MAP.exists()
        or
        FINAL_QC.exists()
        or
        FINAL_SEAL.exists()
    ):

        raise RuntimeError(
            "Final phenotype artifacts already exist. "
            "Refusing overwrite."
        )


    print()
    print(
        "=" * 110
    )

    print(
        "FINALIZING 113-RUN EXTERNAL PHENOTYPE"
    )

    print(
        "=" * 110
    )


    frames = []


    for i, run_base in enumerate(
        expected_runs,
        1,
    ):

        p = per_run_trial_path(
            run_base
        )

        df = pd.read_csv(
            p
        )

        frames.append(
            df
        )

        if (
            i % 10 == 0
            or
            i == EXPECTED_N_RUNS
        ):

            print(
                f"loaded {i}/{EXPECTED_N_RUNS} runs"
            )


    trial = pd.concat(
        frames,
        ignore_index=True,
    )


    trial = trial.sort_values(
        [
            "subject",
            "stim_site",
            "recording_channel",
            "run_base",
            "event_row_index",
        ]
    ).reset_index(
        drop=True
    )


    trial[
        "event_uid"
    ] = (
        trial[
            "run_base"
        ].astype(str)
        +
        ":"
        +
        trial[
            "event_row_index"
        ].astype(str)
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Aggregate ALL eligible trials across runs at once.
    # Never median-of-run-medians.
    # --------------------------------------------------------

    map_df = (
        trial
        .groupby(
            [
                "subject",
                "stim_site",
                "recording_channel",
            ],
            as_index=False,
        )
        .agg(
            n_trials=(
                "event_uid",
                "nunique",
            ),

            n_runs=(
                "run_base",
                "nunique",
            ),

            distance_to_stim=(
                "distance_to_stim",
                "median",
            ),

            baseline_scale_uV_median=(
                "baseline_scale_uV",
                "median",
            ),

            early_rms_z=(
                "early_rms_z",
                "median",
            ),

            late_rms_z=(
                "late_rms_z",
                "median",
            ),

            early_rms_iqr=(
                "early_rms_z",
                lambda x:
                    float(
                        np.percentile(
                            x,
                            75,
                        )
                        -
                        np.percentile(
                            x,
                            25,
                        )
                    ),
            ),

            late_rms_iqr=(
                "late_rms_z",
                lambda x:
                    float(
                        np.percentile(
                            x,
                            75,
                        )
                        -
                        np.percentile(
                            x,
                            25,
                        )
                    ),
            ),
        )
    )


    # --------------------------------------------------------
    # Pure extraction/QC statistics.
    # NO MODEL PERFORMANCE.
    # --------------------------------------------------------

    signal_event_keys = (
        trial[
            [
                "subject",
                "run_base",
                "event_row_index",
            ]
        ]
        .drop_duplicates()
    )


    frozen_event_keys = (
        events[
            [
                "subject",
                "run_base",
                "event_row_index",
            ]
        ]
        .drop_duplicates()
    )


    frozen_sites = set(
        zip(
            sites[
                "subject"
            ].astype(str),

            sites[
                "stim_site"
            ].astype(str),
        )
    )


    observed_sites = set(
        zip(
            map_df[
                "subject"
            ].astype(str),

            map_df[
                "stim_site"
            ].astype(str),
        )
    )


    # Only frozen usable sites are expected.
    # strict_usable_stimulation_sites.csv already represents
    # the frozen site set used to construct the lockbox.
    expected_site_count = len(
        frozen_sites
    )


    missing_sites = sorted(
        frozen_sites
        -
        observed_sites
    )


    subject_map_counts = (
        map_df
        .groupby(
            "subject"
        )[
            "stim_site"
        ]
        .nunique()
        .reindex(
            subjects,
            fill_value=0,
        )
    )


    low_trial_rows = int(
        (
            map_df[
                "n_trials"
            ]
            <
            5
        ).sum()
    )


    final_qc = {
        "created_at_utc":
            iso_now(),

        "status":
            (
                "EXTRACTION_COMPLETE"
                if not missing_sites
                else
                "EXTRACTION_COMPLETE_WITH_MECHANICAL_MISSINGNESS"
            ),

        "subjects_expected":
            EXPECTED_N_SUBJECTS,

        "subjects_observed":
            int(
                map_df[
                    "subject"
                ].nunique()
            ),

        "runs_expected":
            EXPECTED_N_RUNS,

        "runs_extracted":
            EXPECTED_N_RUNS,

        "frozen_events_expected":
            int(
                len(
                    frozen_event_keys
                )
            ),

        "events_with_at_least_one_valid_response":
            int(
                len(
                    signal_event_keys
                )
            ),

        "events_without_valid_response":
            int(
                len(
                    frozen_event_keys
                )
                -
                len(
                    signal_event_keys
                )
            ),

        "trial_response_rows":
            int(
                len(
                    trial
                )
            ),

        "map_rows":
            int(
                len(
                    map_df
                )
            ),

        "frozen_sites_expected":
            int(
                expected_site_count
            ),

        "frozen_sites_with_at_least_one_map_entry":
            int(
                len(
                    observed_sites
                    &
                    frozen_sites
                )
            ),

        "frozen_sites_missing_after_mechanical_qc":
            int(
                len(
                    missing_sites
                )
            ),

        "site_channel_rows_with_fewer_than_5_surviving_trials":
            low_trial_rows,

        "subject_site_count_min":
            int(
                subject_map_counts.min()
            ),

        "subject_site_count_median":
            float(
                subject_map_counts.median()
            ),

        "subject_site_count_max":
            int(
                subject_map_counts.max()
            ),

        "n_trials_min":
            int(
                map_df[
                    "n_trials"
                ].min()
            ),

        "n_trials_median":
            float(
                map_df[
                    "n_trials"
                ].median()
            ),

        "n_trials_max":
            int(
                map_df[
                    "n_trials"
                ].max()
            ),

        "early_rms_z_summary": {
            "median":
                float(
                    map_df[
                        "early_rms_z"
                    ].median()
                ),

            "p01":
                float(
                    map_df[
                        "early_rms_z"
                    ].quantile(
                        0.01
                    )
                ),

            "p99":
                float(
                    map_df[
                        "early_rms_z"
                    ].quantile(
                        0.99
                    )
                ),
        },

        "late_rms_z_summary": {
            "median":
                float(
                    map_df[
                        "late_rms_z"
                    ].median()
                ),

            "p01":
                float(
                    map_df[
                        "late_rms_z"
                    ].quantile(
                        0.01
                    )
                ),

            "p99":
                float(
                    map_df[
                        "late_rms_z"
                    ].quantile(
                        0.99
                    )
                ),
        },

        "missing_sites":
            [
                {
                    "subject":
                        x[0],

                    "stim_site":
                        x[1],
                }

                for x in missing_sites
            ],

        "primary_response_protocol_sha256":
            EXPECTED_RESPONSE_PROTOCOL_SHA256,

        "event_allowlist_sha256":
            EXPECTED_EVENT_ALLOWLIST_SHA256,

        "raw_allowlist_sha256":
            EXPECTED_RAW_ALLOWLIST_SHA256,

        "external_model_inference_run":
            False,

        "external_model_performance_observed":
            False,
    }


    # Remove helper only used for aggregation.
    trial = trial.drop(
        columns=[
            "event_uid"
        ]
    )


    trial.to_csv(
        FINAL_TRIAL,
        index=False,
        compression="gzip",
    )


    map_df.to_csv(
        FINAL_MAP,
        index=False,
        compression="gzip",
    )


    FINAL_QC.write_text(
        json.dumps(
            final_qc,
            indent=2,
        )
        + "\n"
    )


    seal = {
        "seal_id":
            "DS004080-primary-phenotype-v1.0",

        "created_at_utc":
            iso_now(),

        "trial_features_sha256":
            sha256_file(
                FINAL_TRIAL
            ),

        "perturbation_map_sha256":
            sha256_file(
                FINAL_MAP
            ),

        "phenotype_qc_sha256":
            sha256_file(
                FINAL_QC
            ),

        "response_protocol_sha256":
            EXPECTED_RESPONSE_PROTOCOL_SHA256,

        "event_allowlist_sha256":
            EXPECTED_EVENT_ALLOWLIST_SHA256,

        "raw_file_allowlist_sha256":
            EXPECTED_RAW_ALLOWLIST_SHA256,

        "deployment_checkpoint_sha256":
            EXPECTED_DEPLOYMENT_SHA256,

        "external_model_inference_run":
            False,

        "external_model_performance_observed":
            False,
    }


    FINAL_SEAL.write_text(
        json.dumps(
            seal,
            indent=2,
        )
        + "\n"
    )


    print()
    print(
        "=" * 110
    )

    print(
        "DS004080 PRIMARY PHENOTYPE EXTRACTION COMPLETE"
    )

    print(
        "=" * 110
    )

    print(
        "subjects:",
        final_qc[
            "subjects_observed"
        ],
    )

    print(
        "runs:",
        EXPECTED_N_RUNS,
    )

    print(
        "events with response:",
        final_qc[
            "events_with_at_least_one_valid_response"
        ],
        "/",
        EXPECTED_N_EVENTS,
    )

    print(
        "trial rows:",
        len(
            trial
        ),
    )

    print(
        "map rows:",
        len(
            map_df
        ),
    )

    print(
        "frozen sites observed:",
        final_qc[
            "frozen_sites_with_at_least_one_map_entry"
        ],
        "/",
        expected_site_count,
    )

    print(
        "site-channel rows n_trials < 5:",
        low_trial_rows,
    )

    print()
    print(
        "TRIAL FEATURES SHA256:",
        seal[
            "trial_features_sha256"
        ],
    )

    print(
        "PERTURBATION MAP SHA256:",
        seal[
            "perturbation_map_sha256"
        ],
    )

    print(
        "PHENOTYPE QC SHA256:",
        seal[
            "phenotype_qc_sha256"
        ],
    )

    print()
    print(
        "MODEL INFERENCE RUN: NO"
    )

    print(
        "EXTERNAL MODEL PERFORMANCE OBSERVED: NO"
    )

    print()
    print(
        "STATUS: READY FOR PRE-INFERENCE QC AUDIT"
    )


    return True


# ============================================================
# MAIN
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--run-base",
    action="append",
    default=None,
    help=(
        "Process one run_base. "
        "May be supplied multiple times."
    ),
)

parser.add_argument(
    "--force",
    action="store_true",
)

parser.add_argument(
    "--finalize",
    action="store_true",
)

args = parser.parse_args()


if args.run_base:

    selected = args.run_base

else:

    # Default: process every raw-complete run.
    selected = [
        r
        for r in expected_runs
        if run_raw_available(
            r
        )
    ]


print()
print("=" * 110)

print(
    "DS004080 FROZEN PRIMARY PHENOTYPE EXTRACTION"
)

print("=" * 110)

print(
    "formal lockbox:",
    open_record.get(
        "lockbox_status"
    ),
)

print(
    "expected runs:",
    EXPECTED_N_RUNS,
)

print(
    "raw-complete runs currently available:",
    sum(
        run_raw_available(r)
        for r in expected_runs
    ),
)

print(
    "runs selected this invocation:",
    len(
        selected
    ),
)

print(
    "MODEL INFERENCE: DISABLED"
)

print()


for i, run_base in enumerate(
    selected,
    1,
):

    print()
    print(
        f"[{i}/{len(selected)}] "
        f"{run_base}"
    )

    extract_run(
        run_base,
        force=args.force,
    )


progress = collect_progress()


print()
print("=" * 110)

print(
    "EXTRACTION PROGRESS"
)

print("=" * 110)

print(
    "completed:",
    progress[
        "completed_runs"
    ],
    "/",
    EXPECTED_N_RUNS,
)

print(
    "waiting for raw:",
    progress[
        "waiting_for_raw"
    ],
)

print(
    "incomplete/failed:",
    progress[
        "incomplete_or_failed"
    ],
)


if args.finalize:

    finalize()

elif (
    progress[
        "completed_runs"
    ]
    ==
    EXPECTED_N_RUNS
):

    print()
    print(
        "All 113 runs are extracted."
    )

    print(
        "Run again with --finalize "
        "to create the frozen cross-run map."
    )
