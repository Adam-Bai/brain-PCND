from __future__ import annotations

from pathlib import Path
import os
from collections import Counter
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

DATA = (
    ROOT
    / "data_external_lockbox"
    / "ds004080_metadata"
)

OUT = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_metadata_audit_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


DATASET_ID = "ds004080"

EXPECTED_OPENNEURO_VERSION = "1.2.4"

AUDIT_TIME = datetime.now(
    timezone.utc
).isoformat()


# ============================================================
# ABSOLUTE LOCKBOX SAFETY RULE
# ============================================================

FORBIDDEN_SUFFIXES = {
    ".eeg",
    ".edf",
    ".bdf",
    ".set",
    ".fdt",
    ".mef",
    ".mefd",
}


def sha256_file(path: Path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def git_cmd(*args):

    try:

        return subprocess.check_output(
            [
                "git",
                *args,
            ],
            cwd=DATA,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    except Exception:

        return None


def safe_read_tsv(path: Path):

    if path.suffix.lower() in (
        FORBIDDEN_SUFFIXES
    ):

        raise RuntimeError(
            f"FORBIDDEN SIGNAL READ: {path}"
        )

    if "derivatives" in (
        x.lower()
        for x in path.parts
    ):

        raise RuntimeError(
            f"FORBIDDEN DERIVATIVE READ: {path}"
        )

    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )


def safe_json(path: Path):

    if "derivatives" in (
        x.lower()
        for x in path.parts
    ):

        raise RuntimeError(
            f"FORBIDDEN DERIVATIVE READ: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def valid_string_series(s):

    x = (
        s
        .astype(str)
        .str.strip()
    )

    invalid = x.str.lower().isin(
        {
            "",
            "n/a",
            "na",
            "nan",
            "none",
            "null",
        }
    )

    return ~invalid


def normalize_site(x):

    x = str(x).strip()

    x = (
        x
        .replace(" ", "")
        .replace("_", "-")
    )

    return x


# ============================================================
# DATASET PROVENANCE
# ============================================================

if not DATA.exists():

    raise FileNotFoundError(
        DATA
    )


git_commit = git_cmd(
    "rev-parse",
    "HEAD",
)

git_remote = git_cmd(
    "remote",
    "get-url",
    "origin",
)


dataset_description_path = (
    DATA
    / "dataset_description.json"
)

dataset_description = (
    safe_json(
        dataset_description_path
    )
    if dataset_description_path.exists()
    else {}
)


participants_path = (
    DATA
    / "participants.tsv"
)

participants = (
    safe_read_tsv(
        participants_path
    )
    if participants_path.exists()
    else pd.DataFrame()
)


# ============================================================
# SUBJECT DISCOVERY
# ============================================================

subject_dirs = sorted(
    [
        p
        for p in DATA.glob(
            "sub-*"
        )
        if p.is_dir()
    ]
)


print()
print(
    "=" * 110
)
print(
    "DS004080 METADATA-ONLY LOCKBOX AUDIT"
)
print(
    "=" * 110
)

print(
    "subjects discovered:",
    len(subject_dirs)
)

print(
    "git commit:",
    git_commit
)


subject_rows = []
run_rows = []
coord_rows = []
site_rows = []


for subject_dir in subject_dirs:

    subject = (
        subject_dir.name
    )

    event_files = sorted(
        subject_dir.glob(
            "ses-*/ieeg/*_events.tsv"
        )
    )

    channel_files = sorted(
        subject_dir.glob(
            "ses-*/ieeg/*_channels.tsv"
        )
    )

    electrode_files = sorted(
        subject_dir.glob(
            "ses-*/ieeg/*_electrodes.tsv"
        )
    )

    coordsystem_files = sorted(
        subject_dir.glob(
            "ses-*/ieeg/*_coordsystem.json"
        )
    )


    # ========================================================
    # EVENTS / STIMULATION INVENTORY
    # ========================================================

    all_sites = []

    currents = []
    frequencies = []
    pulsewidths = []

    total_rows = 0
    total_spes_rows = 0

    sessions = set()


    for f in event_files:

        ev = safe_read_tsv(f)

        total_rows += len(ev)

        session = next(
            (
                p
                for p in f.parts
                if p.startswith(
                    "ses-"
                )
            ),
            "unknown",
        )

        sessions.add(
            session
        )


        site_col = (
            "electrical_stimulation_site"
        )

        if site_col not in ev.columns:

            run_rows.append({
                "subject":
                    subject,

                "session":
                    session,

                "event_file":
                    str(
                        f.relative_to(
                            DATA
                        )
                    ),

                "n_rows":
                    len(ev),

                "n_spes_rows":
                    0,

                "n_unique_sites":
                    0,

                "has_stim_site_column":
                    False,
            })

            continue


        mask = valid_string_series(
            ev[
                site_col
            ]
        )


        # If sub_type explicitly distinguishes SPES,
        # respect that label.
        if (
            "sub_type"
            in ev.columns
        ):

            sub_type = (
                ev[
                    "sub_type"
                ]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            if (
                sub_type
                ==
                "SPES"
            ).any():

                mask = (
                    mask
                    &
                    (
                        sub_type
                        ==
                        "SPES"
                    )
                )


        spes = ev[
            mask
        ].copy()

        sites = [
            normalize_site(x)
            for x in
            spes[
                site_col
            ].tolist()
        ]

        all_sites.extend(
            sites
        )

        total_spes_rows += (
            len(spes)
        )


        for col, target in [
            (
                "electrical_stimulation_current",
                currents,
            ),
            (
                "electrical_stimulation_frequency",
                frequencies,
            ),
            (
                "electrical_stimulation_pulsewidth",
                pulsewidths,
            ),
        ]:

            if col in spes.columns:

                vals = pd.to_numeric(
                    spes[col],
                    errors="coerce",
                ).dropna()

                target.extend(
                    vals.tolist()
                )


        run_rows.append({
            "subject":
                subject,

            "session":
                session,

            "event_file":
                str(
                    f.relative_to(
                        DATA
                    )
                ),

            "n_rows":
                len(ev),

            "n_spes_rows":
                len(spes),

            "n_unique_sites":
                len(
                    set(
                        sites
                    )
                ),

            "has_stim_site_column":
                True,
        })


    site_counts = Counter(
        all_sites
    )

    unique_sites = len(
        site_counts
    )

    unique_sites_ge5 = sum(
        count >= 5
        for count in
        site_counts.values()
    )

    unique_sites_ge10 = sum(
        count >= 10
        for count in
        site_counts.values()
    )


    for site, count in sorted(
        site_counts.items()
    ):

        site_rows.append({
            "subject":
                subject,

            "stim_site":
                site,

            "n_pulses":
                count,
        })


    # ========================================================
    # CHANNEL INVENTORY
    # ========================================================

    channel_records = []

    for f in channel_files:

        ch = safe_read_tsv(f)

        if "name" not in ch.columns:

            continue

        for _, r in ch.iterrows():

            channel_records.append({
                "name":
                    str(
                        r[
                            "name"
                        ]
                    ),

                "type":
                    str(
                        r.get(
                            "type",
                            "n/a",
                        )
                    ),

                "status":
                    str(
                        r.get(
                            "status",
                            "good",
                        )
                    ),
            })


    channel_df = pd.DataFrame(
        channel_records
    )


    if len(channel_df):

        channel_df[
            "status"
        ] = (
            channel_df[
                "status"
            ]
            .str.lower()
            .str.strip()
        )

        channel_df = (
            channel_df
            .drop_duplicates(
                subset=[
                    "name",
                ]
            )
        )

        n_channels = len(
            channel_df
        )

        n_good = int(
            (
                channel_df[
                    "status"
                ]
                !=
                "bad"
            ).sum()
        )

        channel_types = (
            channel_df[
                "type"
            ]
            .value_counts()
            .to_dict()
        )

        good_names = set(
            channel_df.loc[
                (
                    channel_df[
                        "status"
                    ]
                    !=
                    "bad"
                ),
                "name",
            ]
            .astype(str)
            .tolist()
        )

    else:

        n_channels = 0
        n_good = 0
        channel_types = {}
        good_names = set()


    # ========================================================
    # ELECTRODE COORDINATE INVENTORY
    # ========================================================

    electrode_frames = []

    coord_systems = []

    for f in electrode_files:

        e = safe_read_tsv(f)

        if (
            "name"
            in e.columns
        ):

            electrode_frames.append(
                e
            )


    for f in coordsystem_files:

        c = safe_json(f)

        coord_systems.append(
            str(
                c.get(
                    "iEEGCoordinateSystem",
                    c.get(
                        "EEGCoordinateSystem",
                        "UNKNOWN",
                    ),
                )
            )
        )


    if electrode_frames:

        electrodes = pd.concat(
            electrode_frames,
            ignore_index=True,
        )

        electrodes = (
            electrodes
            .drop_duplicates(
                subset=[
                    "name",
                ]
            )
        )

        for c in [
            "x",
            "y",
            "z",
        ]:

            if c in electrodes.columns:

                electrodes[c] = (
                    pd.to_numeric(
                        electrodes[c],
                        errors="coerce",
                    )
                )


        has_xyz = all(
            c in electrodes.columns
            for c in [
                "x",
                "y",
                "z",
            ]
        )

        if has_xyz:

            coord_ok = (
                electrodes[
                    [
                        "x",
                        "y",
                        "z",
                    ]
                ]
                .notna()
                .all(
                    axis=1
                )
            )

            n_electrodes_xyz = int(
                coord_ok.sum()
            )

        else:

            coord_ok = pd.Series(
                False,
                index=electrodes.index,
            )

            n_electrodes_xyz = 0


        electrode_names_xyz = set(
            electrodes.loc[
                coord_ok,
                "name",
            ]
            .astype(str)
            .tolist()
        )

        if good_names:

            n_good_exact_xyz = len(
                good_names
                &
                electrode_names_xyz
            )

            good_xyz_fraction = (
                n_good_exact_xyz
                /
                len(
                    good_names
                )
            )

        else:

            n_good_exact_xyz = 0
            good_xyz_fraction = np.nan


        n_electrodes = len(
            electrodes
        )

    else:

        n_electrodes = 0
        n_electrodes_xyz = 0
        n_good_exact_xyz = 0
        good_xyz_fraction = np.nan


    # ========================================================
    # PURE METADATA ELIGIBILITY FLAGS
    # ========================================================

    # Site-repeat rule matches development philosophy:
    # stimulation site must have >=5 pulses.
    #
    # k=10 needs 10 context sites + >=1 query site.
    site_budget_ok = (
        unique_sites_ge5
        >=
        11
    )

    # Pre-outcome operational definition of an
    # evaluable recording map.
    recording_budget_ok = (
        n_good
        >=
        20
    )

    coords_present = (
        n_electrodes_xyz
        >
        0
    )

    provisional_k10_eligible = (
        site_budget_ok
        and
        recording_budget_ok
        and
        coords_present
    )


    pulse_counts = np.asarray(
        list(
            site_counts.values()
        ),
        dtype=float,
    )


    subject_rows.append({
        "subject":
            subject,

        "n_sessions":
            len(
                sessions
            ),

        "n_event_files":
            len(
                event_files
            ),

        "n_event_rows":
            total_rows,

        "n_spes_events":
            total_spes_rows,

        "n_unique_stim_sites":
            unique_sites,

        "n_sites_ge5_pulses":
            unique_sites_ge5,

        "n_sites_ge10_pulses":
            unique_sites_ge10,

        "median_pulses_per_site":
            (
                float(
                    np.median(
                        pulse_counts
                    )
                )
                if len(
                    pulse_counts
                )
                else np.nan
            ),

        "min_pulses_per_site":
            (
                float(
                    np.min(
                        pulse_counts
                    )
                )
                if len(
                    pulse_counts
                )
                else np.nan
            ),

        "max_pulses_per_site":
            (
                float(
                    np.max(
                        pulse_counts
                    )
                )
                if len(
                    pulse_counts
                )
                else np.nan
            ),

        "n_channels":
            n_channels,

        "n_good_channels":
            n_good,

        "channel_types":
            json.dumps(
                channel_types,
                sort_keys=True,
            ),

        "n_electrodes":
            n_electrodes,

        "n_electrodes_with_xyz":
            n_electrodes_xyz,

        "n_good_channels_exact_xyz":
            n_good_exact_xyz,

        "good_channel_exact_xyz_fraction":
            good_xyz_fraction,

        "coordinate_systems":
            "|".join(
                sorted(
                    set(
                        coord_systems
                    )
                )
            ),

        "stim_current_median":
            (
                float(
                    np.median(
                        currents
                    )
                )
                if currents
                else np.nan
            ),

        "stim_frequency_median":
            (
                float(
                    np.median(
                        frequencies
                    )
                )
                if frequencies
                else np.nan
            ),

        "stim_pulsewidth_median":
            (
                float(
                    np.median(
                        pulsewidths
                    )
                )
                if pulsewidths
                else np.nan
            ),

        "site_budget_k10_ok":
            site_budget_ok,

        "recording_budget_ok":
            recording_budget_ok,

        "coordinates_present":
            coords_present,

        "provisional_k10_metadata_eligible":
            provisional_k10_eligible,
    })


    coord_rows.append({
        "subject":
            subject,

        "coordinate_systems":
            "|".join(
                sorted(
                    set(
                        coord_systems
                    )
                )
            ),

        "n_electrodes":
            n_electrodes,

        "n_electrodes_with_xyz":
            n_electrodes_xyz,

        "good_channel_exact_xyz_fraction":
            good_xyz_fraction,
    })


# ============================================================
# OUTPUTS
# ============================================================

subject_df = pd.DataFrame(
    subject_rows
)

run_df = pd.DataFrame(
    run_rows
)

coord_df = pd.DataFrame(
    coord_rows
)

site_df = pd.DataFrame(
    site_rows
)


subject_df.to_csv(
    OUT
    / "subject_metadata_eligibility.csv",
    index=False,
)

run_df.to_csv(
    OUT
    / "run_metadata_inventory.csv",
    index=False,
)

coord_df.to_csv(
    OUT
    / "coordinate_inventory.csv",
    index=False,
)

site_df.to_csv(
    OUT
    / "stimulation_site_inventory.csv",
    index=False,
)


eligible = subject_df[
    subject_df[
        "provisional_k10_metadata_eligible"
    ]
].copy()


eligible.to_csv(
    OUT
    / "provisional_k10_eligible_subjects.csv",
    index=False,
)


# ============================================================
# METADATA HASH INVENTORY
# ============================================================

metadata_patterns = [
    "participants.tsv",
    "participants.json",
    "dataset_description.json",
    "events.json",
]


metadata_files = []

for p in metadata_patterns:

    f = DATA / p

    if f.exists():

        metadata_files.append(
            f
        )


for pattern in [
    "sub-*/ses-*/ieeg/*_events.tsv",
    "sub-*/ses-*/ieeg/*_channels.tsv",
    "sub-*/ses-*/ieeg/*_electrodes.tsv",
    "sub-*/ses-*/ieeg/*_coordsystem.json",
    "sub-*/ses-*/ieeg/*_ieeg.json",
]:

    metadata_files.extend(
        sorted(
            DATA.glob(
                pattern
            )
        )
    )


hash_rows = []

for f in sorted(
    set(
        metadata_files
    )
):

    hash_rows.append({
        "path":
            str(
                f.relative_to(
                    DATA
                )
            ),

        "sha256":
            sha256_file(
                f
            ),

        "size_bytes":
            f.stat().st_size,
    })


pd.DataFrame(
    hash_rows
).to_csv(
    OUT
    / "metadata_sha256_inventory.csv",
    index=False,
)


# ============================================================
# AUDIT MANIFEST
# ============================================================

manifest = {
    "audit_time_utc":
        AUDIT_TIME,

    "dataset_id":
        DATASET_ID,

    "expected_openneuro_version":
        EXPECTED_OPENNEURO_VERSION,

    "git_commit":
        git_commit,

    "git_remote":
        git_remote,

    "dataset_description":
        dataset_description,

    "n_subject_directories":
        int(
            len(
                subject_df
            )
        ),

    "n_provisional_k10_eligible":
        int(
            len(
                eligible
            )
        ),

    "eligibility_is_outcome_blind":
        True,

    "signals_read":
        False,

    "derivatives_read":
        False,

    "eligibility_rules": {
        "minimum_sites_with_ge5_pulses":
            11,

        "minimum_good_channels":
            20,

        "coordinates_required":
            True,
    },

    "important_status":
        (
            "This is a metadata-only audit. "
            "No CCEP waveform, response feature, "
            "derivative outcome, or model performance "
            "has been inspected."
        ),
}


(
    OUT
    / "METADATA_AUDIT_MANIFEST.json"
).write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
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
    "DATASET SUMMARY"
)
print(
    "=" * 110
)

print(
    "subjects:",
    len(
        subject_df
    )
)

print(
    "provisional k=10 eligible:",
    len(
        eligible
    )
)


print()
print(
    "=" * 110
)
print(
    "SUBJECT METADATA ELIGIBILITY"
)
print(
    "=" * 110
)

cols = [
    "subject",
    "n_sessions",
    "n_spes_events",
    "n_unique_stim_sites",
    "n_sites_ge5_pulses",
    "median_pulses_per_site",
    "n_good_channels",
    "n_electrodes_with_xyz",
    "coordinate_systems",
    "provisional_k10_metadata_eligible",
]


print(
    subject_df[
        cols
    ]
    .to_string(
        index=False
    )
)


print()
print(
    "=" * 110
)
print(
    "ELIGIBILITY COUNTS"
)
print(
    "=" * 110
)

print(
    "site budget >=11 usable sites:",
    int(
        subject_df[
            "site_budget_k10_ok"
        ].sum()
    ),
    "/",
    len(
        subject_df
    ),
)

print(
    ">=20 good channels:",
    int(
        subject_df[
            "recording_budget_ok"
        ].sum()
    ),
    "/",
    len(
        subject_df
    ),
)

print(
    "coordinates present:",
    int(
        subject_df[
            "coordinates_present"
        ].sum()
    ),
    "/",
    len(
        subject_df
    ),
)

print(
    "provisional eligible:",
    int(
        subject_df[
            "provisional_k10_metadata_eligible"
        ].sum()
    ),
    "/",
    len(
        subject_df
    ),
)


print()
print(
    "NO SIGNALS READ."
)

print(
    "NO DERIVATIVES READ."
)

print(
    "Saved:",
    OUT
)
