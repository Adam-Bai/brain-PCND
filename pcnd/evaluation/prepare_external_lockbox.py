from __future__ import annotations

from pathlib import Path
import os
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import subprocess

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

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

HAP_FEATURE_ROOT = (
    ROOT
    / "results"
    / "clinical_v5"
    / "v3_features"
)

HAP_FREEZE = (
    ROOT
    / "results"
    / "clinical_v9"
    / "v3f_final_strict"
    / "protocol_freeze"
    / "V3F_FINAL_PROTOCOL_MANIFEST.json"
)

OUT_ROOT = (
    ROOT
    / "results"
    / "external_lockbox"
)

DRY_OUT = (
    OUT_ROOT
    / "ds004080_lockbox_freeze_dryrun_v1"
)

SEALED_OUT = (
    OUT_ROOT
    / "ds004080_lockbox_frozen_v1"
)


EXPECTED_HAP_FREEZE_SHA256 = (
    "cb3bd89d6072245ecec7ae608d692f286172fedab517b1b81d821e4e3e710727"
)


# ------------------------------------------------------------
# A priori FieldTrip fsaverage/MNI305 -> MNI152(SPM)
# affine.
#
# IMPORTANT:
# This is an approximate template bridge.
# It is NOT claimed to be an exact mapping to
# MNI152NLin6Sym.
# ------------------------------------------------------------

FSAVERAGE_TO_MNI152 = np.array(
    [
        [
            0.9975,
            -0.0073,
            0.0176,
            -0.0429,
        ],
        [
            0.0146,
            1.0009,
            -0.0024,
            1.5496,
        ],
        [
            -0.0130,
            -0.0093,
            0.9971,
            1.1840,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    ],
    dtype=float,
)


ADAPTER_NAME = (
    "FieldTrip_fsaverage_MNI305_to_MNI152_SPM_affine_v1"
)

ADAPTER_SOURCE = (
    "https://github.com/fieldtrip/fieldtrip/"
    "blob/master/utilities/private/"
    "align_fsaverage2mni.m"
)


PRIMARY_NEAR_FIELD_MM = 10.0

MIN_SITE_PULSES = 5

MIN_RECORDING_CHANNELS_PER_SITE = 20

MIN_USABLE_SITES_K10 = 11

CONTEXT_K = [
    1,
    3,
    5,
    10,
]

N_CONTEXT_REPEATS = 20

LOCKBOX_CONTEXT_SEED = 20260926


FORBIDDEN_SUFFIXES = {
    ".eeg",
    ".edf",
    ".bdf",
    ".set",
    ".fdt",
    ".mef",
    ".mefd",
}


ROI_COLUMNS_HAP = [
    "stim_a_roi",
    "stim_b_roi",
    "recording_roi",
]


parser = argparse.ArgumentParser()

parser.add_argument(
    "--seal",
    action="store_true",
    help=(
        "Write final lockbox seal. "
        "Without this flag the script performs "
        "an outcome-blind dry run only."
    ),
)

args = parser.parse_args()

OUT = (
    SEALED_OUT
    if args.seal
    else DRY_OUT
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAFETY
# ============================================================

def safe_tsv(
    path: Path,
    wanted=None,
):

    if path.suffix.lower() in (
        FORBIDDEN_SUFFIXES
    ):

        raise RuntimeError(
            f"FORBIDDEN SIGNAL READ: {path}"
        )

    if "derivatives" in [
        x.lower()
        for x in path.parts
    ]:

        raise RuntimeError(
            f"FORBIDDEN DERIVATIVE READ: {path}"
        )

    if wanted is None:

        return pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
        )

    header = pd.read_csv(
        path,
        sep="\t",
        nrows=0,
    )

    available = [
        c
        for c in wanted
        if c in header.columns
    ]

    return pd.read_csv(
        path,
        sep="\t",
        usecols=available,
        low_memory=False,
    )


def safe_json(
    path: Path,
):

    if "derivatives" in [
        x.lower()
        for x in path.parts
    ]:

        raise RuntimeError(
            f"FORBIDDEN DERIVATIVE READ: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


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

            h.update(
                block
            )

    return h.hexdigest()


def git_cmd(
    cwd,
    *args_,
):

    try:

        return subprocess.check_output(
            [
                "git",
                *args_,
            ],
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    except Exception:

        return None


# ============================================================
# HELPERS
# ============================================================

def normalize_name(
    x,
):

    return (
        str(x)
        .strip()
        .replace(
            " ",
            "",
        )
    )


def good_string(
    x,
):

    s = str(x).strip()

    return (
        s != ""
        and
        s.lower()
        not in {
            "n/a",
            "na",
            "nan",
            "none",
            "null",
        }
    )


def finite_float(
    x,
):

    try:

        value = float(x)

    except Exception:

        return None

    if not np.isfinite(
        value
    ):

        return None

    return value


def protocol_number(
    x,
):

    value = finite_float(
        x
    )

    if value is None:

        return None

    # deterministic normalization of TSV floats
    return round(
        value,
        9,
    )


def protocol_key(
    row,
):

    return (
        protocol_number(
            row.get(
                "electrical_stimulation_current",
                None,
            )
        ),
        protocol_number(
            row.get(
                "electrical_stimulation_frequency",
                None,
            )
        ),
        protocol_number(
            row.get(
                "electrical_stimulation_pulsewidth",
                None,
            )
        ),
        str(
            row.get(
                "electrical_stimulation_type",
                "NA",
            )
        )
        .strip()
        .lower(),
    )


def parse_pair(
    value,
    electrode_names,
):

    raw = normalize_name(
        value
    )

    names = set(
        electrode_names
    )

    parts = [
        x
        for x in re.split(
            r"[-_;,+/\\\s]+",
            raw,
        )
        if x
    ]

    exact = [
        x
        for x in parts
        if x in names
    ]

    if len(
        exact
    ) == 2:

        return tuple(
            exact
        )

    lower = {
        x.lower(): x
        for x in names
    }

    hits = []

    for part in parts:

        y = lower.get(
            part.lower()
        )

        if y is not None:

            hits.append(y)

    if len(
        hits
    ) == 2:

        return tuple(
            hits
        )

    # conservative fallback
    hits = []

    for name in sorted(
        names,
        key=len,
        reverse=True,
    ):

        pattern = (
            r"(?<![A-Za-z0-9])"
            +
            re.escape(name)
            +
            r"(?![A-Za-z0-9])"
        )

        if re.search(
            pattern,
            raw,
            flags=re.I,
        ):

            hits.append(
                name
            )

    unique = []

    for x in hits:

        if x not in unique:

            unique.append(x)

    if len(
        unique
    ) == 2:

        return tuple(
            unique
        )

    return None


def canonical_site(
    pair,
):

    # V3 uses sign-invariant bipolar orientation,
    # so pair order does not define identity.
    return "--".join(
        sorted(
            pair
        )
    )


def apply_adapter(
    xyz,
):

    xyz = np.asarray(
        xyz,
        dtype=float,
    )

    h = np.concatenate(
        [
            xyz,
            [
                1.0,
            ],
        ]
    )

    out = (
        FSAVERAGE_TO_MNI152
        @ h
    )

    return out[:3]


def hemisphere_class(
    x,
):

    s = str(x).lower().strip()

    if (
        s.startswith("l")
        or
        "left"
        in s
    ):

        return "L"

    if (
        s.startswith("r")
        or
        "right"
        in s
    ):

        return "R"

    return None


def deterministic_seed(
    subject,
    k,
    repeat,
):

    text = (
        f"{LOCKBOX_CONTEXT_SEED}|"
        f"{subject}|"
        f"{k}|"
        f"{repeat}"
    )

    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


# ============================================================
# VERIFY DEVELOPMENT FREEZE
# ============================================================

if not HAP_FREEZE.exists():

    raise FileNotFoundError(
        HAP_FREEZE
    )


hap_hash = sha256_file(
    HAP_FREEZE
)


if (
    hap_hash
    !=
    EXPECTED_HAP_FREEZE_SHA256
):

    raise RuntimeError(
        "HAPwave frozen protocol hash changed.\n"
        f"Expected: "
        f"{EXPECTED_HAP_FREEZE_SHA256}\n"
        f"Actual:   "
        f"{hap_hash}"
    )


print(
    "HAPwave freeze verified:",
    hap_hash
)


# ============================================================
# HAPWAVE ROI VOCABULARY
# ============================================================

hap_roi_values = []

for f in sorted(
    HAP_FEATURE_ROOT.glob(
        "sub-*_pcnd_v3_pairs.csv.gz"
    )
):

    header = pd.read_csv(
        f,
        nrows=0,
    )

    cols = [
        c
        for c in ROI_COLUMNS_HAP
        if c in header.columns
    ]

    x = pd.read_csv(
        f,
        usecols=cols,
    )

    for c in cols:

        hap_roi_values.extend(
            x[c]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )


HAP_ROI_VOCAB = set(
    x
    for x in hap_roi_values
    if good_string(x)
)


print(
    "HAPwave ROI vocabulary:",
    len(
        HAP_ROI_VOCAB
    )
)


# ============================================================
# SUBJECT DISCOVERY
# ============================================================

subjects = sorted(
    p.name
    for p in DATA.glob(
        "sub-*"
    )
    if p.is_dir()
)


# ============================================================
# OUTPUT TABLES
# ============================================================

subject_rows = []

electrode_rows = []

recording_rows = []

site_rows = []

context_rows = []

all_ds_roi_text = []

all_ds_roi_numeric = []


# ============================================================
# PROCESS EACH SUBJECT
# ============================================================

for subject in subjects:

    subdir = (
        DATA
        / subject
    )


    electrode_files = sorted(
        subdir.glob(
            "ses-*/ieeg/*_electrodes.tsv"
        )
    )

    channel_files = sorted(
        subdir.glob(
            "ses-*/ieeg/*_channels.tsv"
        )
    )

    event_files = sorted(
        subdir.glob(
            "ses-*/ieeg/*_events.tsv"
        )
    )


    # --------------------------------------------------------
    # ELECTRODES
    #
    # Explicit usecols intentionally BLIND SOZ / resection.
    # --------------------------------------------------------

    ef = []

    for f in electrode_files:

        e = safe_tsv(
            f,
            wanted=[
                "name",
                "x",
                "y",
                "z",
                "hemisphere",
                "Destrieux_label",
                "Destrieux_label_text",
            ],
        )

        ef.append(
            e
        )


    if not ef:

        raise RuntimeError(
            f"{subject}: no electrodes.tsv"
        )


    elec = pd.concat(
        ef,
        ignore_index=True,
    )


    elec[
        "name"
    ] = (
        elec[
            "name"
        ]
        .astype(str)
        .map(
            normalize_name
        )
    )


    elec = (
        elec
        .drop_duplicates(
            subset=[
                "name",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    for c in [
        "x",
        "y",
        "z",
    ]:

        elec[c] = pd.to_numeric(
            elec[c],
            errors="coerce",
        )


    elec[
        "coord_ok"
    ] = (
        elec[
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


    transformed = []

    for _, r in (
        elec.iterrows()
    ):

        if not r[
            "coord_ok"
        ]:

            transformed.append(
                (
                    np.nan,
                    np.nan,
                    np.nan,
                )
            )

            continue

        transformed.append(
            apply_adapter(
                [
                    r["x"],
                    r["y"],
                    r["z"],
                ]
            )
        )


    transformed = np.asarray(
        transformed,
        dtype=float,
    )


    elec[
        "mni_x"
    ] = transformed[:, 0]

    elec[
        "mni_y"
    ] = transformed[:, 1]

    elec[
        "mni_z"
    ] = transformed[:, 2]


    # ROI compatibility candidates.
    if (
        "Destrieux_label_text"
        in elec.columns
    ):

        all_ds_roi_text.extend(
            elec[
                "Destrieux_label_text"
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )


    if (
        "Destrieux_label"
        in elec.columns
    ):

        all_ds_roi_numeric.extend(
            elec[
                "Destrieux_label"
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )


    # Electrode lookup.
    electrode_lookup = {}

    for _, r in (
        elec.iterrows()
    ):

        name = r[
            "name"
        ]

        if not r[
            "coord_ok"
        ]:

            continue

        roi_text = (
            str(
                r.get(
                    "Destrieux_label_text",
                    "UNK",
                )
            ).strip()
        )

        roi_numeric = (
            str(
                r.get(
                    "Destrieux_label",
                    "UNK",
                )
            ).strip()
        )

        electrode_lookup[
            name
        ] = {
            "fs_xyz":
                np.array(
                    [
                        r["x"],
                        r["y"],
                        r["z"],
                    ],
                    dtype=float,
                ),

            "mni_xyz":
                np.array(
                    [
                        r["mni_x"],
                        r["mni_y"],
                        r["mni_z"],
                    ],
                    dtype=float,
                ),

            "hemisphere":
                r.get(
                    "hemisphere",
                    None,
                ),

            "roi_text":
                roi_text,

            "roi_numeric":
                roi_numeric,
        }


        electrode_rows.append({
            "subject":
                subject,

            "electrode":
                name,

            "fsaverage_x":
                r["x"],

            "fsaverage_y":
                r["y"],

            "fsaverage_z":
                r["z"],

            "adapter_mni_x":
                r["mni_x"],

            "adapter_mni_y":
                r["mni_y"],

            "adapter_mni_z":
                r["mni_z"],

            "hemisphere":
                r.get(
                    "hemisphere",
                    None,
                ),

            "Destrieux_label":
                roi_numeric,

            "Destrieux_label_text":
                roi_text,
        })


    electrode_names = set(
        electrode_lookup.keys()
    )


    # --------------------------------------------------------
    # HEMISPHERE SIGN SANITY
    # --------------------------------------------------------

    hemi_checks = []

    for name, info in (
        electrode_lookup.items()
    ):

        hemi = hemisphere_class(
            info[
                "hemisphere"
            ]
        )

        if hemi is None:

            continue

        x = info[
            "mni_xyz"
        ][0]

        if hemi == "L":

            hemi_checks.append(
                x < 0
            )

        elif hemi == "R":

            hemi_checks.append(
                x > 0
            )


    hemi_sign_fraction = (
        float(
            np.mean(
                hemi_checks
            )
        )
        if hemi_checks
        else np.nan
    )


    # --------------------------------------------------------
    # CHANNELS
    # --------------------------------------------------------

    cf = []

    for f in channel_files:

        ch = safe_tsv(
            f,
            wanted=[
                "name",
                "type",
                "status",
            ],
        )

        cf.append(ch)


    channels = pd.concat(
        cf,
        ignore_index=True,
    )


    channels[
        "name"
    ] = (
        channels[
            "name"
        ]
        .astype(str)
        .map(
            normalize_name
        )
    )


    if (
        "status"
        not in channels.columns
    ):

        channels[
            "status"
        ] = "good"


    if (
        "type"
        not in channels.columns
    ):

        channels[
            "type"
        ] = "UNKNOWN"


    channels[
        "status_norm"
    ] = (
        channels[
            "status"
        ]
        .astype(str)
        .str.lower()
        .str.strip()
    )


    channels[
        "type_norm"
    ] = (
        channels[
            "type"
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    channels = (
        channels
        .drop_duplicates(
            subset=[
                "name",
            ]
        )
    )


    rec_names = []

    for _, r in (
        channels.iterrows()
    ):

        name = r[
            "name"
        ]

        if (
            r[
                "status_norm"
            ]
            ==
            "bad"
        ):

            continue

        if (
            r[
                "type_norm"
            ]
            not in {
                "ECOG",
                "SEEG",
            }
        ):

            continue

        if (
            name
            not in
            electrode_lookup
        ):

            continue

        rec_names.append(
            name
        )


        info = electrode_lookup[
            name
        ]

        recording_rows.append({
            "subject":
                subject,

            "recording_channel":
                name,

            "type":
                r[
                    "type_norm"
                ],

            "fsaverage_x":
                info[
                    "fs_xyz"
                ][0],

            "fsaverage_y":
                info[
                    "fs_xyz"
                ][1],

            "fsaverage_z":
                info[
                    "fs_xyz"
                ][2],

            "adapter_mni_x":
                info[
                    "mni_xyz"
                ][0],

            "adapter_mni_y":
                info[
                    "mni_xyz"
                ][1],

            "adapter_mni_z":
                info[
                    "mni_xyz"
                ][2],

            "roi_text":
                info[
                    "roi_text"
                ],

            "roi_numeric":
                info[
                    "roi_numeric"
                ],
        })


    rec_names = sorted(
        set(
            rec_names
        )
    )


    # --------------------------------------------------------
    # EVENTS — SPES metadata ONLY
    # --------------------------------------------------------

    event_rows = []


    for f in event_files:

        ev = safe_tsv(
            f,
            wanted=[
                "sub_type",
                "trial_type",
                "electrical_stimulation_site",
                "electrical_stimulation_current",
                "electrical_stimulation_frequency",
                "electrical_stimulation_pulsewidth",
                "electrical_stimulation_type",
            ],
        )


        if (
            "electrical_stimulation_site"
            not in ev.columns
        ):

            continue


        mask = (
            ev[
                "electrical_stimulation_site"
            ]
            .map(
                good_string
            )
        )


        if (
            "sub_type"
            in ev.columns
        ):

            subtype = (
                ev[
                    "sub_type"
                ]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            if (
                subtype
                ==
                "SPES"
            ).any():

                mask &= (
                    subtype
                    ==
                    "SPES"
                )


        ev = ev[
            mask
        ].copy()


        for _, r in (
            ev.iterrows()
        ):

            pair = parse_pair(
                r[
                    "electrical_stimulation_site"
                ],
                electrode_names,
            )

            if pair is None:

                continue

            event_rows.append({
                "pair":
                    pair,

                "site":
                    canonical_site(
                        pair
                    ),

                "protocol":
                    protocol_key(
                        r
                    ),
            })


    if not event_rows:

        raise RuntimeError(
            f"{subject}: no parsed SPES events"
        )


    protocol_counts = Counter(
        x[
            "protocol"
        ]
        for x in event_rows
    )


    # deterministic dominant protocol
    dominant_protocol = sorted(
        protocol_counts.items(),
        key=lambda kv: (
            -kv[1],
            str(
                kv[0]
            ),
        ),
    )[0][0]


    dominant_events = [
        x
        for x in event_rows
        if (
            x[
                "protocol"
            ]
            ==
            dominant_protocol
        )
    ]


    site_counts = Counter(
        x[
            "site"
        ]
        for x in dominant_events
    )


    site_to_pair = {}

    for x in dominant_events:

        site_to_pair[
            x[
                "site"
            ]
        ] = x[
            "pair"
        ]


    initial_sites = [
        site
        for site, count
        in site_counts.items()
        if count >= MIN_SITE_PULSES
    ]


    # --------------------------------------------------------
    # STRICT SITE ELIGIBILITY
    # --------------------------------------------------------

    usable_sites = []


    for site in sorted(
        initial_sites
    ):

        pair = site_to_pair[
            site
        ]

        a, b = pair


        if (
            a
            not in electrode_lookup
            or
            b
            not in electrode_lookup
        ):

            continue


        ia = electrode_lookup[a]
        ib = electrode_lookup[b]


        stim_mid_fs = (
            ia[
                "fs_xyz"
            ]
            +
            ib[
                "fs_xyz"
            ]
        ) / 2.0


        stim_mid_mni = (
            ia[
                "mni_xyz"
            ]
            +
            ib[
                "mni_xyz"
            ]
        ) / 2.0


        stim_vec_mni = (
            ib[
                "mni_xyz"
            ]
            -
            ia[
                "mni_xyz"
            ]
        )


        stim_norm = float(
            np.linalg.norm(
                stim_vec_mni
            )
        )


        if (
            not np.isfinite(
                stim_norm
            )
            or
            stim_norm
            <
            1e-9
        ):

            continue


        eligible_rec = []


        for ch in rec_names:

            if ch in {
                a,
                b,
            }:

                continue


            rec = electrode_lookup[
                ch
            ]


            # Primary artifact exclusion uses
            # provided fsaverage mm coordinates.
            dist_fs = float(
                np.linalg.norm(
                    rec[
                        "fs_xyz"
                    ]
                    -
                    stim_mid_fs
                )
            )


            if (
                dist_fs
                <
                PRIMARY_NEAR_FIELD_MM
            ):

                continue


            eligible_rec.append(
                ch
            )


        n_rec = len(
            eligible_rec
        )


        site_is_usable = (
            n_rec
            >=
            MIN_RECORDING_CHANNELS_PER_SITE
        )


        site_rows.append({
            "subject":
                subject,

            "stim_site":
                site,

            "stim_contact_a":
                a,

            "stim_contact_b":
                b,

            "n_pulses_dominant_protocol":
                site_counts[
                    site
                ],

            "n_eligible_recording_channels":
                n_rec,

            "site_usable":
                site_is_usable,

            "stim_mid_fsaverage_x":
                stim_mid_fs[0],

            "stim_mid_fsaverage_y":
                stim_mid_fs[1],

            "stim_mid_fsaverage_z":
                stim_mid_fs[2],

            "stim_mid_adapter_mni_x":
                stim_mid_mni[0],

            "stim_mid_adapter_mni_y":
                stim_mid_mni[1],

            "stim_mid_adapter_mni_z":
                stim_mid_mni[2],

            "stim_pair_distance_adapter_mni":
                stim_norm,

            "stim_a_roi_text":
                ia[
                    "roi_text"
                ],

            "stim_b_roi_text":
                ib[
                    "roi_text"
                ],

            "dominant_protocol":
                json.dumps(
                    dominant_protocol
                ),
        })


        if site_is_usable:

            usable_sites.append(
                site
            )


    usable_sites = sorted(
        usable_sites
    )


    subject_eligible = (
        len(
            usable_sites
        )
        >=
        MIN_USABLE_SITES_K10
    )


    # --------------------------------------------------------
    # FREEZE RANDOM CONTEXTS / QUERY SETS
    # --------------------------------------------------------

    if subject_eligible:

        for k in CONTEXT_K:

            if (
                len(
                    usable_sites
                )
                <= k
            ):

                raise RuntimeError(
                    f"{subject}: "
                    f"not enough sites for k={k}"
                )


            for repeat in range(
                N_CONTEXT_REPEATS
            ):

                seed = deterministic_seed(
                    subject,
                    k,
                    repeat,
                )

                rng = (
                    np.random.default_rng(
                        seed
                    )
                )


                context = sorted(
                    rng.choice(
                        usable_sites,
                        size=k,
                        replace=False,
                    ).tolist()
                )


                context_set = set(
                    context
                )


                query = [
                    x
                    for x in usable_sites
                    if (
                        x
                        not in context_set
                    )
                ]


                context_rows.append({
                    "subject":
                        subject,

                    "context_k":
                        k,

                    "repeat":
                        repeat,

                    "seed":
                        seed,

                    "context_sites":
                        json.dumps(
                            context
                        ),

                    "query_sites":
                        json.dumps(
                            query
                        ),

                    "n_query_sites":
                        len(
                            query
                        ),
                })


    protocol_json = json.dumps(
        {
            "current":
                dominant_protocol[0],

            "frequency":
                dominant_protocol[1],

            "pulsewidth":
                dominant_protocol[2],

            "stimulation_type":
                dominant_protocol[3],
        },
        sort_keys=True,
    )


    rec_per_usable = [
        row[
            "n_eligible_recording_channels"
        ]
        for row in site_rows
        if (
            row[
                "subject"
            ]
            ==
            subject
            and
            row[
                "site_usable"
            ]
        )
    ]


    subject_rows.append({
        "subject":
            subject,

        "dominant_protocol":
            protocol_json,

        "dominant_protocol_events":
            protocol_counts[
                dominant_protocol
            ],

        "n_recording_channels_geometry_valid":
            len(
                rec_names
            ),

        "n_sites_ge5_dominant_protocol":
            len(
                initial_sites
            ),

        "n_strict_usable_sites":
            len(
                usable_sites
            ),

        "min_recording_channels_per_usable_site":
            (
                min(
                    rec_per_usable
                )
                if rec_per_usable
                else 0
            ),

        "median_recording_channels_per_usable_site":
            (
                float(
                    np.median(
                        rec_per_usable
                    )
                )
                if rec_per_usable
                else np.nan
            ),

        "hemisphere_x_sign_fraction":
            hemi_sign_fraction,

        "strict_k10_lockbox_eligible":
            subject_eligible,
    })


# ============================================================
# ROI SCHEMA COMPATIBILITY
# ============================================================

def clean_roi_values(
    values,
):

    return [
        str(x).strip()
        for x in values
        if good_string(x)
    ]


ds_text = clean_roi_values(
    all_ds_roi_text
)

ds_numeric = clean_roi_values(
    all_ds_roi_numeric
)


def weighted_overlap(
    values,
    vocab,
):

    if not values:

        return np.nan

    return float(
        np.mean(
            [
                x in vocab
                for x in values
            ]
        )
    )


text_overlap = weighted_overlap(
    ds_text,
    HAP_ROI_VOCAB,
)

numeric_overlap = weighted_overlap(
    ds_numeric,
    HAP_ROI_VOCAB,
)


if (
    np.nan_to_num(
        text_overlap,
        nan=-1.0,
    )
    >=
    np.nan_to_num(
        numeric_overlap,
        nan=-1.0,
    )
):

    roi_representation = (
        "Destrieux_label_text"
    )

    roi_overlap = text_overlap

else:

    roi_representation = (
        "Destrieux_label"
    )

    roi_overlap = numeric_overlap


# ============================================================
# TABLES
# ============================================================

subject_df = pd.DataFrame(
    subject_rows
)

electrode_df = pd.DataFrame(
    electrode_rows
)

recording_df = pd.DataFrame(
    recording_rows
)

site_df = pd.DataFrame(
    site_rows
)

context_df = pd.DataFrame(
    context_rows
)


cohort = (
    subject_df[
        subject_df[
            "strict_k10_lockbox_eligible"
        ]
    ][
        [
            "subject",
        ]
    ]
    .sort_values(
        "subject"
    )
    .reset_index(
        drop=True
    )
)


subject_df.to_csv(
    OUT
    / "subject_strict_eligibility.csv",
    index=False,
)

electrode_df.to_csv(
    OUT
    / "electrodes_anatomical_adapter.csv",
    index=False,
)

recording_df.to_csv(
    OUT
    / "recording_channel_inventory.csv",
    index=False,
)

site_df.to_csv(
    OUT
    / "strict_usable_stimulation_sites.csv",
    index=False,
)

context_df.to_csv(
    OUT
    / "frozen_context_query_plan.csv",
    index=False,
)

cohort.to_csv(
    OUT
    / "lockbox_cohort.csv",
    index=False,
)


# ============================================================
# ADAPTER RECORD
# ============================================================

adapter = {
    "name":
        ADAPTER_NAME,

    "source_coordinate_system":
        "fsaverage / FreeSurfer MNI305-like volumetric template",

    "source_units":
        "mm",

    "target_coordinate_system":
        "MNI152/SPM as defined by FieldTrip transform",

    "matrix":
        FSAVERAGE_TO_MNI152.tolist(),

    "source":
        ADAPTER_SOURCE,

    "status":
        (
            "Primary a-priori external adapter; "
            "frozen without response outcomes."
        ),

    "important_limitation":
        (
            "The frozen HAPwave V3 geometry uses "
            "MNI152NLin6Sym coordinates. "
            "This standard affine is an approximate "
            "template bridge and is not claimed to "
            "be an exact nonlinear mapping to "
            "MNI152NLin6Sym."
        ),
}


(
    OUT
    / "ANATOMICAL_ADAPTER.json"
).write_text(
    json.dumps(
        adapter,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# METADATA / REPO PROVENANCE
# ============================================================

dataset_git_commit = git_cmd(
    DATA,
    "rev-parse",
    "HEAD",
)

pcnd_git_commit = git_cmd(
    ROOT,
    "rev-parse",
    "HEAD",
)

pcnd_status = git_cmd(
    ROOT,
    "status",
    "--porcelain",
)


# ============================================================
# HARD ANATOMICAL SANITY
# ============================================================

eligible_df = subject_df[
    subject_df[
        "strict_k10_lockbox_eligible"
    ]
]


overall_hemi = (
    float(
        electrode_df.assign(
            hemi_class=
                electrode_df[
                    "hemisphere"
                ].map(
                    hemisphere_class
                )
        )
        .query(
            "hemi_class == 'L' "
            "or hemi_class == 'R'"
        )
        .assign(
            correct=lambda x: np.where(
                x[
                    "hemi_class"
                ]
                ==
                "L",

                x[
                    "adapter_mni_x"
                ]
                <
                0,

                x[
                    "adapter_mni_x"
                ]
                >
                0,
            )
        )[
            "correct"
        ]
        .mean()
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "mode":
        (
            "SEALED"
            if args.seal
            else
            "DRY_RUN"
        ),

    "dataset":
        "OpenNeuro ds004080",

    "dataset_git_commit":
        dataset_git_commit,

    "hapwave_frozen_manifest_sha256":
        hap_hash,

    "pcnd_git_commit":
        pcnd_git_commit,

    "pcnd_git_worktree_clean":
        (
            pcnd_status
            ==
            ""
        ),

    "n_metadata_subjects":
        len(
            subjects
        ),

    "n_strict_k10_eligible":
        int(
            len(
                cohort
            )
        ),

    "minimum_usable_sites":
        (
            int(
                eligible_df[
                    "n_strict_usable_sites"
                ].min()
            )
            if len(
                eligible_df
            )
            else None
        ),

    "median_usable_sites":
        (
            float(
                eligible_df[
                    "n_strict_usable_sites"
                ].median()
            )
            if len(
                eligible_df
            )
            else None
        ),

    "roi_text_exact_overlap_fraction":
        text_overlap,

    "roi_numeric_exact_overlap_fraction":
        numeric_overlap,

    "chosen_roi_representation":
        roi_representation,

    "chosen_roi_exact_overlap_fraction":
        roi_overlap,

    "overall_hemisphere_x_sign_fraction":
        overall_hemi,

    "context_k":
        CONTEXT_K,

    "context_repeats":
        N_CONTEXT_REPEATS,

    "near_field_exclusion_mm":
        PRIMARY_NEAR_FIELD_MM,

    "minimum_site_pulses":
        MIN_SITE_PULSES,

    "minimum_recording_channels_per_site":
        MIN_RECORDING_CHANNELS_PER_SITE,

    "signals_read":
        False,

    "response_derivatives_read":
        False,

    "clinical_outcome_columns_used":
        False,
}


(
    OUT
    / "LOCKBOX_PREPARATION_SUMMARY.json"
).write_text(
    json.dumps(
        summary,
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
    "DS004080 STRICT LOCKBOX "
    + (
        "SEAL"
        if args.seal
        else
        "DRY RUN"
    )
)
print(
    "=" * 120
)


print(
    "metadata subjects:",
    len(
        subjects
    )
)

print(
    "strict k=10 eligible:",
    len(
        cohort
    )
)

if len(
    eligible_df
):

    print(
        "usable sites min / median / max:",
        int(
            eligible_df[
                "n_strict_usable_sites"
            ].min()
        ),
        "/",
        round(
            float(
                eligible_df[
                    "n_strict_usable_sites"
                ].median()
            ),
            1,
        ),
        "/",
        int(
            eligible_df[
                "n_strict_usable_sites"
            ].max()
        ),
    )


print()
print(
    "ROI COMPATIBILITY"
)

print(
    "text exact overlap:",
    round(
        float(
            text_overlap
        ),
        4,
    )
    if np.isfinite(
        text_overlap
    )
    else "NaN"
)

print(
    "numeric exact overlap:",
    round(
        float(
            numeric_overlap
        ),
        4,
    )
    if np.isfinite(
        numeric_overlap
    )
    else "NaN"
)

print(
    "chosen representation:",
    roi_representation
)

print(
    "chosen exact overlap:",
    round(
        float(
            roi_overlap
        ),
        4,
    )
    if np.isfinite(
        roi_overlap
    )
    else "NaN"
)


print()
print(
    "ANATOMICAL SANITY"
)

print(
    "overall hemisphere/x sign agreement:",
    round(
        overall_hemi,
        4,
    )
)


print()
print(
    "STRICT ELIGIBILITY DISTRIBUTION"
)

print(
    subject_df[
        [
            "subject",
            "dominant_protocol_events",
            "n_recording_channels_geometry_valid",
            "n_sites_ge5_dominant_protocol",
            "n_strict_usable_sites",
            "min_recording_channels_per_usable_site",
            "hemisphere_x_sign_fraction",
            "strict_k10_lockbox_eligible",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print(
    "NO SIGNALS READ."
)

print(
    "NO RESPONSE DERIVATIVES READ."
)

print(
    "NO SOZ / RESECTION FIELDS USED."
)


# ============================================================
# SEAL
# ============================================================

if args.seal:

    if (
        pcnd_status
        is not None
        and
        pcnd_status
        !=
        ""
    ):

        raise RuntimeError(
            "Refusing final seal: "
            "PCND git worktree is dirty."
        )


    if (
        len(
            cohort
        )
        == 0
    ):

        raise RuntimeError(
            "No eligible subjects."
        )


    if (
        overall_hemi
        <
        0.95
    ):

        raise RuntimeError(
            "Hemisphere coordinate sanity failed."
        )


    output_files = [
        OUT
        / "subject_strict_eligibility.csv",

        OUT
        / "recording_channel_inventory.csv",

        OUT
        / "strict_usable_stimulation_sites.csv",

        OUT
        / "frozen_context_query_plan.csv",

        OUT
        / "lockbox_cohort.csv",

        OUT
        / "ANATOMICAL_ADAPTER.json",

        OUT
        / "LOCKBOX_PREPARATION_SUMMARY.json",
    ]


    file_hashes = {
        str(
            p.name
        ):
            sha256_file(
                p
            )
        for p in output_files
    }


    manifest = {
        "freeze_version":
            "ds004080-lockbox-v1.0",

        "freeze_time_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "dataset":
            "OpenNeuro ds004080",

        "dataset_git_commit":
            dataset_git_commit,

        "development_protocol_sha256":
            hap_hash,

        "pcnd_git_commit":
            pcnd_git_commit,

        "cohort_rule":
            (
                "All subjects satisfying pre-outcome "
                "strict metadata/anatomical eligibility."
            ),

        "n_frozen_subjects":
            int(
                len(
                    cohort
                )
            ),

        "primary_endpoint":
            {
                "response":
                    "late",

                "context_k":
                    10,

                "repeats":
                    20,

                "metric":
                    "patient-level Spearman",

                "contrast":
                    (
                        "relation minus "
                        "within-recording-channel "
                        "stimulation permutation"
                    ),
            },

        "adapter":
            adapter,

        "eligibility":
            {
                "dominant_spes_protocol_per_subject":
                    True,

                "minimum_repeats_per_site":
                    MIN_SITE_PULSES,

                "minimum_usable_sites":
                    MIN_USABLE_SITES_K10,

                "minimum_recording_channels_per_site":
                    MIN_RECORDING_CHANNELS_PER_SITE,

                "near_field_exclusion_mm":
                    PRIMARY_NEAR_FIELD_MM,

                "exact_stimulation_contacts_excluded":
                    True,
            },

        "context_plan":
            {
                "k":
                    CONTEXT_K,

                "repeats":
                    N_CONTEXT_REPEATS,

                "seed_root":
                    LOCKBOX_CONTEXT_SEED,

                "actual_context_and_query_lists_frozen":
                    True,
            },

        "blindness":
            {
                "signals_read":
                    False,

                "response_derivatives_read":
                    False,

                "soz_or_resection_used":
                    False,
            },

        "files":
            file_hashes,
    }


    manifest_path = (
        OUT
        / "DS004080_LOCKBOX_MANIFEST.json"
    )


    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


    manifest_sha = (
        sha256_file(
            manifest_path
        )
    )


    seal = {
        "freeze_version":
            "ds004080-lockbox-v1.0",

        "manifest_sha256":
            manifest_sha,

        "development_protocol_sha256":
            hap_hash,

        "dataset_git_commit":
            dataset_git_commit,

        "pcnd_git_commit":
            pcnd_git_commit,
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
        "DS004080 LOCKBOX FROZEN"
    )
    print(
        "=" * 120
    )

    print(
        "subjects:",
        len(
            cohort
        )
    )

    print(
        "manifest SHA256:",
        manifest_sha
    )

    print(
        "development SHA256:",
        hap_hash
    )

    print(
        "dataset git commit:",
        dataset_git_commit
    )

    print(
        "PCND git commit:",
        pcnd_git_commit
    )

else:

    print()
    print(
        "DRY RUN ONLY — NO FINAL LOCKBOX SEAL WRITTEN."
    )

    print(
        "Review compatibility before rerunning with --seal."
    )


print()
print(
    "Saved:",
    OUT
)
