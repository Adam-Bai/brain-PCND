from __future__ import annotations

from pathlib import Path
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import json
import re

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
    / "ds004080_anatomical_interface_audit_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


SUBJECTS = sorted(
    p.name
    for p in DATA.glob("sub-*")
    if p.is_dir()
)


FORBIDDEN_SUFFIXES = {
    ".eeg",
    ".edf",
    ".bdf",
    ".set",
    ".fdt",
    ".mef",
    ".mefd",
}


def read_tsv(path: Path):

    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise RuntimeError(
            f"FORBIDDEN SIGNAL READ: {path}"
        )

    if "derivatives" in [
        p.lower()
        for p in path.parts
    ]:
        raise RuntimeError(
            f"FORBIDDEN DERIVATIVE CONTENT READ: {path}"
        )

    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )


def read_json(path: Path):

    if "derivatives" in [
        p.lower()
        for p in path.parts
    ]:
        raise RuntimeError(
            f"FORBIDDEN DERIVATIVE CONTENT READ: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def good_string(x):

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


def normalize_name(x):

    return (
        str(x)
        .strip()
        .replace(" ", "")
    )


def parse_stim_pair(
    value,
    electrode_names,
):

    """
    Conservative metadata-only parser.

    Return exactly two electrode names when the
    stimulation label can be unambiguously matched.
    """

    raw = normalize_name(
        value
    )

    if not raw:
        return None

    names = set(
        electrode_names
    )

    # Common separators.
    pieces = [
        x
        for x in re.split(
            r"[-_;,+/\\\s]+",
            raw,
        )
        if x
    ]

    exact = [
        x
        for x in pieces
        if x in names
    ]

    if len(exact) == 2:
        return tuple(exact)

    # Case-insensitive exact matching.
    lower_map = {
        x.lower(): x
        for x in names
    }

    exact_lower = []

    for x in pieces:

        y = lower_map.get(
            x.lower()
        )

        if y is not None:
            exact_lower.append(
                y
            )

    if len(exact_lower) == 2:
        return tuple(
            exact_lower
        )

    # Last conservative fallback:
    # identify complete electrode names occurring in string.
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

    # unique preserving order
    unique = []

    for x in hits:

        if x not in unique:
            unique.append(x)

    if len(unique) == 2:
        return tuple(unique)

    return None


# ============================================================
# DATASET-WIDE PATH NAME AUDIT
# ============================================================

candidate_paths = []

keywords = [
    "mni",
    "destrieux",
    "atlas",
    "freesurfer",
    "electrode",
]

for p in DATA.rglob("*"):

    rel = str(
        p.relative_to(DATA)
    )

    low = rel.lower()

    if any(
        k in low
        for k in keywords
    ):
        candidate_paths.append({
            "path":
                rel,

            "is_derivative":
                "derivatives"
                in low,

            "suffix":
                p.suffix,
        })


pd.DataFrame(
    candidate_paths
).to_csv(
    OUT
    / "candidate_anatomical_paths.csv",
    index=False,
)


# ============================================================
# SUBJECT AUDIT
# ============================================================

subject_rows = []

electrode_schema_rows = []

ieeg_meta_rows = []

unparsed_rows = []

dataset_column_counter = Counter()


for sub in SUBJECTS:

    subdir = (
        DATA
        / sub
    )

    electrode_files = sorted(
        subdir.glob(
            "ses-*/ieeg/*_electrodes.tsv"
        )
    )

    event_files = sorted(
        subdir.glob(
            "ses-*/ieeg/*_events.tsv"
        )
    )

    coord_files = sorted(
        subdir.glob(
            "ses-*/ieeg/*_coordsystem.json"
        )
    )

    ieeg_files = sorted(
        subdir.glob(
            "ses-*/ieeg/*_ieeg.json"
        )
    )


    # ========================================================
    # ELECTRODES
    # ========================================================

    electrode_frames = []

    for f in electrode_files:

        e = read_tsv(f)

        dataset_column_counter.update(
            e.columns.tolist()
        )

        electrode_schema_rows.append({
            "subject":
                sub,

            "file":
                str(
                    f.relative_to(DATA)
                ),

            "columns":
                "|".join(
                    e.columns.astype(str)
                ),
        })

        electrode_frames.append(e)


    if electrode_frames:

        electrodes = pd.concat(
            electrode_frames,
            ignore_index=True,
        )

        if "name" not in electrodes.columns:

            raise RuntimeError(
                f"{sub}: electrodes.tsv lacks name"
            )

        electrodes[
            "name"
        ] = (
            electrodes[
                "name"
            ]
            .astype(str)
            .map(
                normalize_name
            )
        )

        electrodes = (
            electrodes
            .drop_duplicates(
                subset=[
                    "name",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    else:

        electrodes = pd.DataFrame()


    electrode_names = (
        set(
            electrodes[
                "name"
            ].tolist()
        )
        if len(electrodes)
        else set()
    )


    # ========================================================
    # COORDSYSTEM
    # ========================================================

    coord_systems = []
    coord_units = []
    coord_descriptions = []

    for f in coord_files:

        c = read_json(f)

        coord_systems.append(
            str(
                c.get(
                    "iEEGCoordinateSystem",
                    "UNKNOWN",
                )
            )
        )

        coord_units.append(
            str(
                c.get(
                    "iEEGCoordinateUnits",
                    "UNKNOWN",
                )
            )
        )

        coord_descriptions.append(
            str(
                c.get(
                    "iEEGCoordinateSystemDescription",
                    "",
                )
            )
        )


    # ========================================================
    # COORDINATE COLUMN AUDIT
    # ========================================================

    columns = (
        electrodes.columns.tolist()
        if len(electrodes)
        else []
    )

    lower_col = {
        c.lower(): c
        for c in columns
    }


    has_xyz = all(
        c in lower_col
        for c in [
            "x",
            "y",
            "z",
        ]
    )


    if has_xyz:

        xyz_cols = [
            lower_col["x"],
            lower_col["y"],
            lower_col["z"],
        ]

        xyz = (
            electrodes[
                xyz_cols
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        xyz_ok = (
            xyz
            .notna()
            .all(
                axis=1
            )
        )

        n_xyz = int(
            xyz_ok.sum()
        )

    else:

        n_xyz = 0


    # Find explicit MNI columns if present.
    mni_columns = [
        c
        for c in columns
        if (
            "mni"
            in c.lower()
        )
    ]


    # Detect common MNI triplets.
    candidate_triplets = []

    candidate_sets = [
        (
            "mni_x",
            "mni_y",
            "mni_z",
        ),
        (
            "mni152_x",
            "mni152_y",
            "mni152_z",
        ),
        (
            "x_mni",
            "y_mni",
            "z_mni",
        ),
        (
            "x_mni152",
            "y_mni152",
            "z_mni152",
        ),
    ]

    for triplet in candidate_sets:

        if all(
            c in lower_col
            for c in triplet
        ):

            candidate_triplets.append(
                "|".join(
                    lower_col[c]
                    for c in triplet
                )
            )


    # Atlas / ROI columns.
    atlas_columns = [
        c
        for c in columns
        if any(
            key in c.lower()
            for key in [
                "destrieux",
                "atlas",
                "roi",
                "region",
                "label",
                "aparc",
            ]
        )
    ]


    # ========================================================
    # STIMULATION SITE -> ELECTRODE MATCHING
    # ========================================================

    stimulation_labels = []

    for f in event_files:

        ev = read_tsv(f)

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

        if "sub_type" in ev.columns:

            st = (
                ev[
                    "sub_type"
                ]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            if (
                st
                ==
                "SPES"
            ).any():

                mask &= (
                    st
                    ==
                    "SPES"
                )

        stimulation_labels.extend(
            ev.loc[
                mask,
                "electrical_stimulation_site",
            ]
            .astype(str)
            .tolist()
        )


    unique_stim = sorted(
        set(
            stimulation_labels
        )
    )

    parsed = {}

    for stim in unique_stim:

        pair = parse_stim_pair(
            stim,
            electrode_names,
        )

        parsed[
            stim
        ] = pair

        if pair is None:

            unparsed_rows.append({
                "subject":
                    sub,

                "stimulation_site":
                    stim,
            })


    n_parsed = sum(
        x is not None
        for x in parsed.values()
    )


    stim_pair_match_fraction = (
        n_parsed
        /
        len(
            unique_stim
        )
        if unique_stim
        else np.nan
    )


    # ========================================================
    # IEEG METADATA ONLY
    # ========================================================

    sampling_freqs = []
    line_freqs = []
    references = []
    manufacturers = []

    for f in ieeg_files:

        m = read_json(f)

        sampling_freqs.append(
            m.get(
                "SamplingFrequency",
                None,
            )
        )

        line_freqs.append(
            m.get(
                "PowerLineFrequency",
                None,
            )
        )

        references.append(
            m.get(
                "iEEGReference",
                None,
            )
        )

        manufacturers.append(
            m.get(
                "Manufacturer",
                None,
            )
        )

        ieeg_meta_rows.append({
            "subject":
                sub,

            "file":
                str(
                    f.relative_to(DATA)
                ),

            "SamplingFrequency":
                m.get(
                    "SamplingFrequency",
                    None,
                ),

            "PowerLineFrequency":
                m.get(
                    "PowerLineFrequency",
                    None,
                ),

            "iEEGReference":
                m.get(
                    "iEEGReference",
                    None,
                ),

            "Manufacturer":
                m.get(
                    "Manufacturer",
                    None,
                ),
        })


    # ========================================================
    # INTERFACE CLASSIFICATION
    # ========================================================

    systems_lower = {
        x.lower()
        for x in coord_systems
    }

    direct_mni_system = any(
        (
            "mni152"
            in x
            or
            "mni152lin"
            in x
            or
            "mni152nlin"
            in x
        )
        for x in systems_lower
    )


    direct_mni_columns = (
        len(
            candidate_triplets
        )
        >
        0
    )


    fsaverage_system = (
        "fsaverage"
        in systems_lower
    )


    unit_mm = all(
        str(x).lower()
        in {
            "mm",
            "millimeter",
            "millimeters",
        }
        for x in coord_units
        if x not in {
            None,
            "UNKNOWN",
        }
    )


    subject_rows.append({
        "subject":
            sub,

        "n_electrodes":
            len(
                electrodes
            ),

        "n_xyz":
            n_xyz,

        "coordinate_systems":
            "|".join(
                sorted(
                    set(
                        coord_systems
                    )
                )
            ),

        "coordinate_units":
            "|".join(
                sorted(
                    set(
                        coord_units
                    )
                )
            ),

        "direct_mni_system":
            direct_mni_system,

        "explicit_mni_columns":
            "|".join(
                mni_columns
            ),

        "candidate_mni_triplets":
            "|".join(
                candidate_triplets
            ),

        "direct_mni_columns":
            direct_mni_columns,

        "fsaverage_system":
            fsaverage_system,

        "unit_mm":
            unit_mm,

        "atlas_columns":
            "|".join(
                atlas_columns
            ),

        "n_unique_stim_sites":
            len(
                unique_stim
            ),

        "n_stim_sites_pair_parsed":
            n_parsed,

        "stim_pair_match_fraction":
            stim_pair_match_fraction,

        "sampling_frequencies":
            "|".join(
                sorted(
                    {
                        str(x)
                        for x in sampling_freqs
                        if x is not None
                    }
                )
            ),

        "power_line_frequencies":
            "|".join(
                sorted(
                    {
                        str(x)
                        for x in line_freqs
                        if x is not None
                    }
                )
            ),

        "references":
            "|".join(
                sorted(
                    {
                        str(x)
                        for x in references
                        if x is not None
                    }
                )
            ),

        "manufacturers":
            "|".join(
                sorted(
                    {
                        str(x)
                        for x in manufacturers
                        if x is not None
                    }
                )
            ),
    })


subject_df = pd.DataFrame(
    subject_rows
)


pd.DataFrame(
    electrode_schema_rows
).to_csv(
    OUT
    / "electrode_schema_inventory.csv",
    index=False,
)


pd.DataFrame(
    ieeg_meta_rows
).to_csv(
    OUT
    / "ieeg_metadata_inventory.csv",
    index=False,
)


pd.DataFrame(
    unparsed_rows
).to_csv(
    OUT
    / "unparsed_stimulation_sites.csv",
    index=False,
)


subject_df.to_csv(
    OUT
    / "subject_anatomical_interface.csv",
    index=False,
)


# ============================================================
# DATASET-WIDE SCHEMA SUMMARY
# ============================================================

schema_summary = pd.DataFrame(
    [
        {
            "column":
                col,

            "n_files_with_column":
                count,
        }
        for col, count
        in sorted(
            dataset_column_counter.items()
        )
    ]
)


schema_summary.to_csv(
    OUT
    / "dataset_electrode_schema_summary.csv",
    index=False,
)


# ============================================================
# GLOBAL DECISION
# ============================================================

n_subjects = len(
    subject_df
)

n_fsaverage = int(
    subject_df[
        "fsaverage_system"
    ].sum()
)

n_direct_mni = int(
    (
        subject_df[
            "direct_mni_system"
        ]
        |
        subject_df[
            "direct_mni_columns"
        ]
    ).sum()
)

n_unit_mm = int(
    subject_df[
        "unit_mm"
    ].sum()
)

n_pair_ge_98 = int(
    (
        subject_df[
            "stim_pair_match_fraction"
        ]
        >=
        0.98
    ).sum()
)


atlas_nonempty = (
    subject_df[
        "atlas_columns"
    ]
    .astype(str)
    .str.len()
    >
    0
)

n_atlas = int(
    atlas_nonempty.sum()
)


summary = {
    "audit_time_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "n_subjects":
        n_subjects,

    "n_fsaverage":
        n_fsaverage,

    "n_direct_mni":
        n_direct_mni,

    "n_coordinate_units_mm":
        n_unit_mm,

    "n_stim_pair_match_ge_0_98":
        n_pair_ge_98,

    "n_with_atlas_like_columns":
        n_atlas,

    "signals_read":
        False,

    "response_derivatives_read":
        False,

    "decision_rule":
        (
            "Do not open signal data until a deterministic "
            "outcome-blind anatomical interface to the frozen "
            "V3f feature space has been selected and sealed."
        ),
}


(
    OUT
    / "ANATOMICAL_INTERFACE_AUDIT.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print(
    "=" * 115
)
print(
    "DS004080 ANATOMICAL INTERFACE AUDIT"
)
print(
    "=" * 115
)


print()
print(
    "subjects:",
    n_subjects
)

print(
    "fsaverage coordinate system:",
    n_fsaverage,
    "/",
    n_subjects,
)

print(
    "direct MNI available:",
    n_direct_mni,
    "/",
    n_subjects,
)

print(
    "coordinate units = mm:",
    n_unit_mm,
    "/",
    n_subjects,
)

print(
    "stim pair exact-match >=98%:",
    n_pair_ge_98,
    "/",
    n_subjects,
)

print(
    "atlas-like electrode columns:",
    n_atlas,
    "/",
    n_subjects,
)


print()
print(
    "=" * 115
)
print(
    "COORDINATE / ROI SCHEMA"
)
print(
    "=" * 115
)


show_cols = [
    "subject",
    "coordinate_systems",
    "coordinate_units",
    "direct_mni_columns",
    "candidate_mni_triplets",
    "atlas_columns",
    "stim_pair_match_fraction",
    "sampling_frequencies",
    "power_line_frequencies",
    "references",
]


print(
    subject_df[
        show_cols
    ]
    .to_string(
        index=False
    )
)


print()
print(
    "=" * 115
)
print(
    "DATASET ELECTRODE COLUMNS"
)
print(
    "=" * 115
)

print(
    schema_summary
    .to_string(
        index=False
    )
)


print()
print(
    "=" * 115
)
print(
    "CANDIDATE MNI / ATLAS PATHS"
)
print(
    "=" * 115
)


cand = pd.DataFrame(
    candidate_paths
)

if len(cand):

    print(
        cand
        .head(100)
        .to_string(
            index=False
        )
    )

else:

    print(
        "None found by filename."
    )


print()
print(
    "NO SIGNALS READ."
)

print(
    "NO RESPONSE DERIVATIVES READ."
)

print(
    "Saved:",
    OUT
)
