from __future__ import annotations

from pathlib import Path
import os
import re

import numpy as np
import pandas as pd


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

MNI_FILE = (
    ROOT
    / "data"
    / "ds004696"
    / "derivatives"
    / "MNI"
    / "sub-all_ses-ieeg01_space-MNI152NLin6Sym_electrodes.tsv"
)

META_ROOT = (
    ROOT
    / "results"
    / "clinical_v1"
    / "manifests"
)

MAP_ROOT = (
    ROOT
    / "results"
    / "clinical_v1"
    / "maps"
)

OUT = (
    ROOT
    / "results"
    / "clinical_v4"
    / "mni_harmonized"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def pick_column(
    columns,
    candidates,
    required=True,
):
    lower = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[
                candidate.lower()
            ]

    if required:
        raise RuntimeError(
            "Could not find any of "
            f"{candidates} in columns:\n"
            f"{list(columns)}"
        )

    return None


def normalize_subject(x):
    s = str(x).strip()

    # already BIDS style
    m = re.search(
        r"sub[-_]?(\d+)",
        s,
        flags=re.I,
    )

    if m:
        return (
            "sub-"
            + m.group(1).zfill(2)
        )

    # bare integer
    try:
        return (
            "sub-"
            + str(
                int(float(s))
            ).zfill(2)
        )
    except Exception:
        pass

    return s


def normalize_contact(x):
    return str(x).strip()


# ============================================================
# LOAD MNI TABLE
# ============================================================

mni = pd.read_csv(
    MNI_FILE,
    sep="\t"
)

print()
print("=" * 100)
print("MNI SOURCE")
print("=" * 100)

print(
    "shape:",
    mni.shape
)

print(
    "columns:",
    mni.columns.tolist()
)


subject_col = pick_column(
    mni.columns,
    [
        "subject",
        "subject_id",
        "participant_id",
        "participant",
        "sub",
        "patient",
        "sub_code",
    ]
)

name_col = pick_column(
    mni.columns,
    [
        "name",
        "electrode",
        "contact",
        "label",
        "channel",
    ]
)

x_col = pick_column(
    mni.columns,
    ["x"]
)

y_col = pick_column(
    mni.columns,
    ["y"]
)

z_col = pick_column(
    mni.columns,
    ["z"]
)


print(
    "subject column:",
    subject_col
)

print(
    "contact column:",
    name_col
)

print(
    "xyz columns:",
    x_col,
    y_col,
    z_col
)


mni2 = pd.DataFrame({
    "subject":
        mni[
            subject_col
        ].map(
            normalize_subject
        ),

    "name":
        mni[
            name_col
        ].map(
            normalize_contact
        ),

    "mni_x":
        pd.to_numeric(
            mni[x_col],
            errors="coerce",
        ),

    "mni_y":
        pd.to_numeric(
            mni[y_col],
            errors="coerce",
        ),

    "mni_z":
        pd.to_numeric(
            mni[z_col],
            errors="coerce",
        ),

    "hemisphere":
        mni[
            "hemisphere"
        ].astype(str),

    "seizure_zone":
        mni[
            "seizure_zone"
        ],

    "destrieux_label":
        pd.to_numeric(
            mni[
                "Destrieux_label"
            ],
            errors="coerce",
        ),

    "destrieux_label_text":
        mni[
            "Destrieux_label_text"
        ],
})


# remove exact duplicate rows
mni2 = (
    mni2
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)


# ============================================================
# DUPLICATE AUDIT
# ============================================================

dups = (
    mni2
    .groupby(
        [
            "subject",
            "name",
        ]
    )
    .size()
    .reset_index(
        name="n"
    )
)

dups = dups[
    dups["n"] > 1
]

if len(dups):

    print()
    print("=" * 100)
    print("WARNING: DUPLICATE SUBJECT-CONTACT KEYS")
    print("=" * 100)

    print(
        dups
        .head(50)
        .to_string(
            index=False
        )
    )

    raise RuntimeError(
        "MNI table contains duplicate "
        "(subject, name) keys."
    )


# ============================================================
# SUBJECT AUDIT
# ============================================================

print()
print("=" * 100)
print("MNI SUBJECT COUNTS")
print("=" * 100)

print(
    mni2
    .groupby(
        "subject"
    )[
        "name"
    ]
    .nunique()
    .to_string()
)


SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(
        1,
        9
    )
]


summary_rows = []


# ============================================================
# PER SUBJECT
# ============================================================

for sub in SUBJECTS:

    print()
    print("=" * 100)
    print(sub)
    print("=" * 100)

    sub_mni = (
        mni2[
            mni2[
                "subject"
            ] == sub
        ]
        .copy()
    )

    if len(sub_mni) == 0:
        raise RuntimeError(
            f"No MNI coordinates found for {sub}"
        )


    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    nodes = pd.read_csv(
        META_ROOT
        / f"{sub}_nodes.csv"
    )

    nodes[
        "name"
    ] = nodes[
        "name"
    ].map(
        normalize_contact
    )


    nodes_mni = (
        nodes
        .merge(
            sub_mni,
            on=[
                "name"
            ],
            how="left",
            validate="one_to_one",
        )
    )


    nodes_mni[
        "mni_available"
    ] = (
        nodes_mni[
            [
                "mni_x",
                "mni_y",
                "mni_z",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )


    nodes_mni.to_csv(
        OUT
        / f"{sub}_nodes_mni.csv",
        index=False
    )


    # --------------------------------------------------------
    # STIM SITES
    # --------------------------------------------------------

    sites = pd.read_csv(
        META_ROOT
        / f"{sub}_sites.csv"
    )

    lookup = (
        sub_mni
        .set_index(
            "name"
        )[
            [
                "mni_x",
                "mni_y",
                "mni_z",
            ]
        ]
    )


    site_rows = []


    for _, row in sites.iterrows():

        a = normalize_contact(
            row["stim_a"]
        )

        b = normalize_contact(
            row["stim_b"]
        )


        if (
            a in lookup.index
            and
            b in lookup.index
        ):

            ra = (
                lookup.loc[
                    a
                ]
                .to_numpy(
                    dtype=float
                )
            )

            rb = (
                lookup.loc[
                    b
                ]
                .to_numpy(
                    dtype=float
                )
            )

            ok = (
                np.isfinite(
                    ra
                ).all()
                and
                np.isfinite(
                    rb
                ).all()
            )

        else:

            ok = False


        if ok:

            midpoint = (
                ra + rb
            ) / 2.0

            vec = (
                rb - ra
            )

            norm = float(
                np.linalg.norm(
                    vec
                )
            )

            if norm > 0:
                orient = (
                    vec / norm
                )
            else:
                orient = np.full(
                    3,
                    np.nan
                )


            mx, my, mz = midpoint
            ox, oy, oz = orient

            contact_distance_mni = norm

        else:

            mx = my = mz = np.nan
            ox = oy = oz = np.nan

            contact_distance_mni = np.nan


        rec = row.to_dict()

        rec.update({
            "mni_mid_x": mx,
            "mni_mid_y": my,
            "mni_mid_z": mz,

            "mni_ori_x": ox,
            "mni_ori_y": oy,
            "mni_ori_z": oz,

            "mni_contact_distance":
                contact_distance_mni,

            "mni_geometry_available":
                bool(ok),
        })

        site_rows.append(
            rec
        )


    sites_mni = pd.DataFrame(
        site_rows
    )


    sites_mni.to_csv(
        OUT
        / f"{sub}_sites_mni.csv",
        index=False
    )


    # --------------------------------------------------------
    # PERTURBATION MAP
    # --------------------------------------------------------

    map_df = pd.read_csv(
        MAP_ROOT
        / f"{sub}_perturbation_map.csv.gz"
    )


    node_geo = (
        nodes_mni[
            [
                "name",
                "mni_x",
                "mni_y",
                "mni_z",
            ]
        ]
        .rename(
            columns={
                "name":
                    "recording_channel",

                "mni_x":
                    "rec_mni_x",

                "mni_y":
                    "rec_mni_y",

                "mni_z":
                    "rec_mni_z",
            }
        )
    )


    site_geo = (
        sites_mni[
            [
                "stim_site",

                "mni_mid_x",
                "mni_mid_y",
                "mni_mid_z",

                "mni_ori_x",
                "mni_ori_y",
                "mni_ori_z",

                "mni_geometry_available",
            ]
        ]
    )


    harmonized = (
        map_df
        .merge(
            node_geo,
            on="recording_channel",
            how="left",
            validate="many_to_one",
        )
        .merge(
            site_geo,
            on="stim_site",
            how="left",
            validate="many_to_one",
        )
    )


    harmonized[
        "mni_dx"
    ] = (
        harmonized[
            "rec_mni_x"
        ]
        -
        harmonized[
            "mni_mid_x"
        ]
    )

    harmonized[
        "mni_dy"
    ] = (
        harmonized[
            "rec_mni_y"
        ]
        -
        harmonized[
            "mni_mid_y"
        ]
    )

    harmonized[
        "mni_dz"
    ] = (
        harmonized[
            "rec_mni_z"
        ]
        -
        harmonized[
            "mni_mid_z"
        ]
    )


    harmonized[
        "mni_pair_distance"
    ] = np.sqrt(
        harmonized[
            "mni_dx"
        ] ** 2
        +
        harmonized[
            "mni_dy"
        ] ** 2
        +
        harmonized[
            "mni_dz"
        ] ** 2
    )


    harmonized[
        "mni_pair_available"
    ] = (
        harmonized[
            [
                "rec_mni_x",
                "rec_mni_y",
                "rec_mni_z",

                "mni_mid_x",
                "mni_mid_y",
                "mni_mid_z",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )


    # IMPORTANT:
    # Do NOT replace native distance_to_stim.
    # distance_to_stim remains the physical within-patient
    # distance used for near-field exclusion.
    #
    # mni_pair_distance is only a harmonized anatomical model
    # feature.

    harmonized.to_csv(
        OUT
        / f"{sub}_perturbation_map_mni.csv.gz",
        index=False,
        compression="gzip",
    )


    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    n_nodes = len(
        nodes_mni
    )

    n_nodes_mni = int(
        nodes_mni[
            "mni_available"
        ].sum()
    )

    n_sites = len(
        sites_mni
    )

    n_sites_mni = int(
        sites_mni[
            "mni_geometry_available"
        ].sum()
    )

    n_pairs = len(
        harmonized
    )

    n_pairs_mni = int(
        harmonized[
            "mni_pair_available"
        ].sum()
    )


    summary_rows.append({
        "subject":
            sub,

        "n_nodes":
            n_nodes,

        "n_nodes_mni":
            n_nodes_mni,

        "node_coverage":
            (
                n_nodes_mni
                / n_nodes
                if n_nodes
                else np.nan
            ),

        "n_sites":
            n_sites,

        "n_sites_mni":
            n_sites_mni,

        "site_coverage":
            (
                n_sites_mni
                / n_sites
                if n_sites
                else np.nan
            ),

        "n_pairs":
            n_pairs,

        "n_pairs_mni":
            n_pairs_mni,

        "pair_coverage":
            (
                n_pairs_mni
                / n_pairs
                if n_pairs
                else np.nan
            ),
    })


    print(
        "good SEEG nodes:",
        n_nodes,
        "| MNI:",
        n_nodes_mni,
        "| coverage:",
        round(
            n_nodes_mni
            / n_nodes,
            4
        )
    )

    print(
        "stim sites:",
        n_sites,
        "| MNI:",
        n_sites_mni,
        "| coverage:",
        round(
            n_sites_mni
            / n_sites,
            4
        )
    )

    print(
        "map pairs:",
        n_pairs,
        "| MNI:",
        n_pairs_mni,
        "| coverage:",
        round(
            n_pairs_mni
            / n_pairs,
            4
        )
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT
    / "mni_harmonization_summary.csv",
    index=False
)


print()
print("=" * 100)
print("FINAL MNI COVERAGE")
print("=" * 100)

print(
    summary
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print()
print(
    "Saved to:",
    OUT
)
