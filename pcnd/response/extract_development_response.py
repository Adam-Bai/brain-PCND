from pathlib import Path
import os
import argparse
import json

import numpy as np
import pandas as pd
import mef3io


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

DATA = ROOT / "data" / "ds004696"
META = ROOT / "results" / "clinical_v1" / "manifests"
OUT = ROOT / "results" / "clinical_v1" / "maps"

OUT.mkdir(
    parents=True,
    exist_ok=True
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--subject",
    default="sub-04"
)

parser.add_argument(
    "--max-channels",
    type=int,
    default=None
)

args = parser.parse_args()

SUB = args.subject


# ============================================================
# Load metadata
# ============================================================

events = pd.read_csv(
    META / f"{SUB}_events.csv"
)

nodes = pd.read_csv(
    META / f"{SUB}_nodes.csv"
)

sites = pd.read_csv(
    META / f"{SUB}_sites.csv"
)


ieeg_dir = (
    DATA
    / SUB
    / "ses-ieeg01"
    / "ieeg"
)

mefd = next(
    ieeg_dir.glob("*.mefd")
)

coordsystem_files = list(
    ieeg_dir.glob("*coordsystem.json")
)

coord_units = "unknown"

if coordsystem_files:
    with open(
        coordsystem_files[0],
        "r"
    ) as f:
        cs = json.load(f)

    coord_units = str(
        cs.get(
            "iEEGCoordinateUnits",
            "unknown"
        )
    )


if args.max_channels is not None:
    nodes = nodes.iloc[
        :args.max_channels
    ].copy()


print()
print("=" * 80)
print("PERTURBATION MAP BUILDER")
print("=" * 80)

print("subject:", SUB)
print("events:", len(events))
print("sites:", events[
    "electrical_stimulation_site"
].nunique())
print("recording channels:", len(nodes))
print("coordinate units:", coord_units)
print("MEF:", mefd)


# ============================================================
# Geometry lookup
# ============================================================

node_coord = {}

if set(["x", "y", "z"]).issubset(
    nodes.columns
):
    for _, row in nodes.iterrows():
        xyz = np.array(
            [
                pd.to_numeric(
                    row["x"],
                    errors="coerce"
                ),
                pd.to_numeric(
                    row["y"],
                    errors="coerce"
                ),
                pd.to_numeric(
                    row["z"],
                    errors="coerce"
                ),
            ],
            dtype=float
        )

        node_coord[str(row["name"])] = xyz


site_midpoint = {}

for _, row in sites.iterrows():
    site_midpoint[
        str(row["stim_site"])
    ] = np.array(
        [
            row["mid_x"],
            row["mid_y"],
            row["mid_z"],
        ],
        dtype=float
    )


# ============================================================
# Window definitions
# ============================================================

PRE = (-0.500, -0.020)

EARLY = (
    0.010,
    0.100
)

LATE = (
    0.100,
    0.500
)


def robust_scale(x):
    med = np.nanmedian(x)

    mad = np.nanmedian(
        np.abs(x - med)
    )

    scale = 1.4826 * mad

    if (
        not np.isfinite(scale)
        or scale < 1e-6
    ):
        scale = np.nanstd(x)

    if (
        not np.isfinite(scale)
        or scale < 1e-6
    ):
        scale = 1.0

    return med, scale


def rms(x):
    return float(
        np.sqrt(
            np.nanmean(
                x ** 2
            )
        )
    )


# ============================================================
# Build event lookup
# ============================================================

events = events.reset_index(
    drop=True
)

event_records = []

for idx, row in events.iterrows():
    event_records.append({
        "event_idx": idx,
        "event_id": row["event_id"],
        "sample_start": int(
            row["sample_start"]
        ),
        "stim_site": str(
            row[
                "electrical_stimulation_site"
            ]
        ),
        "stim_a": str(
            row["stim_a"]
        ),
        "stim_b": str(
            row["stim_b"]
        ),
        "split": str(
            row["split"]
        ),
        "current_mA": float(
            row["current_mA"]
        ),
    })


# ============================================================
# Extract response phenotypes
# ============================================================

rows = []


with mef3io.Reader(
    str(mefd),
    n_threads=4
) as reader:

    available = set(
        reader.channels
    )

    valid_nodes = [
        str(x)
        for x in nodes["name"]
        if str(x) in available
    ]

    print(
        "channels found in MEF:",
        len(valid_nodes)
    )

    for ci, ch in enumerate(
        valid_nodes
    ):

        info = reader.info(ch)

        fs = float(
            info[
                "sampling_frequency"
            ]
        )

        start_uutc = int(
            info["start_time"]
        )

        end_uutc = int(
            info["end_time"]
        )

        print(
            f"[{ci+1:03d}/{len(valid_nodes):03d}] "
            f"reading {ch}"
        )

        # Read one channel once
        signal = reader.read(
            ch,
            start_uutc,
            end_uutc
        )

        signal = np.asarray(
            signal,
            dtype=np.float32
        )

        xyz = node_coord.get(
            ch,
            np.full(3, np.nan)
        )

        for e in event_records:

            # Do not use stimulating contacts
            if ch in {
                e["stim_a"],
                e["stim_b"]
            }:
                continue

            s0 = e[
                "sample_start"
            ]

            pre0 = s0 + int(
                round(
                    PRE[0] * fs
                )
            )

            pre1 = s0 + int(
                round(
                    PRE[1] * fs
                )
            )

            early0 = s0 + int(
                round(
                    EARLY[0] * fs
                )
            )

            early1 = s0 + int(
                round(
                    EARLY[1] * fs
                )
            )

            late0 = s0 + int(
                round(
                    LATE[0] * fs
                )
            )

            late1 = s0 + int(
                round(
                    LATE[1] * fs
                )
            )

            if (
                pre0 < 0
                or late1 > len(signal)
            ):
                continue

            baseline = signal[
                pre0:pre1
            ]

            early = signal[
                early0:early1
            ]

            late = signal[
                late0:late1
            ]

            if (
                len(baseline) == 0
                or len(early) == 0
                or len(late) == 0
            ):
                continue

            base_center, base_scale = (
                robust_scale(
                    baseline
                )
            )

            early_z = (
                early - base_center
            ) / base_scale

            late_z = (
                late - base_center
            ) / base_scale

            early_peak_idx = int(
                np.nanargmax(
                    np.abs(
                        early_z
                    )
                )
            )

            late_peak_idx = int(
                np.nanargmax(
                    np.abs(
                        late_z
                    )
                )
            )

            early_peak_ms = (
                EARLY[0] * 1000
                + early_peak_idx
                / fs * 1000
            )

            late_peak_ms = (
                LATE[0] * 1000
                + late_peak_idx
                / fs * 1000
            )

            mid = site_midpoint.get(
                e["stim_site"],
                np.full(3, np.nan)
            )

            if (
                np.isfinite(xyz).all()
                and np.isfinite(mid).all()
            ):
                distance = float(
                    np.linalg.norm(
                        xyz - mid
                    )
                )
            else:
                distance = np.nan

            near_field_10 = (
                bool(distance <= 10.0)
                if (
                    np.isfinite(distance)
                    and coord_units.lower()
                    in {"mm", "millimeter", "millimeters"}
                )
                else False
            )

            rows.append({
                "subject": SUB,
                "event_id": e["event_id"],
                "split": e["split"],

                "stim_site": e["stim_site"],
                "stim_a": e["stim_a"],
                "stim_b": e["stim_b"],
                "current_mA": e["current_mA"],

                "recording_channel": ch,

                "distance_to_stim": distance,
                "coordinate_units": coord_units,
                "near_field_10mm": near_field_10,

                "baseline_center_uV": (
                    float(base_center)
                ),
                "baseline_scale_uV": (
                    float(base_scale)
                ),

                "early_rms_z": rms(
                    early_z
                ),

                "late_rms_z": rms(
                    late_z
                ),

                "early_peak_abs_z": float(
                    np.nanmax(
                        np.abs(
                            early_z
                        )
                    )
                ),

                "late_peak_abs_z": float(
                    np.nanmax(
                        np.abs(
                            late_z
                        )
                    )
                ),

                "early_peak_ms": float(
                    early_peak_ms
                ),

                "late_peak_ms": float(
                    late_peak_ms
                ),
            })


trial_df = pd.DataFrame(
    rows
)


# ============================================================
# Aggregate repeated trials into stimulation map
# ============================================================

map_df = (
    trial_df
    .groupby(
        [
            "subject",
            "split",
            "stim_site",
            "recording_channel",
        ],
        as_index=False
    )
    .agg(
        n_trials=(
            "event_id",
            "nunique"
        ),

        distance_to_stim=(
            "distance_to_stim",
            "median"
        ),

        near_field_10mm=(
            "near_field_10mm",
            "max"
        ),

        early_rms_z=(
            "early_rms_z",
            "median"
        ),

        late_rms_z=(
            "late_rms_z",
            "median"
        ),

        early_peak_abs_z=(
            "early_peak_abs_z",
            "median"
        ),

        late_peak_abs_z=(
            "late_peak_abs_z",
            "median"
        ),

        early_peak_ms=(
            "early_peak_ms",
            "median"
        ),

        late_peak_ms=(
            "late_peak_ms",
            "median"
        ),

        early_rms_iqr=(
            "early_rms_z",
            lambda x:
                float(
                    np.percentile(x, 75)
                    - np.percentile(x, 25)
                )
        ),

        late_rms_iqr=(
            "late_rms_z",
            lambda x:
                float(
                    np.percentile(x, 75)
                    - np.percentile(x, 25)
                )
        ),
    )
)


# ============================================================
# Save
# ============================================================

trial_path = (
    OUT
    / f"{SUB}_trial_response_features.csv.gz"
)

map_path = (
    OUT
    / f"{SUB}_perturbation_map.csv.gz"
)

trial_df.to_csv(
    trial_path,
    index=False,
    compression="gzip"
)

map_df.to_csv(
    map_path,
    index=False,
    compression="gzip"
)


print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    "trial rows:",
    len(trial_df)
)

print(
    "map rows:",
    len(map_df)
)

print(
    "sites:",
    map_df[
        "stim_site"
    ].nunique()
)

print(
    "recording channels:",
    map_df[
        "recording_channel"
    ].nunique()
)

print()
print(
    "split counts:"
)

print(
    map_df[
        ["stim_site", "split"]
    ]
    .drop_duplicates()
    ["split"]
    .value_counts()
)

print()
print("trial file:", trial_path)
print("map file:", map_path)

print()
print(
    map_df[
        [
            "stim_site",
            "recording_channel",
            "n_trials",
            "distance_to_stim",
            "early_rms_z",
            "late_rms_z",
            "early_peak_ms",
            "late_peak_ms",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)
