from __future__ import annotations

from pathlib import Path
import os
from collections import Counter, defaultdict
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

HAP_ROOT = (
    ROOT
    / "results"
    / "clinical_v5"
    / "v3_features"
)

DS_ROOT = (
    ROOT
    / "data_external_lockbox"
    / "ds004080_metadata"
)

OUT = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_roi_bridge_audit_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


HAP_ROI_COLS = [
    "stim_a_roi",
    "stim_b_roi",
    "recording_roi",
]


# ============================================================
# STRING NORMALIZATION
# ============================================================

INVALID = {
    "",
    "nan",
    "na",
    "n/a",
    "none",
    "null",
    "unknown",
    "unk",
}


def clean_raw(x):

    if pd.isna(x):
        return None

    s = str(x).strip()

    if (
        not s
        or
        s.lower()
        in INVALID
    ):
        return None

    return s


def ascii_norm(s):

    s = unicodedata.normalize(
        "NFKD",
        s,
    )

    s = "".join(
        c
        for c in s
        if not unicodedata.combining(c)
    )

    return s


def infer_hemi(raw):

    s = ascii_norm(
        str(raw)
    ).lower()

    # Explicit whole words / prefixes.
    left_patterns = [
        r"(^|[^a-z])left([^a-z]|$)",
        r"^lh([^a-z]|$)",
        r"^lh[_\-.]",
        r"^ctx[-_.]?lh[-_.]",
        r"^wm[-_.]?lh[-_.]",
        r"[-_.]lh$",
    ]

    right_patterns = [
        r"(^|[^a-z])right([^a-z]|$)",
        r"^rh([^a-z]|$)",
        r"^rh[_\-.]",
        r"^ctx[-_.]?rh[-_.]",
        r"^wm[-_.]?rh[-_.]",
        r"[-_.]rh$",
    ]

    for p in left_patterns:

        if re.search(
            p,
            s,
        ):
            return "L"

    for p in right_patterns:

        if re.search(
            p,
            s,
        ):
            return "R"

    return None


def canonical_stem(raw):

    s = ascii_norm(
        str(raw)
    ).lower().strip()

    # Remove common FreeSurfer namespace prefixes.
    s = re.sub(
        r"^(ctx|wm)[-_.]?(lh|rh)[-_.]?",
        "",
        s,
    )

    # Remove textual hemispheres.
    s = re.sub(
        r"\b(left|right)\b",
        "",
        s,
    )

    # Remove lh/rh only when they act as namespace tokens.
    s = re.sub(
        r"^(lh|rh)[-_.]?",
        "",
        s,
    )

    s = re.sub(
        r"[-_.]?(lh|rh)$",
        "",
        s,
    )

    # Normalize separators only.
    #
    # No semantic synonym substitution is allowed
    # in this audit.
    s = re.sub(
        r"[^a-z0-9]+",
        "",
        s,
    )

    if (
        not s
        or
        s in INVALID
    ):
        return None

    return s


def strict_key(raw):

    c = clean_raw(raw)

    if c is None:
        return None

    return (
        infer_hemi(c),
        canonical_stem(c),
    )


# ============================================================
# HAPWAVE VOCAB
# ============================================================

hap_counts = Counter()


for f in sorted(
    HAP_ROOT.glob(
        "sub-*_pcnd_v3_pairs.csv.gz"
    )
):

    header = pd.read_csv(
        f,
        nrows=0,
    )

    cols = [
        c
        for c in HAP_ROI_COLS
        if c in header.columns
    ]

    x = pd.read_csv(
        f,
        usecols=cols,
    )

    for c in cols:

        for value in x[c]:

            value = clean_raw(
                value
            )

            if value is not None:

                hap_counts[
                    value
                ] += 1


# ============================================================
# DS004080 VOCAB
# ============================================================

ds_text_counts = Counter()

ds_numeric_counts = Counter()

ds_rows = []


for f in sorted(
    DS_ROOT.glob(
        "sub-*/ses-*/ieeg/*_electrodes.tsv"
    )
):

    subject = next(
        p
        for p in f.parts
        if p.startswith(
            "sub-"
        )
    )

    header = pd.read_csv(
        f,
        sep="\t",
        nrows=0,
    )

    wanted = [
        c
        for c in [
            "name",
            "hemisphere",
            "Destrieux_label",
            "Destrieux_label_text",
        ]
        if c in header.columns
    ]

    x = pd.read_csv(
        f,
        sep="\t",
        usecols=wanted,
        low_memory=False,
    )

    for _, r in x.iterrows():

        text = clean_raw(
            r.get(
                "Destrieux_label_text"
            )
        )

        numeric = clean_raw(
            r.get(
                "Destrieux_label"
            )
        )

        hemi_meta = clean_raw(
            r.get(
                "hemisphere"
            )
        )

        if text is not None:

            ds_text_counts[
                text
            ] += 1

        if numeric is not None:

            ds_numeric_counts[
                numeric
            ] += 1

        ds_rows.append({
            "subject":
                subject,

            "electrode":
                r.get(
                    "name",
                    None,
                ),

            "hemisphere_metadata":
                hemi_meta,

            "roi_text":
                text,

            "roi_numeric":
                numeric,

            "roi_text_inferred_hemi":
                (
                    infer_hemi(text)
                    if text
                    else None
                ),

            "roi_text_stem":
                (
                    canonical_stem(
                        text
                    )
                    if text
                    else None
                ),
        })


# ============================================================
# BUILD CANONICAL INDICES
# ============================================================

hap_key_to_tokens = defaultdict(
    set
)

hap_stem_to_tokens = defaultdict(
    set
)


for token in hap_counts:

    key = strict_key(
        token
    )

    if key is None:
        continue

    hap_key_to_tokens[
        key
    ].add(
        token
    )

    stem = key[1]

    if stem is not None:

        hap_stem_to_tokens[
            stem
        ].add(
            token
        )


# ============================================================
# DS -> HAP CANDIDATE MATCHES
# ============================================================

mapping_rows = []


for token, count in sorted(
    ds_text_counts.items(),
    key=lambda kv: (
        -kv[1],
        kv[0],
    ),
):

    key = strict_key(
        token
    )

    hemi = (
        key[0]
        if key
        else None
    )

    stem = (
        key[1]
        if key
        else None
    )

    exact = (
        token
        if token in hap_counts
        else None
    )

    strict_candidates = (
        sorted(
            hap_key_to_tokens.get(
                key,
                set(),
            )
        )
        if key
        else []
    )

    stem_candidates = (
        sorted(
            hap_stem_to_tokens.get(
                stem,
                set(),
            )
        )
        if stem
        else []
    )

    # Only call it SAFE when canonical key has
    # exactly one HAP token.
    safe_target = (
        strict_candidates[0]
        if len(
            strict_candidates
        )
        ==
        1
        else None
    )

    mapping_rows.append({
        "ds_roi":
            token,

        "ds_count":
            count,

        "hemi":
            hemi,

        "canonical_stem":
            stem,

        "raw_exact_hap":
            exact,

        "n_strict_candidates":
            len(
                strict_candidates
            ),

        "strict_candidates":
            "|".join(
                strict_candidates
            ),

        "n_stem_candidates":
            len(
                stem_candidates
            ),

        "stem_candidates":
            "|".join(
                stem_candidates
            ),

        "safe_canonical_target":
            safe_target,

        "safe_match":
            safe_target
            is not None,
    })


mapping = pd.DataFrame(
    mapping_rows
)


# ============================================================
# WEIGHTED COVERAGE
# ============================================================

total_ds = int(
    mapping[
        "ds_count"
    ].sum()
)

safe_weight = int(
    mapping.loc[
        mapping[
            "safe_match"
        ],
        "ds_count",
    ].sum()
)


safe_weighted_coverage = (
    safe_weight
    /
    total_ds
    if total_ds
    else np.nan
)


safe_unique_coverage = (
    float(
        mapping[
            "safe_match"
        ].mean()
    )
    if len(
        mapping
    )
    else np.nan
)


# ============================================================
# HEMISPHERE-AWARE SECOND PASS
#
# Some atlases put hemisphere only in a separate column,
# not inside Destrieux_label_text.
# We audit that possibility separately, but do not silently
# use it in the safe mapping.
# ============================================================

ds_electrodes = pd.DataFrame(
    ds_rows
)


def meta_hemi(x):

    if x is None:
        return None

    s = str(x).lower()

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


ds_electrodes[
    "metadata_hemi"
] = (
    ds_electrodes[
        "hemisphere_metadata"
    ].map(
        meta_hemi
    )
)


def row_key(r):

    if r[
        "roi_text"
    ] is None:

        return None

    hemi = (
        r[
            "roi_text_inferred_hemi"
        ]
        or
        r[
            "metadata_hemi"
        ]
    )

    stem = r[
        "roi_text_stem"
    ]

    if stem is None:
        return None

    return (
        hemi,
        stem,
    )


ds_electrodes[
    "metadata_augmented_key"
] = [
    row_key(r)
    for _, r in
    ds_electrodes.iterrows()
]


def candidate_for_key(key):

    if key is None:
        return None

    candidates = sorted(
        hap_key_to_tokens.get(
            key,
            set(),
        )
    )

    if len(
        candidates
    ) == 1:

        return candidates[0]

    return None


ds_electrodes[
    "metadata_augmented_target"
] = (
    ds_electrodes[
        "metadata_augmented_key"
    ].map(
        candidate_for_key
    )
)


augmented_valid = (
    ds_electrodes[
        "roi_text"
    ].notna()
)


augmented_coverage = (
    float(
        ds_electrodes.loc[
            augmented_valid,
            "metadata_augmented_target",
        ]
        .notna()
        .mean()
    )
    if augmented_valid.any()
    else np.nan
)


# ============================================================
# OUTPUT
# ============================================================

hap_table = pd.DataFrame(
    [
        {
            "hap_roi":
                token,

            "count":
                count,

            "hemi":
                infer_hemi(
                    token
                ),

            "canonical_stem":
                canonical_stem(
                    token
                ),
        }
        for token, count in sorted(
            hap_counts.items(),
            key=lambda kv: (
                -kv[1],
                kv[0],
            ),
        )
    ]
)


ds_table = pd.DataFrame(
    [
        {
            "ds_roi":
                token,

            "count":
                count,

            "hemi_from_text":
                infer_hemi(
                    token
                ),

            "canonical_stem":
                canonical_stem(
                    token
                ),
        }
        for token, count in sorted(
            ds_text_counts.items(),
            key=lambda kv: (
                -kv[1],
                kv[0],
            ),
        )
    ]
)


hap_table.to_csv(
    OUT
    / "hapwave_roi_vocabulary.csv",
    index=False,
)


ds_table.to_csv(
    OUT
    / "ds004080_roi_vocabulary.csv",
    index=False,
)


mapping.to_csv(
    OUT
    / "canonical_candidate_mapping.csv",
    index=False,
)


ds_electrodes.to_csv(
    OUT
    / "electrode_level_metadata_augmented_mapping.csv",
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print(
    "=" * 120
)
print(
    "ROI VOCABULARY BRIDGE — OUTCOME BLIND AUDIT"
)
print(
    "=" * 120
)

print(
    "HAP unique ROI tokens:",
    len(
        hap_counts
    )
)

print(
    "DS004080 unique ROI text tokens:",
    len(
        ds_text_counts
    )
)

print(
    "DS004080 unique numeric labels:",
    len(
        ds_numeric_counts
    )
)


print()
print(
    "RAW EXACT OVERLAP"
)

raw_overlap = (
    set(
        hap_counts
    )
    &
    set(
        ds_text_counts
    )
)

print(
    "unique exact:",
    len(
        raw_overlap
    )
)


print()
print(
    "CANONICAL STRICT MATCH"
)

print(
    "weighted safe coverage:",
    round(
        safe_weighted_coverage,
        4,
    )
)

print(
    "unique safe coverage:",
    round(
        safe_unique_coverage,
        4,
    )
)


print()
print(
    "HEMISPHERE-METADATA AUGMENTED MATCH"
)

print(
    "electrode-weighted coverage:",
    round(
        augmented_coverage,
        4,
    )
)


print()
print(
    "=" * 120
)
print(
    "TOP HAPWAVE ROI TOKENS"
)
print(
    "=" * 120
)

print(
    hap_table
    .head(40)
    .to_string(
        index=False
    )
)


print()
print(
    "=" * 120
)
print(
    "TOP DS004080 ROI TOKENS"
)
print(
    "=" * 120
)

print(
    ds_table
    .head(40)
    .to_string(
        index=False
    )
)


print()
print(
    "=" * 120
)
print(
    "TOP UNMATCHED DS004080 TOKENS"
)
print(
    "=" * 120
)

print(
    mapping[
        ~mapping[
            "safe_match"
        ]
    ]
    .sort_values(
        "ds_count",
        ascending=False,
    )
    .head(50)
    [
        [
            "ds_roi",
            "ds_count",
            "hemi",
            "canonical_stem",
            "n_strict_candidates",
            "strict_candidates",
            "n_stem_candidates",
            "stem_candidates",
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
    "NO CLINICAL OUTCOME LABELS USED."
)

print()
print(
    "Saved:",
    OUT
)
