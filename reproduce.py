#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent

EXPECTED_CORE_SHA = (
    "34750f1cee3a56b24d5049866396ff9d"
    "bbf6df8310f2b2e7f3a81f3cfbb7137c"
)

EXPECTED_CKPT_SHA = (
    "399421567b0185383bf8953cdcd6c758"
    "91893b5113fca8a3a41ac190eb68a794"
)

EXPECTED_FEATURE_SHA = (
    "18c8e9d1333941f3f8b51a30829cd74"
    "cf4aa4acb4108aaac5e2d34ab8707b48b"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def pass_msg(msg):
    print(f"[PASS] {msg}")


EXPECTED_INFERENCE_HASHES = {
    "FROZEN_REPEAT_METRICS.csv.gz":
        "3aa6e417972a1b18a511e9d1af70dddd70b4fd72c7db3bed0aadb436b73f49c8",
    "FROZEN_PATIENT_METRIC_SUMMARY.csv":
        "0a4b5940653a82c75db9fabc2b256095be58eca891203ecd9f46d0e9d08f5192",
}


def compare_scientific_value(actual, expected, path):
    """Fail on changed fields, types, nonfinite values, or numerical drift.

    Relative tolerance handles serialization/platform rounding without an
    absolute tolerance that could hide changes in very small P values.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"{path}: field names differ")
        for key in expected:
            compare_scientific_value(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"{path}: list length differs")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_scientific_value(a, b, f"{path}[{i}]")
    elif isinstance(expected, bool):
        if type(actual) is not bool or actual != expected:
            raise RuntimeError(f"{path}: Boolean differs")
    elif isinstance(expected, int):
        if type(actual) is not int or actual != expected:
            raise RuntimeError(f"{path}: integer differs")
    elif isinstance(expected, float):
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isfinite(expected)
                or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)):
            raise RuntimeError(f"{path}: {actual!r} differs from {expected!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise RuntimeError(f"{path}: value differs")


def compare_result(actual, expected):
    # Statistics CSV serialization can differ across library versions.
    # Their hashes are validated against actual generated files separately;
    # their scientific contents are compared with the same numeric tolerance.
    ignored = {"created_at_utc", "patient_delta_sha256", "secondary_results_sha256"}
    for value in (actual, expected):
        for key in ("patient_delta_sha256", "secondary_results_sha256"):
            if not isinstance(value.get(key), str) or len(value[key]) != 64:
                raise RuntimeError(f"result.{key}: missing or invalid output hash")
    a = {k: v for k, v in actual.items() if k not in ignored}
    b = {k: v for k, v in expected.items() if k not in ignored}
    for value in (a, b):
        family = value.get("secondary_family")
        if not isinstance(family, list) or len(family) != 6:
            raise RuntimeError("secondary_family: expected exactly six endpoints")
        names = [row.get("endpoint") for row in family]
        if any(not isinstance(n, str) for n in names) or len(set(names)) != 6:
            raise RuntimeError("secondary_family: missing or duplicate endpoint")
        value["secondary_family"] = {row["endpoint"]: row for row in family}
    compare_scientific_value(a, b, "result")
    pass_msg("all primary, randomization, sensitivity and six secondary fields")


def compare_generated_statistics(out, expected):
    actual = json.loads((out / "PRIMARY_EXTERNAL_RESULT.json").read_text())
    compare_result(actual, expected)
    for name, key in {
        "PRIMARY_PATIENT_DELTAS.csv": "patient_delta_sha256",
        "SECONDARY_FAMILY_RESULTS.csv": "secondary_results_sha256",
    }.items():
        file = require(out / name)
        if sha256(file) != actual[key]:
            raise RuntimeError(f"{name}: generated file/result hash mismatch")
        a = pd.read_csv(file)
        b = pd.read_csv(REPO / "expected_outputs" / name)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            raise RuntimeError(f"{name}: generated CSV schema/length differs")
        identity = b.columns[0]
        if a[identity].duplicated().any() or b[identity].duplicated().any():
            raise RuntimeError(f"{name}: duplicate row identity")
        a = a.sort_values(identity).to_dict("records")
        b = b.sort_values(identity).to_dict("records")
        compare_scientific_value(a, b, name)
        pass_msg(f"{name}: generated hash and all scientific CSV fields")


def verify_inference_hashes(out):
    historical = json.loads(
        (REPO / "frozen_protocol/external/FROZEN_INFERENCE_MANIFEST.json").read_text()
    )
    historical_keys = {
        "FROZEN_REPEAT_METRICS.csv.gz": "repeat_metrics_sha256",
        "FROZEN_PATIENT_METRIC_SUMMARY.csv": "patient_summary_sha256",
    }
    for name, expected in EXPECTED_INFERENCE_HASHES.items():
        if historical[historical_keys[name]] != expected:
            raise RuntimeError(f"{name}: historical manifest hash differs")
        if sha256(require(out / name)) != expected:
            raise RuntimeError(f"{name}: regenerated historical SHA256 mismatch")
        pass_msg(f"{name}: exact historical SHA256")


def audit():
    core = require(
        REPO
        / "frozen_source/scripts/"
        "30_train_pcnd_v3f_final_strict.py"
    )

    ckpt = require(
        REPO
        / "checkpoints/"
        "pcnd_v3f_all8_deployment.pt"
    )

    feature = require(
        REPO
        / "reproduction_inputs/external_features/"
        "DS004080_V3F_EXTERNAL_FEATURES.csv.gz"
    )

    if sha256(core) != EXPECTED_CORE_SHA:
        raise RuntimeError("Frozen V3f source hash mismatch.")
    pass_msg("frozen V3f source SHA256")

    if sha256(ckpt) != EXPECTED_CKPT_SHA:
        raise RuntimeError("Checkpoint hash mismatch.")
    pass_msg("deployment checkpoint SHA256")

    if sha256(feature) != EXPECTED_FEATURE_SHA:
        raise RuntimeError("External feature-table hash mismatch.")
    pass_msg("external feature-table SHA256")

    expected = json.loads(
        require(
            REPO
            / "expected_outputs/"
            "PRIMARY_EXTERNAL_RESULT.json"
        ).read_text()
    )

    p = expected["primary"]

    if p["n_patients"] != 74:
        raise RuntimeError("Unexpected patient count.")

    pass_msg("expected external cohort n=74")

    if not np.isclose(
        p["mean_delta"],
        0.03732822177853643,
        rtol=0,
        atol=1e-15,
    ):
        raise RuntimeError("Primary mean delta mismatch.")

    pass_msg(
        "expected primary mean delta "
        f"{p['mean_delta']:.12f}"
    )

    if (
        p["wins"],
        p["ties"],
        p["losses"],
    ) != (71, 0, 3):
        raise RuntimeError("Primary win/tie/loss mismatch.")

    pass_msg("expected wins/ties/losses = 71/0/3")

    print()
    print("AUDIT COMPLETE")


def build_workspace(
    root: Path,
    use_frozen_patient_summary: bool,
):
    # Historical directory layout expected by frozen scripts.
    (root / "scripts").mkdir(parents=True)
    (
        root
        / "results/clinical_v10/"
        "v3f_all8_deployment"
    ).mkdir(parents=True)

    (
        root
        / "results/external_lockbox/"
        "ds004080_v3f_external_features_v1"
    ).mkdir(parents=True)

    (
        root
        / "results/external_lockbox/"
        "ds004080_final_lockbox_v1"
    ).mkdir(parents=True)

    out = (
        root
        / "results/external_lockbox/"
        "ds004080_frozen_inference_v1"
    )

    out.mkdir(parents=True)

    shutil.copy2(
        REPO
        / "frozen_source/scripts/"
        "30_train_pcnd_v3f_final_strict.py",
        root
        / "scripts/"
        "30_train_pcnd_v3f_final_strict.py",
    )

    shutil.copy2(
        REPO
        / "checkpoints/"
        "pcnd_v3f_all8_deployment.pt",
        root
        / "results/clinical_v10/"
        "v3f_all8_deployment/"
        "pcnd_v3f_all8_deployment.pt",
    )

    feature_dir = (
        root
        / "results/external_lockbox/"
        "ds004080_v3f_external_features_v1"
    )

    for name in [
        "DS004080_V3F_EXTERNAL_FEATURES.csv.gz",
        "EXTERNAL_V3F_FEATURE_MANIFEST.json",
    ]:
        shutil.copy2(
            REPO
            / "reproduction_inputs/external_features/"
            / name,
            feature_dir / name,
        )

    shutil.copy2(
        REPO
        / "frozen_protocol/external/"
        "frozen_context_query_plan.csv",
        root
        / "results/external_lockbox/"
        "ds004080_final_lockbox_v1/"
        "frozen_context_query_plan.csv",
    )

    if use_frozen_patient_summary:
        shutil.copy2(
            REPO
            / "reproduction_inputs/statistics/"
            "FROZEN_PATIENT_METRIC_SUMMARY.csv",
            out / "FROZEN_PATIENT_METRIC_SUMMARY.csv",
        )

        shutil.copy2(
            REPO
            / "frozen_protocol/external/"
            "FROZEN_INFERENCE_MANIFEST.json",
            out / "FROZEN_INFERENCE_MANIFEST.json",
        )

    shutil.copy2(
        REPO
        / "reproduction_inputs/statistics/"
        "PREINFERENCE_STATISTICAL_OPERATIONALIZATION.json",
        out
        / "PREINFERENCE_STATISTICAL_OPERATIONALIZATION.json",
    )

    return out


def patched_script(src: Path, dst: Path, root: Path):
    text = src.read_text()

    old = "/home/dell/jiaxing/bby/pcnd"
    new = str(root)

    if old not in text:
        raise RuntimeError(
            f"Historical ROOT not found in {src}"
        )

    text = text.replace(old, new)

    dst.write_text(text)


def run_statistics():
    with tempfile.TemporaryDirectory(
        prefix="pcnd_stats_"
    ) as tmp:
        root = Path(tmp)

        out = build_workspace(
            root,
            use_frozen_patient_summary=True,
        )

        script = root / "43_primary_statistics.py"

        patched_script(
            REPO
            / "frozen_source/scripts/"
            "43_ds004080_primary_statistics.py",
            script,
            root,
        )

        subprocess.run(
            [sys.executable, str(script)],
            check=True,
        )

        result = json.loads(
            (
                out
                / "PRIMARY_EXTERNAL_RESULT.json"
            ).read_text()
        )

        expected = json.loads(
            (
                REPO
                / "expected_outputs/"
                "PRIMARY_EXTERNAL_RESULT.json"
            ).read_text()
        )

        compare_generated_statistics(out, expected)
        pass_msg("statistics reproduced")
        a = result["primary"]
        pass_msg(f"N = {a['n_patients']}")
        pass_msg(f"mean delta Spearman = {a['mean_delta']:.12f}")
        pass_msg(f"95% CI = {a['bootstrap_95ci_mean']}")
        pass_msg(f"wins/ties/losses = {a['wins']}/{a['ties']}/{a['losses']}")

        print()
        print("STATISTICAL REPRODUCTION COMPLETE")


def run_external():
    with tempfile.TemporaryDirectory(
        prefix="pcnd_external_"
    ) as tmp:
        root = Path(tmp)

        out = build_workspace(
            root,
            use_frozen_patient_summary=False,
        )

        inference_script = (
            root / "42_frozen_inference.py"
        )

        patched_script(
            REPO
            / "frozen_source/scripts/"
            "42_run_ds004080_frozen_v3f_inference.py",
            inference_script,
            root,
        )

        subprocess.run(
            [sys.executable, str(inference_script)],
            check=True,
        )

        # Verify deterministic inference artifacts BEFORE running statistics.
        verify_inference_hashes(out)

        # Add the pre-inference statistics freeze to
        # the newly created inference output directory.
        shutil.copy2(
            REPO
            / "reproduction_inputs/statistics/"
            "PREINFERENCE_STATISTICAL_OPERATIONALIZATION.json",
            out
            / "PREINFERENCE_STATISTICAL_OPERATIONALIZATION.json",
        )

        statistics_script = (
            root / "43_primary_statistics.py"
        )

        patched_script(
            REPO
            / "frozen_source/scripts/"
            "43_ds004080_primary_statistics.py",
            statistics_script,
            root,
        )

        subprocess.run(
            [sys.executable, str(statistics_script)],
            check=True,
        )

        result = json.loads(
            (
                out
                / "PRIMARY_EXTERNAL_RESULT.json"
            ).read_text()
        )

        expected = json.loads(
            (
                REPO
                / "expected_outputs/"
                "PRIMARY_EXTERNAL_RESULT.json"
            ).read_text()
        )

        compare_generated_statistics(out, expected)
        pass_msg("frozen external inference reproduced")
        pass_msg("primary and secondary statistics reproduced")

        print()
        print("EXTERNAL REPRODUCTION COMPLETE")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "audit",
            "statistics",
            "external",
        ],
        default="audit",
    )

    args = parser.parse_args()

    if args.mode == "audit":
        audit()

    elif args.mode == "statistics":
        audit()
        run_statistics()

    elif args.mode == "external":
        audit()
        run_external()


if __name__ == "__main__":
    main()
