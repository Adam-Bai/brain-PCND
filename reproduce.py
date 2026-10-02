#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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

        a = result["primary"]
        b = expected["primary"]

        for key in [
            "n_patients",
            "wins",
            "ties",
            "losses",
        ]:
            if a[key] != b[key]:
                raise RuntimeError(
                    f"{key} mismatch: "
                    f"{a[key]} vs {b[key]}"
                )

        for key in [
            "mean_delta",
            "median_delta",
        ]:
            if not np.isclose(
                a[key],
                b[key],
                rtol=0,
                atol=1e-14,
            ):
                raise RuntimeError(
                    f"{key} mismatch."
                )

        if not np.allclose(
            a["bootstrap_95ci_mean"],
            b["bootstrap_95ci_mean"],
            rtol=0,
            atol=1e-14,
        ):
            raise RuntimeError(
                "Bootstrap CI mismatch."
            )

        pass_msg("statistics reproduced")
        pass_msg(
            f"N = {a['n_patients']}"
        )
        pass_msg(
            "mean delta Spearman = "
            f"{a['mean_delta']:.12f}"
        )
        pass_msg(
            "95% CI = "
            f"{a['bootstrap_95ci_mean']}"
        )
        pass_msg(
            "wins/ties/losses = "
            f"{a['wins']}/{a['ties']}/{a['losses']}"
        )

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

        a = result["primary"]
        b = expected["primary"]

        if a["n_patients"] != 74:
            raise RuntimeError(
                "External reproduction n != 74."
            )

        if not np.isclose(
            a["mean_delta"],
            b["mean_delta"],
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "External primary delta mismatch."
            )

        pass_msg("frozen external inference reproduced")
        pass_msg("primary statistics reproduced")

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
