from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
import torch


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "clinical_v9"
    / "v3f_final_strict"
)

AUDIT_ROOT = (
    RESULT_ROOT
    / "final_statistical_audit"
)

FREEZE_ROOT = (
    RESULT_ROOT
    / "protocol_freeze"
)

FREEZE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


VERSION = "V3f-final-strict-lockbox-v1.0"


# ============================================================
# FILE HASHING
# ============================================================

def sha256_file(
    path: Path,
    chunk_size=1024 * 1024,
):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def hash_if_exists(
    path: Path,
):

    if not path.exists():

        return {
            "path":
                str(path),

            "exists":
                False,

            "sha256":
                None,

            "size_bytes":
                None,
        }

    return {
        "path":
            str(path),

        "exists":
            True,

        "sha256":
            sha256_file(
                path
            ),

        "size_bytes":
            path.stat().st_size,
    }


# ============================================================
# GIT STATE
# ============================================================

def git_command(
    args,
):

    try:

        out = subprocess.check_output(
            [
                "git",
                *args,
            ],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
            text=True,
        )

        return out.strip()

    except Exception:

        return None


git_commit = git_command(
    [
        "rev-parse",
        "HEAD",
    ]
)

git_branch = git_command(
    [
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    ]
)

git_status = git_command(
    [
        "status",
        "--porcelain",
    ]
)


# ============================================================
# DEVELOPMENT COHORT INVENTORY
# ============================================================

FEATURE_ROOT = (
    ROOT
    / "results"
    / "clinical_v5"
    / "v3_features"
)


subject_inventory = []


for i in range(
    1,
    9,
):

    sub = f"sub-{i:02d}"

    f = (
        FEATURE_ROOT
        / f"{sub}_pcnd_v3_pairs.csv.gz"
    )

    if not f.exists():

        raise FileNotFoundError(
            f
        )

    df = pd.read_csv(
        f,
        usecols=[
            "stim_site",
            "recording_channel",
        ],
    )

    subject_inventory.append({
        "subject":
            sub,

        "n_pairs":
            int(
                len(df)
            ),

        "n_stimulation_sites":
            int(
                df[
                    "stim_site"
                ].nunique()
            ),

        "n_recording_channels":
            int(
                df[
                    "recording_channel"
                ].nunique()
            ),

        "feature_table_sha256":
            sha256_file(
                f
            ),
    })


# ============================================================
# VERIFY FINAL AUDIT EXISTS
# ============================================================

required_audit_files = [
    AUDIT_ROOT
    / "patient_condition_summary.csv",

    AUDIT_ROOT
    / "patient_level_paired_deltas.csv",

    AUDIT_ROOT
    / "paired_contrast_statistical_summary.csv",

    AUDIT_ROOT
    / "correspondence_trajectory.csv",

    AUDIT_ROOT
    / "future_lockbox_primary_endpoint_development_snapshot.csv",

    AUDIT_ROOT
    / "STATISTICAL_AUDIT.md",
]


for f in required_audit_files:

    if not f.exists():

        raise FileNotFoundError(
            f
        )


# ============================================================
# HASH CODE + RESULTS
# ============================================================

files_to_freeze = [
    ROOT
    / "scripts"
    / "15_build_pcnd_v3_feature_tables.py",

    ROOT
    / "scripts"
    / "30_train_pcnd_v3f_final_strict.py",

    ROOT
    / "scripts"
    / "31_aggregate_v3f_final.py",

    ROOT
    / "scripts"
    / "32_v3f_final_statistical_audit.py",
]


for i in range(
    1,
    9,
):

    sub = f"sub-{i:02d}"

    files_to_freeze.extend([
        RESULT_ROOT
        / sub
        / "pcnd_v3f_final_strict.pt",

        RESULT_ROOT
        / sub
        / "fold_summary.csv",

        RESULT_ROOT
        / sub
        / "test_metrics_raw.csv",
    ])


files_to_freeze.extend(
    required_audit_files
)


frozen_files = [
    hash_if_exists(f)
    for f in files_to_freeze
]


missing_frozen_files = [
    x["path"]
    for x in frozen_files
    if not x["exists"]
]


if missing_frozen_files:

    raise RuntimeError(
        "Cannot freeze protocol; missing files:\n"
        +
        "\n".join(
            missing_frozen_files
        )
    )


# ============================================================
# MANIFEST
# ============================================================

freeze_time = datetime.now(
    timezone.utc
).isoformat()


manifest = {
    "protocol": {
        "version":
            VERSION,

        "freeze_time_utc":
            freeze_time,

        "target_journal":
            "Brain Stimulation",

        "study_stage":
            "Architecture frozen after HAPwave development; "
            "independent external lockbox not yet opened.",

        "development_cohort_status":
            "HAPwave is DEVELOPMENT ONLY. "
            "It has been repeatedly inspected during adaptive "
            "architecture development and must not be described "
            "as an untouched confirmatory cohort.",
    },


    "scientific_question": {
        "primary":
            "Can sparse intracranial stimulation responses "
            "reconstruct patient-specific perturbational brain "
            "maps beyond a transferable anatomical prior?",

        "mechanistic_subquestion":
            "After accounting for patient-wide amplitude "
            "calibration and persistent recording-node "
            "susceptibility, does stimulation-specific "
            "response correspondence retain predictive "
            "information?",

        "clinical_positioning":
            "The model reconstructs perturbational response "
            "maps from sparse intracranial stimulation. "
            "It is not currently claimed to select therapeutic "
            "targets or improve clinical outcome.",
    },


    "development_cohort": {
        "name":
            "HAPwave",

        "openneuro_dataset":
            "ds004696",

        "n_subjects":
            8,

        "subjects":
            subject_inventory,

        "role":
            "development cohort",

        "adaptive_reuse_disclosure":
            True,
    },


    "response_definition": {
        "primary_features": [
            "early_rms_z",
            "late_rms_z",
        ],

        "early_window_ms": [
            10,
            100,
        ],

        "late_window_ms": [
            100,
            500,
        ],

        "transform":
            "log1p response followed by normalization "
            "fit on internal-training sites only",

        "interpretation_boundary":
            "Early and late windows are operational response "
            "phenotypes. Latency alone is not interpreted as "
            "proof of direct versus polysynaptic transmission.",
    },


    "eligibility_and_preprocessing": {
        "recording_channel_type":
            "SEEG",

        "recording_channel_status":
            "good",

        "primary_near_field_exclusion_mm":
            10,

        "planned_sensitivity_exclusion_mm": [
            5,
            15,
            20,
        ],

        "exact_stimulation_contacts_excluded":
            True,

        "normalization_rule":
            "Within every training fold, stimulation-site "
            "train/validation split is performed FIRST. "
            "All feature and target normalization statistics "
            "are fit on INTERNAL-TRAIN stimulation sites only.",

        "roi_vocabulary_rule":
            "ROI vocabulary is fit on internal-training "
            "sites only; unseen labels map to UNK.",

        "forbidden_model_inputs": [
            "seizure_zone",
            "held-out query responses",
            "independent lockbox outcomes",
        ],
    },


    "model": {
        "name":
            "V3f-final strict hierarchical perturbation model",

        "architecture_status":
            "FROZEN",

        "response_space_equation":
            "z_{s,u,i} = z_pop_{u,i} + c_s + "
            "a_{s,i} + t_{s,u,i}",

        "components": {
            "population":
                "Transferable MNI/anatomical prior.",

            "patient_calibration":
                "Patient-wide robust response offset c_s "
                "estimated from observed context stimulations.",

            "recording_node_susceptibility":
                "Persistent patient-specific recording-node "
                "effect a_{s,i} estimated from observed context.",

            "relational_memory":
                "Neural residual t_{s,u,i} after population, "
                "patient calibration, and node susceptibility "
                "have been removed.",
        },

        "relation_identifiability":
            "Relational component is not interpreted at k=1. "
            "Relation training/evaluation begins at k>=3.",

        "context_budgets": [
            1,
            3,
            5,
            10,
        ],

        "relation_context_budgets": [
            3,
            5,
            10,
        ],

        "test_context_repeats":
            20,

        "population_hidden":
            256,

        "attention_heads":
            8,

        "relation_layers":
            3,

        "roi_embedding_dim":
            48,

        "dropout":
            0.10,

        "population_lr":
            0.0002,

        "relation_lr":
            0.0002,

        "population_steps":
            4000,

        "relation_steps":
            5000,

        "loss_weights": {
            "prediction":
                0.25,

            "correlation":
                0.25,

            "ranking":
                0.10,

            "negative_control":
                0.25,

            "negative_margin":
                0.01,
        },

        "random_seed":
            20260925,
    },


    "negative_control": {
        "name":
            "within-recording-channel stimulation permutation",

        "preserves": [
            "patient identity",
            "recording-channel identity",
            "patient-wide calibration",
            "persistent recording-node susceptibility",
            "response marginal distribution within the "
            "recording channel",
        ],

        "destroys":
            "Correspondence between stimulation identity and "
            "the residual response at the same recording node.",

        "primary_interpretation":
            "relation > relation_perm supports predictive "
            "information in stimulation-response "
            "correspondence beyond global and node effects.",

        "causal_boundary":
            "This contrast does not by itself establish "
            "direct anatomical connectivity or a specific "
            "synaptic pathway.",
    },


    "evaluation": {
        "independent_statistical_unit":
            "patient",

        "repeat_handling":
            "Metrics are calculated within each random-context "
            "repeat, then repeats are averaged within patient "
            "before inferential statistics.",

        "pair_level_inference_prohibited":
            True,

        "metrics": {
            "map_fidelity_primary":
                "Spearman correlation",

            "map_fidelity_secondary":
                "Pearson correlation",

            "amplitude_error":
                "MAE",

            "strong_response_retrieval_exploratory":
                "Top-10% recall",
        },

        "conditions": [
            "population",
            "global",
            "node",
            "relation",
            "relation_perm",
        ],

        "nested_models": {
            "M0":
                "population",

            "M1":
                "population + patient calibration",

            "M2":
                "population + patient calibration "
                "+ recording-node susceptibility",

            "M3":
                "M2 + relational memory",

            "M3_perm":
                "M3 with within-channel stimulation "
                "correspondence destroyed",
        },
    },


    "independent_lockbox_plan": {
        "status":
            "PRE-SPECIFIED BEFORE LOCKBOX OPENING",

        "training_data":
            "All model weights, preprocessing parameters, "
            "and hyperparameters must be learned using "
            "development data only. Independent-lockbox "
            "query outcomes must never be used for training, "
            "normalization, checkpoint selection, "
            "hyperparameter tuning, or architecture changes.",

        "patient_personalization_allowed":
            "Only the designated sparse context stimulation "
            "responses for that lockbox patient may be used "
            "to estimate c_s, a_{s,i}, and relational memory. "
            "Query stimulation responses remain hidden until "
            "evaluation.",

        "primary_endpoint": {
            "target":
                "late response",

            "context_k":
                10,

            "contrast":
                "relation minus relation_perm",

            "metric":
                "patient-level Spearman map correlation",

            "estimand":
                "Mean paired patient-level difference after "
                "averaging the 20 pre-specified random-context "
                "repeats within each patient.",

            "hypothesis_direction":
                "relation > relation_perm",

            "primary_test":
                "two-sided patient-level paired exact "
                "sign-flip/randomization test",

            "alpha":
                0.05,

            "confidence_interval":
                "95% patient-level nonparametric bootstrap "
                "CI for the mean paired difference",

            "claim_support_rule":
                "Primary claim is supported only if the "
                "observed mean difference is positive and "
                "the pre-specified two-sided primary test "
                "has p < 0.05. Effect size and CI are "
                "reported regardless of significance.",
        },


        "key_secondary_family": [
            {
                "target":
                    "late",

                "context_k":
                    10,

                "contrast":
                    "relation vs node",

                "metric":
                    "Spearman",
            },

            {
                "target":
                    "late",

                "context_k":
                    10,

                "contrast":
                    "relation vs relation_perm",

                "metric":
                    "Pearson",
            },

            {
                "target":
                    "late",

                "context_k":
                    10,

                "contrast":
                    "relation vs relation_perm",

                "metric":
                    "MAE",
            },

            {
                "target":
                    "late",

                "context_k":
                    5,

                "contrast":
                    "relation vs relation_perm",

                "metric":
                    "Spearman",
            },

            {
                "target":
                    "late",

                "context_k":
                    3,

                "contrast":
                    "relation vs relation_perm",

                "metric":
                    "Spearman",
            },

            {
                "target":
                    "early",

                "context_k":
                    10,

                "contrast":
                    "relation vs relation_perm",

                "metric":
                    "Spearman",
            },
        ],

        "secondary_multiple_testing":
            "Holm family-wise correction at alpha=0.05 "
            "across the six key secondary tests.",

        "exploratory_endpoints": [
            "Top-10% recall",
            "NDCG/precision/recall analyses if specified "
            "before lockbox outcome inspection",
            "early-response context trajectories",
            "other context-budget interactions",
        ],

        "minimum_reporting":
            [
                "mean paired difference",
                "median paired difference",
                "95% patient bootstrap CI",
                "wins/ties/losses",
                "exact sign-flip p",
                "Wilcoxon sensitivity analysis",
                "effect size",
            ],
    },


    "external_lockbox_eligibility": {
        "primary_k10_requirement":
            "A patient must have sufficient usable "
            "stimulation sites to allocate 10 context sites "
            "and retain an evaluable held-out query map.",

        "recording_requirement":
            "Eligible good intracranial recording channels "
            "must support map-level correlation metrics.",

        "missingness_rule":
            "Eligibility is determined without inspecting "
            "query-response outcomes. Patients may not be "
            "excluded because their final model performance "
            "is poor.",

        "dataset_name":
            "TBD before lockbox opening",

        "lockbox_open_date":
            None,
    },


    "robustness_plan": {
        "near_field_exclusion_mm": [
            5,
            10,
            15,
            20,
        ],

        "training_seed_sensitivity":
            "At least three fixed training seeds; robustness "
            "analyses do not replace the pre-specified "
            "primary model.",

        "preprocessing_sensitivity":
            "Reference/artifact/filtering sensitivity will "
            "be reported when raw CCEP preprocessing is "
            "finalized.",

        "reporting_rule":
            "All planned sensitivity analyses are reported, "
            "including unfavorable results.",
    },


    "claim_boundary": {
        "supported_if_external_replication_succeeds":
            "Sparse patient perturbations contain predictive "
            "stimulation-response correspondence beyond a "
            "shared anatomical prior, patient-wide "
            "calibration, and persistent recording-node "
            "susceptibility.",

        "not_claimed": [
            "direct structural connectivity from latency alone",
            "direct versus polysynaptic mechanism from "
            "early/late timing alone",
            "epileptogenic-zone ground truth",
            "therapeutic target selection",
            "improved surgical outcome",
            "causal clinical benefit",
        ],
    },


    "post_lockbox_rules": {
        "prohibited_after_lockbox_opening": [
            "architecture changes",
            "new feature engineering based on lockbox results",
            "changing early/late response windows",
            "changing the primary context budget",
            "changing the primary metric",
            "changing the primary contrast",
            "tuning loss weights",
            "tuning exclusion thresholds based on performance",
            "changing normalization rules",
            "selecting a preferred seed based on lockbox outcome",
        ],

        "bug_fix_rule":
            "A genuine implementation bug may be corrected "
            "only with a documented protocol amendment. "
            "If the fix was motivated by observed lockbox "
            "performance, that cohort loses confirmatory "
            "lockbox status and a new independent lockbox "
            "is required.",

        "architecture_freeze":
            True,
    },


    "reproducibility": {
        "python":
            sys.version,

        "platform":
            platform.platform(),

        "numpy":
            np.__version__,

        "pandas":
            pd.__version__,

        "scipy":
            scipy.__version__,

        "torch":
            torch.__version__,

        "cuda_available":
            torch.cuda.is_available(),

        "git_commit":
            git_commit,

        "git_branch":
            git_branch,

        "git_worktree_dirty":
            bool(
                git_status
            ),

        "git_status_porcelain":
            git_status,

        "frozen_files":
            frozen_files,
    },
}


# ============================================================
# SAVE JSON
# ============================================================

json_path = (
    FREEZE_ROOT
    / "V3F_FINAL_PROTOCOL_MANIFEST.json"
)


json_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# HUMAN-READABLE MARKDOWN
# ============================================================

md = []

md.append(
    "# V3f Final Protocol Manifest"
)

md.append("")

md.append(
    f"**Version:** {VERSION}"
)

md.append("")

md.append(
    f"**Frozen UTC:** {freeze_time}"
)

md.append("")

md.append(
    "**Target journal:** Brain Stimulation"
)

md.append("")

md.append(
    "## Study status"
)

md.append("")

md.append(
    "HAPwave / OpenNeuro ds004696 is formally closed "
    "as a **development cohort**. Because its eight patients "
    "were repeatedly inspected during adaptive model "
    "development, no HAPwave p-value is treated as an "
    "independent confirmatory result."
)

md.append("")

md.append(
    "The architecture is frozen before opening the "
    "independent external lockbox."
)

md.append("")

md.append(
    "## Frozen model"
)

md.append("")

md.append(
    r"\["
)

md.append(
    r"z_{s,u,i}=z^{pop}_{u,i}+c_s+a_{s,i}+t_{s,u,i}"
)

md.append(
    r"\]"
)

md.append("")

md.append(
    "The four components are the transferable anatomical "
    "population prior, patient-wide calibration, persistent "
    "recording-node susceptibility, and stimulation-specific "
    "relational residual."
)

md.append("")

md.append(
    "The relational component is not interpreted at k=1; "
    "formal relational evaluation begins at k=3."
)

md.append("")

md.append(
    "## Strict relational negative control"
)

md.append("")

md.append(
    "Residual responses are permuted across stimulation "
    "identities **within the same recording channel**. "
    "This preserves patient identity and node susceptibility "
    "while destroying stimulation-response correspondence."
)

md.append("")

md.append(
    "## Independent-lockbox primary endpoint"
)

md.append("")

md.append(
    "**Late response, k=10, patient-level Spearman map "
    "correlation, relation versus within-recording-channel "
    "stimulation permutation.**"
)

md.append("")

md.append(
    "The patient is the inference unit. Twenty random-context "
    "repeats are averaged within patient before the paired "
    "test. The primary statistical test is a two-sided exact "
    "patient-level sign-flip/randomization test at alpha=0.05, "
    "with a 95% patient bootstrap CI for the mean paired "
    "difference."
)

md.append("")

md.append(
    "## Key claim boundary"
)

md.append("")

md.append(
    "A successful external replication supports predictive "
    "information in stimulation-response correspondence "
    "beyond population anatomy, patient calibration, and "
    "persistent node susceptibility. It does not by itself "
    "prove direct anatomical connectivity, a direct versus "
    "polysynaptic mechanism, therapeutic efficacy, or "
    "epileptogenic-zone ground truth."
)

md.append("")

md.append(
    "## Lockbox rule"
)

md.append("")

md.append(
    "After lockbox opening, architecture, preprocessing, "
    "primary response definition, context budget, primary "
    "metric, primary contrast, loss weights, and exclusion "
    "rules may not be changed in response to lockbox results. "
    "An outcome-motivated change converts that cohort into "
    "development data and requires a new independent lockbox."
)

md.append("")

md.append(
    "## Reproducibility"
)

md.append("")

md.append(
    f"- Git commit: `{git_commit}`"
)

md.append(
    f"- Git branch: `{git_branch}`"
)

md.append(
    f"- Dirty worktree: `{bool(git_status)}`"
)

md.append(
    f"- Python: `{sys.version.split()[0]}`"
)

md.append(
    f"- PyTorch: `{torch.__version__}`"
)

md.append(
    f"- NumPy: `{np.__version__}`"
)

md.append(
    f"- SciPy: `{scipy.__version__}`"
)

md.append("")

md.append(
    "Full SHA256 hashes and the complete statistical plan "
    "are stored in the machine-readable JSON manifest."
)


md_path = (
    FREEZE_ROOT
    / "V3F_FINAL_PROTOCOL_MANIFEST.md"
)


md_path.write_text(
    "\n".join(md),
    encoding="utf-8",
)


# ============================================================
# MANIFEST HASH
# ============================================================

manifest_hash = sha256_file(
    json_path
)


seal = {
    "protocol_version":
        VERSION,

    "manifest_path":
        str(json_path),

    "manifest_sha256":
        manifest_hash,

    "freeze_time_utc":
        freeze_time,

    "git_commit":
        git_commit,

    "git_worktree_dirty":
        bool(
            git_status
        ),
}


seal_path = (
    FREEZE_ROOT
    / "FREEZE_SEAL.json"
)


seal_path.write_text(
    json.dumps(
        seal,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print()
print(
    "=" * 110
)
print(
    "V3f FINAL PROTOCOL FROZEN"
)
print(
    "=" * 110
)

print(
    "Version:",
    VERSION
)

print(
    "Git commit:",
    git_commit
)

print(
    "Dirty worktree:",
    bool(
        git_status
    )
)

print(
    "Manifest:",
    json_path
)

print(
    "Manifest SHA256:",
    manifest_hash
)

print(
    "Markdown:",
    md_path
)

print(
    "Seal:",
    seal_path
)

print()
print(
    "HAPwave status: DEVELOPMENT CLOSED."
)

print(
    "Next stage: independent external lockbox."
)
