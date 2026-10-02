from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime, timezone
import hashlib
import json
import subprocess


ROOT = Path(
    os.environ.get(
        "PCND_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

SOURCE05 = (
    ROOT
    / "scripts"
    / "05_build_perturbation_map_v1.py"
)

LOCKBOX_MANIFEST = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_final_lockbox_v1"
    / "DS004080_FINAL_LOCKBOX_MANIFEST.json"
)

DEPLOYMENT_DIR = (
    ROOT
    / "results"
    / "clinical_v10"
    / "v3f_all8_deployment"
)

DEPLOYMENT_CKPT = (
    DEPLOYMENT_DIR
    / "pcnd_v3f_all8_deployment.pt"
)

DEPLOYMENT_MANIFEST = (
    DEPLOYMENT_DIR
    / "ALL8_DEPLOYMENT_MANIFEST.json"
)

OUT = (
    ROOT
    / "results"
    / "external_lockbox"
    / "ds004080_response_protocol_v1"
)

PROTOCOL = (
    OUT
    / "DS004080_RESPONSE_PROTOCOL.json"
)

SEAL = (
    OUT
    / "RESPONSE_PROTOCOL_FREEZE_SEAL.json"
)


EXPECTED_LOCKBOX_SHA256 = (
    "8170555317e6a2b510cb7ef1bbb603ba"
    "6a4220ec92a0b50cc32c29392f66cf6d"
)

EXPECTED_DEPLOYMENT_CKPT_SHA256 = (
    "399421567b0185383bf8953cdcd6c758"
    "91893b5113fca8a3a41ac190eb68a794"
)

EXPECTED_DEPLOYMENT_MANIFEST_SHA256 = (
    "dd69bb8963a678a9459987ebc5e37115"
    "49b42679ed27b1ed57bbbf0628b593b0"
)

EXPECTED_DEVELOPMENT_PROTOCOL_SHA256 = (
    "cb3bd89d6072245ecec7ae608d692f28"
    "6172fedab517b1b81d821e4e3e710727"
)

EXPECTED_ROI_BRIDGE_SHA256 = (
    "fb9e3a493bf05e744c65fae165fc1fe"
    "3d7734b1a6456685c4d470be9c9efbbca"
)


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def git(*args: str) -> str:

    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


# ============================================================
# PRE-FREEZE INTEGRITY
# ============================================================

required = [
    (
        LOCKBOX_MANIFEST,
        EXPECTED_LOCKBOX_SHA256,
        "external lockbox",
    ),
    (
        DEPLOYMENT_CKPT,
        EXPECTED_DEPLOYMENT_CKPT_SHA256,
        "deployment checkpoint",
    ),
    (
        DEPLOYMENT_MANIFEST,
        EXPECTED_DEPLOYMENT_MANIFEST_SHA256,
        "deployment manifest",
    ),
]


for path, expected, label in required:

    if not path.exists():
        raise FileNotFoundError(path)

    actual = sha256_file(path)

    if actual != expected:

        raise RuntimeError(
            f"{label} SHA256 mismatch\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )


if not SOURCE05.exists():
    raise FileNotFoundError(SOURCE05)


tracked_status = git(
    "status",
    "--porcelain",
    "--untracked-files=no",
)

if tracked_status:

    raise RuntimeError(
        "Refusing response-protocol freeze: "
        "tracked PCND worktree is dirty.\n\n"
        + tracked_status
    )


git_commit = git(
    "rev-parse",
    "HEAD",
)

source05_sha = sha256_file(
    SOURCE05
)


# Never silently overwrite a frozen protocol.
if PROTOCOL.exists() or SEAL.exists():

    raise RuntimeError(
        "Response protocol already exists. "
        "Refusing to overwrite frozen artifacts."
    )


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FROZEN RESPONSE PROTOCOL
# ============================================================

protocol = {

    "protocol_id":
        "DS004080-external-response-v1.0",

    "freeze_time_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "FROZEN_BEFORE_RAW_SIGNAL_OPENING",

    "scientific_role": (
        "Outcome-blind external response phenotype "
        "definition for one-shot evaluation of the "
        "frozen PCND V3f deployment model."
    ),

    "provenance": {

        "pcnd_git_commit":
            git_commit,

        "development_response_source":
            str(SOURCE05),

        "development_response_source_sha256":
            source05_sha,

        "development_protocol_sha256":
            EXPECTED_DEVELOPMENT_PROTOCOL_SHA256,

        "external_lockbox_sha256":
            EXPECTED_LOCKBOX_SHA256,

        "roi_bridge_sha256":
            EXPECTED_ROI_BRIDGE_SHA256,

        "deployment_checkpoint_sha256":
            EXPECTED_DEPLOYMENT_CKPT_SHA256,

        "deployment_manifest_sha256":
            EXPECTED_DEPLOYMENT_MANIFEST_SHA256,
    },

    # --------------------------------------------------------
    # Dataset population / eligibility
    # --------------------------------------------------------

    "external_population": {

        "dataset":
            "OpenNeuro ds004080",

        "n_frozen_subjects":
            74,

        "patient_is_inference_unit":
            True,

        "patient_exclusion_based_on_model_performance":
            False,

        "patient_exclusion_based_on_roi_coverage":
            False,

        "eligibility_source": (
            "Previously frozen ds004080 external "
            "lockbox/context plan."
        ),

        "dominant_spes_protocol_per_subject":
            True,

        "minimum_pulses_per_site":
            5,

        "minimum_eligible_recording_channels_per_site":
            20,

        "minimum_usable_sites_for_k10":
            11,

        "context_sizes":
            [1, 3, 5, 10],

        "context_repeats":
            20,

        "context_plan_rows":
            5920,

        "reselection_after_raw_opening":
            False,
    },

    # --------------------------------------------------------
    # Signal semantics
    #
    # Match development target construction.
    # No external-outcome-driven preprocessing.
    # --------------------------------------------------------

    "signal_processing": {

        "additional_bandpass_filter":
            None,

        "additional_highpass_filter":
            None,

        "additional_lowpass_filter":
            None,

        "additional_notch_filter":
            None,

        "additional_rereferencing":
            None,

        "common_average_reference":
            False,

        "additional_resampling":
            None,

        "additional_detrending":
            None,

        "reason": (
            "No such operations were present in "
            "scripts/05_build_perturbation_map_v1.py "
            "for the development response phenotype."
        ),
    },

    # --------------------------------------------------------
    # Event alignment
    # --------------------------------------------------------

    "event_alignment": {

        "primary_event_index":
            "sample_start",

        "sample_start_semantics":
            (
                "Use dataset-provided event sample_start "
                "directly as stimulation time zero."
            ),

        "waveform_peak_realignment":
            False,

        "manual_onset_adjustment":
            False,

        "outcome_dependent_alignment":
            False,

        "if_sample_start_missing":
            (
                "Event is mechanically ineligible for "
                "the primary phenotype; do not infer "
                "onset from response waveform."
            ),
    },

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    "windows_seconds": {

        "baseline":
            [-0.500, -0.020],

        "early":
            [0.010, 0.100],

        "late":
            [0.100, 0.500],

        "sample_conversion":
            "int(round(window_boundary_seconds * fs))",

        "interval_implementation":
            "Python half-open slicing [start:end)",
    },

    # --------------------------------------------------------
    # Baseline normalization
    # --------------------------------------------------------

    "baseline_normalization": {

        "center":
            "nanmedian(baseline)",

        "primary_scale":
            "1.4826 * nanmedian(abs(baseline - center))",

        "fallback_1_condition":
            "scale nonfinite or scale < 1e-6",

        "fallback_1":
            "nanstd(baseline)",

        "fallback_2_condition":
            "scale nonfinite or scale < 1e-6",

        "fallback_2":
            1.0,

        "normalized_signal":
            "(signal_window - baseline_center) / baseline_scale",
    },

    # --------------------------------------------------------
    # Primary response features
    # --------------------------------------------------------

    "trial_features": {

        "early_rms_z": (
            "sqrt(nanmean(early_z ** 2))"
        ),

        "late_rms_z": (
            "sqrt(nanmean(late_z ** 2))"
        ),

        "primary_model_targets": [
            "early_rms_z",
            "late_rms_z",
        ],

        "peak_features_required_for_primary_model":
            False,
    },

    # --------------------------------------------------------
    # Repeated trials
    # --------------------------------------------------------

    "repeat_aggregation": {

        "development_grouping": [
            "subject",
            "split",
            "stim_site",
            "recording_channel",
        ],

        "external_grouping_semantics": (
            "Aggregate repeated eligible pulses belonging "
            "to the already-frozen subject/recording/"
            "stimulation-site protocol unit; do not pool "
            "across distinct acquisition/protocol units."
        ),

        "n_trials":
            "nunique(event_id)",

        "early_rms_z":
            "median",

        "late_rms_z":
            "median",

        "minimum_pulses_for_external_site":
            5,

        "aggregation_after_trial_feature_extraction":
            True,
    },

    # --------------------------------------------------------
    # Spatial exclusions
    # --------------------------------------------------------

    "spatial_rules": {

        "stimulating_contacts_as_recording_channels":
            "exclude",

        "development_near_field_flag":
            "distance_to_stim <= 10.0 mm",

        "external_primary_near_field_rule":
            (
                "Use the already-frozen 10-mm eligibility "
                "rule/context plan; do not regenerate "
                "context/query identities from raw outcomes."
            ),

        "distance_reference":
            "recording contact to stimulation-pair midpoint",

        "outcome_dependent_spatial_exclusion":
            False,
    },

    # --------------------------------------------------------
    # Mechanical signal QC
    # --------------------------------------------------------

    "mechanical_qc": {

        "full_epoch_required":
            True,

        "required_coverage":
            "baseline + early + late windows",

        "empty_window":
            "exclude trial",

        "nonfinite_primary_trial_feature":
            "exclude trial",

        "manual_visual_trial_rejection_primary":
            False,

        "response_amplitude_threshold_primary":
            None,

        "artifact_amplitude_threshold_primary":
            None,

        "model_performance_based_trial_rejection":
            False,

        "notes": (
            "No new response-amplitude or waveform-shape "
            "artifact threshold is introduced into the "
            "primary pipeline because none was present in "
            "the development phenotype implementation. "
            "Stimulating contacts and frozen spatial "
            "near-field exclusions remain excluded."
        ),
    },

    # --------------------------------------------------------
    # Statistical lockbox endpoint
    # --------------------------------------------------------

    "primary_endpoint": {

        "response":
            "late_rms_z",

        "context_k":
            10,

        "contrast":
            "relation - relation_perm",

        "relation_perm_definition":
            (
                "Within the same recording channel, "
                "permute stimulation identity while "
                "preserving patient/node/marginal response."
            ),

        "patient_metric":
            "Spearman map correlation",

        "context_repeats":
            20,

        "within_patient_summary":
            "mean across 20 fixed context repeats",

        "group_effect":
            "mean paired patient-level delta",

        "test":
            "two-sided exact patient-level sign-flip/randomization test",

        "alpha":
            0.05,

        "confidence_interval":
            "95% patient-level bootstrap CI of mean paired delta",

        "support_rule": (
            "Positive mean paired delta and exact "
            "two-sided p < 0.05."
        ),
    },

    "secondary_family": {

        "multiplicity":
            "Holm correction",

        "endpoints": [
            "late k10 relation vs node Spearman",
            "late k10 relation vs relation_perm Pearson",
            "late k10 relation vs relation_perm MAE",
            "late k5 relation vs relation_perm Spearman",
            "late k3 relation vs relation_perm Spearman",
            "early k10 relation vs relation_perm Spearman",
        ],
    },

    # --------------------------------------------------------
    # Sensitivity analyses
    # --------------------------------------------------------

    "robustness_policy": {

        "primary_pipeline_may_not_change_after_opening":
            True,

        "alternative_filtering":
            "robustness only",

        "alternative_reference":
            "robustness only",

        "alternative_near_field_thresholds":
            "robustness only",

        "alternative_response_windows":
            "robustness only",

        "alternative_artifact_rules":
            "robustness only",

        "may_replace_primary_result":
            False,
    },

    # --------------------------------------------------------
    # Lockbox interpretation
    # --------------------------------------------------------

    "claim_boundary": {

        "positive_primary_supports": (
            "Predictive stimulation-response correspondence "
            "beyond the frozen patient-global and "
            "recording-node components under external "
            "patient/center/modality domain shift."
        ),

        "does_not_establish": [
            "direct anatomical connectivity",
            "direct effective connectivity",
            "monosynaptic connectivity",
            "direct-versus-polysynaptic mechanism",
            "therapeutic target efficacy",
            "surgical outcome benefit",
        ],
    },

    "raw_signal_status_at_freeze": {

        "ds004080_raw_signals_opened":
            False,

        "external_response_features_computed":
            False,

        "external_model_performance_observed":
            False,
    },

    "post_opening_rule": (
        "After raw lockbox opening, this primary response "
        "protocol, the frozen cohort/context plan, the "
        "deployment checkpoint, and the primary endpoint "
        "must not be changed in response to ds004080 "
        "electrophysiological outcomes."
    ),
}


PROTOCOL.write_text(
    json.dumps(
        protocol,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


protocol_sha = sha256_file(
    PROTOCOL
)


seal = {

    "protocol_id":
        protocol["protocol_id"],

    "response_protocol_sha256":
        protocol_sha,

    "external_lockbox_sha256":
        EXPECTED_LOCKBOX_SHA256,

    "deployment_checkpoint_sha256":
        EXPECTED_DEPLOYMENT_CKPT_SHA256,

    "deployment_manifest_sha256":
        EXPECTED_DEPLOYMENT_MANIFEST_SHA256,

    "development_protocol_sha256":
        EXPECTED_DEVELOPMENT_PROTOCOL_SHA256,

    "roi_bridge_sha256":
        EXPECTED_ROI_BRIDGE_SHA256,

    "development_response_source_sha256":
        source05_sha,

    "pcnd_git_commit":
        git_commit,

    "raw_signals_opened_before_freeze":
        False,

    "external_performance_observed_before_freeze":
        False,
}


SEAL.write_text(
    json.dumps(
        seal,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


seal_sha = sha256_file(
    SEAL
)


print()
print("=" * 100)
print("DS004080 EXTERNAL RESPONSE PROTOCOL FROZEN")
print("=" * 100)

print(
    "PCND Git commit:",
    git_commit,
)

print(
    "development response source SHA256:",
    source05_sha,
)

print(
    "external lockbox SHA256:",
    EXPECTED_LOCKBOX_SHA256,
)

print(
    "deployment checkpoint SHA256:",
    EXPECTED_DEPLOYMENT_CKPT_SHA256,
)

print()
print(
    "RESPONSE PROTOCOL SHA256:",
    protocol_sha,
)

print(
    "RESPONSE FREEZE SEAL SHA256:",
    seal_sha,
)

print()
print(
    "DS004080 RAW SIGNALS OPENED BEFORE FREEZE: NO"
)

print(
    "EXTERNAL MODEL PERFORMANCE OBSERVED: NO"
)

print()
print(
    "PRIMARY PHENOTYPE:"
)

print(
    "  baseline = [-500, -20] ms"
)

print(
    "  early    = [10, 100] ms"
)

print(
    "  late     = [100, 500] ms"
)

print(
    "  baseline normalization = median / 1.4826*MAD"
)

print(
    "  fallback = nanstd -> 1.0"
)

print(
    "  response = RMS(z)"
)

print(
    "  repeated trials = median"
)

print(
    "  additional filtering/rereferencing/resampling = NONE"
)

print()
print(
    "STATUS: READY FOR ONE-SHOT RAW LOCKBOX OPENING"
)
