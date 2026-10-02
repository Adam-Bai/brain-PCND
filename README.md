# PCND

**Patient-Calibrated Neural Perturbation Map Reconstruction from Sparse Intracranial Stimulation**

PCND is a reproducibility release for reconstructing patient-specific perturbational brain maps from sparse intracranial stimulation responses.

This repository contains:

- byte-preserved frozen source code;
- the frozen deployment checkpoint;
- frozen external-lockbox protocol materials;
- the frozen external feature table;
- repeat-level and patient-level inference outputs;
- prespecified statistical materials;
- portable release-organized source code;
- an executable reproduction workflow.

## Quick Start

The executable chain starts from the released **derived external feature table**.
It does not rerun raw-EEG extraction or development training.

```bash
git clone https://github.com/Adam-Bai/brain-PCND.git
cd brain-PCND
conda env create -f environment-minimal.yml
conda activate pcnd-minimal
python reproduce.py --mode audit
python reproduce.py --mode statistics
```

`audit` checks frozen source, weights and features; `statistics` reruns the
prespecified analysis in a temporary directory and compares **all** scientific
JSON fields, including randomization, sensitivities and six secondary endpoints.
The generated timestamp is excluded. Statistics CSV hashes are checked against
the generated files themselves and CSV values are compared semantically; CSV
serialization can differ across library versions. Inference hashes remain exact. Numerical comparisons use relative tolerance
`1e-12` and zero absolute tolerance, so tiny P values cannot silently disappear.

For full external inference, install PyTorch with a build compatible with your
hardware and NVIDIA driver using the [official installation selector](https://pytorch.org/get-started/locally/), then run:

```bash
python reproduce.py --mode external
```

This mode must match both historical inference SHA256 values before running
statistics. The author's recorded GPU runtime is about 154 minutes on RTX 3090;
that runtime and exact byte reproduction have not been independently established
for other hardware/environments. Audit and statistics do not require a GPU.

The minimal environment describes the CPU dependency set. `environment.yml`,
`requirements-lock.txt`, and `provenance/RUNTIME_INFO.txt` are the **historical
full environment snapshot**, including Linux/CUDA packages and a server prefix;
they are not a universal cross-platform installation recipe.

Release integrity and regression checks:

```bash
python tools/build_release_manifest.py --check
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

`provenance/RELEASE_MANIFEST_v1.0.csv` and `RELEASE_SHA256_v1.0.txt` cover release
files other than themselves. The historical `provenance/SHA256SUMS.txt` is
preserved. Software is Apache-2.0; see `DATA_AND_WEIGHTS_NOTICE.md` for separate
checkpoint and derived-data permissions. Citation metadata is in `CITATION.cff`.

---

The central scientific question is:

> Given the same sparse patient-specific observations and lower-order patient/node information, does preserving the correct stimulation-response correspondence provide additional information for reconstructing held-out perturbational maps?

The supported claim is predictive rather than causal:

> After accounting for a transferable anatomical prior, patient-wide response calibration, and persistent recording-node susceptibility, preserving stimulation-response correspondence improves reconstruction of held-out perturbational maps.

PCND is **not** presented as proof of monosynaptic connectivity, direct anatomical connectivity, therapeutic target selection, or improved clinical outcome.

---

## Study design

### Development cohort

**HAPwave / OpenNeuro `ds004696`**

- 8 development patients
- used for architecture development and exploratory analysis
- repeatedly inspected during adaptive development
- not treated as an untouched confirmatory cohort

The frozen development protocol explicitly records this status.

### External lockbox cohort

**RESPect / OpenNeuro `ds004080`**

- 74 eligible patients
- 4,135 usable stimulation sites
- 5,920 frozen context/query plans
- minimum 5 pulses per usable stimulation site
- minimum 20 eligible recording channels per stimulation site
- minimum 11 usable stimulation sites per patient
- 10 mm primary near-field exclusion
- exact stimulating contacts excluded
- no performance-based patient exclusion
- no ROI-coverage-based patient exclusion

The external lockbox was frozen before external response features or model performance were observed.

---

## Response definition

The frozen response phenotypes are operational stimulation-response measures.

### Baseline window

```text
-500 ms to -20 ms
```

### Early response

```text
10 ms to 100 ms
```

### Late response

```text
100 ms to 500 ms
```

Trial-level responses are robustly standardized using the prestimulation baseline and aggregated using RMS-based response features.

Early and late windows are treated as operational phenotypes. Latency alone is not interpreted as proof of direct versus polysynaptic transmission.

---

## Model

PCND decomposes the response for patient `s`, stimulation site `u`, and recording node `i` as

\[
z_{s,u,i}
=
z^{pop}_{u,i}
+
c_s
+
a_{s,i}
+
t_{s,u,i}.
\]

The components are:

- `z_pop`: transferable anatomical population prior;
- `c_s`: patient-wide response calibration;
- `a_s,i`: persistent patient-specific recording-node susceptibility;
- `t_s,u,i`: stimulation-specific relational residual.

The frozen V3f architecture uses:

- hidden dimension: 256
- attention heads: 8
- relation layers: 3
- ROI embedding dimension: 48
- dropout: 0.10
- population learning rate: `2e-4`
- relation learning rate: `2e-4`
- population training steps: 4,000
- relation training steps: 5,000

Frozen loss weights:

```text
prediction        0.25
correlation       0.25
pairwise ranking  0.10
negative control  0.25
negative margin   0.01
```

The relational component is evaluated for context sizes:

```text
k = 3, 5, 10
```

The external protocol additionally freezes `k = 1` context plans, but the relational component is not interpreted as identifiable at `k = 1`.

---

## Anatomical representation

The model uses a fixed set of 16 geometric pair features:

```text
mni_mid_x
mni_mid_y
mni_mid_z
rec_mni_x
rec_mni_y
rec_mni_z
mni_dx
mni_dy
mni_dz
log_mni_distance
mni_axis_xx
mni_axis_yy
mni_axis_zz
mni_axis_xy
mni_axis_xz
mni_axis_yz
```

External anatomy was bridged using a fixed template-space affine and a frozen ROI mapping procedure.

Unmatched ROI labels are mapped to:

```text
UNK
```

The ROI bridge explicitly prohibits:

- fuzzy string matching;
- contralateral fallback;
- semantic nearest-neighbor remapping;
- outcome-dependent remapping;
- patient exclusion based on ROI coverage.

The external anatomical bridge is treated as an approximate template-space interface rather than an exact nonlinear transformation.

---

## Negative control

The principal negative control is a:

**within-recording-channel stimulation permutation**

It preserves:

- patient identity;
- recording-channel identity;
- patient-wide calibration;
- persistent recording-node susceptibility;
- the marginal response distribution within the recording node.

It disrupts:

- the correspondence between stimulation identity and the relational residual response at the same recording node.

Therefore,

```text
relation > relation_perm
```

supports predictive information in stimulation-response correspondence beyond global and node effects.

This contrast does not itself establish direct anatomical connectivity or a particular synaptic pathway.

---

## Frozen external endpoint

The prespecified primary external endpoint is:

```text
response:       late
context:        k = 10
metric:         patient-level Spearman map correlation
contrast:       relation - relation_perm
inference unit: patient
```

Metrics from the 20 frozen context repeats are averaged within patient before group-level inference.

The patient is the sole independent inferential unit.

---

## Primary external result

External cohort:

```text
N = 74 patients
```

Reproduced primary result:

| Quantity | Value |
|---|---:|
| Mean paired ΔSpearman | 0.03732822177853643 |
| Median paired ΔSpearman | 0.030232770117554916 |
| 95% bootstrap CI | [0.03132607729350565, 0.04359921295990589] |
| Wins / ties / losses | 71 / 0 / 3 |
| Monte-Carlo sign-flip p | 9.9999990000001e-08 |
| Exact binomial sign-test sensitivity p | 7.157428404298838e-18 |
| Wilcoxon sensitivity p | 2.0499718078655237e-13 |

The frozen primary support rule was:

```text
positive mean paired delta
AND
two-sided randomization p < 0.05
```

Result:

```text
PRIMARY SUPPORT RULE MET: True
```

---

## Prespecified secondary family

Six secondary endpoints were evaluated with Holm multiplicity correction.

| Endpoint | Mean delta | Wins | Losses | Holm-adjusted p |
|---|---:|---:|---:|---:|
| late k10 relation vs node Spearman | 0.029925 | 64 | 10 | ~3e-6 |
| late k10 relation vs permutation Pearson | 0.015554 | 63 | 11 | ~3e-6 |
| late k10 relation vs permutation MAE reduction | 0.067325 | 74 | 0 | ~3e-6 |
| late k5 relation vs permutation Spearman | 0.034438 | 70 | 4 | ~3e-6 |
| late k3 relation vs permutation Spearman | 0.026903 | 72 | 2 | ~3e-6 |
| early k10 relation vs permutation Spearman | 0.031096 | 72 | 2 | ~3e-6 |

Detailed outputs are available under:

```text
expected_outputs/
```

---

# Reproducibility

The repository provides three reproduction levels.

---

## Level 1 — Integrity audit

No GPU is required.

Run:

```bash
python reproduce.py --mode audit
```

This verifies:

- byte-preserved frozen V3f source SHA256;
- frozen deployment checkpoint SHA256;
- frozen external feature-table SHA256;
- external cohort size;
- expected primary endpoint metadata;
- primary effect and win/tie/loss counts.

Verified output:

```text
[PASS] frozen V3f source SHA256
[PASS] deployment checkpoint SHA256
[PASS] external feature-table SHA256
[PASS] expected external cohort n=74
[PASS] expected primary mean delta 0.037328221779
[PASS] expected wins/ties/losses = 71/0/3

AUDIT COMPLETE
```

---

## Level 2 — Statistical reproduction

Run:

```bash
python reproduce.py --mode statistics
```

This starts from the released frozen patient-level metric summary and reruns the prespecified statistical analysis.

Primary analysis:

```text
200,000 patient bootstrap resamples
seed = 20260928

10,000,000 Monte-Carlo sign-flip randomizations
seed = 20260927
```

Secondary analysis:

```text
6 prespecified endpoints
2,000,000 Monte-Carlo randomizations per endpoint
Holm multiplicity correction
```

Sensitivity analyses:

```text
exact two-sided binomial sign test
Wilcoxon signed-rank test
```

Verified runtime in the release environment:

```text
~10 seconds
```

Verified output:

```text
N patients: 74

mean delta:
0.03732822177853643

bootstrap 95% CI:
[0.03132607729350565, 0.04359921295990589]

wins / ties / losses:
71 / 0 / 3

Monte-Carlo sign-flip p:
9.9999990000001e-08

PRIMARY SUPPORT RULE MET:
True
```

---

## Level 3 — Full frozen external reproduction

Run:

```bash
python reproduce.py --mode external
```

This starts from:

```text
released external V3f feature table
+
frozen deployment checkpoint
+
byte-preserved frozen V3f source
+
frozen context/query plan
```

and reruns:

```text
74-patient external inference
→ 44,400 repeat-level metric rows
→ 2,220 patient-summary rows
→ primary statistics
→ secondary statistics
```

This workflow was executed successfully in the release environment.

### Exact regenerated inference hashes

Regenerated repeat-level metrics:

```text
FROZEN_REPEAT_METRICS.csv.gz

SHA256
3aa6e417972a1b18a511e9d1af70dddd70b4fd72c7db3bed0aadb436b73f49c8
```

Regenerated patient-level summary:

```text
FROZEN_PATIENT_METRIC_SUMMARY.csv

SHA256
0a4b5940653a82c75db9fabc2b256095be58eca891203ecd9f46d0e9d08f5192
```

Both regenerated hashes exactly match the historical frozen inference manifest.

The author verification record provides byte-level matching of those inference artifacts. The release wrapper now enforces both hashes automatically on every successful external-mode run.

### Verified full reproduction runtime

Hardware:

```text
NVIDIA GeForce RTX 3090
24 GB VRAM
```

Observed wall time:

```text
154 min 6.712 s
```

The workload is mixed CPU/GPU and therefore does not continuously saturate the GPU.

See:

```text
REPRODUCTION_VERIFICATION.md
```

for the full verification record.

---

## Important note on result JSON hashes

The frozen statistics script writes:

```python
created_at_utc = datetime.now(timezone.utc).isoformat()
```

when generating `PRIMARY_EXTERNAL_RESULT.json`.

Therefore, a newly regenerated result JSON is **not expected to be byte-identical** to the historical JSON.

Scientific result fields are compared semantically.

By contrast, the regenerated repeat-level and patient-level inference artifacts are deterministic and reproduced the exact historical SHA256 values.

---

# Frozen statistical operationalization

The originally prespecified target was a two-sided patient-level sign-flip randomization test.

For 74 patients, exact enumeration would require:

\[
2^{74}
\]

sign configurations.

Before external model performance was observed, the test was operationalized as a fixed-seed Monte-Carlo approximation to the same sign-flip randomization distribution.

The frozen implementation uses:

```text
B = 10,000,000
seed = 20260927
p = (extreme + 1) / (B + 1)
```

and reports a Monte-Carlo uncertainty interval for the underlying randomization-tail probability.

This operationalization is stored in:

```text
reproduction_inputs/statistics/
PREINFERENCE_STATISTICAL_OPERATIONALIZATION.json
```

---

# Repository structure

```text
brain-PCND/
├── checkpoints/
│   └── pcnd_v3f_all8_deployment.pt
│
├── expected_outputs/
│   ├── PRIMARY_EXTERNAL_RESULT.json
│   ├── PRIMARY_PATIENT_DELTAS.csv
│   └── SECONDARY_FAMILY_RESULTS.csv
│
├── exploratory/
│   └── run_correspondence_robustness.py
│
├── frozen_protocol/
│   ├── development/
│   ├── external/
│   ├── response/
│   └── roi_bridge/
│
├── frozen_source/
│   └── scripts/
│
├── pcnd/
│   ├── anatomy/
│   ├── evaluation/
│   ├── model/
│   ├── response/
│   └── statistics/
│
├── provenance/
│   ├── FILE_MANIFEST.csv
│   ├── ORIGINAL_FILE_MAP.csv
│   ├── RELEASE_PROVENANCE.md
│   ├── RUNTIME_INFO.txt
│   └── SHA256SUMS.txt
│
├── reproduction_inputs/
│   ├── external_features/
│   └── statistics/
│
├── environment.yml
├── requirements-lock.txt
├── reproduce.py
├── REPRODUCTION_VERIFICATION.md
└── README.md
```

---

# Frozen source versus portable source

This distinction is important.

## `frozen_source/`

Contains byte-preserved historical source files used by the frozen experiment.

These files are retained for provenance and SHA256 verification.

They must not be edited for portability.

The primary frozen V3f source SHA256 is:

```text
34750f1cee3a56b24d5049866396ff9dbbf6df8310f2b2e7f3a81f3cfbb7137c
```

## `pcnd/`

Contains release-organized copies of the research code.

Some files include path-portability adaptations and renamed paths for readability.

These copies must not be used as evidence for the historical frozen-source hash.

The mapping between original and release filenames is recorded in:

```text
provenance/ORIGINAL_FILE_MAP.csv
```

---

# Deployment checkpoint

Frozen deployment checkpoint:

```text
checkpoints/pcnd_v3f_all8_deployment.pt
```

SHA256:

```text
399421567b0185383bf8953cdcd6c75891893b5113fca8a3a41ac190eb68a794
```

The checkpoint was trained using all eight HAPwave development patients with checkpoint selection based only on internal HAPwave validation stimulation sites.

External responses were not used for training, normalization, hyperparameter tuning, checkpoint selection, or architecture selection.

---

# External feature table

Released external feature table:

```text
reproduction_inputs/external_features/
DS004080_V3F_EXTERNAL_FEATURES.csv.gz
```

Rows:

```text
290,758
```

Patients:

```text
74
```

Stimulation sites:

```text
4,135
```

SHA256:

```text
18c8e9d1333941f3f8b51a30829cd74cf4aa4acb4108aaac5e2d34ab8707b48b
```

The external feature manifest records that no model checkpoint was loaded and no external model performance was observed during feature construction.

---

# Environment

A reproducibility environment snapshot is provided in:

```text
environment.yml
requirements-lock.txt
provenance/RUNTIME_INFO.txt
```

For the minimal CPU environment, use:

```bash
conda env create -f environment-minimal.yml
conda activate pcnd-minimal
```

The full `environment.yml` is retained as a historical server snapshot, not a portable installation recommendation.

The exact dependency lock is also preserved in:

```text
requirements-lock.txt
```

GPU execution depends on compatible PyTorch, CUDA, NVIDIA driver, and hardware versions.

---

# Data availability

Raw intracranial electrophysiology is **not redistributed** in this repository.

The source datasets should be obtained from their original OpenNeuro records:

```text
HAPwave
OpenNeuro ds004696

RESPect
OpenNeuro ds004080
```

Users should follow the licenses, terms, and citation requirements of the original datasets.

The present repository distributes derived artifacts required for the documented reproduction workflow.

---

# Confirmatory versus exploratory analyses

The external primary endpoint and six secondary endpoints are preserved as the frozen confirmatory analysis family.

Post-lockbox exploratory robustness analyses are stored separately under:

```text
exploratory/
```

They must not be interpreted as prespecified confirmatory analyses.

The exploratory analyses include correspondence perturbation and robustness controls developed after the confirmatory external analysis was completed.

---

# Historical freeze versus public release

Historical freeze manifests retain their original:

- freeze timestamps;
- development repository commits;
- dataset commits;
- SHA256 identifiers.

The public GitHub repository was assembled later for reproducibility and dissemination.

A public-release Git commit must **not** be interpreted as evidence that the historical lockbox was frozen at the time of the public GitHub commit.

See:

```text
provenance/RELEASE_PROVENANCE.md
```

---

# Reproduction verification status

Current verified status:

```text
Frozen-source integrity audit             PASS
Checkpoint integrity audit                PASS
External feature-table integrity audit    PASS
Patient-level statistical reproduction    PASS
Author-side full 74-patient inference     PASS
Repeat-level historical SHA match          PASS
Patient-summary historical SHA match       PASS
Primary statistical reproduction           PASS
Secondary statistical reproduction         PASS
```

---

# Scientific interpretation boundary

The present study supports reconstruction of patient-specific perturbational response maps.

The results do not independently establish:

- monosynaptic connectivity;
- direct structural connectivity;
- a particular physiological pathway;
- therapeutic efficacy;
- optimal stimulation targets;
- improved surgical outcome.

The relational contrast should be interpreted as predictive information in stimulation-response correspondence after accounting for population, patient-wide, and persistent recording-node effects.

---

# Citation

Use `CITATION.cff` for software citation. Its maintainer entry is not a manuscript author list. Cite the repository/version and the corresponding manuscript when available. No archive DOI is claimed until assigned.

---

# License

Original software is licensed under Apache-2.0 (`LICENSE` and `NOTICE`). Checkpoint and derived-data permissions are separate; see `DATA_AND_WEIGHTS_NOTICE.md`.

Licensing of the original OpenNeuro datasets is governed by their upstream dataset records and is not replaced by this repository.
