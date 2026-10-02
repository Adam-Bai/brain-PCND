# Reproduction verification

## Status

Full frozen external inference and patient-level statistical reproduction
have been rerun successfully from the public-release materials.

This verification is distinct from the historical lockbox freeze.
The historical manifests and timestamps remain unchanged.

## Verification chain

The reproduction used:

1. the byte-preserved frozen V3f source;
2. the frozen all-development deployment checkpoint;
3. the frozen external V3f feature table;
4. the frozen context/query plan;
5. the frozen relation-permutation seed rule;
6. the prespecified patient-level statistical operationalization.

The complete chain was:

external features
→ frozen model inference
→ repeat-level metrics
→ patient-level metrics
→ primary and secondary statistics.

## Exact inference reproduction

Regenerated repeat-level metrics:

SHA256:

`3aa6e417972a1b18a511e9d1af70dddd70b4fd72c7db3bed0aadb436b73f49c8`

Rows: 44,400

Regenerated patient-level metric summary:

SHA256:

`0a4b5940653a82c75db9fabc2b256095be58eca891203ecd9f46d0e9d08f5192`

Rows: 2,220

Both hashes exactly match the historical frozen inference manifest.

## Primary external result

External cohort:

- N = 74 patients

Primary endpoint:

- late response
- context k = 10
- patient-level Spearman map correlation
- relation minus within-recording-channel stimulation permutation

Reproduced result:

- mean paired delta = 0.03732822177853643
- median paired delta = 0.030232770117554916
- 95% bootstrap CI = [0.03132607729350565, 0.04359921295990589]
- wins / ties / losses = 71 / 0 / 3
- Monte-Carlo sign-flip p = 9.9999990000001e-08
- primary support rule = true

The six prespecified secondary endpoints were also reproduced.

## Statistical procedure

Primary inference used:

- 200,000 patient-level bootstrap resamples
- fixed seed 20260928
- 10,000,000 Monte-Carlo sign-flip randomizations
- fixed seed 20260927

Secondary inference used:

- six prespecified endpoints
- 2,000,000 Monte-Carlo randomizations per endpoint
- Holm multiplicity correction

The exact binomial sign test and Wilcoxon signed-rank test were retained
as sensitivity analyses.

## Runtime verification

Full external reproduction was executed on:

- NVIDIA GeForce RTX 3090, 24 GB
- total wall time: 154 min 6.712 s

The inference workload is mixed CPU/GPU and does not require continuous
high GPU utilization.

## Important note on result-file hashes

`PRIMARY_EXTERNAL_RESULT.json` contains a newly generated
`created_at_utc` timestamp on each execution. Therefore the regenerated
JSON file is not expected to be byte-identical to the historical JSON.

Scientific fields are compared semantically instead.

The repeat-level and patient-level inference artifacts are deterministic
and were reproduced with exact historical SHA256 hashes.
