# Software, data and weights notice

## Original software

The original PCND software in `pcnd/`, `frozen_source/`, `exploratory/`,
`reproduce.py`, `tools/`, and `tests/`, and original release documentation
are offered under Apache-2.0 (`LICENSE`). This does not relicense third-party
packages or the upstream datasets. Historical frozen files retain their bytes;
`NOTICE` identifies the software license without changing their hashes.

## Deployment checkpoint

The PCND contributors permit downloading, running, copying and redistributing
`checkpoints/pcnd_v3f_all8_deployment.pt` for research, evaluation and scientific
reproducibility, provided this notice and its provenance accompany it.
The checkpoint is a separate research artifact, not software automatically
covered by the repository's Apache-2.0 license. No broader checkpoint license
is granted by this notice. Contact the repository maintainer for uses outside
this permission. The checkpoint is supplied as-is, without warranty; it has
not been validated for therapeutic target selection or clinical decisions.

## Derived feature and metric tables

`reproduction_inputs/`, `expected_outputs/`, and the tabular artifacts under
`frozen_protocol/` contain derived features, metrics or protocol information.
The contributors permit copying, reuse and redistribution of their
contributions to these artifacts for research and reproducibility with this
notice and source provenance retained. This permission neither replaces nor
restricts the upstream CC0 terms for the underlying dataset contributions.
No new exclusive rights in upstream data are asserted here.

## Upstream records checked on 2026-10-03

Both checked dataset-description records state `License: CC0`:

- HAPwave, OpenNeuro ds004696: [dataset description](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004696/master/dataset_description.json),
  declaring [version 1.0.1](https://doi.org/10.18112/openneuro.ds004696.v1.0.1).
  This checked public metadata is not a new claim about the exact development
  download commit. Please acknowledge Ojeda Valencia and colleagues' source study.
- RESPect, OpenNeuro ds004080: [frozen metadata commit c4fd741](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004080/c4fd7418883e33b024292468eb14da1649f51aae/dataset_description.json),
  declaring [version 1.2.4](https://doi.org/10.18112/openneuro.ds004080.v1.2.4).
  Please acknowledge van Blooijs and colleagues,
  [Developmental trajectory of transmission speed in the human brain](https://doi.org/10.1038/s41593-023-01272-0).

[CC0 public-domain dedication](https://creativecommons.org/publicdomain/zero/1.0/)
remains applicable according to the upstream records. Their acknowledgements
and scientific citations should accompany reuse. Raw HAPwave/RESPect EEG is
not redistributed in this repository. Obtain raw data and its authoritative
metadata from the original OpenNeuro records; no repository software license
replaces those records.
