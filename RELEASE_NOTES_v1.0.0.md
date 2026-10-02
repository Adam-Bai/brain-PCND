# v1.0.0 release notes

This release hardens verification and packaging; it does not change training,
response phenotypes, coordinates, model weights, context plans, patient/site
eligibility, primary/secondary endpoints or frozen results.

- Full semantic comparison of primary inference, Monte Carlo details,
  sensitivity tests and all six secondary endpoints. The generated
  timestamp is excluded; endpoint ordering is irrelevant. Generated statistical
  CSV hashes are self-verified and CSV contents are compared semantically to
  allow harmless library-dependent serialization differences.
- External mode verifies both regenerated inference SHA256 values before
  computing statistics, then applies the complete scientific comparison.
- Complete release manifests are separate from the preserved historical
  38-file checksum list. Regenerate them after any release-file edit.
- Minimal audit/statistics environment and an explicit optional PyTorch
  installation path; the full server environment remains historical.
- Apache-2.0 software license, separate checkpoint/data permission notice,
  upstream CC0 metadata links and repository citation metadata.

Integrity checks and release-validation regression tests are run locally.
A local statistical rerun passed all primary and secondary fields, including
randomization and sensitivity outputs, plus generated CSV semantic checks. It
is recorded separately from the author's prior GPU verification. No third-party complete raw-EEG or GPU reproduction is claimed.
The public executable chain starts at the released external feature table.

A Zenodo DOI is not included until an archive actually assigns it. Citation
metadata describes the repository maintainer, not unconfirmed manuscript authors.
