# Release provenance

This repository distinguishes frozen historical source artifacts from
portable public-release code.

## Frozen historical source

`frozen_source/` contains byte-preserved source files used by the
historical frozen analysis.

The primary V3f source must satisfy:

SHA256:
34750f1cee3a56b24d5049866396ff9dbbf6df8310f2b2e7f3a81f3cfbb7137c

These files must not be edited to improve portability.

## Portable release code

`pcnd/` contains release-organized copies with path portability changes.
These files are not used as evidence for the historical frozen source hash.

## Historical versus public commits

Historical freeze manifests retain their original PCND development commits
and freeze timestamps.

The GitHub public-release commit is a later packaging/reproducibility commit
and must not be interpreted as evidence of pre-lockbox freezing.
