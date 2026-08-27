# Technical Review Guide

This document provides a short path for reviewers assessing the repository as computational research and scientific software.

## What to evaluate

The project implements an end-to-end workflow rather than a single notebook:

```text
external electrophysiology datasets
        |
        v
fixed cohort + data-root contracts
        |
        v
schema-aware loaders and preprocessing
        |
        v
sliding-window VAR(1) fitting
        |
        v
primitive operator metrics + historical/current composites
        |
        v
subject-level statistics / subject-preserving validation
        |
        v
machine-readable result artifacts
        |
        v
artifact, manuscript, and release-contract tests
```

The central engineering design choice is separation of computation, interpretation, and provenance. Historical artifacts are not rewritten merely because terminology or inference code changed. Current code and documentation instead specify which old fields remain numerically valid, which labels are historical, and which execution paths are approved for new runs.

## Five-minute review path

### 1. Mathematical primitives

Open `code/analysis_pipeline/cmcc/features/operator_geometry.py` and `tests/unit/test_operator_geometry.py`.

The unit tests cover spectral radius, eigenvalue spacing, complex eigenvector overlap, historical proximity-score arithmetic, PC1-based Near-Degeneracy behavior, invalid/unpaired-window handling, ambiguous PC1 orientation, participation ratio, effective rank, and recovery of a known synthetic linear system.

The minimum-gap helper defines leading modes internally by eigenvalue magnitude so truncation is invariant to caller ordering. The current ND implementation fails loudly for scientifically ambiguous opposite-sign PC1 loadings rather than silently choosing a convenient orientation.

### 2. Cohort, leakage, and subject boundaries

The canonical primary intracranial cohort is versioned in `cohorts/cogitate_primary.json` and mirrored in `code/config.yaml`. Canonical reproduction must use this fixed manifest rather than discovering subjects from local directory contents. `tests/unit/test_engineering_hardening.py` enforces synchronization between the manifest, YAML configuration, and installed package defaults.

Open `SCIENTIFIC_INTEGRITY.md`, then inspect `code/analysis_pipeline/cmcc/analysis/geometry_embedding_legacy.py` and the current compatibility surface `code/analysis_pipeline/cmcc/analysis/geometry_embedding.py`.

Historical state classifiers use Leave-One-Subject-Out (LOSO) splitting and fit standardization inside each training fold before transforming the held-out subject. Overlapping windows are not treated as independent subjects.

Important semantic note: the historical state-space classifier reads `mean_ep_score`. Older code called that column `nd_score`; numerically it is the legacy proximity statistic. `results/RESULT_SCHEMA_NOTES.md` documents the mapping. The repository does not claim that the historical classifier validates the current PC1 ND score.

### 3. Result and release provenance

Inspect:

- `release_contract_manifest.json`
- `preregistration_spec.json`
- `results/json_results/broadband_comparison.json`
- `results/json_results/exceptional_points.json`
- `results/json_results/ep_propofol_eeg.json`
- `results/json_results/ep_sleep_dynamics.json`
- `results/json_results/geometry_brain_states.json`
- `results/json_results/temporal_precedence.json`

The release-contract manifest pins the canonical configuration, cohort manifest, repository analysis contract, and headline result artifacts by Git blob hash. Tests fail if those files drift without an explicit manifest update. This is a repository-level drift detector; a signed release tag remains the preferred external immutability boundary.

`cmcc.provenance` records the Git commit, full configuration hash and snapshot, Python/platform information, canonical subject list, cohort manifest, and installed distribution versions using package metadata rather than import-name guesses.

### 4. Falsification and robustness

The project contains controls for several plausible failure modes:

- shared-subspace principal component analysis for state-comparison coordinate dependence;
- alternative nearest-neighbor spacing summaries;
- ridge-regularization sweeps;
- non-overlapping-window checks;
- phase-randomized surrogates; and
- negative/null results retained in the public narrative.

The surrogate result that weakens interpretation of absolute spectral sensitivity is intentionally preserved. A failed falsification control is treated as information that narrows the claim.

### 5. Build and test discipline

Continuous Integration is defined in `.github/workflows/ci.yml`. It tests Python 3.10, 3.11, and 3.12, checks dependency consistency, runs Ruff, executes the portable suite with coverage, rejects broad warning suppression in the installable package, builds and installs the wheel, verifies package/distribution version agreement, and separately runs the suite in the pinned Python 3.11 reference environment.

Local broad-environment verification:

```bash
python -m pip install -e ".[dev]"
python -m pip check
ruff check \
  code/analysis_pipeline/cmcc \
  tests \
  code/analysis_pipeline/scripts/run_all_subjects_broadband_canonical.py \
  code/analysis_pipeline/scripts/analysis/_geometry_brain_states.py
python -m pytest -q --cov=cmcc --cov-fail-under=25
python -m pip wheel . --no-deps -w dist
```

Reference-environment verification on Python 3.11:

```bash
python -m pip install -r requirements-reference.txt
python -m pip install -e . --no-deps
python -m pip check
python -m pytest -q
```

Raw research datasets are not required for the portable suite.

## Canonical configuration and data roots

`code/config.yaml` is the canonical public configuration. `cohorts/cogitate_primary.json` freezes the primary 18-subject COGITATE cohort and expected run IDs. Dataset locations are supplied through environment variables rather than private absolute paths:

```bash
export IEEG_DATA_ROOT=/path/to/Cogitate_IEEG_EXP1
export DS004752_DATA_ROOT=/path/to/ds004752
export PROPOFOL_DATA_ROOT=/path/to/ds005620
export SLEEP_DATA_ROOT=/path/to/ANPHY-Sleep
```

Canonical broadband reproduction is strict by default:

```bash
python code/analysis_pipeline/scripts/run_all_subjects_broadband_canonical.py
```

It fails if an expected subject or expected run is missing, or if any required subject analysis does not complete successfully. The optional `--best-effort` mode discovers local subjects and tolerates partial failure, but it is explicitly exploratory and is not a valid release-reproduction path.

The broadband/high-gamma distinction is explicit in configuration. The canonical broadband wrapper validates and records the effective passband while adapting to the retained historical implementation without changing the locked historical artifact.

## Script classification

`code/analysis_pipeline/scripts/README.md` distinguishes canonical public entry points from historical and exploratory scripts retained for provenance. Historical executability is not equivalent to current approval. New canonical paths must have versioned inputs, explicit failure semantics, CI or contract tests, and non-overwriting outputs.

## Historical versus current metrics

Two constructs must remain separate:

```text
legacy proximity = eigenvector overlap / (minimum gap + 1e-10)
```

and the current ND score, which is a first-principal-component projection of standardized transformed eigenvalue crowding and eigenvector conditioning.

The historical `r ~= 0.86` cross-subject result and historical state-classification artifacts use the legacy proximity statistic. The current ND implementation is mathematically tested, but a new subject-level current-ND claim requires a prospectively specified aggregation rule and a new raw-data/result run.

## Hosting controls outside the repository

Two desirable release controls are GitHub repository settings rather than source-code changes: protect `main` with required CI checks and use signed release tags/commits for formal releases. The repository-level tests and hash manifest do not substitute for those hosting controls.

## What remains open

The main outstanding raw-data gate is end-to-end execution of the strict canonical broadband path against the source dataset and reconciliation with the checked-in historical `broadband_comparison.json` artifact. The source data are intentionally not stored in this repository, so that empirical reconciliation cannot be completed by the portable CI suite.

Current ND also requires a prospectively specified subject-level aggregation before any historical subject-level correlation can be re-expressed using that construct. These open items remain explicit because the boundary between verified, historical, and unresolved work is part of the technical record.
