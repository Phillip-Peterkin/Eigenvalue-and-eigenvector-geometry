# Technical Review Guide

This document is a short path for reviewers assessing the repository as computational research and scientific software.

## What to evaluate

The project demonstrates an end-to-end workflow rather than a single notebook:

```text
external electrophysiology datasets
        |
        v
schema-aware loaders and data-root contracts
        |
        v
preprocessing and dimensionality controls
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
machine-readable JSON artifacts
        |
        v
manuscript audit tests and public scientific contracts
```

The most important design choice is separation of *computation* from *interpretation*. Historical artifacts are not rewritten merely because terminology changed. Instead, current code and documentation specify which old fields remain valid numerically and which labels must not be promoted into current constructs.

## Five-minute review path

### 1. Mathematical primitives

Open `code/analysis_pipeline/cmcc/features/operator_geometry.py` and `tests/unit/test_operator_geometry.py`.

The unit tests include known-answer checks for:

- spectral radius;
- minimum eigenvalue spacing;
- real/complex eigenvector overlap;
- historical proximity-score arithmetic;
- PC1-based Near-Degeneracy behavior;
- invalid/unpaired-window handling;
- ambiguous PC1 orientation;
- participation ratio and effective rank; and
- recovery of a known synthetic linear system.

The current ND implementation fails loudly for scientifically ambiguous opposite-sign PC1 loadings rather than silently choosing a convenient orientation.

### 2. Leakage and subject boundaries

Open `SCIENTIFIC_INTEGRITY.md`, then inspect `code/analysis_pipeline/cmcc/analysis/geometry_embedding_legacy.py` and the current compatibility surface `code/analysis_pipeline/cmcc/analysis/geometry_embedding.py`.

Historical state classifiers use leave-one-subject-out splitting and fit standardization inside each training fold before transforming the held-out subject. Overlapping windows are not treated as independent subjects.

Important semantic note: the historical state-space classifier reads `mean_ep_score`. Older code called that column `nd_score`; numerically it is the legacy proximity statistic. `results/RESULT_SCHEMA_NOTES.md` documents the mapping. The repository does not claim that the historical classifier validates the current PC1 ND score.

### 3. Result provenance

Open:

- `results/json_results/broadband_comparison.json`
- `results/json_results/exceptional_points.json`
- `results/json_results/ep_propofol_eeg.json`
- `results/json_results/ep_sleep_dynamics.json`
- `results/json_results/geometry_brain_states.json`
- `results/json_results/temporal_precedence.json` when present

Then inspect `tests/test_manuscript_audit.py`. The purpose of these tests is to prevent prose numbers from drifting away from machine-readable results.

### 4. Falsification and robustness

The project contains controls for several plausible failure modes:

- shared-subspace PCA for state-comparison coordinate dependence;
- alternative nearest-neighbor spacing summaries;
- ridge-regularization sweeps;
- non-overlapping-window checks;
- phase-randomized surrogates; and
- negative/null results retained in the public narrative.

The surrogate result that weakens interpretation of absolute spectral sensitivity is intentionally preserved. A failed falsification control is treated as information that narrows the claim.

### 5. Build and test discipline

Continuous Integration is defined in `.github/workflows/ci.yml`. It installs from a clean runner, checks package dependency consistency, runs Ruff, builds the distributable wheel on the reference Python job, and executes the full portable test suite across supported Python versions.

Local verification:

```bash
python -m pip install -e ".[dev]"
python -m pip check
ruff check code/analysis_pipeline/cmcc tests
python -m pytest -q
python -m pip wheel . --no-deps -w dist
```

Raw research datasets are not required for the portable suite.

## Canonical configuration and data roots

`code/config.yaml` is the canonical public configuration. Dataset locations are supplied through environment variables rather than private absolute paths:

```bash
export IEEG_DATA_ROOT=/path/to/Cogitate_IEEG_EXP1
export DS004752_DATA_ROOT=/path/to/ds004752
export PROPOFOL_DATA_ROOT=/path/to/ds005620
export SLEEP_DATA_ROOT=/path/to/ANPHY-Sleep
```

The broadband/high-gamma distinction is explicit in configuration and the canonical broadband runner validates the effective passband and records provenance.

## Historical versus current metrics

Two constructs must remain separate:

```text
legacy proximity = eigenvector overlap / (minimum gap + 1e-10)
```

and the current ND score, which is a first-principal-component projection of standardized transformed eigenvalue crowding and eigenvector conditioning.

The historical `r ~= 0.86` cross-subject result and the historical state-classification artifacts use the legacy proximity statistic. The current ND implementation is mathematically tested, but a new subject-level current-ND claim requires a prospectively specified aggregation rule and a new raw-data/result run.

## What remains open

The repository is intentionally explicit about unfinished scientific release gates. See `PUBLIC_AUDIT.md` for the authoritative list. The main outstanding raw-data item is end-to-end reproduction of the canonical broadband path against the historical checked-in band-comparison artifact. Current ND also requires a prospectively specified subject-level aggregation before any historical subject-level correlation can be re-expressed using that construct.

These open items are not hidden because technical review is stronger when the boundary between verified, historical, and unresolved work is visible.
