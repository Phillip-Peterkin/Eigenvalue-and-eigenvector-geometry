# Fitted-Operator Geometry as a Complementary Descriptive Axis for Brain-State Discrimination in Human iEEG and Scalp EEG

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Manuscript: CC BY 4.0](https://img.shields.io/badge/Manuscript-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![CI](https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry/actions/workflows/ci.yml/badge.svg)](https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

This repository contains an installable Python analysis package, machine-readable results, robustness and falsification checks, automated tests, and manuscript sources for a descriptive study of fitted linear-operator geometry in human intracranial electroencephalography (iEEG) and scalp electroencephalography (EEG).

The central object is a sliding-window first-order vector autoregressive model, VAR(1). The analysis extracts spectral radius, eigenvalue spacing, eigenvector conditioning, and related summaries from each fitted coefficient matrix. These are treated as properties of fitted local linearizations, not direct measurements of the brain's true dynamical generator.

## Fast technical review

If you are reviewing the project as software or computational research, start with [`TECHNICAL_REVIEW_GUIDE.md`](TECHNICAL_REVIEW_GUIDE.md). It maps the end-to-end architecture to concrete modules, tests, provenance artifacts, leakage controls, failure behavior, and reproduction commands.

For scientific review, read [`PUBLIC_AUDIT.md`](PUBLIC_AUDIT.md) and [`SCIENTIFIC_INTEGRITY.md`](SCIENTIFIC_INTEGRITY.md) before the manuscript. Those files make the project's historical corrections and remaining release gates explicit.

## Scientific scope

The repository intentionally makes narrow claims:

- fitted-operator geometry is descriptive and pipeline-dependent;
- no causal claim is made that fitted geometry generates conscious state, anesthesia state, or sleep stage;
- no claim is made that mathematically exact exceptional points are detected in neural tissue;
- minimum eigenvalue spacing is dimension-dependent and is interpreted comparatively under fixed settings;
- the threshold-derived branching statistic is treated as criticality-related, not as an estimator-independent latent branching parameter;
- OpenNeuro ds004752 is a secondary cross-dataset consistency/generalization analysis, not an independent preregistered replication;
- surrogate analyses constrain absolute sensitivity quantities that can also arise from autoregressive fitting to autocorrelated signals.

## Metric semantics: legacy proximity is not current ND

Earlier analyses stored a quantity under `ep_score` / `mean_ep_score`:

```text
legacy_proximity_score = eigenvector_overlap / (minimum_eigenvalue_gap + 1e-10)
```

The current Near-Degeneracy (ND) score is a different construction. It uses paired valid windows, transformed eigenvalue crowding and eigenvector conditioning, within-analysis-unit standardization, and a sign-normalized first-principal-component projection.

These quantities are not algebraically equivalent. Therefore:

- the historical cross-subject `r ~= 0.86` association belongs to the legacy proximity statistic;
- the checked-in historical `geometry_brain_states.json` artifact contains the legacy proximity values even though an older schema called the feature `nd_score`;
- a simple subject mean of current within-unit standardized ND is not used as a substitute, because that mean is approximately zero by construction;
- any future subject-level current-ND result requires a prospectively specified aggregation rule and a new result artifact.

The current compatibility module documents the historical schema without changing its numbers. See `KEY_MIGRATION.md`, `PUBLIC_AUDIT.md`, and `results/RESULT_SCHEMA_NOTES.md`.

## Main empirical results

The checked-in artifacts support the following descriptive results under the declared pipeline:

| Analysis | Result | Interpretation |
|---|---:|---|
| High-gamma vs broadband branching statistic | paired t(17) = -5.74, p = 8.9e-6 | The two signal definitions differ under this criticality-related estimator |
| Mixed-effects band contrast | coefficient about -0.0173, p = 9.3e-9 | Band difference persists in the mixed-effects analysis |
| Legacy proximity vs branching statistic | r about 0.86 | Historical cross-subject association; not a current-ND result |
| Propofol minimum eigenvalue spacing | d about 0.71 | Fitted spacing differs between awake and sedated states |
| Propofol spectral radius | about 0.9980 awake vs 1.0025 sedated | Small fitted stability-margin shift under sedation |
| Sleep N3 vs REM spacing | d about -2.51 | Strong within-cohort fitted-spacing separation |
| Pre-N3 spectral-radius drift | p = 0.0014; non-overlap p = 0.032 | Small-sample, pipeline-conditional pre-boundary association |
| Historical geometry-feature classifier, awake vs propofol | LOSO AUC = 0.9125 | Subject-preserving discrimination; feature bundle includes legacy proximity field |

The historical sleep classifier reaches AUC = 1.00 for N3 vs REM in the ten-subject cohort. It is reported as a within-cohort upper bound, not an out-of-cohort generalization estimate.

Numbers previously shown in this README as iEEG geometry-only AUC about 0.948 and combined AUC about 0.957 were removed during the August 2026 engineering review because a clean matching checked-in result artifact could not be identified. Public headline numbers must have an inspectable artifact path.

## Quick start

```bash
git clone https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry.git
cd Eigenvalue-and-eigenvector-geometry
python -m pip install -e ".[dev]"
python -m pytest -q
```

The portable suite does not require raw research datasets. It checks mathematical known-answer behavior, synthetic fitted systems, public scientific contracts, and manuscript/result consistency.

## Data

Raw research data are not redistributed here. The analysis record uses four external datasets:

| Dataset | Recording | Role |
|---|---|---|
| COGITATE Experiment 1 | iEEG | Primary intracranial analysis |
| OpenNeuro ds004752 | stereoelectroencephalography | Secondary consistency/generalization analysis |
| OpenNeuro ds005620 | scalp EEG under propofol | Anesthesia state-contrast analysis |
| ANPHY-Sleep | polysomnography / scalp EEG | Sleep-state and pre-transition analysis |

Detailed acquisition links, subject counts, exclusions, and analysis roles are in `REPLICATION_AND_DATA_PROVENANCE.md` and `data/README_data.md`.

Canonical data-root environment variables:

```bash
export IEEG_DATA_ROOT=/path/to/Cogitate_IEEG_EXP1
export DS004752_DATA_ROOT=/path/to/ds004752
export PROPOFOL_DATA_ROOT=/path/to/ds005620
export SLEEP_DATA_ROOT=/path/to/ANPHY-Sleep
```

## Reproducibility layers

1. **Mathematical verification**: synthetic tests exercise spectral radius, spacing, overlap, participation ratio, effective rank, current ND behavior, and fitted VAR recovery on constructed systems.
2. **Artifact verification**: manuscript-audit tests lock reported statistics to checked-in JSON artifacts.
3. **Scientific-contract verification**: public-release tests prevent known terminology and configuration regressions.
4. **Raw-data reproduction**: users with the external datasets can execute the documented canonical runners. Remaining raw-data release gates are listed openly in `PUBLIC_AUDIT.md`.

## Repository layout

```text
.
|-- code/
|   |-- analysis_pipeline/
|   |   |-- cmcc/                  # installable package
|   |   `-- scripts/               # analysis runners
|   `-- config.yaml                # canonical configuration
|-- data/README_data.md
|-- manuscript/
|   |-- main.tex                   # current aligned public manuscript
|   `-- archive/                   # pre-alignment source retained for provenance
|-- results/
|   |-- json_results/              # machine-readable result artifacts
|   `-- RESULT_SCHEMA_NOTES.md     # historical/current field semantics
|-- tests/                         # repository-level audits and unit tests
|-- TECHNICAL_REVIEW_GUIDE.md
|-- PUBLIC_AUDIT.md
|-- SCIENTIFIC_INTEGRITY.md
|-- REPLICATION_AND_DATA_PROVENANCE.md
|-- KEY_MIGRATION.md
|-- CITATION.cff
`-- LICENSE
```

## Review order

A technical reviewer can assess the project quickly in this order:

1. `TECHNICAL_REVIEW_GUIDE.md`
2. `.github/workflows/ci.yml`
3. `tests/unit/test_operator_geometry.py`
4. `tests/test_manuscript_audit.py`
5. `tests/test_public_release_contract.py`
6. `PUBLIC_AUDIT.md`
7. `results/json_results/`
8. `manuscript/main.tex`

## Citation

Citation metadata are in `CITATION.cff`. Software is licensed under the Massachusetts Institute of Technology (MIT) License. Manuscript text and figures are licensed under Creative Commons Attribution 4.0.

## Contact

Phillip Peterkin  
Independent Researcher  
ORCID: 0009-0006-4525-6685
