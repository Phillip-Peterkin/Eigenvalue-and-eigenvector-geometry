# Fitted-Operator Geometry as a Complementary Descriptive Axis for Brain-State Discrimination in Human iEEG and Scalp EEG

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Manuscript: CC BY 4.0](https://img.shields.io/badge/Manuscript-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![CI](https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry/actions/workflows/ci.yml/badge.svg)](https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

This repository contains the analysis code, machine-readable results, robustness checks, tests, and manuscript sources for a descriptive study of fitted linear-operator geometry across human intracranial electroencephalography (iEEG) and scalp electroencephalography (EEG) datasets.

The central object is a sliding-window first-order vector autoregressive model, written as VAR(1). From each fitted coefficient matrix, the analysis extracts quantities such as spectral radius, minimum eigenvalue spacing, eigenvector conditioning, and related geometry summaries. These are treated as properties of fitted local linearizations, not as direct measurements of the brain's true dynamical generator.

## Scientific scope

The public repository is intentionally conservative about what the analyses establish.

- Fitted-operator geometry is descriptive and pipeline-dependent.
- No causal claim is made that operator geometry generates conscious state, anesthesia state, or sleep stage.
- No claim is made that mathematically exact exceptional points are detected in neural tissue.
- Minimum eigenvalue spacing is dimension-dependent and is interpreted only under fixed model and preprocessing settings.
- The Zurich stereoelectroencephalography dataset is a secondary cross-dataset consistency/generalization check, not an independent preregistered replication.
- Surrogate analyses show that some absolute spectral-sensitivity magnitudes can arise from autoregressive fitting to autocorrelated signals. Those quantities are therefore interpreted comparatively, not as standalone neural markers.

See `SCIENTIFIC_INTEGRITY.md`, `REPLICATION_AND_DATA_PROVENANCE.md`, and `ARCHITECTURE.md` for the detailed public audit trail.

## Important metric-status note

Earlier versions of this project used an `ep_score` quantity defined as:

```text
legacy proximity score = eigenvector overlap / (minimum eigenvalue gap + 1e-10)
```

Some checked-in result files retain `ep_score` and `ep_score_mean` keys for provenance and backward compatibility.

The current manuscript uses a distinct Near-Degeneracy (ND) construction based on eigenvalue crowding and eigenvector conditioning. These two quantities must not be treated as algebraically identical. The current repository therefore labels the historical cross-subject `r = 0.86` result as belonging to the legacy proximity-score analysis unless and until it is recomputed and re-audited under the final manuscript ND definition.

This distinction is documented in `KEY_MIGRATION.md` and `PUBLIC_AUDIT.md`. It is preserved deliberately rather than hidden so that the computational history remains inspectable.

## Main empirical results

The checked-in artifacts support the following descriptive results under the declared pipeline:

| Analysis | Result | Interpretation |
|---|---:|---|
| High-gamma versus broadband branching statistic | paired t(17) = -5.74, p = 8.9e-6 | High-gamma and broadband occupy different criticality-related regimes under this estimator |
| Mixed-effects band contrast | coefficient about -0.0173, p = 9.3e-9 | Band difference is also present in the mixed-effects model |
| Legacy geometry-proximity score versus branching statistic | r about 0.86 | Historical cross-subject geometry-criticality association; see metric-status note above |
| Propofol minimum eigenvalue spacing | d about 0.71 | Fitted spacing differs between awake and sedated states |
| Propofol spectral radius | about 0.9980 awake versus 1.0025 sedated | Fitted local dynamics shift toward the unit-circle boundary under sedation |
| Sleep N3 versus rapid-eye-movement spacing | d about -2.51 | Strong within-cohort fitted-geometry separation |
| Pre-N3 spectral-radius drift | p = 0.0014; non-overlapping validation p = 0.032 | Small-sample, pipeline-conditional pre-boundary association |
| Geometry-only iEEG discrimination | leave-one-subject-out area under the curve about 0.948 | Geometry features carry discriminative information within the declared cohort |
| Combined iEEG features | leave-one-subject-out area under the curve about 0.957 | Geometry and criticality features are complementary in this cohort |

The sleep classification result of area under the curve = 1.00 is reported only as a small-sample within-cohort upper bound, not as an out-of-cohort generalization estimate.

## Quick start

```bash
git clone https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry.git
cd Eigenvalue-and-eigenvector-geometry
pip install -e ".[dev]"
pytest
```

The portable test suite does not require the raw research datasets. It checks synthetic mathematical behavior and audits checked-in manuscript/result relationships.

## Data

Raw data are not redistributed in this repository. The analyses use four external datasets:

| Dataset | Recording | Role |
|---|---|---|
| COGITATE Experiment 1 | intracranial electroencephalography | Primary intracranial analysis |
| OpenNeuro ds004752 | stereoelectroencephalography | Secondary cross-dataset generalization/consistency check |
| OpenNeuro ds005620 | scalp electroencephalography under Propofol | State-contrast analysis |
| ANPHY-Sleep | polysomnography / scalp electroencephalography | Sleep-state and pre-transition analysis |

Detailed provenance, subject counts, exclusions, modalities, and source links are in `REPLICATION_AND_DATA_PROVENANCE.md` and `data/README_data.md`.

Canonical environment variables:

```bash
IEEG_DATA_ROOT=/path/to/Cogitate_IEEG_EXP1
DS004752_DATA_ROOT=/path/to/ds004752
PROPOFOL_DATA_ROOT=/path/to/ds005620
SLEEP_DATA_ROOT=/path/to/ANPHY-Sleep
```

## Reproducibility

The repository separates three layers of reproducibility:

1. **Synthetic mathematical verification**: unit tests check quantities such as spectral radius, eigenvalue spacing, eigenvector overlap, participation ratio, effective rank, and operator fitting on constructed inputs with known behavior.
2. **Artifact-level verification**: checked-in JavaScript Object Notation (JSON) results and manuscript audit tests keep reported values tied to machine-readable outputs.
3. **Raw-data reproduction**: users with the external datasets can rerun the full pipeline using the documented data roots and analysis scripts.

The canonical configuration is `code/config.yaml`. Some historical exploratory scripts remain in the repository for provenance but are not part of the canonical reproduction path.

## Repository layout

```text
.
|-- code/
|   |-- analysis_pipeline/
|   |   |-- cmcc/                  # installable analysis package
|   |   `-- scripts/               # pipeline and analysis runners
|   `-- config.yaml                # canonical configuration
|-- data/
|   `-- README_data.md             # data acquisition notes
|-- manuscript/                    # manuscript source and figures
|-- results/
|   |-- json_results/              # machine-readable result artifacts
|   `-- summary_statistics.csv
|-- tests/                         # public audit and synthetic tests
|-- ARCHITECTURE.md
|-- KEY_MIGRATION.md
|-- PUBLIC_AUDIT.md
|-- REPLICATION_AND_DATA_PROVENANCE.md
|-- SCIENTIFIC_INTEGRITY.md
|-- CITATION.cff
`-- LICENSE
```

## Interpretation and review

If you are reviewing the work scientifically, start here:

1. `PUBLIC_AUDIT.md` for known issues, resolved ambiguities, and items that require raw-data recomputation.
2. `REPLICATION_AND_DATA_PROVENANCE.md` for dataset separation and analysis-to-dataset mapping.
3. `KEY_MIGRATION.md` for the legacy `ep_score` versus current ND distinction.
4. `ARCHITECTURE.md` for claim-to-code organization.
5. `results/json_results/` for machine-readable outputs.
6. `tests/` for mathematical and manuscript/result audit checks.

## Citation

Citation metadata are provided in `CITATION.cff`. The manuscript and figures are licensed under Creative Commons Attribution 4.0; software is licensed under the Massachusetts Institute of Technology (MIT) License.

## Contact

Phillip Peterkin  
Independent Researcher  
ORCID: 0009-0006-4525-6685
