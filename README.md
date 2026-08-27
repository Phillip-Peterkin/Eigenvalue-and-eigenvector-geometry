# Fitted-Operator Geometry as a Complementary Descriptive Axis for Brain-State Discrimination in Human iEEG and Scalp EEG

**Phillip Peterkin**  
Independent Researcher, Albany, Oregon, United States  
ORCID: 0009-0006-4525-6685

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Manuscript: CC BY 4.0](https://img.shields.io/badge/Manuscript-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![CI](https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry/actions/workflows/ci.yml/badge.svg)](https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

This repository accompanies the manuscript **Fitted-Operator Geometry as a Complementary Descriptive Axis for Brain-State Discrimination in Human iEEG and Scalp EEG**. It contains the analysis code, machine-readable result artifacts, figures, robustness analyses, reproducibility checks, and manuscript sources used in the study.

[Manuscript source](manuscript/main.tex) | [Technical review guide](TECHNICAL_REVIEW_GUIDE.md) | [Replication and data provenance](REPLICATION_AND_DATA_PROVENANCE.md) | [Public audit](PUBLIC_AUDIT.md)

## Overview

Neural-state analyses commonly summarize recordings through spectral power, complexity, connectivity, and criticality-related statistics. These measures are informative, but they do not directly characterize the geometry of a fitted multivariate dynamical operator.

This study evaluates whether short-window fitted operators provide a complementary descriptive representation of brain state. Human intracranial electroencephalography (iEEG) and scalp electroencephalography (EEG) are analyzed using sliding-window first-order vector autoregressive models, VAR(1). For each window, the fitted coefficient matrix is summarized through eigenvalue and eigenvector quantities, including spectral radius, minimum eigenvalue spacing, and eigenvector conditioning.

The fitted matrices are treated as local linear approximations under a declared preprocessing and estimation pipeline. They are not interpreted as direct measurements of the brain's underlying physical transition operator.

## Analytical framework

For a multichannel signal \(x_t\), the local model is

```text
x(t+1) = A x(t) + error
```

The analysis proceeds as follows:

```text
Multichannel neural recordings
            |
            v
 Dataset-specific preprocessing
            |
            v
      Sliding time windows
            |
            v
       Fit VAR(1) matrix A
            |
            v
  Eigenvalues and eigenvectors
            |
            v
 Spectral and geometric summaries
            |
            v
State, band, and transition analyses
```

Primary fitted-operator quantities include:

- **Spectral radius:** the largest eigenvalue magnitude of the fitted matrix.
- **Minimum eigenvalue spacing:** the smallest pairwise distance between fitted eigenvalues under fixed dimensional settings.
- **Eigenvector condition number:** a measure of sensitivity and non-orthogonality in the fitted eigenvector basis.
- **Composite geometric summaries:** historical and current constructions derived from eigenvalue crowding and eigenvector structure, documented separately where their definitions differ.

## Results

### High-gamma and broadband activity differ under the declared branching-related estimator

In the primary intracranial analysis, the threshold-derived branching statistic was lower in high-gamma activity than in broadband activity, with \(\sigma_{HG}=0.9735\) and \(\sigma_{BB}=0.9908\), paired \(t(17)=-5.74\), \(p=8.9\times10^{-6}\). A mixed-effects band contrast was also significant, with a coefficient of approximately -0.0173 and \(p=9.3\times10^{-9}\).

The branching quantity is treated as a criticality-related statistic under the declared thresholding and preprocessing procedure, not as an estimator-independent measurement of a latent neuronal branching parameter.

### Historical geometry associations are retained with their original metric

An earlier per-window proximity statistic combined closest-pair eigenvector overlap with minimum eigenvalue spacing. Across 18 intracranial electroencephalography subjects, the historical subject-level proximity summary was associated with the branching statistic at approximately \(r=0.86\). Minimum spacing itself was negatively associated with the branching statistic, while other complexity measures showed distinct relationships with the historical geometry summary.

These results are preserved as historical results of the original proximity statistic. They are not relabeled as results of the current Near-Degeneracy score.

### Propofol anesthesia is associated with changes in fitted-operator geometry

In the propofol electroencephalography cohort, minimum eigenvalue spacing differed between awake and sedated states, with an effect size of approximately \(d=0.71\). Spectral radius shifted from approximately 0.9980 while awake to 1.0025 during sedation, and eigenvector conditioning also changed under sedation. The spacing effect remained present under shared-subspace estimation and alternative spacing summaries.

### Sleep stages show distinct fitted-geometric organization

The strongest sleep-state contrast was observed between N3 sleep and rapid eye movement (REM) sleep. Minimum eigenvalue spacing differed with an effect size of approximately \(d=-2.51\). Awake and REM states also differed, while awake and N3 did not show the same headline spacing effect.

### Historical fitted-geometry features discriminate brain states within the studied cohorts

A historical four-feature state representation containing minimum eigenvalue spacing, eigenvector condition number, spectral radius, and the legacy proximity statistic achieved **LOSO AUC = 0.9125** for awake versus propofol. Leave-one-subject-out (LOSO) evaluation keeps each held-out subject separate from model fitting and standardization.

The historical N3-versus-REM sleep classifier reached an area under the receiver operating characteristic curve (AUC) of 1.00 in the ten-subject sleep cohort. This value is treated as a within-cohort upper bound rather than an estimate of external generalization performance.

### Fitted spectral radius changes before scored N2-to-N3 transitions

Across the 120 seconds preceding scored N2-to-N3 transitions, spectral radius changed systematically, with group \(p=0.0014\). A stricter non-overlapping-window validation remained significant at \(p=0.032\). Minimum eigenvalue spacing showed a supportive trend but did not survive the same stricter non-overlap criterion.

The result is interpreted as a pre-boundary association under the fitted pipeline rather than evidence that operator geometry causes the transition.

### Geometry is evaluated against conventional signal structure and falsification controls

The analysis directly tests whether fitted-geometric separation is reducible to conventional signal-power differences. Additional controls include shared-subspace principal component analysis, alternative spacing summaries, ridge-regularization sweeps, non-overlapping windows, and phase-randomized surrogate analyses.

The surrogate analysis produced an important negative result: absolute spectral-sensitivity magnitudes in the tested propofol subset did not exceed the surrogate group mean. That quantity is therefore not promoted as a neural-specific marker.

## Interpretation and scope

The results support fitted-operator geometry as a complementary descriptive axis for neural-state analysis under the declared estimator and preprocessing choices. The principal observation is not that eigenvalues or eigenvectors identify a hidden biological mechanism, but that the organization of fitted local dynamics contains measurable state-associated structure across intracranial, anesthesia, and sleep analyses.

The scalp electroencephalography findings show that state differences can appear in multiple geometric properties, and the pre-transition analysis suggests that some fitted quantities vary before a scored sleep-stage boundary. These findings motivate further testing with larger cohorts, prospectively specified aggregation rules, and external validation data.

Interpretation remains conditional on model order, dimensionality reduction, window length, filtering, finite-sample estimation, and the source recordings themselves. Minimum eigenvalue spacing is dimension-dependent, and composite geometry metrics require explicit definition and provenance.

## Data

Raw research data are not redistributed in this repository. Four external datasets are represented in the analysis record:

| Dataset | Recording | Role |
|---|---|---|
| COGITATE Experiment 1 | intracranial electroencephalography | Primary intracranial analysis |
| OpenNeuro ds004752 | stereoelectroencephalography | Secondary cross-dataset consistency/generalization analysis |
| OpenNeuro ds005620 | scalp electroencephalography under propofol | Anesthesia state-contrast analysis |
| ANPHY-Sleep | polysomnography / scalp electroencephalography | Sleep-state and pre-transition analysis |

Dataset links, subject counts, exclusions, analysis roles, and provenance are documented in [`REPLICATION_AND_DATA_PROVENANCE.md`](REPLICATION_AND_DATA_PROVENANCE.md) and [`data/README_data.md`](data/README_data.md).

## Reproducing the analyses

### Portable test suite

```bash
git clone https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry.git
cd Eigenvalue-and-eigenvector-geometry
python -m pip install -e ".[dev]"
python -m pytest -q
```

The portable suite does not require the raw research datasets. It checks mathematical known-answer behavior, synthetic fitted systems, manuscript-to-result consistency, and public scientific contracts.

### Raw-data configuration

For raw-data reproduction, configure the canonical dataset roots:

```bash
export IEEG_DATA_ROOT=/path/to/Cogitate_IEEG_EXP1
export DS004752_DATA_ROOT=/path/to/ds004752
export PROPOFOL_DATA_ROOT=/path/to/ds005620
export SLEEP_DATA_ROOT=/path/to/ANPHY-Sleep
```

The corresponding runners and configuration are documented in [`TECHNICAL_REVIEW_GUIDE.md`](TECHNICAL_REVIEW_GUIDE.md) and [`code/config.yaml`](code/config.yaml).

## Repository structure

```text
.
|-- code/
|   |-- analysis_pipeline/
|   |   |-- cmcc/                  # installable analysis package
|   |   `-- scripts/               # canonical and historical analysis runners
|   `-- config.yaml                # canonical configuration
|-- data/
|   `-- README_data.md             # dataset and access notes
|-- manuscript/
|   |-- main.tex                   # current public manuscript
|   `-- archive/                   # retained pre-alignment manuscript source
|-- results/
|   |-- figures/                   # result and robustness figures
|   |-- json_results/              # machine-readable result artifacts
|   `-- RESULT_SCHEMA_NOTES.md     # historical/current field semantics
|-- tests/                         # mathematical, scientific-contract, and audit tests
|-- TECHNICAL_REVIEW_GUIDE.md
|-- REPLICATION_AND_DATA_PROVENANCE.md
|-- PUBLIC_AUDIT.md
|-- SCIENTIFIC_INTEGRITY.md
|-- KEY_MIGRATION.md
|-- CITATION.cff
`-- LICENSE
```

## Scientific record and metric provenance

Earlier analyses stored a proximity quantity under `ep_score` and `mean_ep_score`:

```text
legacy_proximity_score = eigenvector_overlap / (minimum_eigenvalue_gap + 1e-10)
```

The current Near-Degeneracy (ND) score is a different construction based on paired valid windows, transformed eigenvalue crowding, eigenvector conditioning, within-analysis-unit standardization, and a sign-normalized first-principal-component projection.

The two quantities are not algebraically equivalent. Consequently, the historical cross-subject association at approximately \(r=0.86\) and the historical state-classification artifacts remain attached to the legacy proximity statistic. A simple subject mean of current within-unit standardized ND is not substituted for those historical results because that mean is approximately zero by construction.

The complete migration record is documented in [`KEY_MIGRATION.md`](KEY_MIGRATION.md), [`PUBLIC_AUDIT.md`](PUBLIC_AUDIT.md), and [`results/RESULT_SCHEMA_NOTES.md`](results/RESULT_SCHEMA_NOTES.md).

## Review and reproducibility record

For software and computational review, begin with [`TECHNICAL_REVIEW_GUIDE.md`](TECHNICAL_REVIEW_GUIDE.md). For scientific review, historical corrections, release gates, subject-boundary rules, and result-integrity controls are documented in [`PUBLIC_AUDIT.md`](PUBLIC_AUDIT.md) and [`SCIENTIFIC_INTEGRITY.md`](SCIENTIFIC_INTEGRITY.md).

The repository separates four reproducibility layers:

1. Mathematical verification using synthetic and known-answer tests.
2. Artifact verification linking reported statistics to checked-in result files.
3. Scientific-contract verification preventing known terminology and configuration regressions.
4. Raw-data reproduction using the documented external datasets and canonical runners.

## Citation

Citation metadata are provided in `CITATION.cff`.

Software is licensed under the Massachusetts Institute of Technology (MIT) License. Manuscript text and figures are licensed under Creative Commons Attribution 4.0.

## Contact

Phillip Peterkin  
Independent Researcher  
ORCID: 0009-0006-4525-6685
