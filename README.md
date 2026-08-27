# Fitted-Operator Geometry as a Complementary Descriptive Axis for Brain-State Discrimination in Human iEEG and Scalp EEG

## The geometry of brain states

**Can the shape of fitted neural dynamics help distinguish wakefulness, anesthesia, and sleep?**

Brain activity is usually described through power, complexity, connectivity, or criticality-related measures. This project asks a complementary question: what can we learn by examining the geometry of a short-window mathematical operator fitted to ongoing neural activity?

Using human intracranial electroencephalography (iEEG) and scalp electroencephalography (EEG), the analysis fits a first-order vector autoregressive model, VAR(1), in sliding windows and tracks how the resulting eigenvalues and eigenvectors change across brain states.

The central result is that **fitted-operator geometry contains measurable information about brain state under the declared analysis pipeline**.

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Manuscript: CC BY 4.0](https://img.shields.io/badge/Manuscript-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![CI](https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry/actions/workflows/ci.yml/badge.svg)](https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## The question

A neural recording can be summarized in many ways. Spectral power asks how much activity exists at different frequencies. Complexity measures ask how structured or compressible the signal is. Criticality-related measures ask whether activity shows scale-free or branching-like organization.

This study asks something different:

> **When the local dynamics of a multichannel neural signal are represented by a fitted matrix, does the geometry of that matrix change with brain state?**

For each short time window, the model is

```text
x(t+1) = A x(t) + error
```

The fitted matrix `A` is then summarized through quantities derived from its eigenvalues and eigenvectors, including:

- **spectral radius**, which describes the largest fitted eigenvalue magnitude;
- **minimum eigenvalue spacing**, which describes how closely packed the fitted eigenvalues are;
- **eigenvector condition number**, which describes sensitivity and non-orthogonality in the fitted eigenvector basis;
- related geometric summaries used for state comparison and robustness testing.

These are treated as properties of fitted local linearizations, not as direct measurements of the brain's true physical dynamical generator.

---

## What emerged

### 1. Anesthesia changes fitted operator geometry

In the propofol EEG cohort, minimum eigenvalue spacing differed between awake and sedated states, with an effect size of approximately **d = 0.71**. Spectral radius also shifted slightly, from approximately **0.9980 while awake to 1.0025 during sedation**.

The historical four-feature fitted-geometry classifier achieved **LOSO AUC = 0.9125** for awake versus propofol, where leave-one-subject-out (LOSO) evaluation keeps each test subject separate from the training set.

![Propofol fitted-geometry state vectors](results/figures/geometry_embedding/state_vectors_propofol.png)

### 2. Sleep stages occupy different fitted-geometric regimes

Sleep showed a different organization. The strongest headline contrast was between N3 deep sleep and rapid eye movement (REM) sleep, where minimum eigenvalue spacing produced an effect size of approximately **d = -2.51**.

The historical within-cohort geometry classifier reached an area under the receiver operating characteristic curve (AUC) of **1.00** for N3 versus REM in the ten-subject sleep cohort. Because this is a small cohort and not an external validation set, it is treated as a within-cohort upper bound rather than a population-level performance guarantee.

![Sleep fitted-geometry state vectors](results/figures/geometry_embedding/state_vectors_sleep.png)

### 3. Geometry changes before a scored sleep-stage transition

Before N2-to-N3 transitions, spectral radius changed systematically across the preceding 120 seconds. The group result was **p = 0.0014**, and a stricter non-overlapping-window validation remained significant at **p = 0.032**.

This is interpreted as a pre-boundary association under the fitted pipeline. It does not establish that the fitted geometry causes the sleep-stage transition.

![Spectral radius before N2-to-N3 transitions](results/figures/temporal_precedence/trajectory_N2_to_N3_spectral_radius.png)

### 4. Geometry is not simply another name for signal power

The project also tests whether fitted-geometric state separation is reducible to conventional power differences. Geometry and power are compared directly rather than assumed to represent the same information.

![Geometry versus power](results/figures/geometry_embedding/geometry_vs_power.png)

---

## Why this is interesting

The study does not argue that eigenvalues or eigenvectors are a hidden code for consciousness. The narrower result is more useful: **a fitted multivariate dynamical representation exposes state structure that can be measured, compared, falsified, and reproduced**.

That matters because two neural states can differ not only in how much activity is present, but potentially in the organization of the fitted local dynamics themselves.

The work therefore treats fitted-operator geometry as a **complementary descriptive axis** alongside power, complexity, and criticality-related measures.

---

## From neural signal to geometry

```text
Multichannel neural recording
            |
            v
     Sliding time windows
            |
            v
      Fit VAR(1) operator A
            |
            v
   Eigenvalues + eigenvectors
            |
            v
  Geometric summary measures
            |
            v
Compare brain states and transitions
```

The same basic analysis logic is used across intracranial and scalp recordings, with dataset-appropriate preprocessing and explicit controls for dimensionality, fitting, leakage, and robustness.

---

## Main empirical results

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

---

## Data

Raw research data are not redistributed in this repository. The analysis record uses four external datasets:

| Dataset | Recording | Role |
|---|---|---|
| COGITATE Experiment 1 | iEEG | Primary intracranial analysis |
| OpenNeuro ds004752 | stereoelectroencephalography | Secondary consistency/generalization analysis |
| OpenNeuro ds005620 | scalp EEG under propofol | Anesthesia state-contrast analysis |
| ANPHY-Sleep | polysomnography / scalp EEG | Sleep-state and pre-transition analysis |

Detailed acquisition links, subject counts, exclusions, and analysis roles are documented in [`REPLICATION_AND_DATA_PROVENANCE.md`](REPLICATION_AND_DATA_PROVENANCE.md) and [`data/README_data.md`](data/README_data.md).

---

## Reproduce the portable test suite

```bash
git clone https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry.git
cd Eigenvalue-and-eigenvector-geometry
python -m pip install -e ".[dev]"
python -m pytest -q
```

The portable suite does not require raw research datasets. It checks mathematical known-answer behavior, synthetic fitted systems, public scientific contracts, and manuscript/result consistency.

For raw-data reproduction, configure the documented dataset roots:

```bash
export IEEG_DATA_ROOT=/path/to/Cogitate_IEEG_EXP1
export DS004752_DATA_ROOT=/path/to/ds004752
export PROPOFOL_DATA_ROOT=/path/to/ds005620
export SLEEP_DATA_ROOT=/path/to/ANPHY-Sleep
```

---

## Scientific boundaries

The public claims are intentionally narrow.

- Fitted-operator geometry is descriptive and pipeline-dependent.
- No causal claim is made that fitted geometry generates conscious state, anesthesia state, or sleep stage.
- No claim is made that mathematically exact exceptional points are detected in neural tissue.
- Minimum eigenvalue spacing is dimension-dependent and is interpreted comparatively under fixed settings.
- The threshold-derived branching statistic is treated as criticality-related, not as an estimator-independent latent branching parameter.
- OpenNeuro ds004752 is a secondary cross-dataset consistency/generalization analysis, not an independent preregistered replication.
- Surrogate analyses constrain absolute sensitivity quantities that can also arise from autoregressive fitting to autocorrelated signals.

---

## Historical metric note

Earlier analyses stored a quantity under `ep_score` / `mean_ep_score`:

```text
legacy_proximity_score = eigenvector_overlap / (minimum_eigenvalue_gap + 1e-10)
```

The current Near-Degeneracy (ND) score is a different construction using paired valid windows, transformed eigenvalue crowding, eigenvector conditioning, within-analysis-unit standardization, and a sign-normalized first-principal-component projection.

The two quantities are not algebraically equivalent. As a result:

- the historical cross-subject `r ~= 0.86` association belongs to the legacy proximity statistic;
- the checked-in historical `geometry_brain_states.json` artifact contains legacy proximity values even though an older schema called the feature `nd_score`;
- a simple subject mean of current within-unit standardized ND is approximately zero by construction and is not substituted for the historical result;
- any future subject-level current-ND result requires a prospectively specified aggregation rule and a new result artifact.

See [`KEY_MIGRATION.md`](KEY_MIGRATION.md), [`PUBLIC_AUDIT.md`](PUBLIC_AUDIT.md), and [`results/RESULT_SCHEMA_NOTES.md`](results/RESULT_SCHEMA_NOTES.md) for the complete record.

---

## For technical and scientific reviewers

The repository includes a separate review layer so the public presentation does not have to carry the entire audit trail on its front page.

For software and computational review, start with [`TECHNICAL_REVIEW_GUIDE.md`](TECHNICAL_REVIEW_GUIDE.md). It maps the end-to-end architecture to concrete modules, tests, provenance artifacts, leakage controls, failure behavior, and reproduction commands.

For scientific review, read [`PUBLIC_AUDIT.md`](PUBLIC_AUDIT.md) and [`SCIENTIFIC_INTEGRITY.md`](SCIENTIFIC_INTEGRITY.md). These documents record historical corrections, open release gates, subject-boundary rules, aggregation rules, and result-integrity controls.

### Reproducibility layers

1. **Mathematical verification:** synthetic tests exercise spectral radius, spacing, overlap, participation ratio, effective rank, current ND behavior, and fitted VAR recovery on constructed systems.
2. **Artifact verification:** manuscript-audit tests lock reported statistics to checked-in JSON artifacts.
3. **Scientific-contract verification:** public-release tests prevent known terminology and configuration regressions.
4. **Raw-data reproduction:** users with the external datasets can execute the documented canonical runners. Remaining raw-data release gates are listed openly in `PUBLIC_AUDIT.md`.

---

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
|   |-- figures/                   # public result figures
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

---

## Manuscript

The full manuscript is available at [`manuscript/main.tex`](manuscript/main.tex).

**Formal title:** *Fitted-Operator Geometry as a Complementary Descriptive Axis for Brain-State Discrimination in Human iEEG and Scalp EEG*

The manuscript contains the complete methods, statistical interpretation, limitations, robustness analyses, and citations.

---

## Citation

Citation metadata are provided in `CITATION.cff`. Software is licensed under the Massachusetts Institute of Technology (MIT) License. Manuscript text and figures are licensed under Creative Commons Attribution 4.0.

## Contact

Phillip Peterkin  
Independent Researcher  
ORCID: 0009-0006-4525-6685
