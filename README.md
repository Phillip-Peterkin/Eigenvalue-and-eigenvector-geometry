# Sample Code for: Criticality-Related Measures and Fitted Operator Geometry Covary Across Brain States in Human Electrophysiology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/Manuscript-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Introduction

This repository contains the code and results that accompany Peterkin (2026) "Criticality-related measures and fitted operator geometry covary across brain states in human electrophysiology." The purpose of the code in this repository is to provide full reproducibility of all analyses reported in the manuscript, and to serve as a reference implementation for operator-geometry analysis of neural time series.

Abstract:

> Criticality-related measures summarize proximity to a dynamical regime that balances stability and responsiveness but do not describe the geometry of the fitted multivariate operators that generate those dynamics. Here we ask whether operator-geometry summaries extracted from sliding-window VAR(1) fits add a complementary descriptive coordinate for variation in human electrophysiology, and whether that coordinate is independent of conventional spectral power. In human intracranial recordings from 18 subjects, high-gamma activity was more subcritical than broadband activity (sigma_HG = 0.9735 vs sigma_BB = 0.9908; mixed-effects p < 10^-8), with complementary dissociations in complexity and long-range temporal correlations. Across subjects, a composite operator-geometry score combining eigenvalue crowding and eigenvector non-orthogonality (ND score) covaried strongly with branching ratio (r = 0.86, p < 10^-5) and inversely with Lempel-Ziv complexity (r = -0.68, p = 0.002); this cross-subject association was stable under leave-one-subject-out resampling (jackknife r range: 0.792-0.889). In scalp EEG, minimum eigenvalue spacing was largely independent of alpha power (propofol dataset) and delta power (sleep dataset), yet it distinguished propofol sedation from wakefulness (d = 0.71) and N3 from REM sleep (d = -2.51); these separations persisted under shared-subspace estimation and across alternative spacing metrics. Spectral radius also drifted significantly in the 120 s before N2-to-N3 sleep transitions (p = 0.001; 9/10 subjects; d = 1.19), and geometry-based LOSO classification separated states with AUC of 0.91 (propofol) and 1.00 (N3 vs REM). A pre-submission adversarial falsification battery (seven attacks) confirmed robustness across label destruction, subject jackknife, spectral confound residualization, window-parameter sweeps, and model competition. Together, fitted operator-geometry summaries provide complementary descriptive coordinates that are partly orthogonal to, and partly aligned with, criticality measures, adding a second axis for characterizing inter-individual and state-dependent variability in these datasets.

## Interpretation Guardrails

Before exploring this repository, please note the following constraints on interpretation:

1. **All geometry metrics are derived from fitted VAR(1) operators** applied to sliding windows of neural data. They are descriptive summaries of local linear dynamics, not mechanistic generative models.
2. **The term "EP" (exceptional point) is used as shorthand** for a composite of eigenvalue proximity and eigenvector non-orthogonality in fitted operators. This study does not claim detection of mathematically exact exceptional points in brain data.
3. **Minimum eigenvalue gap is dimension-dependent.** Comparisons are valid only within fixed preprocessing and model dimension (here: 15 PCA components throughout).
4. **Eigenvector condition numbers and overlaps can be unstable in finite samples.** All metrics should be interpreted comparatively across conditions, not as absolute measurements.
5. **Spectral sensitivity magnitude is not specific to neural temporal structure** under phase-randomized surrogate controls (group p = 0.23). The sensitivity metric tracks operator geometry but its absolute magnitude may reflect spectral properties shared with surrogates.
6. **The ds004752 SEEG analysis is a secondary cross-dataset generalization test** using a separate dataset (OpenNeuro ds004752, Zurich) from the primary iEEG analysis (Cogitate Consortium). The two datasets differ in source, lab, electrode type, paradigm, and subject population, but both use the same software pipeline, so shared pipeline assumptions remain. See `REPLICATION_AND_DATA_PROVENANCE.md` for details.
7. **All reported associations are correlational.** No causal claims are made.

> **Legacy label note:** Some repository code, file names, and JSON keys retain the legacy label `ep_score` for backward compatibility. In the manuscript, this quantity is referred to as **ND score**. These labels refer to the same computed composite statistic unless otherwise noted. See `KEY_MIGRATION.md` for a full mapping.

## Installation (Code)

This repository can be downloaded by entering the following commands:

```bash
cd $target_directory
git clone https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry.git
```

## Installation (Dependencies)

The analysis pipeline requires Python >= 3.10. Install all dependencies with:

```bash
cd Eigenvalue-and-eigenvector-geometry
pip install -r code/requirements.txt
```

Alternatively, use conda:

```bash
conda env create -f code/environment.yml
conda activate operator-geometry
```

| Package | Version | Purpose |
|---------|---------|---------|
| MNE-Python | >= 1.6 | EEG/iEEG data loading and preprocessing |
| NumPy | >= 1.24 | Array operations |
| SciPy | >= 1.11 | Signal processing, statistics |
| scikit-learn | >= 1.3 | PCA, LDA decoding |
| pandas | >= 2.0 | Data manipulation |
| statsmodels | >= 0.14 | Linear mixed-effects models |
| powerlaw | >= 1.5 | Avalanche size distribution fitting |
| antropy | >= 0.1.6 | Lempel-Ziv complexity, DFA |
| neurokit2 | >= 0.2 | Signal complexity measures |
| Matplotlib | >= 3.7 | Figure generation |
| seaborn | >= 0.12 | Statistical visualization |
| h5py | >= 3.9 | HDF5 intermediate storage |
| PyYAML | >= 6.0 | Pipeline configuration |

## Installation (Data)

The full pipeline requires four public datasets. No raw data is included in this repository.

| Dataset | Subjects | Source | Link / DOI | Used For |
|---------|----------|--------|------------|----------|
| COGITATE iEEG Exp. 1 (ECoG) | 18 | Cogitate Consortium | [cogitate-data.ae.mpg.de](https://cogitate-data.ae.mpg.de/) | Primary iEEG analysis |
| Zurich SEEG (ds004752) | 15 | OpenNeuro | [10.18112/openneuro.ds004752.v1.0.1](https://doi.org/10.18112/openneuro.ds004752.v1.0.1) | Secondary cross-dataset generalization test |
| Cambridge Propofol EEG (ds005620) | 20 | OpenNeuro | [10.18112/openneuro.ds005620.v1.0.0](https://doi.org/10.18112/openneuro.ds005620.v1.0.0) | Propofol state contrasts |
| ANPHY-Sleep polysomnography | 10 | OSF | [10.17605/OSF.IO/R26FH](https://doi.org/10.17605/OSF.IO/R26FH) | Sleep state contrasts |

After downloading, set environment variables pointing to your local copies:

```bash
# Linux / macOS
export IEEG_DATA_ROOT=/path/to/Cogitate_IEEG_EXP1
export DS004752_DATA_ROOT=/path/to/ds004752
export PROPOFOL_DATA_ROOT=/path/to/ds005620
export SLEEP_DATA_ROOT=/path/to/ANPHY-Sleep

# Windows
set IEEG_DATA_ROOT=C:\path\to\Cogitate_IEEG_EXP1
set DS004752_DATA_ROOT=C:\path\to\ds004752
set PROPOFOL_DATA_ROOT=C:\path\to\ds005620
set SLEEP_DATA_ROOT=C:\path\to\ANPHY-Sleep
```

See `data/README_data.md` for detailed download instructions and `REPLICATION_AND_DATA_PROVENANCE.md` for dataset provenance.

## Reproducing the Analysis

### Without External Data

All machine-readable statistical results are provided in `results/json_results/` (27 JSON files including a ds004752 subdirectory). A summary of key statistics is in `results/summary_statistics.csv`. These files are sufficient to verify every quantitative claim in the manuscript without re-running the pipeline.

### With External Data

Pipeline configuration is controlled entirely by `code/config.yaml`. All parameters (PCA components = 15, CSD regularization lambda = 10^-5, VAR window = 500 ms, step = 100 ms, random seed = 42) are versioned for reproducibility.

```bash
cd code
bash run_analysis.sh
```

### Steps to Use the Pipeline

1. **Set parameters**: Edit `code/config.yaml` to set data paths, preprocessing parameters, and statistical thresholds. The default configuration reproduces all manuscript results.

2. **Primary iEEG analysis**: Run `scripts/run_all_subjects.py` to process all 18 ECoG subjects from the Cogitate iEEG dataset. This computes per-subject criticality measures (branching ratio, LZc, DFA, tau), fits sliding-window VAR(1) operators, and extracts eigenvalue geometry metrics.

3. **Broadband comparison**: Run `scripts/run_all_subjects_broadband.py` to repeat the analysis on broadband (unfiltered) data for the HG vs BB comparison.

4. **Secondary cross-dataset generalization test**: Run `scripts/run_ds004752.py` to analyze the 15-subject Zurich SEEG dataset (OpenNeuro ds004752) with the same pipeline. This is a separate dataset from the primary Cogitate iEEG analysis, with different subjects, electrodes (SEEG depth vs ECoG surface), paradigm (Sternberg vs visual consciousness), and lab (Zurich vs Cogitate consortium sites), but uses the same software pipeline. See `REPLICATION_AND_DATA_PROVENANCE.md` for details.

5. **Statistical analyses**: Individual analysis scripts in `scripts/analysis/` compute all reported statistics, including band comparisons, operator-geometry correlations, gap-power independence tests, state contrasts, shared-subspace robustness, surrogate controls, jackknife sensitivity, multi-block sleep robustness, temporal precedence of geometry before sleep transitions (`_temporal_precedence.py`), and the pre-submission adversarial falsification battery (`_falsification_battery.py`).

6. **Results**: All outputs are saved as machine-readable JSON files in `results/json_results/`. Key figures are generated in `results/figures/`.

Please make sure to thoroughly read the docstrings in the code to understand the functionality of each module. If you encounter any problems, please report them as issues in the repository.

## Compiling the Manuscript

```bash
cd manuscript
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex  # third pass for cross-references
```

Requires a LaTeX distribution (e.g., MiKTeX, TeX Live) with `biblatex`, `biber`, `elsarticle`, `amsmath`, `booktabs`, `hyperref`, `siunitx`, `threeparttable`, `orcidlink`, and `microtype`.

## Repository Structure

```
.
+-- manuscript/
|   +-- main.tex                     # Full manuscript (LaTeX, Elsevier/elsarticle)
|   +-- references.bib               # BibLaTeX bibliography
|   +-- figures/                     # All 13 manuscript figures (PNG)
|   +-- tables/
|
+-- code/
|   +-- analysis_pipeline/
|   |   +-- cmcc/                    # Installable Python package
|   |   |   +-- analysis/            # Contrasts, decoding, dynamical systems
|   |   |   +-- features/            # Avalanche, branching, complexity, DFA, entropy, PAC
|   |   |   +-- io/                  # Data loading, schema validation
|   |   |   +-- preprocess/          # Epoching, filtering, QC, re-referencing
|   |   |   +-- viz/                 # Plotting modules
|   |   +-- scripts/                 # Pipeline entry points and analysis scripts
|   +-- config.yaml                  # Pipeline configuration
|   +-- requirements.txt             # Python dependencies
|   +-- environment.yml              # Conda environment
|   +-- run_analysis.sh              # Reproduce everything
|
+-- results/
|   +-- json_results/                # 26 machine-readable JSON output files
|   +-- summary_statistics.csv       # Key headline numbers
|
+-- data/
|   +-- README_data.md               # Dataset download links (no raw data)
|
+-- REPLICATION_AND_DATA_PROVENANCE.md
+-- KEY_MIGRATION.md
+-- CITATION.cff
+-- LICENSE
+-- README.md
```

## Key Results

### Primary Findings

| Metric | Value | Context |
|--------|-------|---------|
| Branching ratio sigma (HG vs BB) | t = -5.74, p = 8.9e-6 | HG more subcritical than BB |
| LME band effect on sigma | coef = -0.017, p = 9.3e-9 | Confirmed by mixed-effects model |
| sigma vs ND score | r = 0.860, p = 4.8e-6 | Cross-subject geometry-criticality link |
| LZc vs ND score | r = -0.684, p = 0.002 | Complexity inversely tracks geometry |
| Propofol: spectral radius shift | d = -1.66, p = 4.8e-7 | Toward instability under sedation |
| Propofol: eigenvalue gap | d = 0.71, p = 0.005 | Tighter spacing under sedation |
| Sleep: N3 vs REM gap | d = -2.51, p = 2.4e-5 | REM = tightest spacing |
| Delta-delta (gap vs sensitivity) | r = -0.683, p = 0.0009 | Survives alpha control (r = -0.676) |
| Temporal precedence: spectral radius (N2→N3) | slope p = 0.0014, d = 1.19, 9/10 subjects | Geometry drifts before sleep staging boundary |
| Temporal precedence: eigenvalue gap (N2→N3) | slope p = 0.00063, d = 1.34, 9/10 subjects | Two geometry dimensions pre-transition |
| Temporal precedence: N2→REM (null) | spectral radius p = 0.46, gap p = 0.38 | Selective to N3 descent, not all transitions |
| LOSO classification: propofol vs awake | AUC = 0.913, label-shuffle p = 0.000 | Geometry features classify state above null |
| LOSO classification: N3 vs REM | AUC = 1.000, label-shuffle p = 0.001 | Perfect geometry-based state separation |

### Robustness and Falsification

| Check | Result | Interpretation |
|-------|--------|----------------|
| Shared-subspace PCA (propofol) | d = 0.78 vs 0.71 (per-state) | Effect strengthens under common PCA |
| Surrogate control (200/subject) | real r = 0.076 vs surr r = 0.097, p = 0.23 | Sensitivity not specific to neural structure |
| Jackknife: sigma vs ND score | 18/18 drops significant, r in [0.79, 0.89] | No single subject drives the correlation |
| Multi-block sleep: N3 vs REM | d = 3.03, p = 7.9e-6 (3-block avg) | Gap contrast not driven by block selection |
| Secondary cross-dataset generalization test (SEEG) | spectral sensitivity p ~ 2.5e-8 | Geometry-dynamics relationships generalize to Zurich SEEG dataset |
| Adversarial: label destruction | p = 0.000 (propofol), p = 0.001 (sleep) | Results not due to label structure |
| Adversarial: subject jackknife | propofol AUC [0.903, 0.983]; sleep [1.000, 1.000] | No influential subject drives results |
| Adversarial: temporal jackknife | sign consistent across all LOO iterations | Pre-transition drift not artefactual |
| Adversarial: spectral confounds | 3/4 features survive residualization (|d| >= 0.5) | Geometry effects not explained by power |
| Adversarial: window attacks | slope retains 84% magnitude at 3x decimation | Robust to window-parameter changes |
| Adversarial: model competition | geometry AUC 0.913 vs alpha-power baseline 0.500 | Geometry not reducible to spectral power |
| Adversarial: feature ablation | PARTIAL — spectral radius dominates | Biological finding, not methodological flaw |

## Figures

| Figure | File | Description |
|--------|------|-------------|
| Figure 1 | `lme_paired_bands.png` | Band-specific criticality in intracranial recordings |
| Figure 2 | `ep_summary.png` | Fitted operator geometry aligns with criticality and complexity across subjects |
| Figure 3 | `gap_vs_alpha_discriminating_test.png` | Eigenvalue gap is independent of alpha power in propofol EEG |
| Figure 4 | `ep_propofol_summary.png` | Propofol reorganizes eigenvalue geometry |
| Figure 5 | `delta_delta_scatter.png` | Gap tightening predicts comparative sensitivity loss under propofol |
| Figure 6 | `sleep_dynamics_summary.png` | Sleep follows a different trajectory from propofol |
| Figure 7 | `pca_scatter_sleep.png` | Geometry state-space across wake, N3, and REM sleep |
| Figure 8 | `trajectory_N2_to_N3_spectral_radius.png` | Spectral radius drifts upward before N2-to-N3 sleep transitions |
| Figure 9 | `trajectory_N2_to_N3_eigenvalue_gap.png` | Eigenvalue gap also shows pre-transition drift toward N3 |
| Figure 10 | `slopes_N2_to_N3_spectral_radius.png` | Per-subject spectral radius pre-transition slopes (N2-to-N3) |
| Figure 11 | `spectral_confound_map.png` | Geometry features largely survive spectral confound residualization |
| Figure 12 | `auc_bars.png` | Geometry-based LOSO classification AUC across state contrasts |
| Figure 13 | `falsification_scorecard.png` | Pre-submission adversarial falsification battery scorecard |

**Supplementary figures** (archived in `results/figures/`, not in the main manuscript): `pac_summary.png` (PAC vs tau), `chirality_summary.png` (chirality and non-Hermitian decomposition).

## Contributors

* Phillip Peterkin (Independent Researcher)

> Citation: Peterkin, P. (2025). Criticality-related measures and fitted operator geometry covary across brain states in human electrophysiology. *Manuscript submitted for publication*.

```bibtex
@article{Peterkin2025,
  author  = {Peterkin, Phillip},
  title   = {Criticality-related measures and fitted operator geometry
             covary across brain states in human electrophysiology},
  year    = {2025},
  note    = {Manuscript submitted for publication},
  url     = {https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry}
}
```

See `CITATION.cff` for machine-readable citation metadata.

## License

"Criticality-related measures and fitted operator geometry" Copyright (c) 2025, Phillip Peterkin. All rights reserved.

This repository uses a dual license:

**Code** (Python scripts, configuration files, shell scripts): MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**Manuscript and figures**: Creative Commons Attribution 4.0 International License (CC-BY-4.0)

You are free to share and adapt the material for any purpose, even commercially, under the following terms: You must give appropriate credit, provide a link to the license, and indicate if changes were made.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode
