# Sample Code for: Criticality-Related Measures and Fitted Operator Geometry Covary Across Brain States in Human Electrophysiology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/Manuscript-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Introduction

This repository contains the code and results that accompany Peterkin (2025) "Criticality-related measures and fitted operator geometry covary across brain states in human electrophysiology." The purpose of the code in this repository is to provide full reproducibility of all analyses reported in the manuscript, and to serve as a reference implementation for operator-geometry analysis of neural time series.

Abstract:

> Criticality-related measures are widely used to summarize near-critical brain dynamics. We asked how these measures relate to eigenvalue- and eigenvector-geometry summaries of sliding-window VAR(1) operators fitted under a fixed preprocessing and dimensionality-reduction pipeline. In intracranial recordings, high-gamma activity was more subcritical than broadband activity (branching ratio sigma_HG = 0.974 vs sigma_BB = 0.991; mixed-effects p < 10^-8). Across subjects, a composite operator-geometry score ("EP score," combining eigenvalue proximity and eigenvector non-orthogonality) covaried strongly with branching ratio (r = 0.86, p < 10^-5) and inversely with Lempel-Ziv complexity (r = -0.68, p = 0.002). In scalp EEG, minimum eigenvalue spacing (at fixed model dimension) was largely independent of alpha and delta power yet distinguished propofol sedation from wakefulness (d = 0.71) and REM sleep from N3 (d = -2.51); these separations survived shared-subspace estimation and generalized across alternative local-spacing metrics. Phase-randomized surrogates constrained interpretation of sensitivity: absolute spectral sensitivity was not specific to neural temporal structure, so sensitivity results are interpreted comparatively. Together, fitted operator-geometry summaries provide a complementary set of descriptive coordinates for variation across subjects, frequency bands, and states in these datasets.

## Interpretation Guardrails

Before exploring this repository, please note the following constraints on interpretation:

1. **All geometry metrics are derived from fitted VAR(1) operators** applied to sliding windows of neural data. They are descriptive summaries of local linear dynamics, not mechanistic generative models.
2. **The term "EP" (exceptional point) is used as shorthand** for a composite of eigenvalue proximity and eigenvector non-orthogonality in fitted operators. This study does not claim detection of mathematically exact exceptional points in brain data.
3. **Minimum eigenvalue gap is dimension-dependent.** Comparisons are valid only within fixed preprocessing and model dimension (here: 15 PCA components throughout).
4. **Eigenvector condition numbers and overlaps can be unstable in finite samples.** All metrics should be interpreted comparatively across conditions, not as absolute measurements.
5. **Spectral sensitivity magnitude is not specific to neural temporal structure** under phase-randomized surrogate controls (group p = 0.23). The sensitivity metric tracks operator geometry but its absolute magnitude may reflect spectral properties shared with surrogates.
6. **The ds004752 SEEG analysis is an independent replication** using a completely separate dataset (OpenNeuro ds004752, Zurich) from the primary iEEG analysis (Cogitate Consortium). The two datasets differ in source, lab, electrode type, paradigm, and subject population. See `REPLICATION_AND_DATA_PROVENANCE.md` for details.
7. **All reported associations are correlational.** No causal claims are made.

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
conda activate cmcc
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
| Zurich SEEG (ds004752) | 15 | OpenNeuro | [10.18112/openneuro.ds004752.v1.0.1](https://doi.org/10.18112/openneuro.ds004752.v1.0.1) | Independent replication |
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

All machine-readable statistical results are provided in `results/json_results/` (20 JSON files). A summary of key statistics is in `results/summary_statistics.csv`. These files are sufficient to verify every quantitative claim in the manuscript without re-running the pipeline.

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

4. **Independent replication**: Run `scripts/run_ds004752.py` to analyze the 15-subject Zurich SEEG dataset (OpenNeuro ds004752) with the same pipeline. This is a completely separate dataset from the primary Cogitate iEEG analysis, with different subjects, electrodes (SEEG depth vs ECoG surface), paradigm (Sternberg vs visual consciousness), and lab (Zurich vs Cogitate consortium sites).

5. **Statistical analyses**: Individual analysis scripts in `scripts/analysis/` compute all reported statistics, including band comparisons, operator-geometry correlations, gap-power independence tests, state contrasts, shared-subspace robustness, surrogate controls, jackknife sensitivity, and multi-block sleep robustness.

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
|   +-- figures/                     # All 9 manuscript figures (PNG)
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
|   +-- json_results/                # 20 machine-readable JSON output files
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
| sigma vs geometry score | r = 0.860, p = 4.8e-6 | Cross-subject geometry-criticality link |
| LZc vs geometry score | r = -0.684, p = 0.002 | Complexity inversely tracks geometry |
| Propofol: spectral radius shift | d = -1.66, p = 4.8e-7 | Toward instability under sedation |
| Propofol: eigenvalue gap | d = 0.71, p = 0.005 | Tighter spacing under sedation |
| Sleep: N3 vs REM gap | d = -2.51, p = 2.4e-5 | REM = tightest spacing |
| Delta-delta (gap vs sensitivity) | r = -0.683, p = 0.0009 | Survives alpha control (r = -0.676) |

### Robustness Checks

| Check | Result | Interpretation |
|-------|--------|----------------|
| Shared-subspace PCA (propofol) | d = 0.78 vs 0.71 (per-state) | Effect strengthens under common PCA |
| Surrogate control (200/subject) | real r = 0.076 vs surr r = 0.097, p = 0.23 | Sensitivity not specific to neural structure |
| Jackknife: sigma vs geometry score | 18/18 drops significant, r in [0.79, 0.89] | No single subject drives the correlation |
| Multi-block sleep: N3 vs REM | d = 3.03, p = 7.9e-6 (3-block avg) | Gap contrast not driven by block selection |
| Independent replication (SEEG) | spectral sensitivity p ~ 2.5e-8 | Replicates in separate Zurich dataset |

## Figures

| Figure | Description |
|--------|-------------|
| Figure 1 | Band-specific criticality in intracranial recordings |
| Figure 2 | Fitted operator-geometry summaries covary with criticality and complexity |
| Figure 3 | Eigenvalue gap is independent of alpha power in propofol EEG |
| Figure 4 | Propofol reorganizes eigenvalue geometry of fitted operators |
| Figure 5 | Gap tightening predicts comparative sensitivity loss under propofol |
| Figure 6 | Sleep follows a different trajectory from propofol |
| Figure 7 | Distribution of minimum eigenvalue gap across Wake, N3, and REM |
| Figure 8 | Chirality and non-Hermitian decomposition |
| Figure 9 | Phase-amplitude coupling is linked selectively to tau |

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
