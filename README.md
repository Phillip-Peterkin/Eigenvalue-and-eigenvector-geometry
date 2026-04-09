# Criticality-related measures and fitted operator geometry covary across brain states in human electrophysiology

Covaries with Criticality-Related Measures and Distinguishes Brain States in Human Electrophysiology

Phillip Peterkin | Independent Researcher | [ORCID: 0009-0006-4525-6685](https://orcid.org/0009-0006-4525-6685)

---

## Overview

This repository contains the complete manuscript, analysis pipeline, and machine-readable results for a study that:

1. **Demonstrates band-specific criticality regimes** in intracranial recordings: high-gamma is more subcritical than broadband (branching ratio $\sigma_{\mathrm{HG}}=0.974$ vs. $\sigma_{\mathrm{BB}}=0.991$; mixed-effects $p < 10^{-8}$; $n = 18$ subjects), with dissociations among branching ratio, complexity, and long-range temporal correlations.
2. **Identifies strong cross-subject covariation** between criticality-related measures and fitted operator geometry: branching ratio correlates with a composite geometry score ($r = 0.86$, $p < 10^{-5}$) and inversely with Lempel–Ziv complexity ($r = -0.68$, $p = 0.002$).
3. **Establishes power independence** of minimum eigenvalue spacing: gap–alpha correlations near zero in propofol (max $|r| = 0.103$); gap–delta correlations near zero in sleep (mean $r = -0.001$).
4. **Distinguishes brain states** via fitted operator geometry: propofol tightens eigenvalue spacing ($d = 0.71$); REM sleep shows the tightest spacing of any state ($d = -2.51$ vs. N3), dissociating pharmacological from physiological routes to reduced arousal.
5. **Validates robustness** through shared-subspace PCA (all headline effects survive), alternative nearest-neighbor spacing metrics, phase-randomized surrogates, and PCA dimensionality sweeps.
6. **Replicates key geometry–dynamics relationships** in an independent Zurich SEEG cohort (ds004752; $n = 15$; spectral sensitivity $p \approx 2.5 \times 10^{-8}$).

---

## Repository Structure

```
.
├── manuscript/
│   ├── main.tex                     # Full manuscript (LaTeX, Elsevier/elsarticle format)
│   ├── references.bib               # BibLaTeX bibliography (APA style, biber backend)
│   ├── figures/
│   │   ├── lme_paired_bands.png     # Figure 1: Band-specific criticality (HG vs BB)
│   │   ├── ep_summary.png           # Figure 2: Operator geometry ↔ criticality covariation
│   │   ├── gap_vs_alpha_discriminating_test.png  # Figure 3: Gap–alpha independence
│   │   ├── ep_propofol_summary.png  # Figure 4: Propofol eigenvalue geometry
│   │   ├── delta_delta_scatter.png  # Figure 5: Gap tightening ↔ sensitivity loss
│   │   ├── sleep_dynamics_summary.png   # Figure 6: Sleep trajectory
│   │   ├── sleep_gap_histograms.png     # Figure 7: Gap distributions (W, N3, REM)
│   │   ├── chirality_summary.png    # Figure 8: Chirality & non-Hermitian decomposition
│   │   └── pac_summary.png          # Figure 9: PAC ↔ τ selective coupling
│   └── tables/                      # (empty; tables are inline in main.tex)
│
├── code/
│   ├── analysis_pipeline/           # Full CMCC analysis package
│   │   ├── cmcc/                    # Installable Python package
│   │   │   ├── analysis/            # Contrasts, decoding, dynamical systems, EP advanced
│   │   │   ├── features/            # Avalanche, branching, complexity, DFA, entropy, PAC, powerlaw
│   │   │   ├── io/                  # Data loading, schema validation, artifact handling
│   │   │   ├── preprocess/          # Epoching, filtering, QC, re-referencing, scalp/seizure EEG
│   │   │   ├── viz/                 # Comparisons, distributions, spatial maps, summary plots
│   │   │   ├── config.py            # Configuration loading & validation
│   │   │   └── provenance.py        # Pipeline provenance logging
│   │   └── scripts/
│   │       ├── run_all_subjects.py          # iEEG pipeline (all subjects)
│   │       ├── run_all_subjects_broadband.py # Broadband comparison pipeline
│   │       ├── run_ds004752.py              # Replication dataset pipeline
│   │       ├── run_pipeline.py              # Main orchestrator
│   │       └── analysis/                    # 41 individual analysis scripts
│   │           ├── _broadband_comparison.py
│   │           ├── _exceptional_points.py
│   │           ├── _ep_propofol_eeg.py
│   │           ├── _ep_sleep_dynamics.py
│   │           ├── _ep_shared_subspace_propofol.py
│   │           ├── _ep_shared_subspace_sleep.py
│   │           ├── _ep_robustness_checks.py
│   │           ├── _chirality.py
│   │           ├── _cross_frequency.py
│   │           ├── _gap_vs_alpha_test.py
│   │           ├── _lme_analysis.py
│   │           └── ...              # 30 additional analysis scripts
│   ├── config.yaml                  # Pipeline configuration (all paths genericized)
│   ├── requirements.txt             # Python dependencies
│   ├── environment.yml              # Conda environment specification
│   └── run_analysis.sh              # Shell script to reproduce everything
│
├── results/
│   ├── json_results/                # Machine-readable statistical outputs (17 files)
│   │   ├── broadband_comparison.json
│   │   ├── lme_results.json
│   │   ├── exceptional_points.json
│   │   ├── gap_vs_alpha_test.json
│   │   ├── ep_propofol_eeg.json
│   │   ├── ep_sleep_dynamics.json
│   │   ├── ep_shared_subspace_propofol.json
│   │   ├── ep_shared_subspace_sleep.json
│   │   ├── ep_robustness_checks.json
│   │   ├── chirality.json
│   │   ├── cross_frequency.json
│   │   ├── hypothesis_analysis.json
│   │   ├── theory_synthesis.json
│   │   └── ds004752/
│   │       └── ep_advanced_ds004752.json   # Replication dataset results
│   └── summary_statistics.csv       # Key headline numbers in tabular form
│
├── data/
│   └── README_data.md               # Links to all three public datasets (no raw data)
│
├── README.md
├── LICENSE                          # Dual: MIT (code) + CC-BY-4.0 (manuscript)
├── CITATION.cff                     # Machine-readable citation (Zenodo DOI placeholder)
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry.git
cd Eigenvalue-and-eigenvector-geometry
pip install -r code/requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| MNE-Python | >= 1.6 | EEG/iEEG data loading and preprocessing |
| NumPy | >= 1.24 | Array operations |
| SciPy | >= 1.11 | Signal processing, statistics |
| scikit-learn | >= 1.3 | PCA, LDA decoding |
| pandas | >= 2.0 | Data manipulation |
| statsmodels | >= 0.14 | Linear mixed-effects models |
| powerlaw | >= 1.5 | Avalanche size distribution fitting |
| antropy | >= 0.1.6 | Lempel–Ziv complexity, DFA |
| neurokit2 | >= 0.2 | Signal complexity measures |
| Matplotlib | >= 3.7 | Figure generation |
| seaborn | >= 0.12 | Statistical visualization |
| h5py | >= 3.9 | HDF5 intermediate storage |
| PyYAML | >= 6.0 | Pipeline configuration |

---

## Running the Pipeline

```bash
# Run the full analysis pipeline
cd code
bash run_analysis.sh
```

Pipeline configuration is controlled entirely by `code/config.yaml`. All parameters (PCA components, CSD regularization, VAR window size, statistical thresholds, random seed = 42) are versioned for reproducibility.

---

## Reproducing the Analysis

### Without External Data

All machine-readable statistical results are provided in `results/json_results/` (18 JSON files). A summary of key statistics is in `results/summary_statistics.csv`. These files are sufficient to verify every quantitative claim in the manuscript without re-running the pipeline.

### With External Data

The full pipeline requires three public datasets:

**Step 1:** Download the datasets (see `data/README_data.md` for DOIs and links).

**Step 2:** Set environment variables pointing to your local copies:

```bash
# Linux / macOS
export IEEG_DATA_ROOT=/path/to/Cogitate_IEEG_EXP1
export PROPOFOL_DATA_ROOT=/path/to/ds005620
export SLEEP_DATA_ROOT=/path/to/ANPHY-Sleep

# Windows
set IEEG_DATA_ROOT=C:\path\to\Cogitate_IEEG_EXP1
set PROPOFOL_DATA_ROOT=C:\path\to\ds005620
set SLEEP_DATA_ROOT=C:\path\to\ANPHY-Sleep
```

**Step 3:** Run the pipeline:

```bash
cd code
bash run_analysis.sh
```

---

## Data Sources

| Dataset | Subjects | Source | DOI | Access |
|---------|----------|--------|-----|--------|
| COGITATE iEEG Exp. 1 | 18–19 | OpenNeuro ds004752 | [10.18112/openneuro.ds004752.v1.0.1](https://doi.org/10.18112/openneuro.ds004752.v1.0.1) | CC0 (public) |
| Cambridge Propofol EEG | 20 | OpenNeuro ds005620 | [10.18112/openneuro.ds005620.v1.0.0](https://doi.org/10.18112/openneuro.ds005620.v1.0.0) | CC0 (public) |
| ANPHY-Sleep polysomnography | 10 | OSF | [10.17605/OSF.IO/R26FH](https://doi.org/10.17605/OSF.IO/R26FH) | Public |
| Zurich SEEG (replication) | 15 | OpenNeuro ds004752 | Same as above | CC0 (public) |

---

## Key Results

| Metric | Value | Context |
|--------|-------|---------|
| Branching ratio σ (HG vs BB) | $t = -5.74$, $p = 8.9 \times 10^{-6}$ | HG more subcritical than BB |
| LME band effect on σ | coef $= -0.017$, $p = 9.3 \times 10^{-9}$ | Confirmed by mixed-effects model |
| σ vs EP score | $r = 0.860$, $p = 4.8 \times 10^{-6}$ | Cross-subject geometry–criticality link |
| LZc vs EP score | $r = -0.684$, $p = 0.002$ | Complexity inversely tracks geometry |
| Gap–alpha independence | max $|r| = 0.103$ | No subject > 0.3 |
| Propofol: spectral radius shift | $d = -1.66$, $p = 4.8 \times 10^{-7}$ | Toward instability under sedation |
| Propofol: eigenvalue gap | $d = 0.71$, $p = 0.005$ | Tighter spacing under sedation |
| Sleep: N3 vs REM gap | $d = -2.51$, $p = 2.4 \times 10^{-5}$ | REM = tightest spacing |
| Sleep: Awake vs REM gap | $d = -2.13$, $p = 8.6 \times 10^{-5}$ | REM narrower than wakefulness |
| Delta-delta (gap ↔ sensitivity) | $r = -0.683$, $p = 0.0009$ | Survives alpha control ($r = -0.676$) |
| Shared-subspace propofol gap | $d = 0.78$ vs. $d = 0.71$ (per-state) | Effect strengthens under common PCA |
| Shared-subspace sleep N3–REM | $d = -2.39$ vs. $d = -2.51$ (per-state) | Modest attenuation; FDR significant |
| Surrogate control | real $r = 0.076$ vs. surr $r = 0.101$ | Sensitivity not specific to neural structure |
| Replication: spectral sensitivity | $r \approx 0.097$, $p \approx 2.5 \times 10^{-8}$ | Independent SEEG cohort (ds004752) |
| Replication: rank vs EP score | $r \approx -0.015$, $p \approx 6 \times 10^{-5}$ | Geometry–dimension relationship |

---

## Figures

| Figure | Description |
|--------|-------------|
| Figure 1 | Band-specific criticality in intracranial recordings (paired comparisons, LME) |
| Figure 2 | Fitted operator-geometry summaries covary with criticality and complexity |
| Figure 3 | Eigenvalue gap is independent of alpha power in propofol EEG |
| Figure 4 | Propofol reorganizes eigenvalue geometry |
| Figure 5 | Gap tightening predicts comparative sensitivity loss under propofol |
| Figure 6 | Sleep follows a different trajectory from propofol |
| Figure 7 | Distribution of minimum eigenvalue gap across Wake, N3, and REM |
| Figure 8 | Chirality and non-Hermitian decomposition |
| Figure 9 | Phase–amplitude coupling is linked selectively to τ |

---

## Compiling the Manuscript

```bash
cd manuscript
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex  # third pass for cross-references
```

Requires a LaTeX distribution (e.g., MiKTeX, TeX Live) with `biblatex`, `biber`, `elsarticle`, `amsmath`, `booktabs`, `hyperref`, `siunitx`, `threeparttable`, `orcidlink`, and `microtype`.

---

## Citation

If you use this code, data, or framework, please cite:

```bibtex
@article{Peterkin2025,
  author  = {Peterkin, Phillip},
  title   = {Eigenvalue and eigenvector geometry of fitted linear operators
             covaries with criticality-related measures and distinguishes
             brain states in human electrophysiology},
  year    = {2025},
  note    = {Manuscript submitted for publication},
  url     = {https://github.com/Phillip-Peterkin/Eigenvalue-and-eigenvector-geometry}
}
```

See `CITATION.cff` for machine-readable citation metadata.

---

## License

- **Code**: MIT License
- **Manuscript and figures**: CC-BY-4.0

See `LICENSE` for full text.

---

## Contact

Phillip Peterkin — peterkin.phillip@gmail.com
