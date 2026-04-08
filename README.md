# Operator Geometry of Brain States

**Eigenvalue and eigenvector geometry of fitted linear operators covaries with criticality-related measures and distinguishes brain states in human electrophysiology**

Phillip Peterkin  
Independent Researcher, Albany, Oregon, United States

---

## Project Summary

This repository accompanies the manuscript reporting that criticality-related summary statistics and geometry summaries of fitted linear operators covary systematically in human electrophysiological data. Eigenvalue spacing and eigenvector non-orthogonality measures derived from sliding-window VAR(1) operators provide descriptive coordinates for inter-individual and state-dependent variability that are not reducible to spectral power alone.

## Key Findings

1. **Band-specific criticality**: High-gamma activity is more subcritical than broadband activity in intracranial recordings, with dissociations among branching ratio, complexity, and long-range correlations.
2. **Geometry-criticality covariation**: Branching ratio covaries strongly with a composite geometry score across subjects (r = 0.86, p < 10^-5).
3. **Power independence**: Minimum eigenvalue spacing is largely independent of alpha and delta power.
4. **State discrimination**: Propofol sedation and natural sleep occupy distinguishable regions of fitted operator geometry (propofol gap d = 0.71; sleep N3 vs REM d = -2.51).
5. **Robustness**: Effects survive shared-subspace PCA estimation and generalize across alternative spacing metrics.
6. **Replication**: Key geometry-dynamics relationships replicate in an independent SEEG cohort (ds004752, n = 15).

## Repository Structure

```
operator-geometry-brain-states/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── manuscript/
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   └── tables/
├── code/
│   ├── analysis_pipeline/
│   ├── config.yaml
│   ├── requirements.txt
│   ├── environment.yml
│   └── run_analysis.sh
├── results/
│   ├── json_results/
│   └── summary_statistics.csv
└── data/
    └── README_data.md
```

## How to Reproduce

### Prerequisites

- Python >= 3.10
- Raw data downloaded from the sources listed in `data/README_data.md`

### Setup

```bash
# Option 1: pip
cd code
pip install -r requirements.txt

# Option 2: conda
conda env create -f code/environment.yml
conda activate operator-geometry
```

### Run the full pipeline

```bash
cd code
bash run_analysis.sh
```

### Pre-computed results

All machine-readable statistical results are provided in `results/json_results/`. A summary of key statistics is in `results/summary_statistics.csv`. These files are sufficient to verify every quantitative claim in the manuscript without re-running the pipeline.

## Data Availability

Raw data are publicly available from three sources:

| Dataset | Source | DOI |
|---------|--------|-----|
| Zurich SEEG (iEEG) | OpenNeuro ds004752 | [10.18112/openneuro.ds004752.v1.0.1](https://doi.org/10.18112/openneuro.ds004752.v1.0.1) |
| Cambridge Propofol EEG | OpenNeuro ds005620 | [10.18112/openneuro.ds005620.v1.0.0](https://doi.org/10.18112/openneuro.ds005620.v1.0.0) |
| Sleep EEG | OSF | [10.17605/OSF.IO/R26FH](https://doi.org/10.17605/OSF.IO/R26FH) |

See `data/README_data.md` for detailed instructions.

## Citation

If you use this code or data in your work, please cite:

> Peterkin, P. (2025). Eigenvalue and eigenvector geometry of fitted linear operators covaries with criticality-related measures and distinguishes brain states in human electrophysiology. *Preprint*.

See `CITATION.cff` for machine-readable citation metadata.

## License

- **Code**: MIT License
- **Manuscript and figures**: CC-BY-4.0

See `LICENSE` for full text.

## Software Dependencies

Core dependencies (see `code/requirements.txt` for pinned versions):

- MNE-Python >= 1.6
- NumPy >= 1.24
- SciPy >= 1.11
- scikit-learn >= 1.3
- pandas >= 2.0
- statsmodels >= 0.14
- powerlaw >= 1.5
- antropy >= 0.1.6
- neurokit2 >= 0.2
- Matplotlib >= 3.7
- seaborn >= 0.12
- h5py >= 3.9
- PyYAML >= 6.0

All analyses used Python 3.10+ with random seed fixed at 42.
