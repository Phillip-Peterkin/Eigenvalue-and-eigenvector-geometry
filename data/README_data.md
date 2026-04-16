# Data Availability

This repository does not include raw data. All datasets used in this study are publicly available from the following sources.

## Datasets

### 1. COGITATE iEEG Experiment 1 (Primary iEEG analysis)

- **Source**: Cogitate Consortium
- **Direct link**: https://cogitate-data.ae.mpg.de/
- **Reference**: Melloni et al., 2023
- **Used for**: Primary intracranial criticality and operator-geometry analysis (n=18 ECoG subjects)

### 2. Zurich SEEG (Secondary cross-dataset generalization test)

- **OpenNeuro**: ds004752
- **DOI**: [10.18112/openneuro.ds004752.v1.0.1](https://doi.org/10.18112/openneuro.ds004752.v1.0.1)
- **Direct link**: https://openneuro.org/datasets/ds004752
- **License**: CC0
- **Reference**: Dimakopoulos et al., eLife 2022
- **Used for**: Secondary cross-dataset generalization test of operator-geometry analysis (n=15 SEEG subjects)

### 3. Cambridge Propofol EEG

- **OpenNeuro**: ds005620
- **DOI**: [10.18112/openneuro.ds005620.v1.0.0](https://doi.org/10.18112/openneuro.ds005620.v1.0.0)
- **Direct link**: https://openneuro.org/datasets/ds005620
- **License**: CC0
- **Used for**: Propofol sedation vs. wakefulness operator-geometry analysis (n=20 subjects)

### 4. Sleep EEG

- **OSF**: R26FH
- **DOI**: [10.17605/OSF.IO/R26FH](https://doi.org/10.17605/OSF.IO/R26FH)
- **Direct link**: https://osf.io/r26fh/
- **Used for**: Natural sleep state analysis — Awake, N3, and REM (n=10 subjects)

## Download Instructions

1. Download each dataset to a local directory.
2. Update paths in `code/config.yaml` to point to your local copies.
3. Run `code/run_analysis.sh` to reproduce all results.
