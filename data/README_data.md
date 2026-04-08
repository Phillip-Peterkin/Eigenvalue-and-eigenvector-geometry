# Data Availability

This repository does not include raw data. All datasets used in this study are publicly available from the following sources.

## Datasets

### 1. Zurich SEEG (intracranial EEG)

- **OpenNeuro**: ds004752
- **DOI**: [10.18112/openneuro.ds004752.v1.0.1](https://doi.org/10.18112/openneuro.ds004752.v1.0.1)
- **Direct link**: https://openneuro.org/datasets/ds004752
- **License**: CC0
- **Used for**: Intracranial criticality analysis (n=18 subjects) and independent replication (n=15 subjects)

### 2. Cambridge Propofol EEG

- **OpenNeuro**: ds005620
- **DOI**: [10.18112/openneuro.ds005620.v1.0.0](https://doi.org/10.18112/openneuro.ds005620.v1.0.0)
- **Direct link**: https://openneuro.org/datasets/ds005620
- **License**: CC0
- **Used for**: Propofol sedation vs. wakefulness operator-geometry analysis (n=20 subjects)

### 3. Sleep EEG

- **OSF**: R26FH
- **DOI**: [10.17605/OSF.IO/R26FH](https://doi.org/10.17605/OSF.IO/R26FH)
- **Direct link**: https://osf.io/r26fh/
- **Used for**: Natural sleep state analysis — Awake, N3, and REM (n=10 subjects)

## Download Instructions

1. Download each dataset to a local directory.
2. Update paths in `code/config.yaml` to point to your local copies.
3. Run `code/run_analysis.sh` to reproduce all results.
