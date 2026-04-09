# Replication and Data Provenance

## Overview

This document clarifies which datasets, subject subsets, recording modalities, and experimental paradigms are used in each analysis reported in the manuscript and this repository.

## Datasets

All analyses use publicly available data from four separate sources:

| Dataset | Source | Recording | Subjects | Paradigm | Used In |
|---------|--------|-----------|----------|----------|---------|
| COGITATE iEEG Exp. 1 | [Cogitate Consortium](https://cogitate-data.ae.mpg.de/) | ECoG grids/strips (intracranial) | 18 (CE, CF, CG sites) | Visual consciousness (face/object/letter/false-font) | Primary iEEG analysis |
| Zurich SEEG (ds004752) | [OpenNeuro](https://openneuro.org/datasets/ds004752) | SEEG depth electrodes | 15 (sub-01 through sub-15) | Verbal working memory (Sternberg task) | Independent replication |
| Cambridge Propofol (ds005620) | [OpenNeuro](https://openneuro.org/datasets/ds005620) | 65-channel scalp EEG | 20 (1 excluded: sub-1037) | Resting state under graded propofol sedation | Propofol state-contrast analysis |
| ANPHY-Sleep | [OSF](https://osf.io/r26fh/) | 93-channel scalp EEG (10-20/10-05) | 10 | Overnight polysomnography | Sleep state-contrast analysis |

## Provenance Note: Primary iEEG vs Replication SEEG

The primary iEEG analysis and the replication SEEG analysis use **completely separate datasets** from **different sources**.

### What differs between the two analyses:

| Property | Primary (Cogitate ECoG) | Replication (Zurich SEEG) |
|----------|------------------------|---------------------------|
| Dataset source | Cogitate Consortium data portal | OpenNeuro ds004752 |
| Subjects | 18 subjects (CE103, CE110, CF102, CF103, CF104, CF106, CG100, CG101, CG103, CG105, CG107, CG108, CG109, CG110, CG111, CG113, CG114, CG115) | 15 subjects (sub-01 through sub-15) |
| Electrode type | ECoG surface grids and strips | SEEG depth electrodes |
| Recording sites | Lateral cortex (primarily posterior) | Hippocampal / temporal depth |
| Paradigm | Visual consciousness (category perception, task relevance) | Verbal working memory (Sternberg set-size 4/6/8) |
| Condition contrasts | Face vs object vs letter, task-relevant vs irrelevant | Set size (4/6/8), match (IN/OUT) |
| Lab / geography | Cogitate consortium (North American and European sites) | Schweizerische Epilepsie-Klinik, Zurich |
| Line noise | 60 Hz (North American sites) | 50 Hz (European site, Zurich) |
| Sampling rates | 1024-2048 Hz | 2000 or 4096 Hz |
| Re-referencing | Laplacian (surface grid geometry) | None (SEEG geometry precludes surface Laplacian) |
| Sessions used | DurR1 through DurR5 runs | ses-01 only |
| Reference | Melloni et al., 2023 (COGITATE) | Dimakopoulos et al., eLife 2022 |

### Correct characterization

The Zurich SEEG analysis constitutes an **independent replication** of the operator-geometry pipeline on a separate dataset. The two datasets:
- Come from different data sources (Cogitate Consortium vs OpenNeuro)
- Were collected by different labs
- Use different electrode types (ECoG vs SEEG)
- Test different paradigms (visual consciousness vs verbal working memory)
- Include entirely different patient populations

This is a genuine out-of-sample generalization test.

## Analysis-Dataset Mapping

| Analysis | Dataset | Subjects | Key Output |
|----------|---------|----------|------------|
| Band-specific criticality (Table 1) | Cogitate iEEG (ECoG) | 18 | broadband_comparison.json |
| Operator geometry vs criticality (Fig 2) | Cogitate iEEG (ECoG) | 18 | exceptional_points.json |
| Jackknife sensitivity | Cogitate iEEG (ECoG) | 18 | jackknife_sensitivity.json |
| Gap vs alpha independence (Fig 3) | ds005620 (Propofol) | 20 | gap_vs_alpha_test.json |
| Propofol state contrasts (Fig 4-5) | ds005620 (Propofol) | 20 | ep_propofol_eeg.json |
| Phase-randomized surrogates | ds005620 (Propofol) | 5 subset | ep_robustness_checks.json |
| Shared-subspace PCA (Propofol) | ds005620 (Propofol) | 20 | ep_shared_subspace_propofol.json |
| Sleep state contrasts (Fig 6-7) | ANPHY-Sleep (OSF) | 10 | ep_sleep_dynamics.json |
| Multi-block sleep robustness | ANPHY-Sleep (OSF) | 10 | sleep_multiblock_robustness.json |
| Shared-subspace PCA (Sleep) | ANPHY-Sleep (OSF) | 10 | ep_shared_subspace_sleep.json |
| Independent replication | ds004752 (Zurich SEEG) | 15 | ep_advanced_ds004752.json |
| Chirality analysis (Fig 8) | Cogitate iEEG (ECoG) | 18 | chirality.json |
| PAC analysis (Fig 9) | Cogitate iEEG (ECoG) | 18 | cross_frequency.json |

## Implications for Interpretation

1. The independent replication in ds004752 strengthens the claim that operator-geometry signatures are not artifacts of a specific electrode type, paradigm, lab, or dataset. The same pipeline applied to an entirely separate dataset with different electrode modality and cognitive task yields consistent geometry-dynamics relationships.

2. All three within-dataset robustness analyses (jackknife, expanded surrogates, multi-block sleep) address analysis-level robustness within their respective primary datasets.
