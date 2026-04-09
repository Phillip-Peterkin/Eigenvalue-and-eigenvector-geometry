# Replication and Data Provenance

## Overview

This document clarifies which datasets, subject subsets, recording modalities, and experimental paradigms are used in each analysis reported in the manuscript and this repository.

## Datasets

All analyses use publicly available data from three sources:

| Dataset ID | Source | Recording | Subjects | Paradigm | Used In |
|-----------|--------|-----------|----------|----------|---------|
| ds004752 (Cogitate iEEG Exp. 1) | OpenNeuro | ECoG grids/strips (intracranial) | 18 (CE, CF, CG sites) | Visual consciousness (face/object/letter/false-font) | Primary iEEG analysis |
| ds004752 (Zurich SEEG subset) | OpenNeuro | SEEG depth electrodes | 15 (sub-01 through sub-15) | Verbal working memory (Sternberg task) | Cross-cohort generalization |
| ds005620 (Cambridge Propofol) | OpenNeuro | 65-channel scalp EEG | 20 (1 excluded: sub-1037) | Resting state under graded propofol sedation | Propofol state-contrast analysis |
| ANPHY-Sleep (OSF) | OSF | 93-channel scalp EEG (10-20/10-05) | 10 | Overnight polysomnography | Sleep state-contrast analysis |

## Critical Provenance Note: ds004752

The primary intracranial EEG analysis and the cross-cohort generalization analysis **both draw from the same OpenNeuro dataset (ds004752)**. They are NOT independent datasets.

### What differs between the two analyses:

| Property | Primary (Cogitate ECoG) | Cross-Cohort (Zurich SEEG) |
|----------|------------------------|---------------------------|
| Subjects | 18 subjects (CE103, CE110, CF102, CF103, CF104, CF106, CG100, CG101, CG103, CG105, CG107, CG108, CG109, CG110, CG111, CG113, CG114, CG115) | 15 subjects (sub-01 through sub-15) |
| Electrode type | ECoG surface grids and strips | SEEG depth electrodes |
| Recording sites | Lateral cortex (primarily posterior) | Hippocampal / temporal depth |
| Paradigm | Visual consciousness (category perception, task relevance) | Verbal working memory (Sternberg set-size 4/6/8) |
| Condition contrasts | Face vs object vs letter, task-relevant vs irrelevant | Set size (4/6/8), match (IN/OUT) |
| Line noise | 60 Hz (North American sites) | 50 Hz (European site, Zurich) |
| Sampling rates | 1024-2048 Hz | 2000 or 4096 Hz |
| Re-referencing | Laplacian (surface grid geometry) | None (SEEG geometry precludes surface Laplacian) |
| Sessions used | DurR1 through DurR5 runs | ses-01 only |

### Correct characterization

The Zurich SEEG analysis is a **within-dataset cross-cohort generalization**, not an cross-cohort generalization. The two subject groups share:
- The same OpenNeuro deposit (ds004752)
- The same data-collection consortium context (COGITATE)

They differ in:
- Subject identities (non-overlapping)
- Electrode modality (surface vs depth)
- Experimental paradigm (visual vs verbal)
- Recording site (lab and geography)
- Preprocessing choices (re-referencing, line noise)

This is a meaningful generalization test across recording modality and paradigm within the same multi-site data resource, but it does not constitute an cross-cohort generalization from a separate study.

## Analysis-Dataset Mapping

| Analysis | Dataset | Subjects | Key Output |
|----------|---------|----------|------------|
| Band-specific criticality (Table 1) | ds004752 (ECoG) | 18 | broadband_comparison.json |
| Operator geometry vs criticality (Fig 2) | ds004752 (ECoG) | 18 | exceptional_points.json |
| Jackknife sensitivity | ds004752 (ECoG) | 18 | jackknife_sensitivity.json |
| Gap vs alpha independence (Fig 3) | ds005620 (Propofol) | 20 | gap_vs_alpha_test.json |
| Propofol state contrasts (Fig 4-5) | ds005620 (Propofol) | 20 | ep_propofol_eeg.json |
| Phase-randomized surrogates | ds005620 (Propofol) | 5 subset | ep_robustness_checks.json |
| Shared-subspace PCA (Propofol) | ds005620 (Propofol) | 20 | ep_shared_subspace_propofol.json |
| Sleep state contrasts (Fig 6-7) | ANPHY-Sleep (OSF) | 10 | ep_sleep_dynamics.json |
| Multi-block sleep robustness | ANPHY-Sleep (OSF) | 10 | sleep_multiblock_robustness.json |
| Shared-subspace PCA (Sleep) | ANPHY-Sleep (OSF) | 10 | ep_shared_subspace_sleep.json |
| Cross-cohort generalization | ds004752 (SEEG) | 15 | ep_advanced_ds004752.json |
| Chirality analysis (Fig 8) | ds004752 (ECoG) | 18 | chirality.json |
| PAC analysis (Fig 9) | ds004752 (ECoG) | 18 | cross_frequency.json |

## Implications for Interpretation

1. The cross-cohort generalization strengthens the claim that operator-geometry signatures are not artifacts of a specific electrode type or paradigm, but it cannot rule out dataset-level confounds shared across ds004752 (e.g., common preprocessing pipeline at the acquisition stage, shared BIDS curation decisions).

2. True cross-cohort generalization would require a dataset collected by a different group, with different equipment, at a different institution, uploaded independently.

3. All three robustness analyses (jackknife, expanded surrogates, multi-block sleep) operate within their respective primary datasets and address analysis-level robustness, not dataset-level independence.
