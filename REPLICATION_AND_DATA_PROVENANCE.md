# Replication and Data Provenance

## Overview

This document records which datasets, subject groups, recording modalities, and analysis roles are used in the public repository. The term "replication" is reserved for analyses that actually reproduce the same target effect under an appropriately independent design. Cross-dataset consistency or generalization checks are labeled as such.

## Datasets

| Dataset | Source | Recording | Approximate public analysis sample | Paradigm | Repository role |
|---|---|---|---:|---|---|
| COGITATE intracranial electroencephalography Experiment 1 | COGITATE Consortium | Electrocorticography and intracranial recordings | analysis-dependent | Visual consciousness paradigm | Primary intracranial analysis |
| Zurich ds004752 | OpenNeuro | Stereoelectroencephalography | 15 | Verbal working memory | Secondary cross-dataset consistency/generalization check |
| Cambridge ds005620 | OpenNeuro | Scalp electroencephalography | 20 analyzed in principal state contrast | Resting state under Propofol sedation | Propofol state analysis |
| ANPHY-Sleep | Open Science Framework | Polysomnography / scalp electroencephalography | 10 | Overnight sleep | Sleep state and pre-transition analysis |

Exact subject inclusion can differ by feature availability and analysis. The manuscript and machine-readable outputs should be consulted for the sample size attached to each result.

## Primary intracranial versus Zurich stereoelectroencephalography

The COGITATE and Zurich datasets are genuinely separate datasets with different participants, acquisition environments, electrode modalities, tasks, and laboratories. That makes the Zurich analysis useful as a cross-dataset stress test of parts of the analysis framework.

It does **not** make every Zurich result a replication of the COGITATE findings.

In particular, task-condition contrasts did not transfer under the current pipeline. The public manuscript therefore treats Zurich as a secondary consistency/generalization analysis. Geometry-dynamics relationships that show similar direction across the two datasets are supportive, but they share software and analysis assumptions and are not presented as independent preregistered replication.

## Analysis-to-dataset map

| Analysis | Dataset | Role | Main machine-readable output |
|---|---|---|---|
| High-gamma versus broadband criticality-related summaries | COGITATE intracranial data | Primary | `broadband_comparison.json` |
| Historical geometry-proximity versus criticality analysis | COGITATE intracranial data | Historical primary analysis retained for provenance | `exceptional_points.json` |
| Leave-one-subject-out sensitivity of historical geometry association | COGITATE intracranial data | Robustness | `jackknife_sensitivity.json` |
| Minimum-gap versus alpha-power control | Cambridge Propofol | Control | `gap_vs_alpha_test.json` |
| Propofol fitted-geometry state contrasts | Cambridge Propofol | Primary state contrast | `ep_propofol_eeg.json` |
| Phase-randomized sensitivity controls | Cambridge Propofol subset | Falsification / limitation | `ep_robustness_checks.json` |
| Shared-subspace Propofol analysis | Cambridge Propofol | Robustness | `ep_shared_subspace_propofol.json` |
| Sleep fitted-geometry state contrasts | ANPHY-Sleep | Primary state contrast | `ep_sleep_dynamics.json` |
| Pre-transition sleep analysis | ANPHY-Sleep | Small-sample temporal analysis | `temporal_precedence.json` |
| Multi-block sleep analysis | ANPHY-Sleep | Robustness | `sleep_multiblock_robustness.json` |
| Shared-subspace sleep analysis | ANPHY-Sleep | Robustness | `ep_shared_subspace_sleep.json` |
| Adversarial falsification battery | Propofol + ANPHY-Sleep | Stress test | `falsification_battery.json` |
| Zurich cross-dataset analysis | OpenNeuro ds004752 | Secondary consistency/generalization | `ep_advanced_ds004752.json` |
| Chirality-style trajectory summaries | COGITATE intracranial data | Exploratory | `chirality.json` |
| Phase-amplitude coupling analysis | COGITATE intracranial data | Exploratory / secondary | `cross_frequency.json` |

## Interpretation constraints

1. **Separate dataset is not synonymous with replication.** The Zurich analysis uses independent data but tests a different task and recording modality. Results are described as cross-dataset consistency/generalization unless they reproduce the same prespecified target effect.

2. **Shared pipeline assumptions remain.** Applying the same software to multiple datasets is useful for testing portability, but common preprocessing and estimator choices can create common biases.

3. **Propofol and sleep are different observables from intracranial high-gamma.** Scalp electroencephalography analyses use a different acquisition scale and preprocessing pathway. Their value is cross-state and cross-modality consistency, not literal measurement equivalence.

4. **Sleep results are small-sample.** The ANPHY-Sleep cohort contains 10 participants. Classification values, especially an area under the curve of 1.00, are treated as within-cohort descriptive upper bounds rather than prospective generalization estimates.

5. **Temporal precedence is not causal control.** The pre-N3 spectral-radius result is a pre-boundary association under scored sleep-stage transitions. It is not evidence that the fitted operator metric causes the transition.

6. **Historical metric names remain in outputs.** Files beginning with `ep_` and keys such as `ep_score` are retained for provenance. See `KEY_MIGRATION.md` before interpreting them.

## Reproduction notes

Raw data are not redistributed by this repository. Dataset source links and acquisition instructions are maintained in `data/README_data.md`. Canonical local data roots are configured through environment variables documented in the main README.

For public manuscript review, `PUBLIC_AUDIT.md` should be read alongside this file. It records the remaining metric-alignment and broadband-reproduction items that require explicit resolution or continued disclosure.
