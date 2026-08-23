# Scientific Integrity and Contribution Rules

This repository contains research code. Changes are reviewed not only for software correctness but also for whether they preserve the meaning of the scientific analysis.

## Core principles

- Every confirmatory analysis must map to a stated scientific question and a versioned configuration.
- Exploratory analyses must remain labeled exploratory.
- No change may silently alter subject inclusion, aggregation level, preprocessing, model dimension, temporal windowing, or statistical family.
- Geometry quantities describe fitted first-order vector autoregressive operators. They are not ground-truth neural generators.
- Historical labels such as `ep_score` are provenance labels, not permission to collapse mathematically distinct quantities into one name.
- Negative controls, failed replications, and surrogate results remain part of the public record when they constrain interpretation.

## Data and leakage rules

- Dataset loaders must validate required metadata, units, dimensions, and sampling rate.
- Cross-validation must preserve subject boundaries unless a different design is explicitly justified.
- Future-state prediction must use only information available before the prediction target.
- Principal Component Analysis or other dimensionality reduction used inside predictive evaluation must be fitted only on the appropriate training data unless the analysis is explicitly descriptive rather than predictive.
- State-specific versus shared-subspace Principal Component Analysis must never be switched silently.

## Reproducibility rules

- Canonical analyses load parameters from versioned configuration files.
- Random seeds are fixed and recorded where stochastic procedures are used.
- Locked confirmatory result files are not overwritten casually. A changed scientific result should be written as a new artifact or accompanied by an explicit amendment explaining why the previous artifact changed.
- Result artifacts should record the subject set, configuration or configuration hash, random seed where applicable, and software version information when practical.

## Testing rules

- Mathematical helper functions require synthetic tests with known expected behavior.
- Bug fixes require regression tests when the failure can be captured deterministically.
- Manuscript quantitative claims should remain auditable against machine-readable outputs.
- Audit tests are bookkeeping safeguards. They do not replace mathematical tests of the estimator itself.

## Public terminology

Use descriptive terminology unless stronger language is mathematically and empirically justified.

Preferred examples:

- "fitted-operator geometry" rather than "neural generator geometry"
- "minimum eigenvalue spacing" rather than "distance to an exceptional point"
- "criticality-related branching statistic" when estimator dependence matters
- "secondary cross-dataset generalization check" rather than "replication" when the design is not a direct independent replication
- "pre-boundary association" rather than "control variable" when the evidence is temporal ordering alone

## Release discipline

Before a manuscript-facing release, review `PUBLIC_AUDIT.md`. Open scientific alignment items must either be resolved or remain explicitly disclosed in the public documentation.
