# JSON Key Interpretation and Metric Migration Guide

## Purpose

Historical result files in `results/json_results/` use keys containing `EP`, shorthand inherited from an earlier exceptional-point framing. Those keys are preserved so that old outputs remain auditable. They should not be interpreted as evidence that mathematically exact exceptional points were detected in neural data.

This document also separates the historical proximity score from the current manuscript Near-Degeneracy (ND) construction.

## Two distinct geometry quantities

### Historical proximity score

Several checked-in result artifacts use:

```text
ep_score = eigenvector_overlap / (minimum_eigenvalue_gap + 1e-10)
```

This is a heuristic proximity score. It combines a small eigenvalue gap with non-orthogonality of the closest eigenvector pair.

### Current manuscript Near-Degeneracy score

The current manuscript defines a distinct score from:

- eigenvalue crowding, `c = -log10(gap + epsilon)`;
- eigenvector non-orthogonality, `k = log10(condition_number + epsilon)`;
- within-condition standardization;
- projection onto the first principal component of the two-feature covariance;
- a sign convention requiring crowding and non-orthogonality to load in the same positive direction.

The historical proximity score and the current ND score are conceptually related but mathematically different. They must not be described as the same computed statistic.

## Important status of the historical r ~= 0.86 result

The checked-in `exceptional_points.json` artifact contains the historical cross-subject correlation stored under `sigma_vs_ep_score`. That result was produced from the legacy proximity-score pipeline.

Until the correlation is recomputed from a prospectively defined subject-level summary of the final manuscript ND construction, public documentation should describe `r ~= 0.86` as a legacy geometry-proximity association rather than as a validated correlation with the final ND score.

A simple subject mean of a within-subject z-scored ND time series is not an acceptable replacement because its mean is approximately zero by construction.

## Key mappings

### `exceptional_points.json`

| JSON key | Interpretation |
|---|---|
| `ep_score_mean` | Mean historical proximity score, `overlap / (gap + 1e-10)` |
| `min_eigenvalue_gap_mean` | Mean minimum eigenvalue spacing |
| `sigma_vs_ep_score` | Branching statistic versus historical proximity score |
| `lzc_vs_ep_score` | Lempel-Ziv complexity versus historical proximity score |
| `tau_vs_ep_score` | Autocorrelation-time summary versus historical proximity score |
| `sigma_vs_min_gap` | Branching statistic versus minimum eigenvalue spacing |
| `lzc_vs_min_gap` | Complexity versus minimum eigenvalue spacing |

### `ep_propofol_eeg.json`

| JSON key | Interpretation |
|---|---|
| `ep_score` | Historical geometry-proximity quantity retained for provenance |
| `spectral_sensitivity` | Spectral-radius/eigenvalue-spacing sensitivity summary under the fitted-operator pipeline |
| `eigenvalue_gap` | Minimum eigenvalue spacing |

### `ep_sleep_dynamics.json`

| JSON key | Interpretation |
|---|---|
| `ep_score` | Historical geometry-proximity quantity retained for provenance |
| `eigenvalue_gap` | Minimum eigenvalue spacing |

### `ep_robustness_checks.json`

| JSON key | Interpretation |
|---|---|
| `surrogate_control` | Phase-randomized surrogate comparison constraining specificity of spectral-sensitivity magnitude |
| `pca_robustness` | Sensitivity to Principal Component Analysis dimensionality |
| `partial_regression_alpha` | Alpha-power-controlled partial correlation |

### `ep_advanced_ds004752.json`

| JSON key | Interpretation |
|---|---|
| `spectral_sensitivity` | Secondary cross-dataset consistency/generalization analysis |
| `svd_dimension` | Effective-rank versus historical geometry-proximity relationship |
| `state_contrast` | Exploratory task-condition contrasts in the Zurich stereoelectroencephalography dataset |

### `jackknife_sensitivity.json`

| JSON key | Interpretation |
|---|---|
| `sigma_vs_ep_score` | Leave-one-subject-out sensitivity of branching statistic versus historical proximity score |
| `lzc_vs_ep_score` | Leave-one-subject-out sensitivity of complexity versus historical proximity score |

## Function names

Historical function names remain callable for backward compatibility. Naming does not change the statistic they compute.

| Function | Status and interpretation |
|---|---|
| `compute_ep_proximity_timecourse()` | Historical name for the proximity-score pipeline. New code should prefer `compute_geometry_proximity_timecourse()`. |
| `compute_geometry_proximity_timecourse()` | Preferred descriptive name for the historical proximity-score pipeline. It does not compute the manuscript ND score. |
| `detect_exceptional_points()` | Historical detector name. It returns candidates based on the historical proximity score. |
| `detect_near_degeneracies()` | Backward-compatible alias for `detect_exceptional_points()`. Despite its name, it still returns the historical proximity-score result and must not be described as the manuscript ND implementation. |
| `compute_nd_score()` | Current manuscript window-level ND construction from eigenvalue crowding and eigenvector condition number. |

No detector function is currently designated as a preferred ND-candidate detector. Reserve that terminology for a future implementation whose selection rule is explicitly based on the final ND construction.

## Future migration rule

If the final manuscript ND analysis is recomputed, write a new machine-readable result artifact with explicit ND naming and metadata. Do not overwrite the historical `exceptional_points.json` file. The goal is to preserve the provenance chain while making the final analysis unambiguous.
