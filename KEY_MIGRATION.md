# JSON Key Interpretation Guide

## Purpose

The JSON result files in `results/json_results/` use key names that include "EP" (exceptional point) shorthand. These keys are preserved for reproducibility and backward compatibility with the manuscript's notation. This document maps each key to its cautious interpretation.

## Key Mappings

### exceptional_points.json

| JSON Key | Cautious Interpretation |
|----------|----------------------|
| `ep_score_mean` | Mean geometry score (composite of eigenvalue proximity and eigenvector non-orthogonality) |
| `min_eigenvalue_gap_mean` | Mean minimum nearest-neighbor eigenvalue spacing |
| `sigma_vs_ep_score` | Branching ratio vs geometry score correlation |
| `lzc_vs_ep_score` | Lempel-Ziv complexity vs geometry score correlation |
| `tau_vs_ep_score` | Autocorrelation time vs geometry score correlation |
| `sigma_vs_min_gap` | Branching ratio vs minimum eigenvalue gap correlation |
| `lzc_vs_min_gap` | Complexity vs minimum eigenvalue gap correlation |

### ep_propofol_eeg.json

| JSON Key | Cautious Interpretation |
|----------|----------------------|
| `ep_score` | Geometry score per window |
| `spectral_sensitivity` | Spectral radius sensitivity to eigenvalue perturbation |
| `eigenvalue_gap` | Minimum nearest-neighbor eigenvalue spacing |

### ep_sleep_dynamics.json

| JSON Key | Cautious Interpretation |
|----------|----------------------|
| `ep_score` | Geometry score per window |
| `eigenvalue_gap` | Minimum nearest-neighbor eigenvalue spacing |

### ep_robustness_checks.json

| JSON Key | Cautious Interpretation |
|----------|----------------------|
| `surrogate_control` | Phase-randomized surrogate comparison (tests specificity of sensitivity magnitude to neural temporal structure) |
| `pca_robustness` | Sensitivity to PCA dimensionality choice |
| `partial_regression_alpha` | Alpha-power-controlled partial correlation |

### ep_advanced_ds004752.json

| JSON Key | Cautious Interpretation |
|----------|----------------------|
| `spectral_sensitivity` | Cross-cohort generalization of spectral radius sensitivity (not independent replication) |
| `svd_dimension` | Cross-cohort generalization of effective rank vs geometry score relationship |
| `state_contrast` | Task condition contrasts (set-size effects in verbal working memory) |

### jackknife_sensitivity.json

| JSON Key | Cautious Interpretation |
|----------|----------------------|
| `sigma_vs_ep_score` | Jackknife of branching ratio vs geometry score (leave-one-subject-out) |
| `lzc_vs_ep_score` | Jackknife of complexity vs geometry score |

### chirality.json

| JSON Key | Cautious Interpretation |
|----------|----------------------|
| `chirality_index` | Directional consistency of eigenvector phase rotation |
| `sigma_vs_chirality_index` | Branching ratio vs chirality correlation |

## Why Keys Are Not Renamed

Renaming JSON keys would break:
1. Verification of manuscript claims against result artifacts
2. Any downstream analysis code that reads these files
3. The reproducibility chain from raw data to published numbers

The `ep_` prefix in keys should be read as shorthand for "operator-geometry" or "near-degeneracy geometry" throughout.

## Python Function Name Changes

| Old Name (preserved, with deprecation warning) | New Preferred Name |
|-----------------------------------------------|-------------------|
| `compute_ep_proximity_timecourse()` | `compute_geometry_proximity_timecourse()` |
| `detect_exceptional_points()` | `detect_near_degeneracies()` |

Both old and new names are available in `cmcc.analysis.dynamical_systems`. The old names will continue to work indefinitely.
