# Result Schema Notes

This file records historical result-field semantics that cannot be inferred safely from key names alone.

## `ep_score` / `mean_ep_score`

Historical operator-geometry result files use:

```text
ep_score = eigenvector_overlap / (minimum_eigenvalue_gap + 1e-10)
```

This is the **legacy proximity statistic**. It is not the current PC1-based Near-Degeneracy (ND) score.

## `geometry_brain_states.json` and the historical `nd_score` label

The checked-in `geometry_brain_states.json` artifact lists the geometry feature bundle as:

```text
[eigenvalue_gap, condition_number, nd_score, spectral_radius]
```

However, the historical extraction implementation populated the third column from `mean_ep_score`. Therefore the third numerical feature is the legacy proximity statistic despite the old `nd_score` label.

This artifact is retained unchanged for provenance. Do not reinterpret its classifier results as validation of the current ND construction.

Current code retains the historical implementation in:

```text
code/analysis_pipeline/cmcc/analysis/geometry_embedding_legacy.py
```

The public compatibility surface:

```text
code/analysis_pipeline/cmcc/analysis/geometry_embedding.py
```

documents the mismatch and provides semantic-label adapters for new code without changing historical numerical values.

## Current ND

The current implementation is in `code/analysis_pipeline/cmcc/features/operator_geometry.py`. It uses paired valid gap/condition-number windows, transformed features, within-analysis-unit standardization, and a sign-normalized first-principal-component projection. It is a different statistic from historical `ep_score`.

Any future JSON artifact using current ND should use an explicit schema name such as `nd_pc1_score` and should record the normalization unit, loading vector, input validity count, configuration hash, and software commit.
