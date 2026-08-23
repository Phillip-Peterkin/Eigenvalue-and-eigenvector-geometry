# Result Schema Notes

This file records historical result-field and inference semantics that cannot be inferred safely from key names alone.

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

The public compatibility/execution surface:

```text
code/analysis_pipeline/cmcc/analysis/geometry_embedding.py
```

documents the mismatch and provides semantic-label adapters for new code without changing historical numerical values.

## Historical classifier uncertainty

The historical state-classification point AUC uses leave-one-subject-out predictions with fold-internal standardization. Those point predictions remain the provenance basis for the checked-in artifact.

The historical bootstrap confidence-interval code, however, sampled subject IDs with replacement and then converted the draw to an `np.isin` membership mask. Repeated sampled subjects were therefore collapsed instead of contributing repeated subject blocks. Historical finite permutation p-values also used an uncorrected exceedance fraction that could report zero with a finite null sample.

These historical uncertainty values are retained unchanged in the locked artifact. New public execution corrects both behaviors:

- subject-block bootstrap row construction preserves multiplicity when a subject is drawn more than once; and
- finite permutation p-values use `(exceedances + 1) / (B + 1)`.

Accordingly, historical point AUC values may be cited with their declared cohort and feature semantics, while historical classifier confidence intervals and finite-null p-values should be treated as provenance values rather than the preferred current uncertainty procedure. A fresh run should write a new artifact instead of overwriting `geometry_brain_states.json`.

## Current ND

The current implementation is in `code/analysis_pipeline/cmcc/features/operator_geometry.py`. It uses paired valid gap/condition-number windows, transformed features, within-analysis-unit standardization, and a sign-normalized first-principal-component projection. It is a different statistic from historical `ep_score`.

Any future JSON artifact using current ND should use an explicit schema name such as `nd_pc1_score` and should record the normalization unit, loading vector, input validity count, configuration hash, and software commit.
