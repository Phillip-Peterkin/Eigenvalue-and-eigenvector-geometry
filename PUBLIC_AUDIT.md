# Public Repository Audit

This audit records the scientific and engineering state of the public repository after the August 2026 hardening reviews. The goal is not to erase historical mistakes. It is to make the boundary between current definitions, historical artifacts, verified controls, and open raw-data gates explicit.

## Status summary

| Item | Status | Public interpretation |
|---|---|---|
| Manuscript title and metric semantics | Resolved in current source | `README.md`, `CITATION.cff`, and `manuscript/main.tex` use the same title and distinguish legacy proximity from current ND |
| Pre-alignment manuscript source | Archived for provenance | `manuscript/archive/main_pre_alignment_2026-08-23.tex`; not the current scientific contract |
| Legacy `ep_score` vs current Near-Degeneracy score | Resolved semantically | Distinct mathematical quantities; no algebraic equivalence claim |
| Historical `r ~= 0.86` geometry-criticality correlation | Preserved with qualification | Belongs to the legacy proximity statistic |
| Historical state-classifier `nd_score` field | Documented schema debt | Numerical field is legacy `mean_ep_score`; see `results/RESULT_SCHEMA_NOTES.md` |
| Historical classifier uncertainty | Corrected for new runs | Historical point LOSO AUC retained; old bootstrap/null uncertainty is provenance-only because duplicate subject draws were collapsed and finite null p-values lacked +1 correction |
| Current ND window-level implementation | Implemented and synthetically tested | PC1-based paired-valid-window construct with explicit orientation/validity rules |
| Subject-level current-ND correlation | Open scientific item | Requires a prospective aggregation/normalization rule and a new result artifact |
| Broadband vs high-gamma configuration | Code path corrected | Canonical config and runner are explicit; raw-data end-to-end reproduction remains open |
| Exceptional-point terminology | Resolved in interpretation | Historical naming only; no exact exceptional-point claim |
| Zurich ds004752 analysis | Resolved in interpretation | Secondary consistency/generalization analysis, not independent replication |
| Spectral-sensitivity surrogate result | Resolved in interpretation | Absolute magnitude is not treated as a standalone neural marker |
| Sleep AUC = 1.00 | Resolved in interpretation | Small-sample within-cohort result, not external generalization |
| Pre-N3 spectral-radius drift | Retained with limitation | Small-sample, pipeline-conditional pre-boundary association |
| Pre-N3 minimum-gap drift | Supporting only | Not promoted without the non-overlap support achieved by spectral radius |
| README AUC 0.948 / 0.957 claims | Removed | A clean matching checked-in result artifact was not identified during review |

## 1. Historical proximity and current ND

Historical result files store:

```text
ep_score = eigenvector_overlap / (minimum_eigenvalue_gap + 1e-10)
```

The current Near-Degeneracy (ND) implementation instead transforms minimum gap and eigenvector condition number, standardizes paired valid windows within the declared analysis unit, and projects onto the first principal component. Same-sign PC1 loadings are oriented positive. Ambiguous opposite-sign loadings fail loudly.

These statistics share a scientific motivation but are not the same mathematical object.

## 2. Why subject-mean ND cannot replace the historical subject statistic

Within-unit standardization makes the mean of each standardized ND component approximately zero within that same unit. Consequently, a simple subject mean of current within-subject ND is also approximately zero by construction and is not a meaningful drop-in replacement for the historical subject-level proximity score.

Any future subject-level current-ND analysis must freeze an aggregation rule before evaluating the target association. Suitable approaches might include a common cohort-level normalization/loading rule, a raw-scale subject statistic, or another prospectively specified summary. The selected rule must be justified and written to a new machine-readable artifact.

## 3. Historical state-classification schema and uncertainty

`results/json_results/geometry_brain_states.json` contains a feature label `nd_score`, but the historical extraction code populated that column from `mean_ep_score`. The numerical values are therefore legacy proximity values.

The original implementation is retained as `code/analysis_pipeline/cmcc/analysis/geometry_embedding_legacy.py`. The current `geometry_embedding.py` documents the semantics and provides explicit semantic-label adapters for new code. The historical JSON is not silently rewritten because doing so would alter a locked result artifact without recomputation.

A second historical issue was found in the classifier uncertainty code. Subject IDs were sampled with replacement for bootstrap confidence intervals, but an `np.isin` membership mask then collapsed repeated subject draws. The historical finite-permutation p-value also used an uncorrected exceedance fraction. These issues do not alter the stored leave-one-subject-out point AUC. They do mean the historical classifier confidence intervals and finite-null p-values are retained as provenance rather than preferred current inference.

The current public execution path corrects both issues: repeated sampled subjects contribute repeated subject observation blocks, and finite permutation p-values use `(exceedances + 1) / (B + 1)`. Regression tests enforce both behaviors. Fresh results must be written to a new artifact instead of overwriting the historical JSON.

## 4. Broadband reproduction

The public configuration now declares separate high-gamma `[70, 150]` Hz and broadband `[1, 200]` Hz passbands. The canonical broadband runner validates the effective passband after Nyquist adjustment, uses the documented data-root contract, records band provenance, and fails if no subject succeeds.

The remaining gate is empirical: rerun the canonical corrected path on the source data and compare the resulting subject-level and group-level values with the historical `broadband_comparison.json` artifact. Until that is completed, the checked-in comparison is a historical result rather than proof that the corrected runner reproduces it bit-for-bit.

## 5. Branching-statistic interpretation

The repository uses a threshold-derived branching-related statistic. Public text describes it as criticality-related under the declared estimator. It is not promoted as a direct, estimator-independent measurement of a latent neuronal branching parameter.

## 6. Sleep-transition interpretation

The spectral-radius pre-N3 effect survives the documented non-overlapping-window validation and is retained as a pipeline-conditional pre-boundary association. Temporal precedence alone is not evidence of causal control. The minimum-gap pre-transition result is treated as supportive because it did not survive the same stricter non-overlap gate.

## 7. Public release gate

Before a tagged manuscript release intended to represent a fully rerunnable raw-data analysis, all of the following should be true:

1. `python -m pytest -q` passes from a clean environment.
2. CI is green across the supported Python matrix.
3. package dependency consistency passes with `python -m pip check`.
4. a wheel builds successfully from `pyproject.toml`.
5. README, citation metadata, and current manuscript use the same title.
6. current manuscript claims map to machine-readable result artifacts.
7. historical proximity and current ND are never silently equated.
8. any new subject-level current-ND claim uses a prospectively frozen aggregation rule.
9. new classifier uncertainty uses multiplicity-preserving subject bootstrap and finite-permutation +1 correction.
10. the canonical broadband runner is rerun against source data and its output is reconciled with the historical artifact.
11. exploratory datasets are not described as independent replications unless they satisfy that design standard.

## Why this file exists

A research repository is more credible when corrections are inspectable. This file therefore remains part of the reproducibility record even as items are resolved.
