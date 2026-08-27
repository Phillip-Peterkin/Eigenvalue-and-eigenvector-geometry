# Public Repository Audit

This audit records the scientific and engineering state of the public repository after the August 2026 hardening reviews. The goal is not to erase historical mistakes. It is to make the boundary between current definitions, historical artifacts, verified controls, and open raw-data gates explicit.

## Status summary

| Item | Status | Public interpretation |
|---|---|---|
| Manuscript title and metric semantics | Resolved in current source | `README.md`, `CITATION.cff`, and `manuscript/main.tex` use the same title and distinguish legacy proximity from current ND |
| Primary COGITATE cohort selection | Resolved in code | Fixed 18-subject cohort and expected runs are versioned in `cohorts/cogitate_primary.json`; canonical reproduction does not infer the cohort from local directories |
| Canonical batch failure semantics | Resolved in code | Strict broadband reproduction fails if any required subject/run is missing or any required subject does not complete successfully; `--best-effort` is exploratory only |
| Canonical config/default drift | Resolved with regression test | `code/config.yaml` and `cmcc.config.DEFAULTS` must remain equal |
| Reference software environment | Added | `requirements-reference.txt` provides a pinned Python 3.11 reviewer baseline while `pyproject.toml` retains supported library ranges |
| Package version/provenance metadata | Resolved | Package version is read from distribution metadata; provenance uses distribution names and records config/cohort/Git/platform/package information |
| Numerical warning policy | Hardened | Runtime warnings fail the portable test suite; broad warning suppression is rejected in the installable package |
| Primitive eigenvalue-gap ordering | Resolved | Leading modes are defined internally by eigenvalue magnitude, making truncated minimum-gap calculation invariant to input ordering |
| VAR fitting input contract | Hardened for new code | `cmcc.analysis.validated_var.estimate_var_operator` rejects malformed/non-finite inputs and invalid window, step, or regularization values before delegating to the retained estimator |
| Release-contract drift detection | Added | `release_contract_manifest.json` pins canonical configuration, cohort/spec files, and headline result artifacts by Git blob hash |
| Pre-alignment manuscript source | Archived for provenance | `manuscript/archive/main_pre_alignment_2026-08-23.tex`; not the current scientific contract |
| Legacy `ep_score` vs current Near-Degeneracy score | Resolved semantically | Distinct mathematical quantities; no algebraic equivalence claim |
| Historical `r ~= 0.86` geometry-criticality correlation | Preserved with qualification | Belongs to the legacy proximity statistic |
| Historical state-classifier `nd_score` field | Documented schema debt | Numerical field is legacy `mean_ep_score`; see `results/RESULT_SCHEMA_NOTES.md` |
| Historical classifier uncertainty | Corrected for new runs | Historical point LOSO AUC retained; old bootstrap/null uncertainty is provenance-only because duplicate subject draws were collapsed and finite null p-values lacked +1 correction |
| Current ND window-level implementation | Implemented and synthetically tested | PC1-based paired-valid-window construct with explicit orientation/validity rules |
| Subject-level current-ND correlation | Open scientific item | Requires a prospective aggregation/normalization rule and a new result artifact |
| Broadband vs high-gamma raw-data reconciliation | Open empirical item | Execution path is hardened and strict; corrected raw-data output still needs reconciliation with the historical checked-in artifact |
| Exceptional-point terminology | Resolved in interpretation | Historical naming only; no exact exceptional-point claim |
| Zurich ds004752 analysis | Resolved in interpretation | Secondary consistency/generalization analysis, not independent replication |
| Spectral-sensitivity surrogate result | Resolved in interpretation | Absolute magnitude is not treated as a standalone neural marker |
| Sleep AUC = 1.00 | Resolved in interpretation | Small-sample within-cohort result, not external generalization |
| Pre-N3 spectral-radius drift | Retained with limitation | Small-sample, pipeline-conditional pre-boundary association |
| Pre-N3 minimum-gap drift | Supporting only | Not promoted without the non-overlap support achieved by spectral radius |
| README AUC 0.948 / 0.957 claims | Removed | A clean matching checked-in result artifact was not identified during review |
| GitHub `main` protection / signed releases | Hosting control still recommended | These are repository/account settings rather than source-code guarantees; enable required CI checks and signed formal release tags externally |

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

## 4. Cohort and canonical execution hardening

The historical repository allowed the nominally canonical broadband runner to discover whatever `ECOG` directories happened to exist beneath a local data root. That meant two reviewers with different directory contents could execute the same commit and configuration yet analyze different cohorts.

The primary 18-subject cohort is now frozen in `cohorts/cogitate_primary.json`, mirrored in `code/config.yaml`, and enforced by regression tests. Strict canonical reproduction consumes that fixed cohort and expected run set. Filesystem discovery is available only through explicit `--best-effort` mode and is not a release-reproduction path.

Strict mode also changes batch failure semantics. A run no longer succeeds merely because at least one subject completed. Missing expected subjects/runs and any non-OK required subject result make the canonical run fail after the audit summaries are written.

## 5. Configuration, environment, and provenance

`code/config.yaml` and `cmcc.config.DEFAULTS` previously represented overlapping but non-identical schemas. They are now synchronized by test, and validation covers both high-gamma and broadband passbands, cohort fields, run identifiers, epoch/baseline parameters, statistical parameters, decoding counts, and output configuration.

Library dependencies in `pyproject.toml` retain compatible lower bounds for normal installation. `requirements-reference.txt` separately pins the direct Python 3.11 reference environment for reviewers. CI tests both the supported Python matrix and this reference environment.

Run provenance now reads installed package versions from distribution metadata. This avoids incorrect import-name assumptions such as `scikit-learn` versus `sklearn` and `PyYAML` versus `yaml`. The provenance record also includes the full configuration hash/snapshot, Git commit, platform, canonical subjects, and cohort manifest.

## 6. Numerical and test hardening

Pytest no longer globally hides `RuntimeWarning` or `UserWarning`. Runtime warnings are errors in the portable suite, while other warnings remain visible. CI rejects unqualified `warnings.filterwarnings("ignore")` in the installable `cmcc` package. Historical/exploratory scripts are retained separately and classified in `code/analysis_pipeline/scripts/README.md` rather than being represented as canonical merely because they are executable.

The public minimum-eigenvalue-gap helper now defines leading modes by descending eigenvalue magnitude before mode truncation and maps the selected pair back to caller indices. This removes an implicit ordering dependency from the public numerical API.

New analysis code can use `cmcc.analysis.validated_var.estimate_var_operator`, which validates dimensionality, finite input values, positive step size, non-negative regularization, window/channel constraints, and finite key outputs before delegating to the retained VAR(1) implementation.

## 7. Release-contract integrity

`release_contract_manifest.json` pins the Git blob hashes of the canonical configuration, cohort manifest, repository analysis contract, and headline result artifacts. `tests/unit/test_engineering_hardening.py` recomputes those blob hashes using Git and fails on unexpected drift.

This mechanism is deliberately a drift detector, not a substitute for repository governance. Formal releases should additionally use protected `main`, required CI checks, and signed immutable release tags where the GitHub account settings permit them.

## 8. Broadband reproduction

The public configuration declares separate high-gamma `[70, 150]` Hz and broadband `[1, 200]` Hz passbands. The strict canonical broadband wrapper validates effective passbands after per-recording Nyquist adjustment, consumes the fixed cohort, uses the documented data-root contract, records band/cohort/execution provenance, and fails on incomplete canonical execution.

The remaining gate is empirical: rerun the strict corrected path on the source data and compare the resulting subject-level and group-level values with the historical `broadband_comparison.json` artifact. Until that is completed, the checked-in comparison remains a historical result rather than proof that the corrected runner reproduces it bit-for-bit.

## 9. Branching-statistic interpretation

The repository uses a threshold-derived branching-related statistic. Public text describes it as criticality-related under the declared estimator. It is not promoted as a direct, estimator-independent measurement of a latent neuronal branching parameter.

## 10. Sleep-transition interpretation

The spectral-radius pre-N3 effect survives the documented non-overlapping-window validation and is retained as a pipeline-conditional pre-boundary association. Temporal precedence alone is not evidence of causal control. The minimum-gap pre-transition result is treated as supportive because it did not survive the same stricter non-overlap gate.

## 11. Public release gate

Before a tagged manuscript release intended to represent a fully rerunnable raw-data analysis, all of the following should be true:

1. the portable test suite passes from a clean environment and the pinned Python 3.11 reference environment;
2. CI is green across the supported Python matrix;
3. package dependency consistency passes with `python -m pip check`;
4. the built wheel installs and reports the same version as distribution metadata;
5. canonical config, package defaults, and cohort manifest remain synchronized;
6. release-contract blob hashes match the declared manifest;
7. README, citation metadata, and current manuscript use the same title;
8. current manuscript claims map to machine-readable result artifacts;
9. historical proximity and current ND are never silently equated;
10. any new subject-level current-ND claim uses a prospectively frozen aggregation rule;
11. new classifier uncertainty uses multiplicity-preserving subject bootstrap and finite-permutation +1 correction;
12. the strict canonical broadband runner is rerun against source data and its output is reconciled with the historical artifact;
13. exploratory datasets are not described as independent replications unless they satisfy that design standard; and
14. formal release governance uses protected `main`, required CI, and signed release tags where supported by repository settings.

## Why this file exists

A research repository is more credible when corrections are inspectable. This file therefore remains part of the reproducibility record even as items are resolved.
