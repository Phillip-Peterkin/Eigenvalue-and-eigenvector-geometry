# Public Repository Audit

This document records issues discovered during the August 2026 public-hardening review of the repository. The purpose is to make the scientific state of the public codebase explicit rather than to hide historical changes behind renamed variables or rewritten prose.

## Audit principle

The current manuscript is the interpretive source of truth. Historical code and result artifacts are retained when they are necessary for provenance, but historical labels are not silently promoted into current manuscript quantities.

## Status summary

| Item | Status | Public interpretation |
|---|---|---|
| Legacy `ep_score` versus current Near-Degeneracy score | Open scientific alignment item | Distinct quantities; do not treat them as algebraically identical |
| Historical `r ~= 0.86` geometry-criticality correlation | Preserved with qualification | Currently tied to the legacy proximity-score artifact unless recomputed under the final Near-Degeneracy definition |
| Exceptional-point terminology | Resolved in interpretation | Historical naming only; no claim of exact exceptional points in neural tissue |
| Zurich ds004752 analysis | Resolved in interpretation | Secondary cross-dataset consistency/generalization check, not an independent replication |
| Spectral-sensitivity surrogate result | Resolved in interpretation | Absolute magnitude is not treated as a standalone neural marker |
| Sleep area under the curve = 1.00 | Resolved in interpretation | Small-sample within-cohort upper bound, not an out-of-cohort generalization estimate |
| Pre-N3 spectral-radius drift | Retained with limitation | Small-sample, pipeline-conditional pre-boundary association |
| Pre-N3 minimum-gap drift | Supporting only | Should not be promoted to a primary validated pre-transition result without the same non-overlapping-window support |

## 1. Legacy proximity score and current Near-Degeneracy score

The repository contains a historical per-window proximity score:

```text
ep_score = eigenvector_overlap / (minimum_eigenvalue_gap + 1e-10)
```

The current manuscript instead defines a Near-Degeneracy score from eigenvalue crowding and eigenvector conditioning. The current manuscript form uses transformed, standardized geometry features and data-derived first-principal-component loadings.

These are related only at the level of scientific intent. They are not the same mathematical statistic.

Accordingly:

- checked-in JSON keys such as `ep_score` and `ep_score_mean` are preserved for provenance;
- documentation must identify them as legacy proximity quantities;
- the historical cross-subject correlation near `r = 0.86` must not be relabeled as the final manuscript Near-Degeneracy score without recomputation;
- any future recomputation must write a new result artifact rather than overwriting the historical file in place.

## 2. Mean-of-standardized-score warning

A within-subject z-scored quantity has an approximately zero within-subject mean by construction. Therefore, a cross-subject analysis based on the simple mean of a within-subject standardized Near-Degeneracy score is not a meaningful replacement for the historical subject-level proximity score.

A valid final subject-level Near-Degeneracy analysis must define its aggregation explicitly. Suitable options include a common loading/normalization learned across the analysis cohort under a frozen rule, a raw-scale subject summary, or another prospectively specified subject-level statistic. The choice must be justified before the historical result is relabeled.

## 3. Broadband versus high-gamma configuration

The public manuscript distinguishes high-gamma and broadband observables. The canonical public configuration must therefore expose both passbands explicitly, and the broadband runner must consume the broadband setting rather than reusing the high-gamma key.

Until this is verified end to end on raw data, users should treat the checked-in broadband result artifact as the historical analysis output and should inspect the runner/configuration before attempting a fresh raw-data reproduction.

## 4. Branching statistic interpretation

The repository uses a branching-related statistic derived from thresholded neural activity. Public text should describe this as a criticality-related summary under the declared estimator. It should not be treated as a direct, estimator-independent measurement of the latent neuronal branching parameter.

## 5. Sleep transition interpretation

The sleep analysis is based on scored stage boundaries and a small cohort. The spectral-radius pre-N3 effect survives the documented non-overlapping-window validation and is retained as a pipeline-conditional pre-boundary association. Temporal precedence alone is not treated as evidence of causal control of the transition.

## 6. Public-release gate

Before a tagged public release intended for manuscript submission, the following checks should pass:

1. `pytest` passes from a clean environment.
2. Continuous Integration is visibly green on the release commit.
3. `README.md`, `CITATION.cff`, manuscript title, and repository description use the same study title.
4. The final Near-Degeneracy implementation matches the manuscript definition exactly.
5. The historical `r ~= 0.86` claim is either recomputed under that final definition or labeled everywhere as a legacy-proximity result.
6. The broadband runner reads an explicit broadband configuration and reproduces the checked-in band comparison from raw data.
7. Machine-readable result artifacts record configuration, subject set, random seed, and software/version metadata sufficient to identify the run.
8. No exploratory dataset is described as an independent replication unless it actually satisfies that design standard.

## Why this file exists

A public research repository should make uncertainty and historical changes inspectable. This audit file is therefore part of the reproducibility record. Resolving an item should update its status here and point to the commit or new result artifact that resolved it.
