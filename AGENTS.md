# Operator Geometry Repository Rules

This file contains machine-facing repository guidance. Human-facing scientific rules live in `SCIENTIFIC_INTEGRITY.md` and take precedence when wording differs.

## Scientific integrity

- All analysis code must map to a stated scientific question.
- Exploratory outputs must remain labeled exploratory.
- Confirmatory claims must map to versioned parameters and machine-readable result artifacts.
- Confirmatory analysis must not reuse tuning choices derived from evaluation data.
- No silent averaging across subjects, sessions, trials, channels, or recording sites. Aggregation rules must be explicit.
- Geometry metrics summarize fitted first-order vector autoregressive operators. Do not describe them as ground-truth neural generators or mathematically exact exceptional points.
- Do not equate the historical `ep_score` proximity heuristic with the current manuscript Near-Degeneracy score. See `KEY_MIGRATION.md` and `PUBLIC_AUDIT.md`.

## Data contracts

- Every dataset loader must validate required metadata, units, sampling rate, and shape semantics.
- Required metadata should include subject and session or run identity where applicable.
- Time values must specify units and reference frame.
- Local dataset roots resolve through the documented environment-variable contract. Do not introduce machine-local absolute path defaults.

## Leakage prevention

- Predictive split functions must preserve subject and session boundaries unless a different design is explicitly justified.
- Leave-one-subject-out evaluation must never place the same subject in both training and evaluation folds.
- Future-state prediction must use only information available before the target time.
- Label creation must not use forbidden future information.
- State-specific versus shared-subspace Principal Component Analysis must never be switched silently when reporting a locked claim.
- Surrogate, label-shuffle, and jackknife controls are part of the scientific evidence when a claim depends on them.

## Testing requirements

- Mathematical helper functions require synthetic tests with known behavior.
- Preprocessing transforms require unit or regression coverage when practical.
- Model pipelines require smoke tests.
- Bug fixes require regression tests when the failure can be reproduced deterministically.
- Public quantitative claims should remain auditable against checked-in result artifacts.
- Audit tests are bookkeeping safeguards and do not replace mathematical estimator tests.

## Reproducibility

- Canonical executable analyses load configuration from versioned files.
- Random seeds must be set and logged where stochastic procedures are used.
- Result artifacts should include enough metadata to identify subject set, configuration, random seed, and software version when practical.
- Do not overwrite historical locked artifacts simply to make terminology match a newer manuscript. Write a new artifact or explicit amendment when the underlying statistic changes.

## Documentation

- Public functions should document purpose, units, assumptions, and failure modes.
- Modules should include scientific rationale where it clarifies interpretation.
- Historical labels remain documented as historical labels. Backward compatibility does not establish mathematical equivalence.

## Preferred structure

- `code/analysis_pipeline/cmcc/io`
- `code/analysis_pipeline/cmcc/preprocess`
- `code/analysis_pipeline/cmcc/features`
- `code/analysis_pipeline/cmcc/analysis`
- `code/analysis_pipeline/cmcc/viz`
- `tests/` and `code/analysis_pipeline/tests/`
- `code/config.yaml`
- `results/json_results/`
- `PUBLIC_AUDIT.md`
- `SCIENTIFIC_INTEGRITY.md`

## Code review stance

- Reject code that is computationally convenient but scientifically ambiguous.
- Prefer explicit names over short names.
- Prefer auditable intermediate artifacts over hidden transformations.
- Prefer failing loudly when required configuration or data roots are missing over inventing silent defaults.
