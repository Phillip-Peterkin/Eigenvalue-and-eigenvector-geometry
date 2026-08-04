Never use abbreviations. Explain them in educational ways at an undergraduate level with parenthesis so there can be understanding.

# Operator Geometry Pipeline Rules

These rules govern analysis code in this repository. They exist to prevent silent scientific mistakes that look like ordinary engineering shortcuts.

## Scientific integrity
- Never use abbreviations. Explain them in educational ways at an undergraduate level with parenthesis so there can be understanding.
- All analysis code must map to a stated scientific question.
- Exploratory outputs must be labeled exploratory. Confirmatory claims must map to frozen parameters in `preregistration_spec.json` and `code/config.yaml`.
- Confirmatory analysis must not reuse tuning choices derived from evaluation data.
- No silent averaging across subjects, sessions, trials, channels, or recording sites. Aggregation rules must be explicit in code and artifacts.
- Geometry metrics summarize fitted first-order vector autoregressive (VAR(1)) operators. Do not describe them as ground-truth neural generators or as mathematically exact exceptional points unless that claim is separately proven.

## Data contracts
- Every dataset loader must validate schema, required metadata, units, and shape semantics.
- Required metadata should include subject_id and session_id (or run identifier) where applicable.
- Time values must specify units and reference frame.
- Sampling rate must be explicit in Hertz (Hz).
- Local dataset roots resolve only through the documented environment variables in `cmcc.data_roots` (for example, `IEEG_DATA_ROOT`). No machine-local absolute path defaults.

## Leakage prevention
- Any split function must preserve subject and session boundaries unless the task explicitly justifies otherwise and documents the justification.
- Leave-one-subject-out (LOSO) and related subject-level cross-validation must never place the same subject in both training and evaluation folds.
- Time series tasks that predict future state must use causal splits: features may use only information available before the prediction target time.
- Label creation must not use future information (for example, sleep-stage labels used as targets must not be rebuilt from post-onset windows that the predictor is forbidden to see).
- Shared-subspace principal component analysis (PCA) fits used for confirmatory state contrasts must be documented as such; do not silently switch between per-state and shared subspaces when reporting a locked claim.
- Falsification and surrogate controls (phase-randomized surrogates, label shuffles, jackknife) are confirmatory stress tests, not optional decorations. When a claim depends on them, the corresponding JSON artifact must remain auditable.

## Testing requirements
- Every preprocessing transform needs at least one unit test.
- Every model pipeline needs a smoke test.
- Every bug fix needs a regression test.
- All critical alignment logic needs edge-case tests.
- Manuscript quantitative claims must remain locked to checked-in results under `results/json_results/` via the manuscript audit suite, unless a deliberate amendment updates both the manuscript and the artifact in the same change.

## Reproducibility
- Every executable analysis must load configuration from a versioned file (`code/config.yaml` for the canonical path).
- Random seeds must be set and logged.
- Output artifacts must include metadata sufficient to reproduce the run (seed, config hash or path, subject list, software versions when available).
- Derived outputs must be written to versioned results directories. Do not overwrite locked confirmatory JSON without an explicit amendment note.

## Documentation
- Public functions must document purpose, units, assumptions, and failure modes.
- Modules must include scientific rationale, not just implementation notes.
- Legacy labels such as `ep_score` must remain mapped to the manuscript term near-degeneracy (ND) score in `KEY_MIGRATION.md`.

## Preferred structure
- `code/analysis_pipeline/cmcc/io`
- `code/analysis_pipeline/cmcc/preprocess`
- `code/analysis_pipeline/cmcc/features`
- `code/analysis_pipeline/cmcc/analysis`
- `code/analysis_pipeline/cmcc/viz`
- `tests/` and `code/analysis_pipeline/tests/`
- `code/config.yaml`
- `results/json_results/`
- `preregistration_spec.json`

## Code review stance
- Reject code that is fast but scientifically ambiguous.
- Prefer explicit names over short names.
- Prefer auditable intermediate files over hidden in-memory transformations.
- Prefer failing loudly when dataset roots or configs are missing over inventing silent defaults.
