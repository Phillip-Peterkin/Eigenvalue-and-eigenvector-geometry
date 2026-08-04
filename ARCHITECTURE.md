# Architecture

Companion code for:

> Peterkin, P. (2026). *Fitted Operator Geometry Reveals Brain-State Structure and Sleep Transitions.*

## Layout

| Path | Role |
|------|------|
| `code/analysis_pipeline/cmcc/` | Installable analysis package (`pip install -e ".[dev]"`) |
| `code/config.yaml` | Canonical versioned pipeline parameters |
| `code/analysis_pipeline/scripts/` | End-to-end runners (need local datasets) |
| `tests/` | Manuscript-to-JSON audit + citation/repo/unit hygiene |
| `code/analysis_pipeline/tests/` | Unit tests for geometry / falsification / amplification |
| `results/json_results/` | Checked-in machine-readable statistics for zero-data verification |
| `manuscript/` | LaTeX manuscript and figures |

## Quick verification (no private data)

```bash
pip install -e ".[dev]"
pytest
```

This runs manuscript audit tests against `results/json_results/` and synthetic unit tests for operator-geometry helpers. Scripts import `cmcc` from the installed package — there is no `sys.path` hack to a missing `src/` directory.

## Dataset root contract

| Variable | Dataset |
|----------|---------|
| `IEEG_DATA_ROOT` | Cogitate iEEG Experiment 1 |
| `DS004752_DATA_ROOT` | Zurich SEEG (OpenNeuro ds004752) |
| `PROPOFOL_DATA_ROOT` | Cambridge propofol EEG (ds005620) |
| `SLEEP_DATA_ROOT` | ANPHY-Sleep polysomnography |
| `RESULTS_ROOT` | Optional analysis output root |

Resolution is implemented in `cmcc.data_roots`. Legacy aliases (`COGITATE_IEEG_ROOT`, `DS005620_ROOT`, `ANPHY_SLEEP_ROOT`) are accepted but not documented for new use.

## Entrypoints

```bash
pip install -e ".[dev]"
python code/analysis_pipeline/scripts/run_pipeline.py
python code/analysis_pipeline/scripts/run_all_subjects.py
```

Primary runners load `code/config.yaml`. Some exploratory seizure scripts still reference historical `configs/*.yaml` filenames that are not shipped; treat those as research notebooks in script form, not the canonical public path.

## Scientific integrity notes

- Geometry metrics summarize **fitted VAR(1) operators**, not ground-truth neural generators.
- Manuscript audit tests lock quantitative claims to checked-in JSON; they are confirmatory bookkeeping, not a substitute for synthetic unit tests of the estimators.
- No silent averaging helpers are part of the public package API; aggregation rules are explicit in analysis scripts.
