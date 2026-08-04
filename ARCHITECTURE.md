# Architecture

Companion code for:

> Peterkin, P. (2026). *Fitted Operator Geometry Reveals Brain-State Structure and Sleep Transitions.*

## Layout

| Path | Role |
|------|------|
| `code/analysis_pipeline/cmcc/` | Installable analysis package (`pip install -e ".[dev]"`) |
| `code/config.yaml` | Versioned pipeline parameters |
| `code/analysis_pipeline/scripts/` | End-to-end runners (need local datasets) |
| `tests/` | Manuscript-to-JSON audit + citation/repo hygiene |
| `code/analysis_pipeline/tests/` | Unit tests for geometry / falsification / amplification |
| `results/json_results/` | Checked-in machine-readable statistics for zero-data verification |
| `manuscript/` | LaTeX manuscript and figures |

## Quick verification (no private data)

```bash
pip install -e ".[dev]"
pytest
```

This runs manuscript audit tests against `results/json_results/` and synthetic unit tests for operator-geometry helpers.

## Data-dependent runs

Set dataset roots via environment variables (`IEEG_DATA_ROOT`, `DS004752_DATA_ROOT`, `PROPOFOL_DATA_ROOT`, `SLEEP_DATA_ROOT`, and related script-specific variables). Scripts no longer ship machine-local absolute path defaults.
