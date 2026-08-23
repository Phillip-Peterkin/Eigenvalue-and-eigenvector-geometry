# Architecture

Companion code for:

> Peterkin, P. (2026). *Fitted-Operator Geometry as a Complementary Descriptive Axis for Brain-State Discrimination in Human iEEG and Scalp EEG.*

## Layout

| Path | Role |
|---|---|
| `code/analysis_pipeline/cmcc/` | Installable analysis package |
| `code/config.yaml` | Canonical versioned pipeline parameters |
| `code/analysis_pipeline/scripts/` | End-to-end and analysis-specific runners |
| `tests/` | Public release, manuscript-to-result, and synthetic mathematical tests |
| `code/analysis_pipeline/tests/` | Additional unit and analysis tests |
| `results/json_results/` | Checked-in machine-readable historical and current result artifacts |
| `manuscript/` | Manuscript source and figures |
| `PUBLIC_AUDIT.md` | Known issues, resolved ambiguities, and release gate |
| `SCIENTIFIC_INTEGRITY.md` | Scientific contribution and leakage rules |
| `KEY_MIGRATION.md` | Historical `ep_score` versus current Near-Degeneracy terminology |
| `REPLICATION_AND_DATA_PROVENANCE.md` | Dataset and analysis provenance |
| `preregistration_spec.json` | Frozen public repository analysis contract |

## Quick verification without raw datasets

```bash
pip install -e ".[dev]"
pytest
```

The portable suite includes synthetic mathematical tests plus checks that public claims remain tied to machine-readable artifacts and documented metric definitions.

## Dataset-root contract

| Variable | Dataset |
|---|---|
| `IEEG_DATA_ROOT` | COGITATE intracranial electroencephalography Experiment 1 |
| `DS004752_DATA_ROOT` | Zurich stereoelectroencephalography, OpenNeuro ds004752 |
| `PROPOFOL_DATA_ROOT` | Cambridge Propofol scalp electroencephalography, OpenNeuro ds005620 |
| `SLEEP_DATA_ROOT` | ANPHY-Sleep polysomnography |
| `RESULTS_ROOT` | Optional analysis output root |

Resolution is implemented in `cmcc.data_roots`. No public reproduction path should depend on a machine-local absolute path.

## Canonical entry points

```bash
pip install -e ".[dev]"
python code/analysis_pipeline/scripts/run_pipeline.py
python code/analysis_pipeline/scripts/run_all_subjects.py
python code/analysis_pipeline/scripts/run_all_subjects_broadband_canonical.py
```

The canonical broadband wrapper requires the explicit `preprocessing.broadband_passband` configuration entry and prevents a fresh broadband run from silently reusing the high-gamma passband. The older broadband script is retained for provenance.

## Scientific object map

The primary fitted object is a first-order vector autoregressive matrix estimated in sliding windows. Publicly reported geometry summaries include:

- spectral radius;
- minimum eigenvalue spacing;
- eigenvector condition number;
- related derived geometry summaries.

The historical `ep_score` is a proximity heuristic based on closest-pair overlap divided by eigenvalue spacing. The current manuscript Near-Degeneracy score is a distinct composite based on transformed crowding and eigenvector conditioning. See `KEY_MIGRATION.md` and `PUBLIC_AUDIT.md` before using either quantity in a new analysis.

## Scientific integrity notes

- Fitted operators are descriptive local linearizations, not identified neural mechanisms.
- Synthetic tests validate mathematical helper behavior independently of checked-in empirical results.
- Manuscript/result audit tests are bookkeeping safeguards and do not substitute for estimator validation.
- Exploratory scripts and negative results remain visible when they constrain interpretation.
- Subject-preserving predictive evaluation and explicit aggregation rules are required by `SCIENTIFIC_INTEGRITY.md`.
