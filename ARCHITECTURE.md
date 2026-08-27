# Architecture

Companion code for:

> Peterkin, P. (2026). *Fitted-Operator Geometry as a Complementary Descriptive Axis for Brain-State Discrimination in Human iEEG and Scalp EEG.*

## Layout

| Path | Role |
|---|---|
| `code/analysis_pipeline/cmcc/` | Installable analysis package |
| `code/config.yaml` | Canonical versioned pipeline parameters |
| `cohorts/cogitate_primary.json` | Fixed primary intracranial cohort and expected runs |
| `code/analysis_pipeline/scripts/` | Canonical plus retained historical/exploratory runners |
| `code/analysis_pipeline/scripts/README.md` | Execution-status policy for scripts |
| `tests/` | Public release, contract, manuscript-to-result, and synthetic tests |
| `code/analysis_pipeline/tests/` | Additional unit and analysis tests |
| `results/json_results/` | Checked-in machine-readable historical and current result artifacts |
| `release_contract_manifest.json` | Git-blob hashes for canonical contract files and headline result artifacts |
| `requirements-reference.txt` | Pinned Python 3.11 reference environment |
| `manuscript/` | Manuscript source and figures |
| `PUBLIC_AUDIT.md` | Known issues, resolved ambiguities, and release gate |
| `SCIENTIFIC_INTEGRITY.md` | Scientific contribution and leakage rules |
| `KEY_MIGRATION.md` | Historical `ep_score` versus current Near-Degeneracy terminology |
| `REPLICATION_AND_DATA_PROVENANCE.md` | Dataset and analysis provenance |
| `preregistration_spec.json` | Frozen post-hoc public repository analysis contract |

## Reproducibility layers

The repository separates five engineering layers:

1. **Configuration:** `code/config.yaml` is synchronized with `cmcc.config.DEFAULTS` by regression tests.
2. **Cohort selection:** `cohorts/cogitate_primary.json` freezes the primary 18-subject cohort and expected runs. Canonical reproduction cannot infer the scientific cohort from local directory contents.
3. **Numerical contracts:** `cmcc.features.operator_geometry` validates primitive metric inputs; `cmcc.analysis.validated_var.estimate_var_operator` provides the strict public VAR(1) fitting entry point.
4. **Artifact integrity:** `release_contract_manifest.json` pins canonical contract files and headline result artifacts by Git blob hash.
5. **Environment:** broad supported dependencies remain in `pyproject.toml`; `requirements-reference.txt` supplies a pinned Python 3.11 reviewer baseline.

## Quick verification without raw datasets

```bash
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest -q --cov=cmcc --cov-fail-under=25
```

The portable suite includes synthetic mathematical tests, numerical-input contracts, package/version consistency checks, cohort/config synchronization, release-artifact hash checks, and manuscript-to-result safeguards.

For the pinned reference baseline on Python 3.11:

```bash
python -m pip install -r requirements-reference.txt
python -m pip install -e . --no-deps
python -m pip check
python -m pytest -q
```

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
python code/analysis_pipeline/scripts/run_pipeline.py
python code/analysis_pipeline/scripts/run_all_subjects_broadband_canonical.py
python code/analysis_pipeline/scripts/analysis/_geometry_brain_states.py
```

The canonical broadband runner is strict by default. It consumes the fixed cohort from `code/config.yaml` / `cohorts/cogitate_primary.json`, requires the expected run set, and fails the batch if any required subject does not complete successfully. `--best-effort` is explicitly exploratory.

The broadband wrapper still adapts the explicit `broadband_passband` to an older implementation internally so historical computation can be preserved, but that translation is isolated to the compatibility wrapper and is recorded in provenance.

Historical and exploratory executables are retained rather than deleted. Their status is documented in `code/analysis_pipeline/scripts/README.md`; executable does not mean canonical.

## Scientific object map

The primary fitted object is a first-order vector autoregressive matrix estimated in sliding windows. New library code should prefer `cmcc.analysis.validated_var.estimate_var_operator`, which validates dimensions, finite inputs, window/step sizes, and regularization before delegating to the retained estimator.

Publicly reported geometry summaries include:

- spectral radius;
- minimum eigenvalue spacing;
- eigenvector condition number;
- related derived geometry summaries.

The historical `ep_score` is a proximity heuristic based on closest-pair overlap divided by eigenvalue spacing. The current manuscript Near-Degeneracy score is a distinct composite based on transformed crowding and eigenvector conditioning. See `KEY_MIGRATION.md` and `PUBLIC_AUDIT.md` before using either quantity in a new analysis.

## Scientific integrity notes

- Fitted operators are descriptive local linearizations, not identified neural mechanisms.
- Synthetic tests validate mathematical helper behavior independently of checked-in empirical results.
- Runtime warnings are no longer globally hidden by pytest; numerical runtime warnings fail the portable suite unless narrowly justified.
- Historical result artifacts are never silently overwritten by corrected execution paths.
- Exploratory scripts and negative results remain visible when they constrain interpretation.
- Subject-preserving predictive evaluation and explicit aggregation rules are required by `SCIENTIFIC_INTEGRITY.md`.
- GitHub branch protection and signed formal release tags are hosting/release controls and should supplement, not be replaced by, repository-level tests.
