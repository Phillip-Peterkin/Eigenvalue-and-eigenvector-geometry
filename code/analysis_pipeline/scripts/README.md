# Analysis script execution policy

This directory contains both current public entry points and historical or exploratory analysis scripts retained for computational provenance. File presence alone does not make a script part of the canonical reproduction contract.

## Canonical public entry points

The current review path is limited to explicitly documented entry points:

- `run_pipeline.py` for the configured single-analysis pipeline.
- `run_all_subjects.py` for the historical high-gamma batch path where required by provenance.
- `run_all_subjects_broadband_canonical.py` for canonical broadband reproduction. This runner is strict by default, consumes the fixed cohort in `code/config.yaml` / `cohorts/cogitate_primary.json`, and fails if any expected subject or expected run is unavailable or unsuccessful. `--best-effort` is an exploratory convenience and is not valid for release reproduction.
- `analysis/_geometry_brain_states.py` for the corrected historical-proximity state-space inference path. It preserves the locked historical artifact and writes a distinct corrected result.

The authoritative reviewer commands are maintained in `TECHNICAL_REVIEW_GUIDE.md`.

## Historical and exploratory scripts

Other scripts, including files with names such as `legacy`, `diagnostic`, `sweep`, `detector`, `realtime`, `hardening`, and one-off analysis names, are retained because deleting them would erase computational provenance. Some predate the current warning, naming, configuration, or inference contracts. They must not be substituted for the documented canonical entry points simply because they remain executable.

Historical scripts may contain broad warning suppression or machine-era compatibility behavior. That behavior is not accepted in the installable `cmcc` package or canonical public runners. New production code must use narrow, justified warning handling and current configuration/data-root contracts.

## Change rule

A script becomes canonical only when all of the following are true:

1. it is named in `TECHNICAL_REVIEW_GUIDE.md`;
2. its configuration and cohort inputs are versioned;
3. failure semantics are explicit;
4. the path is covered by CI or portable contract tests; and
5. its output is written to a new artifact rather than silently overwriting a historical result.
