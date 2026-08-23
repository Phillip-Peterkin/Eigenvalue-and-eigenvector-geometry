# Manuscript source status

`main.tex` is the current public manuscript source and uses the same study title and metric semantics as the repository README and `CITATION.cff`.

During the August 2026 public audit, the repository identified that several historical results stored under `ep_score` / `mean_ep_score` had been described in an earlier manuscript source as results of the later Near-Degeneracy (ND) construct. They are not the same statistic.

For transparency, the pre-alignment manuscript source is retained at:

- `archive/main_pre_alignment_2026-08-23.tex`

That archived source is provenance only and is not the current scientific contract. In particular, its subject-level `r ~= 0.86` statements used the historical proximity statistic even where the prose called it ND.

The current `main.tex`:

- uses the current public title;
- defines the legacy proximity statistic and the PC1-based current ND separately;
- attributes historical cross-subject correlations to the legacy statistic;
- identifies the historical state-classification feature bundle as containing the legacy proximity field;
- preserves null and limiting robustness findings; and
- avoids treating a simple within-unit mean of standardized ND as a valid subject-level statistic.

See `../PUBLIC_AUDIT.md`, `../KEY_MIGRATION.md`, and `../SCIENTIFIC_INTEGRITY.md` for the corresponding repository-level contracts.
