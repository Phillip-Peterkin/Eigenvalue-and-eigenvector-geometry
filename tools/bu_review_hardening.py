from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "main.tex"
README = ROOT / "README.md"
AUDIT = ROOT / "PUBLIC_AUDIT.md"

NEW_TITLE = (
    "Fitted-Operator Geometry as a Complementary Descriptive Axis for "
    "Brain-State Discrimination in Human iEEG and Scalp EEG"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    out, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return out


text = MANUSCRIPT.read_text(encoding="utf-8")

text = replace_once(
    text,
    "Fitted Operator Geometry Reveals Brain-State Structure and Sleep Transitions",
    NEW_TITLE,
    "manuscript title",
)

new_abstract = r"""Conventional state markers in electrophysiology compress multivariate dynamics into a small number of univariate summaries. We ask whether fitted-operator summaries carry complementary state-discrimination information under a declared analysis pipeline. Using sliding-window VAR(1) fits across human intracranial and scalp electrophysiology datasets, we report three descriptive findings. In iEEG, the threshold-derived branching statistic was lower in high-gamma than broadband activity ($\sigma_{\mathrm{HG}} = 0.9735$ vs. $\sigma_{\mathrm{BB}} = 0.9908$; paired $p = 8.9 \times 10^{-6}$). Geometry-only leave-one-subject-out discrimination reached area under the receiver operating characteristic curve (AUC) $\approx 0.948$, while combined geometry and criticality features reached AUC $\approx 0.957$ within the declared cohort. In scalp EEG, minimum eigenvalue spacing distinguished propofol sedation from wakefulness ($d = 0.71$) and REM from N3 sleep ($d = -2.51$), while within-state spacing-power correlations were small. In sleep recordings, spectral radius shifted before scored N2-to-N3 transitions ($p = 0.0014$; non-overlapping validation $p = 0.032$). Historical cross-subject correlations involving the repository's legacy geometry-proximity score are retained for provenance but are not relabeled as results of the current Near-Degeneracy (ND) construction. All results are conditional on the declared estimator and preprocessing pipeline and are interpreted as descriptive, complementary features rather than identified neural mechanisms."""

text = sub_once(
    text,
    r"Criticality metrics are widely used to summarize near-critical brain dynamics,.*?underlying neural mechanisms are not identifiable from these fits alone\.",
    new_abstract,
    "abstract",
)

new_significance = r"""Conventional electrophysiology markers summarize power, complexity, and criticality but do not directly describe the geometry of fitted multivariate operators. This study asks whether fitted spectral radius, eigenvalue spacing, and eigenvector-conditioning features add descriptive or discriminative information under a fixed computational pipeline. Across iEEG and scalp EEG cohorts, geometry features complement conventional feature families in subject-preserving classification and show state-dependent effects that survive several robustness checks. The repository also preserves a historical geometry-proximity statistic whose cross-subject associations motivated later work; those associations are explicitly kept separate from the current Near-Degeneracy (ND) definition. The contribution is therefore methodological and comparative: a tested, reproducible framework for asking whether fitted-operator geometry adds information, with negative results, surrogate constraints, and unresolved items retained in the public audit trail."""

text = sub_once(
    text,
    r"Criticality measures show brain-state change, but not fitted multivariate geometry\..*?Operator geometry thus provides descriptive coordinates for state-dependent variability beyond conventional spectral measures\.",
    new_significance,
    "significance statement",
)

text = text.replace(
    "Yes to all three. High-gamma is more subcritical than broadband, with dissociations among branching ratio, complexity, and correlations. Branching ratio covaries strongly with the operator-geometry score. In scalp EEG, eigenvalue spacing distinguishes propofol sedation and sleep stages while remaining largely independent of alpha/delta power. Effects persist under shared-subspace estimation and across spacing metrics.",
    "The analyses support a narrower descriptive conclusion. High-gamma and broadband differ under the declared branching-related estimator, fitted-geometry features add discriminative information in subject-preserving validation, and scalp EEG geometry effects persist under shared-subspace estimation and alternative spacing metrics. Historical cross-subject correlations with the legacy geometry-proximity score are retained as provenance-qualified results and are not attributed to the current ND construction.",
)

legacy_results = r"""The fitted-operator analyses also contain a historical cross-subject geometry association that must be interpreted using the statistic that was actually computed. Across 18 subjects, the legacy geometry-proximity score, defined as eigenvector overlap divided by minimum eigenvalue gap plus $10^{-10}$, covaried strongly with the branching statistic ($r = 0.860$, $p = 4.8 \times 10^{-6}$). Leave-one-subject-out jackknife recomputation of that historical association produced $r$ values from 0.792 to 0.889. The branching statistic also correlated with tighter minimum eigenvalue spacing ($r = -0.588$, $p = 0.0102$). These are descriptive associations among fitted summaries; the $r = 0.860$ result is not a result of the current PC1-based Near-Degeneracy (ND) score.

Complexity showed the opposite pattern for the same historical proximity statistic. Lempel--Ziv complexity correlated negatively with the legacy geometry-proximity score ($r = -0.684$, $p = 0.00174$) and positively with minimum eigenvalue spacing ($r = 0.745$, $p = 3.9 \times 10^{-4}$). The avalanche exponent $\tau$ also correlated with the legacy score ($r = 0.526$, $p = 0.025$), though less strongly than the branching statistic did. The current ND construct remains available for window-level analyses and future prospectively specified subject-level aggregation, but these historical subject-level correlations are not relabeled as ND results."""

text = sub_once(
    text,
    r"The geometry of the fitted operators tracked the most widely used criticality and complexity measures across subjects\..*?Taken together, criticality and operator geometry occupy a shared descriptive space, but these are empirical associations between fitted summaries, not evidence for a unique underlying operator or a mathematically exact exceptional point \\citep\{Bergholtz2021\}\.",
    legacy_results,
    "cross-subject geometry results",
)

# Correct the historical Figure 2 interpretation without touching window-level ND uses elsewhere.
text = text.replace("Subject mean ND score", "Subject mean legacy proximity score")
text = text.replace("Mean ND score by subject", "Mean legacy proximity score by subject")
text = sub_once(
    text,
    r"The ND score is a principled composite derived from the two geometry summaries.*?Pearson correlations\s*\(r\) and jackknife resampling ranges are reported\.",
    "The top-row subject-level quantity is the historical geometry-proximity score (eigenvector overlap divided by minimum eigenvalue gap plus $10^{-10}$), retained for provenance. It is not the current PC1-based Near-Degeneracy (ND) score. Pearson correlations ($r$) and leave-one-subject-out jackknife ranges are reported for the historical statistic.",
    "figure 2 metric description",
)
text = text.replace("versus the ND score.", "versus the legacy geometry-proximity score.")

text = text.replace(
    "(ii) cross-subject covariation between branching ratio and ND score,",
    "(ii) the provenance-qualified cross-subject association between branching ratio and the legacy geometry-proximity score,",
)
text = text.replace(
    "but not by ND score ($r=0.058$, $p=0.82$)",
    "but not by the legacy geometry-proximity score ($r=0.058$, $p=0.82$)",
)
text = text.replace(
    "effective rank showed a small negative relationship with ND score",
    "effective rank showed a small negative relationship with the legacy geometry-proximity score",
)
text = text.replace(
    "Effective rank vs. ND score",
    "Effective rank vs. legacy proximity score",
)

text = text.replace(
    "Subjects near-critical by avalanche statistics sit in crowded, non-orthogonal fitted-operator space. The ND score covaried strongly with branching ratio ($r=0.86$). Spacing moved with it.",
    "Subjects with larger values of the branching-related statistic also showed larger values of the historical geometry-proximity score ($r=0.86$), while minimum spacing moved in the complementary direction. This is a provenance-qualified association among fitted summaries, not a result of the current ND definition.",
)

text = sub_once(
    text,
    r"\\paragraph\{Why ``ND score'' is only shorthand\}.*?the term is chosen for readability rather than to claim EP physics\.",
    r"\\paragraph{Legacy proximity versus current ND.} The historical cross-subject statistic is the ratio of eigenvector overlap to minimum eigenvalue gap plus a small numerical constant. The current Near-Degeneracy (ND) construct is a separate PC1-based composite of transformed eigenvalue crowding and eigenvector conditioning. The two are not algebraically equivalent, and the historical $r=0.86$ association is therefore reported only for the legacy proximity statistic. Neither construct implies a mathematically exact exceptional point, topological invariant, or identified neural mechanism.",
    "discussion metric distinction",
)

text = text.replace(
    "Kreiss constants tracked eigenvector condition number ($r=0.91$) but not the ND score ($r=0.058$).",
    "Kreiss constants tracked eigenvector condition number ($r=0.91$) but not the legacy geometry-proximity score ($r=0.058$).",
)
text = text.replace(
    "(H2) cross-subject association between $\\sigma$ and ND score;",
    "(H2) historical cross-subject association between $\\sigma$ and the legacy geometry-proximity score, retained as provenance-qualified rather than promoted to a final-ND endpoint;",
)
text = text.replace(
    "mathematical coupling between ND score and effective rank",
    "mathematical coupling between the historical geometry-proximity score and effective rank",
)
text = text.replace(
    "their rank--ND-score correlation served as the null distribution",
    "their rank--legacy-proximity correlation served as the null distribution",
)

# Ensure the corrected manuscript does not retain the specific known mislabeling.
for forbidden in (
    "the ND score, a composite of eigenvalue crowding and eigenvector non-orthogonality, covaried strongly with the branching ratio",
    "The ND score covaried strongly with branching ratio ($r=0.86$)",
    "(H2) cross-subject association between $\\sigma$ and ND score;",
):
    if forbidden in text:
        raise RuntimeError(f"forbidden manuscript wording remains: {forbidden}")

MANUSCRIPT.write_text(text, encoding="utf-8")

# Add a fast technical-review path without turning the repository into admissions marketing.
readme = README.read_text(encoding="utf-8")
marker = "## Scientific scope\n"
review_block = """## Technical review path\n\nFor a fast engineering-oriented review, start with [`TECHNICAL_REVIEW_GUIDE.md`](TECHNICAL_REVIEW_GUIDE.md). It maps the end-to-end data/analysis architecture to concrete modules, tests, failure controls, provenance artifacts, and reproducibility commands. The guide is intentionally program-agnostic and is designed for technical reviewers who want to assess software-engineering and computational-research quality before reading the full manuscript.\n\n"""
if review_block not in readme:
    if marker not in readme:
        raise RuntimeError("README scientific-scope marker not found")
    readme = readme.replace(marker, review_block + marker, 1)
README.write_text(readme, encoding="utf-8")

# Record that manuscript terminology is now synchronized while leaving raw-data gates open.
audit = AUDIT.read_text(encoding="utf-8")
status_header = "| Item | Status | Public interpretation |\n|---|---|---|\n"
status_row = "| Manuscript title and legacy-metric labeling | Resolved in source | `manuscript/main.tex`, README, and citation metadata use the same title; historical `r ~= 0.86` associations are labeled as legacy proximity results |\n"
if status_row not in audit:
    if status_header not in audit:
        raise RuntimeError("PUBLIC_AUDIT status table marker not found")
    audit = audit.replace(status_header, status_header + status_row, 1)
AUDIT.write_text(audit, encoding="utf-8")

print("Engineering-review manuscript/documentation patch completed.")
