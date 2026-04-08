"""Criticality-decoding correlation analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix as sk_confusion_matrix


@dataclass
class DecodingResult:
    """Result of criticality-decoding analysis.

    Attributes
    ----------
    n_channels_used : int
        Number of channels in this analysis.
    selection : str
        How channels were selected: "most_critical", "least_critical", "random".
    accuracy : float
        Mean cross-validated accuracy.
    accuracy_std : float
        Standard deviation across folds.
    confusion_matrix : np.ndarray
        Aggregated confusion matrix.
    channel_names : list[str]
        Names of channels used.
    fold_accuracies : list[float]
        Per-fold accuracy values.
    """

    n_channels_used: int
    selection: str
    accuracy: float
    accuracy_std: float
    confusion_matrix: np.ndarray
    channel_names: list[str]
    fold_accuracies: list[float] = field(default_factory=list)


@dataclass
class SelectionComparisonResult:
    """Statistical comparison between channel selection strategies.

    Attributes
    ----------
    n_channels : int
        Number of channels used in this comparison.
    accuracy_most_critical : float
        Mean accuracy for most-critical channels.
    accuracy_least_critical : float
        Mean accuracy for least-critical channels.
    accuracy_random : float
        Mean accuracy for random channels.
    diff_most_vs_least : float
        Accuracy difference (most - least).
    p_value_most_vs_least : float
        Permutation test p-value for most vs. least.
    diff_most_vs_random : float
        Accuracy difference (most - random).
    p_value_most_vs_random : float
        Permutation test p-value for most vs. random.
    """

    n_channels: int
    accuracy_most_critical: float
    accuracy_least_critical: float
    accuracy_random: float
    diff_most_vs_least: float
    p_value_most_vs_least: float
    diff_most_vs_random: float
    p_value_most_vs_random: float


def rank_channels_by_criticality(
    channel_names: list[str],
    tau_values: dict[str, float],
    target_tau: float = 1.5,
) -> list[str]:
    """Rank channels by proximity to critical exponent.

    Parameters
    ----------
    channel_names : list[str]
        All channel names.
    tau_values : dict[str, float]
        Channel name -> tau exponent mapping.
    target_tau : float
        Target critical exponent.

    Returns
    -------
    list[str]
        Channel names sorted by |tau - target_tau| (most critical first).
    """
    scored = []
    for ch in channel_names:
        tau = tau_values.get(ch, float("nan"))
        if not np.isnan(tau):
            scored.append((ch, abs(tau - target_tau)))

    scored.sort(key=lambda x: x[1])
    return [ch for ch, _ in scored]


def decode_with_channel_subset(
    data: np.ndarray,
    labels: np.ndarray,
    channel_indices: list[int],
    block_labels: np.ndarray,
    classifier: str = "lda",
) -> DecodingResult:
    """Decode stimulus category using a subset of channels with leave-one-block-out CV.

    Each channel is z-scored using its own mean and standard deviation
    computed across all trials and timepoints in the training set only.
    This normalizes amplitude differences between channels while
    preserving temporal structure within each channel, and prevents
    leakage of test-set distributional information.

    Parameters
    ----------
    data : np.ndarray, shape (n_trials, n_channels, n_timepoints)
        Epoch data.
    labels : np.ndarray, shape (n_trials,)
        Category labels.
    channel_indices : list[int]
        Indices of channels to use.
    block_labels : np.ndarray, shape (n_trials,)
        Block labels for leave-one-block-out CV.
    classifier : str
        "lda" or "svm".

    Returns
    -------
    DecodingResult
    """
    n_trials = data.shape[0]
    selected = data[:, channel_indices, :]

    unique_blocks = np.unique(block_labels)
    fold_accuracies = []
    all_true = []
    all_pred = []

    for test_block in unique_blocks:
        test_mask = block_labels == test_block
        train_mask = ~test_mask

        if train_mask.sum() < 2 or test_mask.sum() < 1:
            continue

        train_labels = labels[train_mask]
        if len(np.unique(train_labels)) < 2:
            continue

        X_train_raw = selected[train_mask]
        X_test_raw = selected[test_mask]
        y_train = labels[train_mask]
        y_test = labels[test_mask]

        n_train = X_train_raw.shape[0]
        n_test = X_test_raw.shape[0]
        n_ch = X_train_raw.shape[1]
        n_t = X_train_raw.shape[2]

        ch_mean = X_train_raw.mean(axis=(0, 2), keepdims=True)
        ch_std = X_train_raw.std(axis=(0, 2), keepdims=True)
        ch_std[ch_std == 0] = 1.0

        X_train_normed = (X_train_raw - ch_mean) / ch_std
        X_test_normed = (X_test_raw - ch_mean) / ch_std

        X_train = X_train_normed.reshape(n_train, n_ch * n_t)
        X_test = X_test_normed.reshape(n_test, n_ch * n_t)

        if classifier == "lda":
            clf = LinearDiscriminantAnalysis()
        else:
            clf = SVC(kernel="linear")

        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(acc)
            all_true.extend(y_test.tolist())
            all_pred.extend(y_pred.tolist())
        except Exception:
            continue

    if not fold_accuracies:
        return DecodingResult(
            n_channels_used=len(channel_indices),
            selection="custom",
            accuracy=float("nan"),
            accuracy_std=float("nan"),
            confusion_matrix=np.array([]),
            channel_names=[],
            fold_accuracies=[],
        )

    cm = sk_confusion_matrix(all_true, all_pred) if all_true else np.array([])

    return DecodingResult(
        n_channels_used=len(channel_indices),
        selection="custom",
        accuracy=float(np.mean(fold_accuracies)),
        accuracy_std=float(np.std(fold_accuracies)),
        confusion_matrix=cm,
        channel_names=[],
        fold_accuracies=fold_accuracies,
    )


def _permutation_test_accuracy(
    accs_a: list[float],
    accs_b: list[float],
    n_perm: int = 1000,
    seed: int = 42,
) -> float:
    """Permutation test for difference in mean accuracy between two groups.

    Parameters
    ----------
    accs_a : list[float]
        Per-fold accuracies for group A.
    accs_b : list[float]
        Per-fold accuracies for group B.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    float
        Two-tailed p-value.
    """
    a = np.array([x for x in accs_a if not np.isnan(x)])
    b = np.array([x for x in accs_b if not np.isnan(x)])

    if len(a) < 1 or len(b) < 1:
        return float("nan")

    observed = abs(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = abs(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
        if perm_diff >= observed:
            count += 1

    return (count + 1) / (n_perm + 1)


def criticality_decoding_analysis(
    data: np.ndarray,
    labels: np.ndarray,
    block_labels: np.ndarray,
    channel_names: list[str],
    tau_values: dict[str, float],
    n_channels_list: list[int] | None = None,
    classifier: str = "lda",
    n_random: int = 10,
    seed: int = 42,
    target_tau: float = 1.5,
    n_perm_comparison: int = 1000,
) -> dict[str, Any]:
    """Run full criticality-decoding correlation analysis.

    Parameters
    ----------
    data : np.ndarray, shape (n_trials, n_channels, n_timepoints)
        Epoch data.
    labels : np.ndarray
        Category labels.
    block_labels : np.ndarray
        Block labels.
    channel_names : list[str]
        Channel names.
    tau_values : dict[str, float]
        Per-channel tau exponents.
    n_channels_list : list[int]
        Number of channels to test.
    classifier : str
        "lda" or "svm".
    n_random : int
        Number of random channel subsets to average.
    seed : int
        Random seed.
    target_tau : float
        Critical exponent target for ranking.
    n_perm_comparison : int
        Number of permutations for selection strategy comparison test.

    Returns
    -------
    dict
        Keys: "most_critical", "least_critical", "random" -> list[DecodingResult],
              "comparisons" -> list[SelectionComparisonResult].
    """
    if n_channels_list is None:
        n_channels_list = [5, 10, 20]

    ranked = rank_channels_by_criticality(channel_names, tau_values, target_tau)
    ch_to_idx = {ch: i for i, ch in enumerate(channel_names)}

    results: dict[str, Any] = {
        "most_critical": [],
        "least_critical": [],
        "random": [],
        "comparisons": [],
    }

    rng = np.random.default_rng(seed)

    for n_ch in n_channels_list:
        if n_ch > len(ranked):
            continue

        top_channels = ranked[:n_ch]
        top_indices = [ch_to_idx[ch] for ch in top_channels]
        r_top = decode_with_channel_subset(
            data, labels, top_indices, block_labels, classifier
        )
        r_top.selection = "most_critical"
        r_top.channel_names = top_channels
        results["most_critical"].append(r_top)

        bottom_channels = ranked[-n_ch:]
        bottom_indices = [ch_to_idx[ch] for ch in bottom_channels]
        r_bottom = decode_with_channel_subset(
            data, labels, bottom_indices, block_labels, classifier
        )
        r_bottom.selection = "least_critical"
        r_bottom.channel_names = bottom_channels
        results["least_critical"].append(r_bottom)

        random_accs_all = []
        random_fold_accs_all = []
        for _ in range(n_random):
            rand_channels = rng.choice(ranked, size=n_ch, replace=False).tolist()
            rand_indices = [ch_to_idx[ch] for ch in rand_channels]
            r = decode_with_channel_subset(
                data, labels, rand_indices, block_labels, classifier
            )
            if not np.isnan(r.accuracy):
                random_accs_all.append(r.accuracy)
                random_fold_accs_all.extend(r.fold_accuracies)

        r_avg = DecodingResult(
            n_channels_used=n_ch,
            selection="random",
            accuracy=float(np.mean(random_accs_all)) if random_accs_all else float("nan"),
            accuracy_std=float(np.std(random_accs_all)) if random_accs_all else float("nan"),
            confusion_matrix=np.array([]),
            channel_names=[],
            fold_accuracies=random_fold_accs_all,
        )
        results["random"].append(r_avg)

        p_most_vs_least = _permutation_test_accuracy(
            r_top.fold_accuracies, r_bottom.fold_accuracies,
            n_perm=n_perm_comparison, seed=seed,
        )
        p_most_vs_random = _permutation_test_accuracy(
            r_top.fold_accuracies, random_fold_accs_all,
            n_perm=n_perm_comparison, seed=seed,
        )

        comparison = SelectionComparisonResult(
            n_channels=n_ch,
            accuracy_most_critical=r_top.accuracy,
            accuracy_least_critical=r_bottom.accuracy,
            accuracy_random=r_avg.accuracy,
            diff_most_vs_least=r_top.accuracy - r_bottom.accuracy
            if not (np.isnan(r_top.accuracy) or np.isnan(r_bottom.accuracy))
            else float("nan"),
            p_value_most_vs_least=p_most_vs_least,
            diff_most_vs_random=r_top.accuracy - r_avg.accuracy
            if not (np.isnan(r_top.accuracy) or np.isnan(r_avg.accuracy))
            else float("nan"),
            p_value_most_vs_random=p_most_vs_random,
        )
        results["comparisons"].append(comparison)

    return results
