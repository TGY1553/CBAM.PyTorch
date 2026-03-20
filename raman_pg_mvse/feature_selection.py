from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder


@dataclass
class IntervalSelectionResult:
    importances: np.ndarray
    smoothed_importances: np.ndarray
    intervals: List[Tuple[float, float, int, int]]
    selected_indices: np.ndarray
    stability_scores: Optional[np.ndarray] = None
    stability_indices: Optional[np.ndarray] = None


class PeakIntervalSelector:
    def __init__(
        self,
        model_type: str = "extratrees",
        smoothing_window: int = 9,
        min_interval_width: int = 8,
        max_intervals: int = 8,
        importance_threshold_quantile: float = 0.85,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.smoothing_window = smoothing_window
        self.min_interval_width = min_interval_width
        self.max_intervals = max_intervals
        self.importance_threshold_quantile = importance_threshold_quantile
        self.random_state = random_state
        self.result_: Optional[IntervalSelectionResult] = None

    def fit(self, X: np.ndarray, y: np.ndarray, wavenumbers: np.ndarray) -> "PeakIntervalSelector":
        estimator = self._build_estimator()
        estimator.fit(X, y)
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        sigma = max(self.smoothing_window / 3.0, 1.0)
        smoothed = gaussian_filter1d(importances, sigma=sigma)
        threshold = float(np.quantile(smoothed, self.importance_threshold_quantile))
        peaks, properties = find_peaks(smoothed, height=threshold)
        heights = properties.get("peak_heights", np.array([]))
        order = np.argsort(heights)[::-1] if len(heights) else np.array([], dtype=int)

        intervals: List[Tuple[float, float, int, int]] = []
        covered = np.zeros(len(wavenumbers), dtype=bool)
        for idx in order[: self.max_intervals * 2]:
            peak_idx = peaks[idx]
            left = peak_idx
            right = peak_idx
            while left > 0 and smoothed[left] >= threshold * 0.6:
                left -= 1
            while right < len(smoothed) - 1 and smoothed[right] >= threshold * 0.6:
                right += 1
            if right - left + 1 < self.min_interval_width:
                pad = self.min_interval_width - (right - left + 1)
                left = max(0, left - pad // 2)
                right = min(len(smoothed) - 1, right + pad - pad // 2)
            if covered[left : right + 1].mean() > 0.5:
                continue
            covered[left : right + 1] = True
            intervals.append((float(wavenumbers[left]), float(wavenumbers[right]), int(left), int(right)))
            if len(intervals) >= self.max_intervals:
                break

        if not intervals:
            top_indices = np.argsort(smoothed)[::-1][: self.min_interval_width]
            left = int(np.min(top_indices))
            right = int(np.max(top_indices))
            intervals.append((float(wavenumbers[left]), float(wavenumbers[right]), left, right))

        selected_indices = np.unique(np.concatenate([np.arange(l, r + 1) for _, _, l, r in intervals]))
        self.result_ = IntervalSelectionResult(
            importances=importances,
            smoothed_importances=smoothed,
            intervals=intervals,
            selected_indices=selected_indices,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("PeakIntervalSelector must be fitted before transform().")
        return X[:, self.result_.selected_indices]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
        self.fit(X, y, wavenumbers)
        return self.transform(X)

    def _build_estimator(self):
        if self.model_type == "randomforest":
            return RandomForestClassifier(n_estimators=400, random_state=self.random_state, class_weight="balanced")
        return ExtraTreesClassifier(n_estimators=500, random_state=self.random_state, class_weight="balanced")


class StabilitySparseSelector:
    def __init__(
        self,
        repeats: int = 20,
        top_k: int = 80,
        c: float = 0.3,
        l1_ratio: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.repeats = repeats
        self.top_k = top_k
        self.c = c
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.stability_scores_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StabilitySparseSelector":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=self.repeats, random_state=self.random_state)
        counts = np.zeros(X.shape[1], dtype=float)
        for train_idx, _ in cv.split(X, y_encoded):
            clf = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=self.c,
                l1_ratio=self.l1_ratio,
                class_weight="balanced",
                max_iter=5000,
                multi_class="ovr",
                random_state=self.random_state,
            )
            clf.fit(X[train_idx], y_encoded[train_idx])
            coef = np.abs(clf.coef_)
            if coef.ndim == 2:
                coef = coef.max(axis=0)
            counts += (coef > 1e-8).astype(float)

        scores = counts / (self.repeats * 3)
        self.stability_scores_ = scores
        self.selected_indices_ = np.argsort(scores)[::-1][: min(self.top_k, X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("StabilitySparseSelector must be fitted before transform().")
        return X[:, self.selected_indices_]


def save_selected_intervals(path: Path, intervals: Sequence[Tuple[float, float, int, int]]) -> None:
    pd.DataFrame(
        [
            {
                "start_wavenumber": start,
                "end_wavenumber": end,
                "start_index": left,
                "end_index": right,
            }
            for start, end, left, right in intervals
        ]
    ).to_csv(path, index=False)


def save_selected_features(path: Path, wavenumbers: np.ndarray, indices: np.ndarray, scores: Optional[np.ndarray] = None) -> None:
    payload = {"feature_index": indices, "wavenumber": wavenumbers[indices]}
    if scores is not None:
        payload["score"] = scores[indices]
    pd.DataFrame(payload).to_csv(path, index=False)


def plot_interval_importance(result: IntervalSelectionResult, wavenumbers: np.ndarray, output_prefix: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(wavenumbers, result.importances, label="importance", alpha=0.35)
    plt.plot(wavenumbers, result.smoothed_importances, label="smoothed importance", linewidth=2)
    for start, end, _, _ in result.intervals:
        plt.axvspan(start, end, color="tab:red", alpha=0.18)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Importance")
    plt.title("Peak-guided informative interval discovery")
    plt.legend(frameon=False)
    plt.tight_layout()
    for suffix in (".png", ".svg"):
        plt.savefig(output_prefix.with_suffix(suffix), dpi=300)
    plt.close()
