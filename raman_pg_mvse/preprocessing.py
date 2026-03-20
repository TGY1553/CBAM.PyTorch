from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessArtifacts:
    wavenumbers: np.ndarray
    selected_indices: np.ndarray


class SpectralPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        pipeline_steps: Sequence[str],
        wavenumbers: np.ndarray,
        wavenumber_range: Optional[Tuple[float, float]] = None,
        sg_window_length: int = 11,
        sg_polyorder: int = 3,
        derivative_window_length: int = 11,
        derivative_polyorder: int = 3,
        use_standard_scaler: bool = False,
    ) -> None:
        self.pipeline_steps = list(pipeline_steps)
        self.original_wavenumbers = np.asarray(wavenumbers)
        self.wavenumber_range = wavenumber_range
        self.sg_window_length = sg_window_length
        self.sg_polyorder = sg_polyorder
        self.derivative_window_length = derivative_window_length
        self.derivative_polyorder = derivative_polyorder
        self.use_standard_scaler = use_standard_scaler
        self.scaler_: Optional[StandardScaler] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.processed_wavenumbers_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_work = np.asarray(X, dtype=float)
        wavenumbers = self.original_wavenumbers.copy()
        selected_indices = np.arange(len(wavenumbers))

        if "crop" in self.pipeline_steps and self.wavenumber_range is not None:
            lo, hi = self.wavenumber_range
            mask = (wavenumbers >= lo) & (wavenumbers <= hi)
            selected_indices = selected_indices[mask]
            wavenumbers = wavenumbers[mask]
            X_work = X_work[:, mask]

        for step in self.pipeline_steps:
            if step in {"crop", "standardize"}:
                continue
            X_work = self._apply_step(X_work, step, fit=True)

        if self.use_standard_scaler or "standardize" in self.pipeline_steps:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_work)

        self.selected_indices_ = selected_indices
        self.processed_wavenumbers_ = wavenumbers
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("SpectralPreprocessor must be fitted before calling transform().")
        X_work = np.asarray(X, dtype=float)[:, self.selected_indices_]
        for step in self.pipeline_steps:
            if step in {"crop", "standardize"}:
                continue
            X_work = self._apply_step(X_work, step, fit=False)
        if self.scaler_ is not None:
            X_work = self.scaler_.transform(X_work)
        return X_work

    def _safe_window(self, width: int, n_features: int) -> int:
        width = min(width, n_features if n_features % 2 == 1 else n_features - 1)
        return max(width, 3 if width >= 3 else 1)

    def _apply_step(self, X: np.ndarray, step: str, fit: bool) -> np.ndarray:
        if step == "sg_smooth":
            window = self._safe_window(self.sg_window_length, X.shape[1])
            poly = min(self.sg_polyorder, max(1, window - 1))
            return savgol_filter(X, window_length=window, polyorder=poly, axis=1)
        if step == "first_derivative":
            window = self._safe_window(self.derivative_window_length, X.shape[1])
            poly = min(self.derivative_polyorder, max(1, window - 1))
            return savgol_filter(X, window_length=window, polyorder=poly, deriv=1, axis=1)
        if step == "l2norm":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return X / norms
        raise ValueError(f"Unknown preprocessing step: {step}")

    @property
    def artifacts_(self) -> PreprocessArtifacts:
        if self.selected_indices_ is None or self.processed_wavenumbers_ is None:
            raise RuntimeError("Preprocessor artifacts are unavailable before fit().")
        return PreprocessArtifacts(
            wavenumbers=self.processed_wavenumbers_,
            selected_indices=self.selected_indices_,
        )


def build_preprocessors(config, wavenumbers: np.ndarray, wavenumber_range: Optional[Tuple[float, float]] = None) -> Dict[str, SpectralPreprocessor]:
    preprocessors = {}
    for name, steps in config.candidate_pipelines.items():
        preprocessors[name] = SpectralPreprocessor(
            pipeline_steps=steps,
            wavenumbers=wavenumbers,
            wavenumber_range=wavenumber_range,
            sg_window_length=config.sg_window_length,
            sg_polyorder=config.sg_polyorder,
            derivative_window_length=config.derivative_window_length,
            derivative_polyorder=config.derivative_polyorder,
            use_standard_scaler=config.standardize_for_linear_models,
        )
    return preprocessors
