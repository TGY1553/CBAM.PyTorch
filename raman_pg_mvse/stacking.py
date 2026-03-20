from __future__ import annotations

"""Stacked PG-MVSE classifier.

This module intentionally keeps a single top-level ``from __future__`` import.
The previous user-reported error was caused by an accidental duplicate future-import
later in the file; this rewrite keeps the module body clean and import-safe.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from feature_selection import PeakIntervalSelector, StabilitySparseSelector
from model_pca_branch import BranchedPCAMLPWrapper
from models_baseline import _build_tree_booster
from preprocessing import SpectralPreprocessor


@dataclass
class PGMVSEArtifacts:
    intervals: List[Tuple[float, float, int, int]]
    selected_indices: np.ndarray
    importances: np.ndarray
    smoothed_importances: np.ndarray
    feature_importance: Optional[np.ndarray]


class PGMVSEClassifier:
    def __init__(self, preprocess_config, fs_config, model_config, wavenumbers: np.ndarray, logger, wavenumber_range=None) -> None:
        self.preprocess_config = preprocess_config
        self.fs_config = fs_config
        self.model_config = model_config
        self.wavenumbers = wavenumbers
        self.logger = logger
        self.wavenumber_range = wavenumber_range
        self.label_encoder_ = LabelEncoder()
        self.meta_model_ = LogisticRegression(max_iter=3000, class_weight="balanced", multi_class="auto")
        self.view_models_: Dict[str, object] = {}
        self.preprocessors_: Dict[str, SpectralPreprocessor] = {}
        self.selector_: Optional[PeakIntervalSelector] = None
        self.stability_selector_: Optional[StabilitySparseSelector] = None
        self.artifacts_: Optional[PGMVSEArtifacts] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PGMVSEClassifier":
        y_encoded = self.label_encoder_.fit_transform(y)

        prep_a = SpectralPreprocessor(
            pipeline_steps=self.preprocess_config.candidate_pipelines["P2"],
            wavenumbers=self.wavenumbers,
            wavenumber_range=self.wavenumber_range,
            sg_window_length=self.preprocess_config.sg_window_length,
            sg_polyorder=self.preprocess_config.sg_polyorder,
            derivative_window_length=self.preprocess_config.derivative_window_length,
            derivative_polyorder=self.preprocess_config.derivative_polyorder,
        ).fit(X, y)
        X_a = prep_a.transform(X)
        wav_a = prep_a.artifacts_.wavenumbers

        selector = PeakIntervalSelector(
            model_type=self.fs_config.importance_model,
            smoothing_window=self.fs_config.importance_smoothing_window,
            min_interval_width=self.fs_config.min_interval_width,
            max_intervals=self.fs_config.max_intervals,
            importance_threshold_quantile=self.fs_config.importance_threshold_quantile,
            random_state=self.model_config.random_seed,
        ).fit(X_a, y_encoded, wav_a)
        X_a_selected = selector.transform(X_a)

        stability_selector = None
        if self.fs_config.stability_selection_enabled:
            stability_selector = StabilitySparseSelector(
                repeats=self.fs_config.stability_cv_repeats,
                top_k=min(self.fs_config.stability_top_k, X_a_selected.shape[1]),
                c=self.fs_config.stability_c,
                l1_ratio=self.fs_config.stability_l1_ratio,
                random_state=self.model_config.random_seed,
            ).fit(X_a_selected, y)
            X_a_selected = stability_selector.transform(X_a_selected)

        prep_b = SpectralPreprocessor(
            pipeline_steps=self.preprocess_config.candidate_pipelines["P3"],
            wavenumbers=self.wavenumbers,
            wavenumber_range=self.wavenumber_range,
            sg_window_length=self.preprocess_config.sg_window_length,
            sg_polyorder=self.preprocess_config.sg_polyorder,
            derivative_window_length=self.preprocess_config.derivative_window_length,
            derivative_polyorder=self.preprocess_config.derivative_polyorder,
        ).fit(X, y)
        X_b = prep_b.transform(X)
        X_b_selected = X_b[:, selector.result_.selected_indices]
        if stability_selector is not None:
            X_b_selected = stability_selector.transform(X_b_selected)

        prep_c = SpectralPreprocessor(
            pipeline_steps=self.preprocess_config.candidate_pipelines["P2"],
            wavenumbers=self.wavenumbers,
            wavenumber_range=self.wavenumber_range,
            sg_window_length=self.preprocess_config.sg_window_length,
            sg_polyorder=self.preprocess_config.sg_polyorder,
            derivative_window_length=self.preprocess_config.derivative_window_length,
            derivative_polyorder=self.preprocess_config.derivative_polyorder,
            use_standard_scaler=True,
        ).fit(X, y)
        X_c = prep_c.transform(X)

        meta_features = self._build_oof_meta_features(X_a_selected, X_b_selected, X_c, y, y_encoded)
        self.meta_model_.fit(meta_features, y_encoded)

        view_a = self._fit_view_a(X_a_selected, y_encoded)
        view_b, view_b_importance = self._fit_view_b(X_b_selected, y_encoded)
        view_c = self._fit_view_c(X_c, y)

        self.preprocessors_ = {"A": prep_a, "B": prep_b, "C": prep_c}
        self.selector_ = selector
        self.stability_selector_ = stability_selector
        self.view_models_ = {"A": view_a, "B": view_b, "C": view_c}

        final_selected = selector.result_.selected_indices
        if stability_selector is not None:
            final_selected = final_selected[stability_selector.selected_indices_]
        self.artifacts_ = PGMVSEArtifacts(
            intervals=selector.result_.intervals,
            selected_indices=final_selected,
            importances=selector.result_.importances,
            smoothed_importances=selector.result_.smoothed_importances,
            feature_importance=view_b_importance,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.meta_model_.predict_proba(self._build_meta_features(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        encoded = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder_.inverse_transform(encoded)

    def _build_oof_meta_features(
        self,
        X_a: np.ndarray,
        X_b: np.ndarray,
        X_c: np.ndarray,
        y: np.ndarray,
        y_encoded: np.ndarray,
    ) -> np.ndarray:
        inner_cv = StratifiedKFold(
            n_splits=self.model_config.inner_folds,
            shuffle=True,
            random_state=self.model_config.random_seed,
        )
        n_classes = len(self.label_encoder_.classes_)
        meta = np.zeros((len(y), n_classes * 3), dtype=float)
        for train_idx, valid_idx in inner_cv.split(X_a, y_encoded):
            model_a = self._fit_view_a(X_a[train_idx], y_encoded[train_idx])
            model_b, _ = self._fit_view_b(X_b[train_idx], y_encoded[train_idx])
            model_c = self._fit_view_c(X_c[train_idx], y[train_idx])
            meta[valid_idx] = np.hstack(
                [
                    model_a.predict_proba(X_a[valid_idx]),
                    self._predict_tree_proba(model_b, X_b[valid_idx]),
                    model_c.predict_proba(X_c[valid_idx]),
                ]
            )
        return meta

    def _build_meta_features(self, X: np.ndarray) -> np.ndarray:
        prep_a = self.preprocessors_["A"]
        prep_b = self.preprocessors_["B"]
        prep_c = self.preprocessors_["C"]
        selector = self.selector_
        stability = self.stability_selector_

        X_a = selector.transform(prep_a.transform(X))
        if stability is not None:
            X_a = stability.transform(X_a)

        X_b = prep_b.transform(X)[:, selector.result_.selected_indices]
        if stability is not None:
            X_b = stability.transform(X_b)

        X_c = prep_c.transform(X)
        return np.hstack(
            [
                self.view_models_["A"].predict_proba(X_a),
                self._predict_tree_proba(self.view_models_["B"], X_b),
                self.view_models_["C"].predict_proba(X_c),
            ]
        )

    def _fit_view_a(self, X: np.ndarray, y_encoded: np.ndarray):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(random_state=self.model_config.random_seed)),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=self.model_config.random_seed,
                    ),
                ),
            ]
        )
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=self.model_config.svm_param_grid,
            cv=self.model_config.inner_folds,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        grid.fit(X, y_encoded)
        self.logger.info("View A best params: %s", grid.best_params_)
        return grid.best_estimator_

    def _fit_view_b(self, X: np.ndarray, y_encoded: np.ndarray):
        backend_name, model = _build_tree_booster(self.model_config.random_seed)
        model.fit(X, y_encoded)
        importance = getattr(model, "feature_importances_", None)
        self.logger.info("View B booster backend: %s", backend_name)
        return model, importance

    def _fit_view_c(self, X: np.ndarray, y: np.ndarray):
        model = BranchedPCAMLPWrapper(
            epochs=self.model_config.branched_mlp_epochs,
            batch_size=self.model_config.branched_mlp_batch_size,
            lr=self.model_config.branched_mlp_lr,
            weight_decay=self.model_config.branched_mlp_weight_decay,
            dropout=self.model_config.branched_mlp_dropout,
            patience=self.model_config.branched_mlp_patience,
            random_state=self.model_config.random_seed,
        )
        model.fit(X, y)
        return model

    @staticmethod
    def _predict_tree_proba(model, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return model.predict(X)
