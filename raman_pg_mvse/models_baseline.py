from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from model_pca_branch import BranchedPCAMLPWrapper, Shallow1DCNNWrapper


class PLSDAWrapper(BaseEstimator, ClassifierMixin):
    """Approximate multi-class PLS-DA using one-vs-rest PLSRegression."""

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.model_: Optional[OneVsRestClassifier] = None
        self.label_encoder_: Optional[LabelEncoder] = None

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.model_ = OneVsRestClassifier(
            PLSRegression(n_components=min(self.n_components, X.shape[1], X.shape[0] - 1))
        )
        self.model_.fit(X, y_encoded)
        return self

    def predict_proba(self, X):
        raw = np.column_stack([est.predict(X).ravel() for est in self.model_.estimators_])
        raw = np.clip(raw, 0, None)
        row_sum = raw.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return raw / row_sum

    def predict(self, X):
        prob = self.predict_proba(X)
        return self.label_encoder_.inverse_transform(np.argmax(prob, axis=1))


class FeatureNameSafeClassifier(BaseEstimator, ClassifierMixin):
    """Wrap tabular estimators that emit feature-name warnings on ndarray predict input."""

    def __init__(self, estimator: Any):
        self.estimator = estimator
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X, y):
        X_fit = self._to_frame(X, fit=True)
        self.estimator.fit(X_fit, y)
        return self

    def predict(self, X):
        return self.estimator.predict(self._to_frame(X, fit=False))

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._to_frame(X, fit=False))

    @property
    def feature_importances_(self):
        return getattr(self.estimator, "feature_importances_", None)

    def _to_frame(self, X, fit: bool):
        if isinstance(X, pd.DataFrame):
            if fit or self.feature_names_ is None:
                self.feature_names_ = list(X.columns)
            return X
        if fit or self.feature_names_ is None:
            self.feature_names_ = [f"f_{idx}" for idx in range(np.asarray(X).shape[1])]
        return pd.DataFrame(np.asarray(X), columns=self.feature_names_)


@dataclass
class ModelBundle:
    name: str
    estimator: Any
    needs_probabilities: bool = True


def _build_tree_booster(random_state: int):
    try:
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
            min_child_samples=1,
        )
        return "LightGBM", FeatureNameSafeClassifier(estimator)
    except Exception:
        try:
            from xgboost import XGBClassifier

            return (
                "XGBoost",
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=random_state,
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                ),
            )
        except Exception:
            return (
                "RandomForestFallback",
                RandomForestClassifier(n_estimators=400, random_state=random_state, class_weight="balanced"),
            )


def build_baseline_models(model_config) -> Dict[str, ModelBundle]:
    tree_name, tree_model = _build_tree_booster(model_config.random_seed)

    models = {
        "PCA-LDA": ModelBundle(
            name="PCA-LDA",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=0.98, random_state=model_config.random_seed)),
                    ("lda", LinearDiscriminantAnalysis()),
                ]
            ),
        ),
        "PLS-DA": ModelBundle(name="PLS-DA", estimator=PLSDAWrapper(n_components=8)),
        "PCA-SVM": ModelBundle(
            name="PCA-SVM",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=0.98, random_state=model_config.random_seed)),
                    (
                        "svm",
                        SVC(
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=model_config.random_seed,
                        ),
                    ),
                ]
            ),
        ),
        "RandomForest": ModelBundle(
            name="RandomForest",
            estimator=RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=model_config.random_seed,
            ),
        ),
        tree_name: ModelBundle(name=tree_name, estimator=tree_model),
        "Shallow1D-CNN": ModelBundle(
            name="Shallow1D-CNN",
            estimator=Shallow1DCNNWrapper(
                epochs=model_config.cnn_epochs,
                batch_size=model_config.cnn_batch_size,
                lr=model_config.cnn_lr,
                weight_decay=model_config.cnn_weight_decay,
                patience=model_config.cnn_patience,
                random_state=model_config.random_seed,
            ),
        ),
        "BranchedPCA-MLP": ModelBundle(
            name="BranchedPCA-MLP",
            estimator=BranchedPCAMLPWrapper(
                epochs=model_config.branched_mlp_epochs,
                batch_size=model_config.branched_mlp_batch_size,
                lr=model_config.branched_mlp_lr,
                weight_decay=model_config.branched_mlp_weight_decay,
                dropout=model_config.branched_mlp_dropout,
                patience=model_config.branched_mlp_patience,
                random_state=model_config.random_seed,
            ),
        ),
    }
    return models


def clone_estimator(estimator: Any) -> Any:
    try:
        return clone(estimator)
    except Exception:
        return copy.deepcopy(estimator)
