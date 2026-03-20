from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from models_baseline import clone_estimator
from preprocessing import SpectralPreprocessor
from stacking import PGMVSEClassifier


@dataclass
class EvaluationResult:
    summary: pd.DataFrame
    fold_results: pd.DataFrame
    predictions: pd.DataFrame
    confusion_matrices: Dict[str, np.ndarray]
    oof_probabilities: Dict[str, np.ndarray]


def get_cv_splitter(y: np.ndarray, groups: Optional[np.ndarray], n_splits: int, random_state: int):
    if groups is not None:
        try:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        except Exception:
            pass
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    lb = LabelBinarizer()
    lb.fit(labels)
    try:
        y_true_bin = lb.transform(y_true)
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        metrics["roc_auc_ovr"] = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc_ovr"] = np.nan
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    for cls, value in zip(labels, per_class_f1):
        metrics[f"f1_{cls}"] = value
    return metrics


def evaluate_baselines(dataset, preprocessors: Dict[str, SpectralPreprocessor], baseline_models, config, logger) -> EvaluationResult:
    labels = np.asarray(dataset.labels)
    sample_ids = np.asarray(dataset.sample_ids)
    splitter = get_cv_splitter(labels, dataset.groups, config.model.outer_folds, config.model.random_seed)
    classes = np.unique(labels)

    summary_rows: List[Dict[str, float]] = []
    fold_rows: List[Dict[str, float]] = []
    prediction_rows: List[Dict[str, object]] = []
    confusion_matrices: Dict[str, np.ndarray] = {}
    oof_probabilities: Dict[str, np.ndarray] = {}

    # use P2 as the common baseline preprocessing to keep comparisons fair
    preprocessor = preprocessors["P2"]

    for model_name, bundle in baseline_models.items():
        logger.info("Evaluating baseline model: %s", model_name)
        all_probs = np.zeros((len(labels), len(classes)), dtype=float)
        all_preds = np.empty(len(labels), dtype=object)
        fold_metrics: List[Dict[str, float]] = []

        split_iter = splitter.split(dataset.X_raw, labels, dataset.groups) if dataset.groups is not None else splitter.split(dataset.X_raw, labels)
        for fold_id, (train_idx, test_idx) in enumerate(split_iter, start=1):
            X_train_raw = dataset.X_raw[train_idx]
            X_test_raw = dataset.X_raw[test_idx]
            y_train = labels[train_idx]
            y_test = labels[test_idx]

            prep = SpectralPreprocessor(
                pipeline_steps=preprocessor.pipeline_steps,
                wavenumbers=dataset.wavenumbers,
                wavenumber_range=preprocessor.wavenumber_range,
                sg_window_length=preprocessor.sg_window_length,
                sg_polyorder=preprocessor.sg_polyorder,
                derivative_window_length=preprocessor.derivative_window_length,
                derivative_polyorder=preprocessor.derivative_polyorder,
                use_standard_scaler=preprocessor.use_standard_scaler,
            ).fit(X_train_raw, y_train)
            X_train = prep.transform(X_train_raw)
            X_test = prep.transform(X_test_raw)

            estimator = clone_estimator(bundle.estimator)
            estimator.fit(X_train, y_train)
            if hasattr(estimator, "predict_proba"):
                prob = estimator.predict_proba(X_test)
            else:
                pred = estimator.predict(X_test)
                prob = np.eye(len(classes))[np.searchsorted(classes, pred)]
            pred = classes[np.argmax(prob, axis=1)] if not hasattr(estimator, "predict") else estimator.predict(X_test)

            all_probs[test_idx] = prob
            all_preds[test_idx] = pred
            metrics = compute_metrics(y_test, pred, prob, classes)
            metrics.update({"model": model_name, "fold": fold_id})
            fold_rows.append(metrics)
            fold_metrics.append(metrics)

        confusion_matrices[model_name] = confusion_matrix(labels, all_preds, labels=classes)
        oof_probabilities[model_name] = all_probs
        overall = compute_metrics(labels, all_preds, all_probs, classes)
        overall.update({"model": model_name})
        summary_rows.append(overall)

        for sid, true_label, pred_label, probs in zip(sample_ids, labels, all_preds, all_probs):
            row = {
                "sample_id": sid,
                "true_label": true_label,
                "pred_label": pred_label,
                "model": model_name,
            }
            for cls, p in zip(classes, probs):
                row[f"prob_{cls}"] = p
            prediction_rows.append(row)

    return EvaluationResult(
        summary=pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False),
        fold_results=pd.DataFrame(fold_rows),
        predictions=pd.DataFrame(prediction_rows),
        confusion_matrices=confusion_matrices,
        oof_probabilities=oof_probabilities,
    )


def evaluate_pg_mvse(dataset, config, logger):
    labels = np.asarray(dataset.labels)
    splitter = get_cv_splitter(labels, dataset.groups, config.model.outer_folds, config.model.random_seed)
    classes = np.unique(labels)
    probs = np.zeros((len(labels), len(classes)), dtype=float)
    preds = np.empty(len(labels), dtype=object)
    fold_rows = []
    artifacts = []

    split_iter = splitter.split(dataset.X_raw, labels, dataset.groups) if dataset.groups is not None else splitter.split(dataset.X_raw, labels)
    for fold_id, (train_idx, test_idx) in enumerate(split_iter, start=1):
        model = PGMVSEClassifier(
            config.preprocess,
            config.feature_selection,
            config.model,
            dataset.wavenumbers,
            logger,
            wavenumber_range=config.data.wavenumber_range,
        )
        model.fit(dataset.X_raw[train_idx], labels[train_idx])
        fold_prob = model.predict_proba(dataset.X_raw[test_idx])
        fold_pred = model.predict(dataset.X_raw[test_idx])
        probs[test_idx] = fold_prob
        preds[test_idx] = fold_pred
        metrics = compute_metrics(labels[test_idx], fold_pred, fold_prob, classes)
        metrics.update({"model": "PG-MVSE", "fold": fold_id})
        fold_rows.append(metrics)
        artifacts.append(model.artifacts_)

    summary = pd.DataFrame([compute_metrics(labels, preds, probs, classes) | {"model": "PG-MVSE"}])
    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame({"sample_id": dataset.sample_ids, "true_label": labels, "pred_label": preds})
    for idx, cls in enumerate(classes):
        predictions[f"prob_{cls}"] = probs[:, idx]
    confusion = {"PG-MVSE": confusion_matrix(labels, preds, labels=classes)}
    oof = {"PG-MVSE": probs}
    return EvaluationResult(summary=summary, fold_results=fold_results, predictions=predictions, confusion_matrices=confusion, oof_probabilities=oof), artifacts[-1]
