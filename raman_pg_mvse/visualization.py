from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


def _save_current_figure(output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        plt.savefig(output_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close()


def plot_mean_spectra(wavenumbers: np.ndarray, X_raw: np.ndarray, X_processed: np.ndarray, output_prefix: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(wavenumbers, X_raw.mean(axis=0), label="Raw mean spectrum", linewidth=1.8)
    if X_processed.shape[1] == len(wavenumbers):
        x_axis = wavenumbers
    else:
        x_axis = np.linspace(wavenumbers.min(), wavenumbers.max(), X_processed.shape[1])
    plt.plot(x_axis, X_processed.mean(axis=0), label="Processed mean spectrum", linewidth=1.8)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Mean spectra before and after preprocessing")
    plt.legend(frameon=False)
    plt.tight_layout()
    _save_current_figure(output_prefix)


def plot_pca_scores(X: np.ndarray, labels: np.ndarray, output_prefix: Path) -> None:
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    for cls in np.unique(labels):
        mask = labels == cls
        plt.scatter(scores[mask, 0], scores[mask, 1], label=str(cls), s=40, alpha=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title("PCA score plot")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    _save_current_figure(output_prefix)


def plot_confusion_matrices(confusion_matrices: Dict[str, np.ndarray], labels: np.ndarray, output_dir: Path) -> None:
    class_names = np.unique(labels)
    for model_name, cm in confusion_matrices.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"Confusion matrix - {model_name}")
        plt.tight_layout()
        _save_current_figure(output_dir / f"confusion_{model_name.replace(' ', '_')}")


def plot_baseline_comparison(summary_df: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = summary_df.sort_values("macro_f1", ascending=True)
    plt.figure(figsize=(8, max(4, 0.5 * len(plot_df))))
    plt.barh(plot_df["model"], plot_df["macro_f1"], color="tab:blue", alpha=0.8)
    plt.xlabel("Macro-F1")
    plt.title("Baseline and PG-MVSE comparison")
    plt.tight_layout()
    _save_current_figure(output_prefix)


def plot_feature_importance(importance: Optional[np.ndarray], output_prefix: Path, title: str = "Feature importance") -> None:
    if importance is None:
        return
    top_idx = np.argsort(importance)[::-1][:30]
    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(len(top_idx)), importance[top_idx], color="tab:orange")
    plt.xlabel("Ranked feature index")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    _save_current_figure(output_prefix)


def plot_stacking_summary(summary_df: pd.DataFrame, output_prefix: Path) -> None:
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "roc_auc_ovr"]
    plot_df = summary_df.set_index("model")[metrics]
    plot_df.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Score")
    plt.title("Model performance summary")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_current_figure(output_prefix)
