from __future__ import annotations

"""
PG-MVSE Raman classification project.

How to prepare label file:
1. Fill outputs/.../label_template.csv or create a csv/xlsx file with columns:
   sample_id,label
   1,Sanxiao
   2,Colgate
2. Set config.data.label_path to that file.
3. If you later have repeated spectra per physical item, provide config.data.groups_path
   with columns sample_id,group so the code can switch to group-aware CV.
"""

from pathlib import Path

import pandas as pd

from config import DEFAULT_CONFIG, ExperimentConfig
from data_loader import build_dataset
from evaluation import evaluate_baselines, evaluate_pg_mvse
from feature_selection import plot_interval_importance, save_selected_features, save_selected_intervals
from models_baseline import build_baseline_models
from preprocessing import build_preprocessors
from utils import save_json, set_global_seed, setup_logger
from visualization import (
    plot_baseline_comparison,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_mean_spectra,
    plot_pca_scores,
    plot_stacking_summary,
)


def run_experiment(config: ExperimentConfig) -> None:
    config.data.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(config.data.output_dir)
    set_global_seed(config.model.random_seed)
    logger.info("Starting PG-MVSE Raman experiment.")

    dataset = build_dataset(
        spectra_path=config.data.spectra_path,
        label_path=config.data.label_path,
        groups_path=config.data.groups_path,
        sample_id_column=config.data.sample_id_column,
        label_column=config.data.label_column,
        logger=logger,
    )

    preprocessors = build_preprocessors(
        config.preprocess,
        dataset.wavenumbers,
        wavenumber_range=config.data.wavenumber_range,
    )
    preview_preprocessor = preprocessors["P2"].fit(dataset.X_raw, dataset.labels)
    X_preview = preview_preprocessor.transform(dataset.X_raw)

    plot_mean_spectra(dataset.wavenumbers, dataset.X_raw, X_preview, config.data.output_dir / "mean_spectra")
    plot_pca_scores(X_preview, dataset.labels, config.data.output_dir / "pca_scores")

    baseline_models = build_baseline_models(config.model)
    baseline_result = evaluate_baselines(dataset, preprocessors, baseline_models, config, logger)

    pg_result, pg_artifacts = evaluate_pg_mvse(dataset, config, logger)

    combined_summary = pd.concat([baseline_result.summary, pg_result.summary], ignore_index=True)
    combined_folds = pd.concat([baseline_result.fold_results, pg_result.fold_results], ignore_index=True)
    combined_predictions = pd.concat([baseline_result.predictions, pg_result.predictions.assign(model="PG-MVSE")], ignore_index=True)

    combined_summary.to_csv(config.data.output_dir / "metrics_summary.csv", index=False)
    combined_folds.to_csv(config.data.output_dir / "fold_results.csv", index=False)
    combined_predictions.to_csv(config.data.output_dir / "final_predictions.csv", index=False)
    save_json(config.data.output_dir / "config_used.json", config.to_serializable_dict())

    save_selected_intervals(config.data.output_dir / "selected_intervals.csv", pg_artifacts.intervals)
    save_selected_features(
        config.data.output_dir / "selected_features.csv",
        preview_preprocessor.artifacts_.wavenumbers,
        pg_artifacts.selected_indices,
        pg_artifacts.smoothed_importances,
    )
    plot_interval_importance(pg_artifacts, preview_preprocessor.artifacts_.wavenumbers, config.data.output_dir / "importance_intervals")
    plot_confusion_matrices(
        baseline_result.confusion_matrices | pg_result.confusion_matrices,
        dataset.labels,
        config.data.output_dir,
    )
    plot_feature_importance(pg_artifacts.feature_importance, config.data.output_dir / "view_b_feature_importance", title="View B feature importance")
    plot_baseline_comparison(combined_summary, config.data.output_dir / "baseline_comparison")
    plot_stacking_summary(combined_summary, config.data.output_dir / "stacking_performance_summary")

    logger.info("Experiment finished. Results saved to %s", config.data.output_dir)


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    # Example override section: update these paths before running on your machine.
    # config.data.spectra_path = Path(r"data/raman_data.xlsx")  # or Path("data/raman_data")
    # config.data.label_path = Path(r"data/labels.csv")
    # config.data.wavenumber_range = (400.0, 1800.0)
    try:
        run_experiment(config)
    except FileNotFoundError as exc:
        print("\n[PG-MVSE setup error]", exc)
        print("Hint: put spectra in data/raman_data.xlsx and labels in data/labels.csv, then rerun.")
