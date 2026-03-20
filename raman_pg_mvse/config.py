from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    # Prefer a project-local path so the code runs after copying the folder to another machine.
    spectra_path: Path = Path("data/raman_data")
    # Default to a project-local label file so users can fill it once and rerun without editing code.
    label_path: Optional[Path] = Path("data/labels")
    output_dir: Path = Path("outputs/pg_mvse_run")
    sample_id_column: str = "sample_id"
    label_column: str = "label"
    groups_path: Optional[Path] = None
    wavenumber_range: Optional[Tuple[float, float]] = None


@dataclass
class PreprocessConfig:
    sg_window_length: int = 11
    sg_polyorder: int = 3
    derivative_window_length: int = 11
    derivative_polyorder: int = 3
    standardize_for_linear_models: bool = False
    candidate_pipelines: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "P1": ["crop", "l2norm"],
            "P2": ["crop", "sg_smooth", "l2norm"],
            "P3": ["crop", "sg_smooth", "first_derivative", "l2norm"],
        }
    )


@dataclass
class FeatureSelectionConfig:
    importance_model: str = "extratrees"
    importance_smoothing_window: int = 9
    min_interval_width: int = 8
    max_intervals: int = 8
    importance_threshold_quantile: float = 0.85
    stability_selection_enabled: bool = True
    stability_cv_repeats: int = 20
    stability_top_k: int = 80
    stability_c: float = 0.3
    stability_l1_ratio: float = 0.5


@dataclass
class ModelConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    random_seed: int = 42
    test_size: float = 0.2
    svm_param_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "pca__n_components": [5, 10, 20, 30],
            "svm__C": [0.5, 1.0, 2.0, 5.0],
            "svm__gamma": ["scale", 0.01, 0.05, 0.1],
        }
    )
    lightgbm_param_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "n_estimators": [100, 200],
            "max_depth": [-1, 3, 5],
            "learning_rate": [0.03, 0.05, 0.1],
        }
    )
    rf_param_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "n_estimators": [200, 400],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 4],
        }
    )
    branched_mlp_epochs: int = 250
    branched_mlp_batch_size: int = 16
    branched_mlp_lr: float = 1e-3
    branched_mlp_weight_decay: float = 1e-4
    branched_mlp_dropout: float = 0.25
    branched_mlp_patience: int = 25
    cnn_epochs: int = 200
    cnn_batch_size: int = 16
    cnn_lr: float = 1e-3
    cnn_weight_decay: float = 1e-4
    cnn_patience: int = 20


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_serializable_dict(self) -> Dict[str, Any]:
        def _convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_convert(v) for v in value]
            return value

        return _convert(asdict(self))


DEFAULT_CONFIG = ExperimentConfig()
