from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


SUPPORTED_TABLE_SUFFIXES = (".xlsx", ".xls", ".csv")


@dataclass
class RamanDataset:
    wavenumbers: np.ndarray
    X_raw: np.ndarray
    sample_ids: np.ndarray
    labels: np.ndarray
    groups: Optional[np.ndarray]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def _candidate_paths(path: Path, extra_roots: Iterable[Path]) -> list[Path]:
    candidates: list[Path] = []

    def _push(candidate: Path) -> None:
        candidate = Path(candidate)
        if candidate not in candidates:
            candidates.append(candidate)

    _push(path)
    if not path.suffix:
        for suffix in SUPPORTED_TABLE_SUFFIXES:
            _push(path.with_suffix(suffix))

    for root in extra_roots:
        _push(root / path)
        if not path.suffix:
            for suffix in SUPPORTED_TABLE_SUFFIXES:
                _push((root / path).with_suffix(suffix))

        # Also look inside common local data folders for copied project layouts.
        for subdir in ("data", "dataset", "datasets"):
            _push(root / subdir / path.name)
            if not path.suffix:
                for suffix in SUPPORTED_TABLE_SUFFIXES:
                    _push((root / subdir / path.name).with_suffix(suffix))

    return candidates


def resolve_input_path(path: Path, logger, purpose: str) -> Path:
    path = Path(path)
    script_root = Path(__file__).resolve().parent
    search_roots = [Path.cwd(), script_root, script_root.parent]
    for candidate in _candidate_paths(path, search_roots):
        if candidate.exists():
            if candidate != path:
                logger.info("Resolved %s path from %s to %s", purpose, path, candidate)
            return candidate

    searched = "\n  - ".join(str(candidate) for candidate in _candidate_paths(path, search_roots)[:12])
    raise FileNotFoundError(
        f"{purpose} file not found. Configured path: {path}\n"
        f"Tried these locations:\n  - {searched}\n"
        "Please place the file in the project data folder or update the config path."
    )


def load_spectra_matrix(path: Path, logger) -> Dict[str, np.ndarray]:
    resolved_path = resolve_input_path(path, logger, purpose="Spectra")
    df = _read_table(resolved_path)
    if df.shape[1] < 2:
        raise ValueError("The spectra file must contain at least one wavenumber column and one sample column.")

    raw_sample_ids = df.columns[1:]
    if raw_sample_ids.duplicated().any():
        dupes = raw_sample_ids[raw_sample_ids.duplicated()].tolist()
        raise ValueError(f"Duplicate sample IDs found in header: {dupes}")

    wavenumbers = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if wavenumbers.isna().any():
        raise ValueError("Non-numeric values detected in the first (wavenumber) column.")

    matrix = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        bad_count = int(matrix.isna().sum().sum())
        raise ValueError(f"Detected {bad_count} non-numeric or missing spectral entries.")

    if not np.all(np.diff(wavenumbers.to_numpy()) > 0):
        logger.warning("Wavenumber axis is not strictly increasing. Sorting by wavenumber.")
        order = np.argsort(wavenumbers.to_numpy())
        wavenumbers = wavenumbers.iloc[order].reset_index(drop=True)
        matrix = matrix.iloc[order].reset_index(drop=True)

    logger.info("Loaded spectra matrix from %s with %d wavenumbers and %d samples.", resolved_path, df.shape[0], df.shape[1] - 1)
    logger.info("No duplicated sample IDs, non-numeric values, or missing spectral entries remain after checks.")

    return {
        "resolved_path": resolved_path,
        "wavenumbers": wavenumbers.to_numpy(dtype=float),
        "X_raw": matrix.to_numpy(dtype=float).T,
        "sample_ids": raw_sample_ids.astype(str).to_numpy(),
    }


def generate_label_template(sample_ids: np.ndarray, path: Path) -> None:
    template = pd.DataFrame({"sample_id": sample_ids, "label": ["TODO_FILL_LABEL"] * len(sample_ids)})
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        template.to_excel(path, index=False)
    else:
        template.to_csv(path, index=False)


def generate_default_label_files(sample_ids: np.ndarray, base_dir: Path) -> tuple[Path, Path]:
    template_path = base_dir / "label_template.csv"
    labels_path = base_dir / "labels.csv"
    generate_label_template(sample_ids, template_path)
    if not labels_path.exists():
        generate_label_template(sample_ids, labels_path)
    return template_path, labels_path


def load_labels(label_path: Optional[Path], sample_ids: np.ndarray, sample_id_column: str, label_column: str, logger) -> np.ndarray:
    if label_path is None:
        raise FileNotFoundError(
            "No label file configured. Please set config.data.label_path and provide a file with columns "
            f"'{sample_id_column}' and '{label_column}'."
        )

    resolved_label_path = resolve_input_path(label_path, logger, purpose="Label")
    label_df = _read_table(resolved_label_path)
    required = {sample_id_column, label_column}
    missing = required.difference(label_df.columns)
    if missing:
        raise ValueError(f"Label file is missing required columns: {sorted(missing)}")

    label_df[sample_id_column] = label_df[sample_id_column].astype(str)
    if label_df[sample_id_column].duplicated().any():
        dupes = label_df.loc[label_df[sample_id_column].duplicated(), sample_id_column].tolist()
        raise ValueError(f"Duplicate sample IDs found in label file: {dupes}")

    mapping = dict(zip(label_df[sample_id_column], label_df[label_column]))
    missing_samples = [sid for sid in sample_ids if sid not in mapping]
    if missing_samples:
        raise ValueError(f"Missing labels for sample IDs: {missing_samples[:10]}")

    return np.array([mapping[sid] for sid in sample_ids])


def load_groups(groups_path: Optional[Path], sample_ids: np.ndarray, sample_id_column: str, logger, group_column: str = "group") -> Optional[np.ndarray]:
    if groups_path is None:
        return None
    resolved_groups_path = resolve_input_path(groups_path, logger, purpose="Groups")
    group_df = _read_table(resolved_groups_path)
    required = {sample_id_column, group_column}
    missing = required.difference(group_df.columns)
    if missing:
        raise ValueError(f"Groups file is missing required columns: {sorted(missing)}")
    group_df[sample_id_column] = group_df[sample_id_column].astype(str)
    mapping = dict(zip(group_df[sample_id_column], group_df[group_column]))
    return np.array([mapping.get(sid, sid) for sid in sample_ids])


def build_dataset(spectra_path: Path, label_path: Optional[Path], groups_path: Optional[Path], sample_id_column: str, label_column: str, logger) -> RamanDataset:
    loaded = load_spectra_matrix(spectra_path, logger)
    sample_ids = loaded["sample_ids"]
    resolved_spectra_path = loaded["resolved_path"]
    template_path, default_labels_path = generate_default_label_files(sample_ids, resolved_spectra_path.parent)
    if label_path is None:
        raise FileNotFoundError(
            "Label file is required for supervised learning. Templates have been generated at: "
            f"{template_path} and {default_labels_path}. Fill {default_labels_path.name} with sample_id,label pairs and rerun."
        )

    try:
        labels = load_labels(label_path, sample_ids, sample_id_column, label_column, logger)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nA ready-to-fill label file has been generated at: {default_labels_path}. "
            f"Please fill the '{label_column}' column and rerun."
        ) from exc
    groups = load_groups(groups_path, sample_ids, sample_id_column, logger)
    return RamanDataset(
        wavenumbers=loaded["wavenumbers"],
        X_raw=loaded["X_raw"],
        sample_ids=sample_ids,
        labels=labels,
        groups=groups,
    )
