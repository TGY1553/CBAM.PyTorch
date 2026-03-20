# PG-MVSE Raman Classification Project

This folder contains a reproducible Raman spectroscopy classification framework for toothbrush-handle evidence.

## Label file format
Create a CSV/XLSX file with columns:

```csv
sample_id,label
1,Sanxiao
2,Colgate
```

Place your spectra file at `data/raman_data.xlsx` (or `data/raman_data.csv`). The project now also looks for `data/labels.csv` by default; if it is missing, a ready-to-fill template will be generated there for you.

## Notes
- The code prevents leakage by fitting preprocessing and feature selection only on training folds.
- If repeated spectra become available later, you can provide `groups_path` to activate group-aware CV.
- `PLS-DA` is implemented as a practical `PLSRegression + one-vs-rest` approximation for multi-class settings.
