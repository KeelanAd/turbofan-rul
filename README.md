# Predicting when a jet engine needs maintenance

Remaining-Useful-Life (RUL) prediction for turbofan engines using NASA's C-MAPSS dataset.

> 🚧 Work in progress.

## The problem
<!-- Plain English: what RUL is, and why predicting it matters (unplanned removals, safety, cost). -->

## Dataset
NASA C-MAPSS *Turbofan Engine Degradation Simulation*, subset **FD001** (one operating condition, one fault mode).

The raw files are not committed. To reproduce:
1. Download the C-MAPSS dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
2. Put `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` in `data/`.

## Method
<!-- Exploration → dropping flat sensors → RUL labels (clipped) → scaling → baseline → tree models with rolling features. -->

## Results
| Model | Test RMSE (cycles) |
|-------|--------------------|
| Linear regression (baseline) | _TBD_ |
| Random forest | _TBD_ |

<!-- 1–2 plots from figures/ -->

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
```

## Project layout
```
data/            raw C-MAPSS files (not committed)
notebooks/       exploration
src/data.py      loading, RUL labels, dropping flat sensors
src/features.py  rolling features
src/train.py     models, RMSE table, plots
figures/         saved plots
```

## What I learned
<!-- Written by Keelan. -->

## What I'd do next
<!-- Written by Keelan. -->
