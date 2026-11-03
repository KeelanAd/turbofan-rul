# Predicting when a jet engine needs maintenance

Remaining-Useful-Life (RUL) prediction for turbofan engines using NASA's C-MAPSS dataset.

> 🚧 Work in progress.

## The problem
<!-- Plain English: what RUL is, and why predicting it matters (unplanned removals, safety, cost). -->

## Dataset
NASA C-MAPSS *Turbofan Engine Degradation Simulation*, subset **FD001** (one operating condition, one fault mode).

Each file has 26 space-separated columns and no header:

| File column | Name in code | What it is |
|-------------|--------------|------------|
| 1 | `unit` | engine id |
| 2 | `cycle` | time step (one flight cycle) |
| 3–5 | `op_setting_1` … `op_setting_3` | operational settings |
| 6–26 | `sensor_1` … `sensor_21` | sensor readings (file column = sensor number + 5) |

The raw files are not committed. To reproduce:
1. Download the C-MAPSS dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
2. Put the unzipped `CMaps` folder inside `data/`.

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
