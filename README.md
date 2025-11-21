# AI-RAN KPI Forecasting

Small ML project for forecasting telecom RAN KPIs (per-cell, univariate forecasting).

This repository contains a single-file runnable forecasting script (`ai-ran-kpi-forecasting.py`) that:
- Loads generic RAN KPI CSVs or Telecom Italia MI challenge data
- Engineers time & lag features
- Trains an XGBoost (or RandomForest fallback) regressor
- Produces a short-horizon autoregressive forecast

**Quick Start**

Prerequisites
- Python 3.8+ recommended
- Make a virtual environment and activate it (Windows PowerShell shown):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install dev/test dependencies (optional):

```powershell
python -m pip install -r requirements-dev.txt
```

Run the example smoke test with pytest:

```powershell
pytest -q
```

Running the script

The main entrypoint is `ai-ran-kpi-forecasting.py`. The script expects `--data` and supports two dataset types:

- `generic`: a CSV with a timestamp column, a `cell_id` column and one or more numeric KPI columns.
- `telecom-italia-mi`: directory or CSVs from the Telecom Italia Big Data Challenge (MI dataset).

Example (generic CSV):

```powershell
python ai-ran-kpi-forecasting.py --dataset-type generic --data .\data\ran_kpi_sample.csv --timestamp-col timestamp --cell-id-col cell_id --kpi-col prb_dl_util --cell-id CELL_001 --horizon 24
```

Example (Telecom Italia MI directory):

```powershell
python ai-ran-kpi-forecasting.py --dataset-type telecom-italia-mi --data .\data\telecom_italia_mi --aggregate hourly --kpi-col internet_traffic --horizon 24
```

Notes / tips
- If `xgboost` is installed the script uses `XGBRegressor`; otherwise it falls back to `RandomForestRegressor` from scikit-learn.
- For MLflow logging, set `MLFLOW_TRACKING_URI` to a local folder, e.g. `file:./mlruns` before running.
- Keep test fixtures and small sample data under `tests/fixtures/` to avoid committing large datasets.

Contributing
- See `.github/copilot-instructions.md` for AI-agent guidance and repository conventions.

If you want, I can add a small `train.py` wrapper, a sample test fixture, or a GitHub Actions workflow that runs `pytest` on PRs.
📡 AI-RAN KPI Forecasting

Adaptive Forecasting Engine for RAN Traffic, Mobility & Utilization KPIs

This repository provides a modular time-series forecasting pipeline for Radio Access Network (RAN) KPIs. It supports both:

Generic RAN KPI CSV Files

Telecom Italia “Telecommunications – SMS, Call, Internet – MI” Dataset (Big Data Challenge)

The system uses lag features, calendar features, and machine learning models (XGBoost or RandomForest fallback) to produce short-term and mid-term forecasts for per-cell KPIs such as:

PRB utilization

Throughput (DL/UL)

RRC connections

Internet traffic volume

SMS / Call counters

Any numeric KPI column

🚀 Features
✔ Supports Multiple Dataset Types

Generic RAN CSV

Telecom Italia MI traffic dataset (Option 2)

✔ Automatic Feature Engineering

Time features: hour, day, week, cyclical encodings

Lag features: configurable (1, 2, 3, 6, 12 steps, etc.)

✔ Time-Aware Modeling

Temporal (strict forward-only) train/test split

Autoregressive forecasting with lag roll-forward

✔ Model Options

XGBoost (default)

RandomForest if XGBoost is not installed

✔ Forecasting Horizon

Configurable (default: 24 steps)

📁 Directory Structure
ai-ran-kpi-forecasting/
│
├── ai_ran_kpi_forecasting.py       # Main forecasting engine
├── data/
│   ├── ran_kpi_sample.csv          # Example generic dataset (user-provided)
│   ├── telecom_italia_mi/          # Telecom Italia MI CSV files
│
└── README.md

🧠 Dataset Options
Option 1 — Generic RAN KPI CSV

Your file must include:

timestamp — datetime

cell_id — cell identifier

one or more numeric KPIs

Example:

timestamp,cell_id,prb_dl_util,throughput_mbps,rrc_users
2024-01-01 00:00:00,CELL_001,45.2,12.3,122
2024-01-01 00:05:00,CELL_001,50.1,14.9,138
...


Run:

python ai_ran_kpi_forecasting.py \
    --dataset-type generic \
    --data ./data/ran_kpi_sample.csv \
    --timestamp-col timestamp \
    --cell-id-col cell_id \
    --kpi-col prb_dl_util \
    --cell-id CELL_001 \
    --horizon 24

Option 2 — Telecom Italia “SMS, Call, Internet – MI” Dataset

This is a public telecom dataset containing:

Square id (grid cell)

Time interval (UNIX ms, 10-min)

SMS-in/out activity

Call-in/out activity

Internet traffic activity

Download from:
https://dandelion.eu/datamine/open-big-data/

Organize as:

data/telecom_italia_mi/
    2013-11-01.csv
    2013-11-02.csv
    ...


Run:

python ai_ran_kpi_forecasting.py \
    --dataset-type telecom-italia-mi \
    --data ./data/telecom_italia_mi \
    --aggregate hourly \
    --kpi-col internet_traffic \
    --horizon 24

Notes:

The loader automatically normalizes column names

Converts UNIX ms → UTC timestamps

Aggregates over country codes

Supports:

10-minute granularity

1-hour resampling (--aggregate hourly)

🔧 Key Arguments
Argument	Description
--dataset-type	generic or telecom-italia-mi
--data	CSV file (generic) or directory (TI MI)
--timestamp-col	For generic datasets
--cell-id-col	For generic datasets
--kpi-col	KPI to forecast; inferred if omitted
--cell-id	Specific cell; auto-selects densest
--lags	Autoregressive lag steps
--horizon	Forecast steps
--test-size	Train/test split (time-ordered)
--aggregate	For TI MI (10min, hourly)
📊 Example Forecast Output
timestamp                forecast_step   y_hat
2024-01-10 12:00:00+00:00      1          125.33
2024-01-10 13:00:00+00:00      2          130.77
2024-01-10 14:00:00+00:00      3          128.44
...
