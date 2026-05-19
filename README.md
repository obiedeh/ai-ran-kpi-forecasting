# AI-RAN KPI Forecasting

Edge network intelligence — inference-aware workload placement and KPI forecasting at the network edge.

This repository is the forecasting substrate underneath that frame: deterministic KPI forecasting on RAN-style telemetry today, with a data contract and roadmap shaped around edge-AI workload placement decisions. See [Implemented vs planned](#implemented-vs-planned) for the honest boundary between what works today and what is on the roadmap.

## Core Stack

**Implemented:** Python · Pandas · deterministic ridge regression · time-series forecasting · RAN KPI analysis · reproducible reports

**Planned / integration path:** operational forecast reports · telemetry validation · edge/AI-RAN deployment context

<p>
  <img src="https://img.shields.io/badge/Python-3.x-blue" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-dataframes-150458" alt="Pandas" />
  <img src="https://img.shields.io/badge/Ridge%20Regression-deterministic-F7931E" alt="Deterministic ridge regression" />
  <img src="https://img.shields.io/badge/Time%20Series-forecasting-4B8BBE" alt="Time Series Forecasting" />
  <img src="https://img.shields.io/badge/AI--RAN-telemetry-76B900" alt="AI-RAN telemetry" />
  <img src="https://img.shields.io/badge/Reports-reproducible-555555" alt="Reproducible reports" />
</p>

## What this repo does

- Loads RAN-style telemetry (PRB utilization, throughput, active users, latency, and — per the data contract — edge GPU and memory utilization) and builds temporal + lag features for forward-looking KPI forecasting.
- Trains a deterministic ridge-regression baseline and produces a multi-step autoregressive forecast, writing metrics, plots, and machine-readable evidence per run.
- Generates three scenario evidence packs (congestion, backhaul saturation, outage recovery) so the forecasting layer can be exercised against the operational patterns the downstream inference-aware placement logic will consume.

## Forecasting approach

- **Model**: ridge regression (`ridge_linear`) on lag + time features — deterministic, fast, reviewable, and used as the credibility baseline rather than the ceiling.
- **Rollout**: multi-step autoregressive forecast over a configurable horizon (default 24 steps).
- **Features**: time-of-day / day-of-week temporal features plus configurable lag features over the target KPI.
- **Split**: forward-only train/test (no shuffle), so metrics reflect true forward forecasting and not temporal leakage.

## Measured metrics

Source: [`reports/forecast_examples/latest/metrics.json`](reports/forecast_examples/latest/metrics.json). Model: `ridge_linear`. KPI: `prb_dl_util` on `CELL_001`. Sample dataset: 49 rows × 6 columns (1 cell, hourly, `data/ran_kpi_sample.csv`).

| Metric | Value | Status |
| --- | ---: | --- |
| RMSE | 0.8368 | measured |
| MAE | 0.6954 | measured |
| MAPE | 0.82% | measured |
| Forecasting accuracy on Telecom Italia MI | `<TO MEASURE>` | Plan: place the Telecom Italia MI dataset under `data/telecom_italia_mi/` (loader already supports it), run `make run-telecom`, capture RMSE/MAE/MAPE into `reports/forecast_examples/telecom_italia/metrics.json`. |
| Forecasting accuracy on `edge_gpu_util_pct` (RMSE / MAE / MAPE) | 3.62 / 2.86 / 6.38% | measured ([reports/forecast_examples/edge_ai/gpu_util/metrics.json](reports/forecast_examples/edge_ai/gpu_util/metrics.json)) — `ridge_linear`, CELL_001, 24-step horizon on synthetic 504-row, 3-cell telemetry; reproduce via `make forecast-edge-ai` |
| Forecasting accuracy on `edge_memory_util_pct` (RMSE / MAE / MAPE) | 3.21 / 2.65 / 4.44% | measured ([reports/forecast_examples/edge_ai/memory_util/metrics.json](reports/forecast_examples/edge_ai/memory_util/metrics.json)) — same setup as the GPU row |
| End-to-end forecast wall-clock (s) | `<TO MEASURE>` | Plan: time `make run-sample` end-to-end on the dev machine and write to `reports/forecast_examples/latest/timing.json`. |

The measured numbers are from deterministic sample telemetry — they prove the pipeline is reproducible end-to-end, not that the model has been validated against live operator data. See [Credibility Boundary](#credibility-boundary).

Reproduce locally:

```bash
make install-dev
make run-sample
cat reports/forecast_examples/latest/metrics.json
```

## Implemented vs planned

**Implemented and demonstrated today:**

- Generic RAN KPI CSV loading and per-cell, per-KPI forecasting
- Telecom Italia MI dataset loader (supported, not yet benchmarked — see the `<TO MEASURE>` row above)
- Deterministic ridge-regression forecasting with autoregressive rollout
- Scenario evidence packs (congestion, backhaul saturation, outage recovery)
- Static evidence portal and publish page for reviewer inspection
- Reproducible report artifacts: CSV, SVG, Markdown, JSON

**Planned (per [PROJECT_STATUS.md](PROJECT_STATUS.md)):**

- Congestion-risk scoring, anomaly detection, and multi-KPI correlation — Phase 3
- Edge inference workload trace simulation — Phase 4
- Inference-aware workload placement recommendation logic — Phase 4
- Private 5G robotics use-case scenario — Phase 4

The repository tracks its own credibility in `PROJECT_STATUS.md` and does not claim Phase 3 or Phase 4 capabilities until they have working code paths and evidence artifacts.

## Credibility Boundary

This project demonstrates offline AI-RAN-style KPI forecasting and operational evidence generation. It does not claim:

- live RAN integration
- autonomous network control
- production closed-loop optimization
- real operator feedback integration
- private network topology access

Those boundaries are intentional and keep the repository credible as a public portfolio artifact.

## Prerequisites

- **Python 3.11 or newer** (see `pyproject.toml`)
- **git** to clone the repository
- **make** (Linux / macOS) — Windows users: see [Without make](#without-make-windows-or-direct-python) below
- The Makefile uses `PYTHON ?= python` and does **not** auto-create a virtual environment — activate one yourself before running `make install-dev`

## Quickstart

**1. Clone the repo**

```bash
git clone https://github.com/obiedeh/ai-ran-kpi-forecasting.git
cd ai-ran-kpi-forecasting
```

**2. Create a virtual environment and install**

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
```

`make install-dev` installs both `requirements.txt` (runtime) and `requirements-dev.txt` (dev, lint, test).

**3. Run the sample forecast**

```bash
make run-sample
```

Writes the sample forecast bundle to `reports/forecast_examples/latest/` (see [Understanding the outputs](#understanding-the-outputs)).

**4. Generate scenario evidence packs and the static portal**

```bash
make scenario-demo
make scenario-backhaul
make scenario-outage
make portal
make publish
```

Open `reports/index.html` or `reports/publish/latest/index.html` for the reviewer-facing evidence portal.

### Without make (Windows or direct Python)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python ai-ran-kpi-forecasting.py run-sample --output-dir reports/forecast_examples/latest
pytest -q
```

On Linux / macOS, replace the activate line with `source .venv/bin/activate`.

After `pip install -e .`, you can also invoke the CLI as `ai-ran-kpi-forecasting run-sample` thanks to the entry point declared in `pyproject.toml`.

## Understanding the outputs

Every run of `make run-sample` writes a complete forecast bundle to `reports/forecast_examples/latest/`. Every quantitative claim in this README is sourced directly from `metrics.json`.

| File | What it tells you |
| --- | --- |
| `metrics.json` | Machine-readable RMSE / MAE / MAPE — the source of the [Measured metrics](#measured-metrics) table above. |
| `metrics_summary.md` | Human-readable run summary (model, cell, KPI, metrics). |
| `prb_dl_util_forecast.csv` | Forecasted KPI values over the configured horizon. |
| `prb_dl_util_holdout.csv` | Holdout-set actuals used to compute the metrics. |
| `prb_dl_util_forecast.svg` | Visual forecast vs. holdout overlay. |
| `prb_dl_util_impact.svg` | Visual impact / residual chart. |
| `feature_importance.csv` | Per-feature ridge coefficients ranked by absolute weight. |
| `feature_importance.svg` | Visual feature importance bar chart. |

Scenario packs add their own bundles under `reports/scenarios/latest/{congestion,backhaul,outage}/`, each with baseline telemetry, scenario telemetry, forecast outputs, metrics, plot, and a dashboard HTML.

## CLI Usage

Run the reproducible sample:

```bash
python ai-ran-kpi-forecasting.py run-sample
```

Run a generic CSV forecast:

```bash
python ai-ran-kpi-forecasting.py forecast \
  --dataset-type generic \
  --data ./data/ran_kpi_sample.csv \
  --timestamp-col timestamp \
  --cell-id-col cell_id \
  --kpi-col prb_dl_util \
  --cell-id CELL_001 \
  --horizon 24 \
  --output-dir reports/forecast_examples/latest
```

Generate synthetic telemetry:

```bash
python ai-ran-kpi-forecasting.py generate-synthetic --output data/synthetic_ran_kpi.csv
```

Generate scenario dashboards and the evidence portal:

```bash
make scenario-demo
make scenario-backhaul
make scenario-outage
make portal
make publish
```

## Data Contract

Generic CSV input requires `timestamp`, `cell_id`, and one or more numeric KPI columns — for example `prb_dl_util`, `throughput_dl_mbps`, `rrc_users`, `latency_ms`, `edge_gpu_util_pct`, or `edge_memory_util_pct`.

See [DATA_CONTRACT.md](DATA_CONTRACT.md) for the full schema, column overrides, recommended KPI fields, and the Telecom Italia MI normalization map.

## Make Targets

| Target | What it does |
| --- | --- |
| `make install` | Install runtime requirements |
| `make install-dev` | Install dev requirements (lint, test, etc.) |
| `make test` | Run pytest |
| `make lint` | Run ruff |
| `make run-sample` | Run the reproducible sample forecast |
| `make run-generic` | Run a generic CSV forecast against `data/ran_kpi_sample.csv` |
| `make run-telecom` | Run a forecast against the Telecom Italia MI dataset |
| `make synthetic` | Regenerate synthetic telemetry under `data/` |
| `make scenario-demo` | Generate the congestion scenario evidence pack |
| `make scenario-backhaul` | Generate the backhaul saturation scenario pack |
| `make scenario-outage` | Generate the cell-outage scenario pack |
| `make portal` | Build the static evidence portal at `reports/index.html` |
| `make publish` | Build the publish page at `reports/publish/latest/index.html` |

## Repository Structure

```text
.
|-- ai-ran-kpi-forecasting.py          # thin compatibility wrapper
|-- configs/sample_config.yaml
|-- data/ran_kpi_sample.csv
|-- docs/ARCHITECTURE.md
|-- reports/
|   |-- README.md
|   |-- sample_metrics_report.md
|   |-- forecast_examples/
|   |-- scenarios/
|   |-- publish/
|   `-- index.html
|-- src/ai_ran_kpi_forecasting/
|   |-- data.py
|   |-- features.py
|   |-- models.py
|   |-- forecast.py
|   |-- metrics.py
|   |-- visualization.py
|   `-- cli.py
|-- tests/
|-- DATA_CONTRACT.md
|-- PORTFOLIO_DELIVERABLES.md
|-- PROJECT_STATUS.md
|-- Makefile
`-- pyproject.toml
```

`src/ai_ran_kpi_forecasting/forecasting.py` remains as a compatibility import for older callers.

## Deliverables

See [PORTFOLIO_DELIVERABLES.md](PORTFOLIO_DELIVERABLES.md) for the reviewer-facing evidence map.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the honest implemented/planned breakdown and credibility rules.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline diagram and module responsibilities.
