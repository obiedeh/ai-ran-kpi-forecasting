# Architecture

## System Purpose

This repository is the supporting AI-RAN and edge-infrastructure telemetry workflow in the portfolio. It keeps the focus practical: KPI CSV ingestion, temporal forecasting, scenario evidence, advisory policy candidates, and reproducible operational reports.

## Current Implementation Status

- **Implemented:** CSV telemetry loading, feature generation, deterministic ridge regression, autoregressive forecasting, metrics, SVG/Markdown/JSON report artifacts, scenario evidence packs, and tests.
- **Runnable scaffold:** `make run-sample`, `make run-scenarios`, and CLI-driven report generation.
- **Planned integration path:** operational forecast reports, telemetry validation, edge workload context, and future Non-RT RIC integration work when a target environment exists.
- **Future validation:** larger public/operator-style datasets and timed end-to-end run evidence.

## Main Components

- `src/ai_ran_kpi_forecasting/data.py`: CSV loading, sample generation, and dataset normalization.
- `features.py`: time and lag feature generation.
- `models.py`: deterministic ridge baseline implemented in-repo.
- `forecast.py`: temporal split, training, rollout, and orchestration.
- `metrics.py`: RMSE, MAE, and MAPE calculation.
- `reports.py` and `visualization.py`: report, plot, and machine-readable artifact generation.
- `reports/`: current evidence outputs and scenario packs.

## Runtime Flow

The current path loads a KPI CSV, builds temporal and lag features, trains the deterministic ridge baseline, rolls forward a forecast horizon, evaluates hold-out metrics, and writes reproducible artifacts.

## Data / Telemetry Flow

RAN-style KPI rows become normalized time-series records. Feature generation adds temporal and lag columns. Forecasting produces predictions, metrics, plots, feature weights, and summaries under `reports/`.

## Deployment Modes

- **Local development:** Python package, tests, Makefile commands, and sample report generation.
- **Offline operational analysis:** CSV-driven KPI forecasting and scenario evidence packs.
- **Planned edge/AI infrastructure context:** telemetry validation and forecast reports that can support future edge workload placement decisions.
- **Not claimed:** live RAN control, autonomous optimization, production closed-loop deployment, or operator network integration.

## Evidence Artifacts

- Current evidence lives in `reports/forecast_examples/` and `reports/scenarios/`.
- Reviewer placeholders live in `artifacts/sample-inputs/`, `artifacts/sample-outputs/`, `artifacts/logs/`, and `artifacts/reports/`.
- Diagram sources live in `docs/diagrams/`.

## Known Limitations

- Measured metrics are from deterministic sample telemetry unless otherwise stated.
- Telecom Italia MI is benchmark-ready, pending local public dataset files. No benchmark metric is claimed yet.
- This repo supports the edge AI infrastructure story; it is not the flagship Physical AI platform.

## Next Validation Step

Add one fresh operational forecast report with timing, assumptions, dataset notes, limitations, and a clear edge-infrastructure handoff.
