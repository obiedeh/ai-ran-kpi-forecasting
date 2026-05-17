# Portfolio Deliverables

This file is the reviewer-facing evidence map for the repository. It separates runnable proof from future roadmap items so the project can be evaluated quickly.

## One-Command Checks

```bash
make install-dev
make test
make run-sample
make scenario-demo
make scenario-backhaul
make scenario-outage
make portal
make publish
```

The GitHub Actions workflow runs the same core checks and verifies that report artifacts are generated.

## Core Deliverables

| Deliverable | Evidence |
|---|---|
| Generic RAN KPI forecasting pipeline | `src/ai_ran_kpi_forecasting/forecast.py` |
| Deterministic sample telemetry | `data/ran_kpi_sample.csv` |
| CLI entry point | `ai-ran-kpi-forecasting.py` and `src/ai_ran_kpi_forecasting/cli.py` |
| Unit tests | `tests/` |
| CI workflow | `.github/workflows/ci.yml` |
| Data contract | `DATA_CONTRACT.md` |
| Architecture notes | `docs/ARCHITECTURE.md` |
| Project honesty/status | `PROJECT_STATUS.md` |

## Evidence Artifacts

| Artifact | Path |
|---|---|
| Forecast metrics JSON | `reports/forecast_examples/latest/metrics.json` |
| Human-readable metrics | `reports/forecast_examples/latest/metrics_summary.md` |
| Forecast CSV | `reports/forecast_examples/latest/prb_dl_util_forecast.csv` |
| Holdout CSV | `reports/forecast_examples/latest/prb_dl_util_holdout.csv` |
| Forecast plot | `reports/forecast_examples/latest/prb_dl_util_forecast.svg` |
| Impact plot | `reports/forecast_examples/latest/prb_dl_util_impact.svg` |
| Feature importance CSV | `reports/forecast_examples/latest/feature_importance.csv` |
| Feature importance SVG | `reports/forecast_examples/latest/feature_importance.svg` |
| Evidence portal | `reports/index.html` |
| Publish page | `reports/publish/latest/index.html` |
| Publish manifest | `reports/publish/latest/manifest.json` |

## Scenario Packs

| Scenario | Evidence |
|---|---|
| Congestion | `reports/scenarios/latest/congestion/dashboard/dashboard.html` |
| Backhaul saturation | `reports/scenarios/latest/backhaul/dashboard/dashboard.html` |
| Cell outage recovery | `reports/scenarios/latest/outage/dashboard/dashboard.html` |

Each scenario includes baseline telemetry, scenario telemetry, forecast outputs, metrics, plots, and a dashboard summary.

## Current Sample Metrics

From `reports/forecast_examples/latest/metrics.json`:

| Metric | Value |
|---|---:|
| RMSE | 0.8368 |
| MAE | 0.6954 |
| MAPE | 0.8204% |

These numbers are from deterministic sample telemetry and are intended as reproducibility evidence, not claims about live operator performance.

## Credibility Boundary

This project demonstrates offline AI-RAN-style forecasting and operational evidence generation. It does not claim:

- live RAN integration
- autonomous network control
- production closed-loop optimization
- real operator feedback integration
- private network topology access

Those boundaries are intentional and keep the repository credible as a public portfolio artifact.

