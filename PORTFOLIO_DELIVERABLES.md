# Portfolio Deliverables — AI-RAN KPI Forecasting (Non-RT RIC rApp pattern)

Reviewer-facing evidence map. This repo implements the **AI-for-RAN at the operational layer** rApp pattern — schema-typed KPM input, multi-model forecasting, A1-policy output, scenario evidence packs. Companion to the L1 PHY repo [`wireless-link-intelligence-system`](https://github.com/obiedeh/wireless-link-intelligence-system). Pattern, not deployment — see [docs/AI_RAN_INTEGRATION.md](docs/AI_RAN_INTEGRATION.md) for the boundary.

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

This repo demonstrates the **Non-RT RIC rApp pattern** for AI-for-RAN KPI forecasting on synthetic + small-public telemetry. It does not claim:

- live RAN integration
- live Non-RT RIC integration (FlexRIC / OSC RIC / Nokia MantaRay / Ericsson IAP / Mavenir)
- E2 / A1 / O1 / R1 protocol implementations on the wire (the contracts are documented; the wire protocol is not exercised)
- autonomous network control or closed-loop policy enforcement
- production-grade rApp lifecycle (Helm packaging, R1 service registration over real ORAN endpoints)

The deliverable is the **pattern + schemas + measured forecasting evidence** — enough that a senior AI-RAN engineer reading the repo can confirm this person understands how AI-for-RAN deploys, not enough to drop into a live RIC. See [docs/AI_RAN_INTEGRATION.md](docs/AI_RAN_INTEGRATION.md) for the integration recipe with FlexRIC / OSC RIC.

