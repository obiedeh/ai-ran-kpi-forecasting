# Sample Benchmark Report

This repo uses small, reproducible benchmark artifacts to make AI-RAN forecast behavior visible.

## Run Context

- Dataset: `data/ran_kpi_sample.csv`
- Cell: `CELL_001`
- Target KPI: `prb_dl_util`
- Horizon: `24`
- Lags: `1,2,3,6,12`

## Expected Outputs

- `reports/forecast_examples/latest/prb_dl_util_forecast.csv`
- `reports/forecast_examples/latest/prb_dl_util_holdout.csv`
- `reports/forecast_examples/latest/prb_dl_util_forecast.svg`
- `reports/forecast_examples/latest/feature_importance.csv`
- `reports/forecast_examples/latest/feature_importance.svg`
- `reports/forecast_examples/latest/metrics.json`
- `reports/forecast_examples/latest/metrics_summary.md`

## Benchmark Notes

The benchmark is intentionally small and deterministic enough to run in CI.
It is designed for operational visibility, not leaderboard-style model chasing.
