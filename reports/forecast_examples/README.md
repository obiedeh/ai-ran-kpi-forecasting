# Forecast Evidence Reports

This directory stores reproducible forecasting evidence generated from the AI-RAN KPI forecasting workflow.

The goal is to make model behavior visible and inspectable instead of relying on abstract claims.

Planned report artifacts include:

- forecast output CSVs
- KPI trend plots
- hold-out evaluation metrics
- SVG forecast and feature-importance visualizations
- optional SHAP summary plots when dependencies are installed
- anomaly visibility examples
- congestion-risk summaries
- AI-RAN operational scenario outputs
- baseline vs congestion scenario dashboards

---

## Recommended Evidence Files

```text
reports/forecast_examples/
├── latest/
│   ├── metrics_summary.md
│   ├── metrics.json
│   ├── *_forecast.csv
│   ├── *_forecast.svg
│   ├── *_impact.svg
│   ├── feature_importance.csv
│   ├── feature_importance.svg
│   └── shap_summary.svg
├── sample_benchmark.md
├── prb_dl_util_forecast.csv
├── prb_dl_util_forecast.svg
├── metrics_summary.md
├── feature_importance.svg
├── congestion_risk_report.md
└── anomaly_detection_example.md
```

---

## Current Direction

The current pipeline focuses on:

- temporal forecasting
- lag-aware feature engineering
- short-horizon KPI prediction
- operational visibility for AI-native telecom systems

Future extensions will expand toward:

- AI-RAN operational intelligence
- edge workload forecasting
- private 5G robotics readiness
- inference-aware infrastructure planning
- distributed edge observability

The repo keeps the reporting surface small and reproducible so these artifacts can be generated in CI or on a laptop without a notebook workflow.

For broader operational evidence, see `reports/scenarios/latest/` after running the congestion scenario demo.
The top-level entry point is `reports/index.html`.
The release-friendly landing page is `reports/publish/latest/index.html`.
