# Forecast Evidence Reports

This directory stores reproducible forecasting evidence generated from the AI-RAN KPI forecasting workflow.

The goal is to make model behavior visible and inspectable instead of relying on abstract claims.

Planned report artifacts include:

- forecast output CSVs
- KPI trend plots
- hold-out evaluation metrics
- SHAP feature importance visualizations
- anomaly visibility examples
- congestion-risk summaries
- AI-RAN operational scenario outputs

---

## Recommended Evidence Files

```text
reports/forecast_examples/
├── prb_dl_util_forecast.csv
├── prb_dl_util_forecast.png
├── metrics_summary.md
├── feature_importance.png
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
