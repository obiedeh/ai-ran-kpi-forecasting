# Project Status

This document separates what currently exists from what is planned. The goal is to keep the repository credible, measurable, and useful as it evolves from a forecasting script into an AI-native telecom intelligence platform.

---

## Current State

The repository currently provides a runnable forecasting workflow for RAN-style KPI time series.

Implemented capabilities:

- generic RAN KPI CSV support
- Telecom Italia MI dataset support
- time-based feature engineering
- lag feature generation
- deterministic ridge-regression forecasting path
- autoregressive forecast rollout
- focused tests for data loading, lag features, temporal splitting, forecast shape, and metrics
- reproducible sample report artifacts
- committed scenario evidence packs for congestion, backhaul saturation, and outage recovery
- static evidence portal and publish page for reviewer inspection

Current maturity level:

> Portfolio-complete offline forecasting and evidence-generation baseline

This is not a live production platform. The current value is that it establishes a reproducible offline forecasting layer, report artifacts, and scenario evidence packs that future observability, anomaly detection, and AI-RAN decision-support features can build on.

---

## Credibility Rules

This repository should not claim capabilities that are not backed by code, reports, plots, tests, or reproducible examples.

Use this standard:

| Claim Type | Required Evidence |
|---|---|
| Forecasting works | forecast output, test run, sample plot |
| Model quality improved | baseline comparison and metrics |
| Operationally useful | scenario, threshold, alert, or decision-support example |
| AI-RAN aligned | clear connection to RAN telemetry, edge workload, or private 5G use case |
| Production-ready | tests, CI, configs, reproducible runs, documentation, failure handling |

---

## Near-Term Upgrade Plan

### Phase 1: Evidence Layer

Goal: make the current forecasting workflow visible and inspectable.

Deliverables:

- add sample forecast output under `reports/`
- add forecast plot examples
- add baseline metrics table
- add one reproducible command that generates results from sample data
- document assumptions and limitations
- add scenario dashboards and a static evidence portal

Target evidence files:

```text
reports/
|-- README.md
|-- sample_metrics_report.md
`-- forecast_examples/
    `-- latest/
        |-- prb_dl_util_forecast.csv
        |-- prb_dl_util_forecast.svg
        |-- metrics.json
        `-- metrics_summary.md
```

Current evidence map:

- `PORTFOLIO_DELIVERABLES.md`
- `reports/index.html`
- `reports/publish/latest/index.html`
- `reports/scenarios/latest/congestion/dashboard/dashboard.html`
- `reports/scenarios/latest/backhaul/dashboard/dashboard.html`
- `reports/scenarios/latest/outage/dashboard/dashboard.html`

---

### Phase 2: Repository Hardening

Goal: make the project look and behave like an engineering repo, not a one-off script.

Deliverables:

- move reusable code into `src/ai_ran_kpi_forecasting/`
- keep CLI entrypoint thin
- add config-driven runs
- add unit tests for data loading, feature engineering, and forecasting
- add GitHub Actions CI
- add linting and formatting

Target structure:

```text
src/ai_ran_kpi_forecasting/
|-- data.py
|-- features.py
|-- models.py
|-- forecast.py
|-- metrics.py
|-- visualization.py
`-- cli.py
```

---

### Phase 3: Network Intelligence Layer

Goal: move beyond raw prediction into operational telecom insight.

Deliverables:

- congestion-risk scoring
- threshold-based alerting
- anomaly detection baseline
- multi-KPI correlation examples
- cell-level ranking by risk
- operational recommendation templates

Example output:

```text
CELL_014: elevated congestion risk over next 6 hours
Likely driver: rising PRB utilization + traffic volume trend
Recommended action: inspect capacity, workload placement, or local hotspot behavior
```

---

### Phase 4: AI-RAN and Edge Workload Extension

Goal: connect network forecasting to edge AI deployment decisions.

Deliverables:

- simulated edge inference workload traces
- latency and utilization forecasting
- workload placement recommendation logic
- private 5G robotics use case scenario
- AI-RAN operations architecture diagram

Example use case:

> Forecast whether a private 5G edge site has enough headroom to support a robotics/VLM inference workload without pushing latency or utilization beyond safe operating thresholds.

---

## Roadmap Checklist

### Foundation

- [x] Generic KPI CSV workflow
- [x] Telecom Italia dataset option
- [x] Lag/time feature engineering
- [x] Deterministic ridge-regression model path
- [x] Basic autoregressive forecasting
- [x] Reproducible sample forecast artifact
- [x] Metrics report
- [x] Forecast visualization

### Engineering Quality

- [x] Package structure under `src/`
- [x] Sample YAML configuration
- [x] Unit tests for core modules
- [x] CI workflow
- [x] Makefile or task runner
- [x] Clear data contract

### Telecom Intelligence

- [ ] Congestion risk score
- [ ] Anomaly detection baseline
- [ ] Multi-KPI scenario
- [ ] Cell/site risk ranking
- [ ] Operational recommendation output

### AI-RAN Extension

- [ ] Edge workload trace simulation
- [ ] Inference-aware capacity planning example
- [ ] Private 5G robotics scenario
- [ ] AI-RAN architecture diagram
- [ ] Dashboard or report export

---

## Current Limitations

Known limitations:

- full observability dashboard is out of scope
- no real operator feedback loop yet
- no production deployment packaging yet
- no live RAN data source integration yet
- AI-RAN workload placement logic is planned, not implemented

These limitations are intentional and tracked so the repo stays honest.

---

## Definition of Production-Grade for This Repo

This repository should only be described as production-grade when it has:

- reproducible runs
- CI passing
- documented data contracts
- tests for core pipeline functions
- model metrics and baseline comparison
- explainability or feature importance outputs
- operational scenario examples
- clear failure modes and limitations
- deployable CLI or package interface

Until then, describe it as:

> An evolving engineering project for AI-native telecom forecasting and operational intelligence.
