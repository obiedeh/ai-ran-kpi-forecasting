# Project Status

This document separates what currently exists from what is planned. The goal is to keep the repository credible, measurable, and useful as a Non-RT RIC rApp pattern for AI-for-RAN KPI forecasting and advisory A1 policy generation.

---

## Current State

The repository currently provides a runnable forecasting workflow for RAN-style KPI time series plus generated evidence artifacts for operator review.

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
- static evidence portal, top-level dashboard, and publish page for reviewer inspection
- schema-typed KPM-style input contract
- schema-typed advisory A1 policy candidate output
- rApp manifest documenting identity, inputs, outputs, and boundaries
- R1-style dataflow demo from KPM-style input to forecast to A1 candidate

Current maturity level:

> Portfolio-complete Non-RT RIC rApp pattern for offline KPI forecasting, advisory policy candidates, and evidence generation

This is not a live RIC deployment. The current value is that it establishes a reproducible offline forecasting layer, typed contracts, advisory policy output, report artifacts, and scenario evidence packs that future observability, benchmark, and integration work can build on.

Telecom Italia MI benchmark status:

- Loader path: `ai_ran_kpi_forecasting.data.load_telecom_italia_mi`
- Make target: `make run-telecom`
- Output target: `reports/forecast_examples/telecom_italia_mi/`
- Published result: Benchmark-ready: pending local public dataset files. No benchmark metric claimed yet.

---

## Credibility Rules

This repository should not claim capabilities that are not backed by code, reports, plots, tests, or reproducible examples. The current deliverable is a tested evidence pattern, not a live network system.

Use this standard:

| Claim Type | Required Evidence |
|---|---|
| Forecasting works | forecast output, test run, sample plot |
| Model quality improved | baseline comparison and metrics |
| Operationally useful | scenario, threshold, advisory policy, or decision-support example |
| AI-RAN aligned | clear connection to RAN telemetry, typed rApp contracts, edge workload, or private 5G use case |
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

- [x] Congestion risk tiering in generated portal/dashboard
- [ ] Anomaly detection baseline
- [x] Multi-scenario evidence packs
- [ ] Cell/site risk ranking
- [x] Advisory A1 policy candidate output

### AI-RAN Extension

- [x] Edge workload trace simulation
- [ ] Inference-aware capacity planning example
- [ ] Private 5G robotics scenario
- [x] AI-RAN architecture diagram
- [x] Dashboard or report export

---

## Current Limitations

Known limitations:

- live observability dashboard is out of scope
- no real operator feedback loop yet
- no production deployment packaging yet
- no live RAN data source integration yet
- no live Non-RT RIC deployment
- Telecom Italia MI is benchmark-ready, pending local public dataset files; no benchmark metric is claimed yet

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
