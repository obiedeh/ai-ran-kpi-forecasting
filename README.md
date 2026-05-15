# AI-RAN KPI Forecasting

AI-native forecasting and operational intelligence for telecom RAN and edge infrastructure.

This repository explores how machine learning can support proactive operations across AI-RAN, private 5G, and edge AI environments by forecasting network KPIs, identifying emerging congestion patterns, and improving infrastructure visibility before failures occur.

The focus is not just forecasting a metric.

The focus is building a foundation for:

- AI-native RAN observability
- edge workload intelligence
- congestion and utilization forecasting
- inference-aware network operations
- proactive telecom operations support
- future AI-RAN and 6G operational workflows

---

## Why This Matters

As AI workloads move closer to the edge, telecom infrastructure is becoming tightly coupled with inference systems, robotics platforms, distributed compute, and latency-sensitive applications.

Modern AI-RAN environments must handle:

- fluctuating inference demand
- variable user mobility
- latency-sensitive edge workloads
- resource contention
- energy and thermal constraints
- unpredictable traffic patterns

Forecasting network behavior ahead of time enables:

- smarter workload placement
- proactive scaling decisions
- congestion mitigation
- improved edge resource utilization
- more resilient AI-native telecom operations

This repository is an engineering exploration of those operational intelligence workflows.

---

## Core Capabilities

### Forecasting and Network Intelligence

- per-cell KPI forecasting
- temporal trend analysis
- autoregressive forecasting workflows
- short-horizon and mid-horizon prediction
- anomaly-oriented KPI visibility

### Supported KPI Types

- PRB utilization
- throughput (DL/UL)
- RRC connections
- internet traffic volume
- SMS and call activity
- custom numeric network KPIs

### Feature Engineering

- lag-based temporal features
- cyclical time encodings
- hour/day/week seasonality
- time-aware train/test separation

### Modeling

- XGBoost forecasting
- RandomForest fallback pipeline
- forward-only temporal validation
- autoregressive roll-forward prediction

---

## Architecture

```text
RAN / Edge Telemetry
        |
        v
Telemetry Ingestion
        |
        v
Feature Engineering
(time, lag, cyclical patterns)
        |
        v
Forecasting Engine
(XGBoost / RandomForest)
        |
        v
Operational Intelligence Layer
- congestion forecasting
- utilization visibility
- anomaly awareness
- workload planning
        |
        v
AI-RAN / Edge Operations
```

---

## Repository Structure

```text
.
├── ai-ran-kpi-forecasting.py
├── src/ai_ran_kpi_forecasting/
│   ├── cli.py
│   ├── data.py
│   ├── features.py
│   ├── forecasting.py
│   ├── metrics.py
│   ├── reports.py
│   ├── visualization.py
│   └── explainability.py
├── data/
├── reports/forecast_examples/
├── reports/scenarios/
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Supported Dataset Types

### 1. Generic RAN KPI CSV

Expected fields:

- timestamp
- cell_id
- one or more numeric KPI columns

Example:

```csv
timestamp,cell_id,prb_dl_util,throughput_mbps,rrc_users
2024-01-01 00:00:00,CELL_001,45.2,12.3,122
2024-01-01 00:05:00,CELL_001,50.1,14.9,138
```

---

### 2. Telecom Italia Big Data Challenge Dataset

Supports the Telecom Italia:

> Telecommunications - SMS, Call, Internet - MI Dataset

Capabilities include:

- timestamp normalization
- traffic aggregation
- KPI extraction
- temporal forecasting workflows

Dataset source:

https://dandelion.eu/datamine/open-big-data/

---

## Quick Start

### Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Optional development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

---

### Run Smoke Tests

```bash
pytest -q
```

### Generate Synthetic Telemetry

```bash
python ai-ran-kpi-forecasting.py generate-synthetic --output data/synthetic_ran_kpi.csv
```

### Generate Congestion Scenario Dashboard

```bash
python ai-ran-kpi-forecasting.py scenario-demo --output-dir reports/scenarios/latest/congestion
```

### Generate Backhaul Scenario Dashboard

```bash
python ai-ran-kpi-forecasting.py scenario-demo --scenario-type backhaul --output-dir reports/scenarios/latest/backhaul
```

### Generate Outage Scenario Dashboard

```bash
python ai-ran-kpi-forecasting.py scenario-demo --scenario-type outage --output-dir reports/scenarios/latest/outage
```

### Generate Evidence Portal

```bash
python ai-ran-kpi-forecasting.py portal --output reports/index.html
```

### Generate Publish Page

```bash
python ai-ran-kpi-forecasting.py publish --output-dir reports/publish/latest
```

The scenario dashboard includes telecom-style KPI cards, pre-shock versus shock-window telemetry, and side-by-side baseline/congestion report panels.

---

## Example Usage

### Generic RAN KPI Dataset

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

---

### Telecom Italia Dataset

```bash
python ai-ran-kpi-forecasting.py forecast \
  --dataset-type telecom-italia-mi \
  --data ./data/telecom_italia_mi \
  --aggregate hourly \
  --kpi-col internet_traffic \
  --horizon 24 \
  --output-dir reports/forecast_examples/latest
```

---

## Engineering Direction

This repository is evolving toward a broader AI-native telecom intelligence platform focused on:

- AI-RAN operations
- edge AI infrastructure awareness
- network telemetry intelligence
- anomaly detection
- workload-aware forecasting
- operational decision support
- distributed edge systems

Planned future directions include:

- SHAP-based explainability
- multi-KPI forecasting
- anomaly detection pipelines
- Grafana-style observability outputs
- workload-placement experimentation
- AI-RAN operational dashboards
- GPU-accelerated training paths

The current implementation keeps those ideas lean: optional SHAP output, reproducible benchmark bundles, and a small command-line workflow rather than a notebook-first workflow.

The scenario demo adds a telecom-style baseline versus congestion dashboard with KPI cards, forecast evidence, and pre/post impact plots.

---

## Positioning

This project is part of a broader engineering focus around:

- Physical AI
- Edge AI
- Agentic AI
- LLM/RAG systems
- AI-native infrastructure
- AI-RAN and private 5G

The goal is to build practical, deployable AI systems that operate across telecom networks, edge infrastructure, and real-world operational environments.
