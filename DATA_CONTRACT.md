# Data Contract

This document defines the expected data inputs for the AI-RAN KPI forecasting pipeline.

The goal is to keep the project reproducible and prevent silent data assumptions from becoming hidden model failures.

---

## Supported Dataset Modes

The current pipeline supports two modes:

1. Generic RAN KPI CSV
2. Telecom Italia MI dataset

It also includes a synthetic telemetry generator for reproducible demos, benchmarks, and CI checks.

---

## 1. Generic RAN KPI CSV

### Required Columns

| Column | Type | Required | Description |
|---|---:|---:|---|
| `timestamp` | datetime/string | yes | Observation timestamp. Parsed as UTC. |
| `cell_id` | string | yes | Logical cell, sector, site, or grid identifier. |
| KPI column | numeric | yes | Target metric to forecast, for example `prb_dl_util`. |

Column names can be overridden with CLI flags:

```bash
--timestamp-col <column_name>
--cell-id-col <column_name>
--kpi-col <column_name>
```

---

### Recommended KPI Columns

Recommended fields for AI-RAN and private 5G scenarios:

| KPI | Description | Example Use |
|---|---|---|
| `prb_dl_util` | Downlink physical resource block utilization | Congestion forecasting |
| `prb_ul_util` | Uplink physical resource block utilization | Uplink pressure detection |
| `throughput_dl_mbps` | Downlink throughput | User experience trend |
| `throughput_ul_mbps` | Uplink throughput | Upload workload trend |
| `rrc_users` | Active connected users | Mobility and load pattern |
| `latency_ms` | Access or application latency | Edge workload readiness |
| `packet_loss_pct` | Packet loss percentage | Service degradation risk |
| `edge_gpu_util_pct` | Edge accelerator utilization | Inference-aware placement |
| `edge_memory_util_pct` | Edge memory utilization | Runtime stability risk |

---

### Minimal Example

```csv
timestamp,cell_id,prb_dl_util,throughput_mbps,rrc_users
2024-01-01 00:00:00,CELL_001,45.2,12.3,122
2024-01-01 00:05:00,CELL_001,50.1,14.9,138
2024-01-01 00:10:00,CELL_001,54.8,15.7,141
```

---

## 2. Telecom Italia MI Dataset

The Telecom Italia MI dataset is supported as a public proxy for city-scale telecom activity patterns.

Expected raw fields include:

| Raw Field | Normalized Field |
|---|---|
| `Square id` | `cell_id` |
| `Time interval` | `time_interval` |
| `SMS-in activity` | `sms_in` |
| `SMS-out activity` | `sms_out` |
| `Call-in activity` | `call_in` |
| `Call-out activity` | `call_out` |
| `Internet traffic activity` | `internet_traffic` |

The loader converts UNIX millisecond timestamps into UTC timestamps and can aggregate to hourly buckets.

---

## Data Quality Expectations

Before modeling, input data should satisfy:

- timestamps are parseable
- each selected cell has enough sequential records for lag features
- target KPI is numeric
- missing timestamps are understood or documented
- sampling frequency is reasonably consistent
- extreme outliers are reviewed before training

---

## Known Limitations

Current limitations:

- one target KPI is forecast at a time
- multi-cell graph relationships are not modeled yet
- exogenous features are limited
- missing interval imputation is basic
- live streaming telemetry is not implemented yet

Future work should add stronger validation, data profiling, and schema enforcement.

---

## Synthetic Telemetry

The `generate-synthetic` CLI command produces a small, deterministic telemetry table with:

- `timestamp`
- `cell_id`
- `prb_dl_util`
- `throughput_mbps`
- `rrc_users`
- `latency_ms`

This output is meant for smoke tests, report generation, and local development, not for claims about real telecom performance.
