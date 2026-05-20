# R1 → forecast → A1 dataflow demo

This bundle is the end-to-end output of the rApp's data plane on
**synthetic / sample** telemetry. The wire protocol (R1 service
registration, A1-P transport) is **not** exercised.

| Stage | Artifact |
|---|---|
| R1 input | `data\ran_kpi_sample.csv` (matches `schemas/kpm_input_v1.json`) |
| Forecast | [`forecast.csv`](forecast.csv) — 24 steps |
| Metrics | [`metrics.json`](metrics.json) — RMSE 0.8368 |
| A1 output | [`a1_policy_candidate.json`](a1_policy_candidate.json) (matches `schemas/a1_policy_v1.json`) |

Action recommended: **no_action**

Reproduce: `python scripts/simulate_r1_dataflow.py --output-dir reports\r1_dataflow_demo`
