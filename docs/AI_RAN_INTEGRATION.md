# AI-for-RAN Integration Boundary

This repo implements the Non-RT RIC KPI-forecasting rApp pattern. It is pattern, not deployment.

The useful part is the shape of the work:

```text
KPM-style telemetry
-> temporal forecasting
-> scenario evidence
-> advisory A1 policy candidate
-> operator review
```

The repo does not connect to a live RAN, register with a live Non-RT RIC, execute A1 transport, or enforce policy. It produces the contracts and artifacts that a real integration would need to inspect first.

## What Is Exercised

| Component | Status | Evidence |
|---|---|---|
| KPM-shaped CSV input parsing | exercised | `data.py`, `schemas/kpm_input_v1.json` |
| Temporal forecasting | exercised and measured | `reports/forecast_examples/latest/` |
| Three-model comparison | exercised and measured | `reports/model_comparison/` |
| Forecast to advisory A1 candidate | exercised and schema-validated | `a1_policy.py`, `schemas/a1_policy_v1.json` |
| R1-style dataflow simulation | exercised on sample data | `scripts/simulate_r1_dataflow.py` |
| rApp manifest shape | documented | `rapp_manifest.yaml` |
| Scenario evidence packs | generated | `reports/scenarios/latest/` |

## What Is Not Exercised

| Boundary | What remains |
|---|---|
| Live RAN telemetry | A real source of decoded KPM measurements |
| Live Non-RT RIC registration | Target-specific packaging and service registration |
| A1-P transport | Wire-protocol producer integration |
| Closed-loop control | Downstream Near-RT RIC/xApp enforcement |
| Operator validation | Real operational review and calibration |

## What a Real Integration Would Still Need

A real integration would need target-specific packaging, authentication, telemetry subscription wiring, A1 transport, runtime observability, drift checks, and policy emission monitoring.

Those pieces are not included because they depend on the selected Non-RT RIC environment. The boundary is intentional: this repo shows how the forecasting and policy-output logic should be shaped before wiring it to a live control plane.

## References

- AI-RAN Alliance: https://ai-ran.org/
- O-RAN Alliance: https://www.o-ran.org/
- O-RAN WG2 Non-RT-RIC Architecture
- O-RAN E2SM-KPM Service Model
- O-RAN A1-P interface specification
