# Tech brief — AI-RAN KPI Forecasting (Non-RT RIC rApp pattern)

A one-page brief for a senior tech-leader or hiring-manager review. Read this if you want to see whether the repo is more than a KPI forecasting script: schema-typed input, time-ordered evaluation, advisory A1 policy output, evidence-led reporting, and a clear boundary before live RIC integration.

Every claim below links to a committed artifact.

---

## What this repo demonstrates

The operating question is straightforward: can a Non-RT intelligence layer see KPI degradation early enough to recommend a policy review before the cell is already in trouble?

This repo targets **AI-for-RAN at the operational layer**: the Non-RT RIC rApp pattern for KPI forecasting, scenario evidence, and advisory policy candidates.

**This is the rApp pattern, not a deployed rApp.** The manifest, schemas, ML pipeline, and A1-policy output are real and tested; the wire-protocol integration with a live Non-RT RIC is intentionally not exercised. The point is not that Ridge regression is novel. The point is that the forecasting workflow is wrapped in the contracts, evidence artifacts, and policy boundaries expected from an AI-for-RAN operational intelligence pattern.

---

## Evidence summary

| What | Value | Source |
|---|---|---|
| **rApp manifest** (identity, R1 data subscriptions, A1 policy outputs, ML artifacts, boundary block) | shipped | [`rapp_manifest.yaml`](rapp_manifest.yaml) |
| **KPM input schema** (JSON Schema for the KPM-shaped measurements the rApp consumes) | shipped | [`schemas/kpm_input_v1.json`](schemas/kpm_input_v1.json) |
| **A1 policy schema** (JSON Schema for the traffic-steering policy candidates the rApp emits) | shipped | [`schemas/a1_policy_v1.json`](schemas/a1_policy_v1.json) |
| **Three-model head-to-head** (same KPI, same temporal split, same horizon) | Ridge 0.84 RMSE / GBR 2.88 / MLP 22.59 (underfits) | [`reports/model_comparison/comparison_metrics.md`](reports/model_comparison/comparison_metrics.md) |
| **Forecast → A1 policy converter** | exercised + schema-validated | [`src/ai_ran_kpi_forecasting/a1_policy.py`](src/ai_ran_kpi_forecasting/a1_policy.py) |
| **End-to-end R1 → forecast → A1 dataflow demo** | exercised on sample CSV | [`reports/r1_dataflow_demo/`](reports/r1_dataflow_demo/) |
| **Scenario evidence packs** | congestion + backhaul saturation + cell outage | [`reports/scenarios/latest/`](reports/scenarios/latest/) |
| Sample forecast metrics (ridge baseline, PRB DL util) | RMSE 0.84 · MAE 0.70 · MAPE 0.82 % | [`reports/forecast_examples/latest/metrics.json`](reports/forecast_examples/latest/metrics.json) |
| Telecom Italia MI benchmark | Benchmark-ready: pending local public dataset files. No benchmark metric claimed yet. | will land in `reports/forecast_examples/telecom_italia_mi/` |
| Tests | **40 / 40** (incl. 9 for A1 policy + 12 for model comparison) | [`tests/`](tests/) + `pytest -q` |
| CI | green on Linux | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| End-to-end reproducible | `make verify` regenerates every committed artifact | [`Makefile`](Makefile) |

---

## The rApp pattern, end-to-end

```
        KPM measurements (R1 input)        ◄── schemas/kpm_input_v1.json
                  │
                  ▼
          ┌───────────────────┐
          │  data.py +         │
          │  features.py       │  no-shuffle temporal split,
          │                    │  lag features, time features
          └─────────┬──────────┘
                    │
                    ▼
        ┌────────────────────────────────────────┐
        │  models.py — three-model registry      │
        │   Ridge / GradientBoosting / MLP       │
        │   same train/test split per model      │
        └────────────────────┬───────────────────┘
                             │
                             ▼
                   ForecastRunResult
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  a1_policy.py                          │  ◄── schemas/a1_policy_v1.json
        │  build_a1_policy_candidate(result,     │
        │                            threshold)  │
        └────────────────────┬───────────────────┘
                             │
                             ▼
                A1 traffic-steering policy candidate (JSON)
                             │
                             ▼
              [Near-RT RIC would consume here — out of scope]
```

The pipeline is exercised end-to-end by `scripts/simulate_r1_dataflow.py` on the sample CSV. The wire-protocol layer (R1 service registration, A1-P transport, KPM ASN.1 encoding) is intentionally not part of the deliverable — that's vendor-RIC-specific integration work documented in [`docs/AI_RAN_INTEGRATION.md`](docs/AI_RAN_INTEGRATION.md).

---

## Why three models, sklearn-only?

Three models — **Ridge**, **GradientBoostingRegressor**, **MLPRegressor** — all from scikit-learn. Zero new dependencies on top of the existing stack. The point is not to win a leaderboard; the point is to surface "which family of model wins on which KPI on which dataset" honestly rather than cherry-picking.

On the seeded 49-row sample:

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| `ridge_linear` | **0.84** | **0.70** | **0.82 %** |
| `gradient_boosting` | 2.88 | 2.63 | 3.16 % |
| `mlp` | 22.59 | 19.64 | 26.96 % (underfits) |

Honest reading: with 49 rows × 1 cell, the linear baseline dominates. Tree ensembles need more data to express their non-linearity advantage; small neural networks need both more data and tuning. This is a small-data finding, and it stays visible instead of being hidden behind only the winning model.

The Telecom Italia MI public benchmark is the next step where GBR and MLP may catch up or pull ahead. The loader and `make run-telecom` path are implemented, but no benchmark result is claimed until local public dataset files are available and `reports/forecast_examples/telecom_italia_mi/metrics.json` is generated.

---

## What makes this defensible as an AI-for-RAN rApp signal

| Practice | Evidence |
|---|---|
| **Schema-typed input / output** | `schemas/kpm_input_v1.json` + `schemas/a1_policy_v1.json` — the boundaries around the model, codified in JSON Schema |
| **rApp manifest with explicit boundary block** | `rapp_manifest.yaml` lists what's exercised vs not (live R1 wire protocol, live A1-P, OSC SMO integration — all honestly "not exercised") |
| **Forecast carries audit trail into the policy** | `forecast_basis` block in every A1 policy candidate records the model name, predicted peak, threshold crossed, and metrics-ref so operators can audit the policy back to its training run |
| **No-shuffle temporal split** | All three models trained on the same forward-only split — no leakage |
| **No oracle features** | Lag features use past values only; time features come from the timestamp column |
| **Honest no-action policy** | When the forecast doesn't cross the threshold, the policy candidate is still emitted with `action: no_action` and a rationale — the policy plane has an audit trail of "we looked, nothing to do" |
| **Integration boundary documented** | The repo explains what a real Non-RT RIC integration would require without claiming any live RIC, vendor RIC, or wire-protocol deployment. |

---

## Credibility Boundary

This repo demonstrates the Non-RT RIC rApp pattern for AI-for-RAN KPI forecasting on synthetic + small-public telemetry. It does **not** claim:

- live RAN integration
- live Non-RT RIC integration
- E2 / A1 / O1 / R1 protocol implementations on the wire (the contracts are documented in JSON Schema; ASN.1 encoding over E2/O1 and A1-P transport are not exercised)
- autonomous network control or closed-loop policy enforcement
- production rApp lifecycle (Helm packaging, R1 service registration via dms_cli)

The Telecom Italia MI public benchmark is **Benchmark-ready: pending local public dataset files. No benchmark metric claimed yet.** The loader is implemented per `DATA_CONTRACT.md`; run `make run-telecom REPORT_DIR=reports/forecast_examples/telecom_italia_mi` against the actual dataset to generate metrics. Until that artifact exists, this repo does not claim Telecom Italia MI benchmark accuracy.

See [`docs/AI_RAN_INTEGRATION.md`](docs/AI_RAN_INTEGRATION.md) for the integration recipe and the explicit "pattern, not deployment" rationale.

---

## Try it in five minutes

```bash
git clone https://github.com/obiedeh/ai-ran-kpi-forecasting.git
cd ai-ran-kpi-forecasting
python -m venv .venv
source .venv/bin/activate          # PowerShell: .\.venv\Scripts\Activate.ps1
make install-dev
make verify
```

`make verify` runs **ruff lint → 40 pytest tests → sample forecast → three-model comparison → R1 dataflow demo → three scenario evidence packs → evidence portal → publish page → artifact-existence checks**. GitHub Actions CI runs the same recipe on Ubuntu / Python 3.11 on every push.

Inspect:

- [`reports/index.html`](reports/index.html) — evidence portal
- [`reports/model_comparison/comparison_metrics.md`](reports/model_comparison/comparison_metrics.md) — three-model head-to-head
- [`reports/r1_dataflow_demo/dataflow_summary.md`](reports/r1_dataflow_demo/dataflow_summary.md) — R1 → forecast → A1 dataflow
- [`reports/scenarios/latest/`](reports/scenarios/latest/) — congestion / backhaul / outage dashboards

---

## What production deployment would require (honest priority list)

1. **R1 service registration** — deployment-specific registration for this rApp's data subscriptions and policy outputs with a target Non-RT RIC.
2. **Wire-protocol transport** — ASN.1 encoders/decoders for E2SM-KPM, A1-P producer SDK integration. Vendor-specific in production.
3. **Real KPI dataset benchmark** — Telecom Italia MI is the cheapest credible benchmark; full operator dataset access is the next tier.
4. **Model lifecycle** — training data drift detection, retraining scheduling, A/B between model versions, rollback policy.
5. **Observability** — metrics on forecast errors, policy emission rates, model staleness, and structured logging for the policy audit trail.
6. **Multi-cell scaling** — current pipeline is single-cell per invocation; production needs batched multi-cell forecasting and per-site routing of policies.

These are not claimed in this repo because they depend on a specific Non-RT RIC environment and operator data source. The boundary is clear: this repo is the **evidence + pattern**, not the **deployable rApp**.

For the evidence checklist see [`PORTFOLIO_DELIVERABLES.md`](PORTFOLIO_DELIVERABLES.md). For implemented-vs-planned tracking see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).
