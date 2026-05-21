# AI-RAN KPI Forecasting — Non-RT RIC rApp pattern

**AI-for-RAN at the operational layer.** A Non-RT RIC rApp pattern for KPI forecasting on RAN telemetry: deterministic ridge baseline plus tree-ensemble and small-neural comparisons, scenario evidence packs for congestion / backhaul / outage, and an A1-policy output contract so a downstream Near-RT RIC could consume the forecasts as policy enrichment.

The deliverable is the engineering pattern — *not* a deployed rApp on a live RIC. Every claim is backed by a committed artifact: schemas, manifest, code, reports.

> **▶ [Open the live evidence portal](https://obiedeh.github.io/ai-ran-kpi-forecasting/reports/index.html)** &nbsp;·&nbsp; [Tech brief](TECH_BRIEF.md) &nbsp;·&nbsp; [AI-RAN integration](docs/AI_RAN_INTEGRATION.md) &nbsp;·&nbsp; [Source](https://github.com/obiedeh/ai-ran-kpi-forecasting)

---

## Why this exists

The AI-RAN movement (AI-RAN Alliance, founded Feb 2024 by NVIDIA, NTT DOCOMO, Microsoft Azure for Operators, SoftBank, Nokia, Ericsson, Samsung, and others) splits AI in the radio stack into three pillars:

- **AI-and-RAN** — RAN and AI workloads share GPU infrastructure
- **AI-on-RAN** — AI services delivered over the RAN
- **AI-for-RAN** — AI used to *improve* the RAN itself

The companion repo [`wireless-link-intelligence-system`](https://github.com/obiedeh/wireless-link-intelligence-system) targets **AI-for-RAN at the PHY layer** (channel estimation, OFDM, INT8 ONNX deployment). This repo targets **AI-for-RAN at the operational layer**: forecasting RAN KPIs hours ahead so a Non-RT RIC can pre-position policies before congestion / backhaul saturation / outage actually arrives.

The Non-RT RIC + rApp framing is how AI-for-RAN actually deploys in 2026 — every operator with a real AI-RAN programme (AT&T, Verizon, DT, Vodafone, NTT, Rakuten, SoftBank) runs ML workloads as rApps on a Non-RT RIC. This repo implements the *rApp pattern* on synthetic data; integration with FlexRIC / OSC RIC / a vendor RIC is documented but not exercised. See [Credibility Boundary](#credibility-boundary).

**The discipline is the deliverable.** The forecasting math is textbook; the dataset is synthetic with a public-benchmark plan; the rApp manifest + A1-policy contract are honest pattern, not deployment.

---

## Headline Evidence

| Signal | Value | Source |
|---|---|---|
| **Three-model head-to-head** (same KPI, same temporal split) | Ridge **wins** on tiny 49-row sample · GBR runner-up · MLP underfits | `reports/model_comparison/comparison_metrics.md` |
| **rApp pattern artifacts** | manifest + KPM input schema + A1 policy output schema | `rapp_manifest.yaml` · `schemas/` |
| Scenario evidence packs | congestion · backhaul saturation · cell outage | `reports/scenarios/latest/` |
| Sample forecast metrics (ridge baseline, PRB DL util) | RMSE 0.84 · MAE 0.70 · MAPE 0.82% | `reports/forecast_examples/latest/metrics.json` |
| Edge-AI KPI forecast (synthetic) | GPU util RMSE 3.62 · memory util RMSE 3.21 | `reports/forecast_examples/edge_ai/` |
| Telecom Italia MI benchmark | `<TO MEASURE>` *(public dataset, deferred — see below)* | will land in `reports/forecast_examples/telecom_italia_mi/` |
| End-to-end forecast wall-clock | ~20 s on Windows / Python 3.13 | `reports/forecast_examples/latest/timing.json` |
| Test suite | 19 tests · CI green on Linux | `.github/workflows/ci.yml` |
| End-to-end reproducible | `make verify` regenerates every committed artifact under `reports/` | `Makefile` |

Full numbers + methodology in [Measured metrics](#measured-metrics) below. Limitations in [Credibility Boundary](#credibility-boundary).

---

## Operational Intelligence Dashboard

The live evidence portal is organized as a Non-RT RIC operational forecasting dashboard, not a notebook export:

- KPI forecasting summary: horizon, peak PRB forecast, best measured model, scenario count, and validation status.
- Congestion interpretation: project-defined Stable / Elevated / Congested / Critical risk tiers over existing forecast outputs.
- Policy recommendation layer: typed KPM input to forecast to A1 policy candidate, framed as advisory decision support only.
- Model comparison: Ridge / GradientBoosting / MLP metrics shown side-by-side, including weak MLP performance.
- Scenario evidence: congestion, backhaul saturation, and cell outage dashboards generated from existing synthetic telemetry artifacts.
- Engineering boundaries: synthetic KPI environment, no live RAN connection, no deployed rApp/xApp, no autonomous control.
- Reproducible workflow: `make verify` regenerates reports, scenario dashboards, portal, publish bundle, model comparison, and R1-to-A1 demo.

---

## Engineering practices that matter here

The concrete decisions that separate a clean Non-RT RIC rApp pattern from a forecasting tutorial:

- **Non-shuffled temporal split.** All train/test splits respect time order — no shuffle, no leakage. The metrics reflect *forward forecasting*, not held-out interpolation. Enforced in `forecast.py::temporal_train_test_split` and tested.
- **No-leakage feature engineering.** Lag features are constructed from past values only; time-of-day / day-of-week features are computed from the timestamp column. There is no feature in the matrix that the model couldn't compute online with only past KPM measurements.
- **Schema-typed inputs and outputs.** `schemas/kpm_input_v1.json` formalises the KPM measurement contract the rApp consumes (matches the existing `DATA_CONTRACT.md`); `schemas/a1_policy_v1.json` formalises the A1-policy output the rApp emits. The schemas are the bridge between the forecaster and the broader RIC architecture.
- **Three-model comparison, same split, same data.** Ridge (deterministic linear), GradientBoostingRegressor (tree ensemble), MLPRegressor (small neural) — all from scikit-learn, zero new dependencies. The point is that no single model is cherry-picked.
- **Honest weak result on the channel side.** The L1 companion repo's channel classifier scores 0.472 — surfaced, not hidden. This repo's analog: forecasting accuracy is measured only on synthetic and small samples; the Telecom Italia MI public benchmark is explicitly deferred with a reproduction recipe rather than a fabricated number.
- **Deterministic seeded RNG everywhere.** `np.random.default_rng(seed)` threads through synthetic-data generation, model training, and scenario perturbation. `make verify` regenerates byte-identical CSV/JSON output across runs.
- **Scenario-led evidence.** Congestion / backhaul saturation / cell-outage are the canonical operational use cases for KPI-forecasting rApps; this repo ships dashboards for each.
- **rApp pattern, not rApp claim.** The repo emits A1-policy JSON candidates and ships an rApp manifest, but does not integrate with a live RIC. The integration recipe is documented in `docs/AI_RAN_INTEGRATION.md`. Pattern, not deployment.

If you are evaluating AI-for-RAN engineering: these are the signals that distinguish a real rApp pattern from a forecasting tutorial with telecom terminology.

---

## What this repo does

| Layer | Implementation |
|---|---|
| **KPM input parser** (`data.py`) | Loads RAN-style telemetry — PRB utilization, throughput, RRC users, latency, SINR, packet loss, plus optional edge-AI KPIs (GPU / memory utilization) per `DATA_CONTRACT.md` |
| **Feature engineering** (`features.py`) | Time-of-day / day-of-week features + configurable lag windows over the target KPI |
| **Forecasting models** (`models.py`) | Ridge baseline + GradientBoostingRegressor + MLPRegressor on the same temporal split |
| **Multi-step rollout** (`forecast.py`) | Configurable horizon (default 24 steps), autoregressive, forward-only |
| **A1-policy emitter** (`a1_policy.py`) | Converts a forecast into an A1 traffic-steering policy JSON candidate per `schemas/a1_policy_v1.json` |
| **Scenario generators** (`data.py`) | Congestion / backhaul saturation / cell outage — perturbation overlays on baseline telemetry |
| **Evidence reports** (`reports.py`) | CSV + SVG + JSON + Markdown bundle per run, plus per-scenario HTML dashboards and a top-level evidence portal |
| **rApp manifest** (`rapp_manifest.yaml`) | Declares this rApp's identity, data subscriptions, and policy outputs — the contract for slotting into a Non-RT RIC |

---

## Measured metrics

Source: [`reports/forecast_examples/latest/metrics.json`](reports/forecast_examples/latest/metrics.json). Baseline model: `ridge_linear`. KPI: `prb_dl_util` on `CELL_001`. Sample dataset: 49 rows × 6 columns (hourly, `data/ran_kpi_sample.csv`).

| Metric | Value | Status |
| --- | ---: | --- |
| RMSE | 0.8368 | measured (ridge baseline) |
| MAE | 0.6954 | measured |
| MAPE | 0.82% | measured |
| Three-model head-to-head on the sample KPI (same temporal split) | Ridge 0.84 RMSE · GBR 2.88 RMSE · MLP 22.59 RMSE (underfits) | [`reports/model_comparison/comparison_metrics.md`](reports/model_comparison/comparison_metrics.md). Honest small-data finding: with 49 rows × 1 cell, the linear baseline dominates. Tree ensembles and small neural nets need more data to compete — exactly why the Telecom Italia MI benchmark matters as the next step. |
| Forecasting accuracy on Telecom Italia MI | `<TO MEASURE>` (Linux/storage-deferred) | Plan: download the Telecom Italia "Big Data Challenge — Milan" SMS/Call/Internet dataset (one daily file ≈ 150 MB), place under `data/telecom_italia_mi/`, run `make run-telecom` (loader already implemented per `DATA_CONTRACT.md`), capture RMSE/MAE/MAPE to `reports/forecast_examples/telecom_italia/metrics.json` |
| Forecasting accuracy on `edge_gpu_util_pct` (RMSE / MAE / MAPE) | 3.62 / 2.86 / 6.38% | measured ([reports/forecast_examples/edge_ai/gpu_util/metrics.json](reports/forecast_examples/edge_ai/gpu_util/metrics.json)) — `ridge_linear`, CELL_001, 24-step horizon on synthetic 504-row, 3-cell telemetry; reproduce via `make forecast-edge-ai` |
| Forecasting accuracy on `edge_memory_util_pct` (RMSE / MAE / MAPE) | 3.21 / 2.65 / 4.44% | measured ([reports/forecast_examples/edge_ai/memory_util/metrics.json](reports/forecast_examples/edge_ai/memory_util/metrics.json)) — same setup as the GPU row |
| End-to-end forecast wall-clock (s) | 19.7 (Windows · Python 3.13) | measured ([reports/forecast_examples/latest/timing.json](reports/forecast_examples/latest/timing.json)) — machine-dependent; reproduce by timing `make run-sample` on your machine |

The measured numbers are from deterministic sample telemetry — they prove the pipeline is reproducible end-to-end, not that the model has been validated against live operator data. See [Credibility Boundary](#credibility-boundary).

Reproduce locally:

```bash
make install-dev
make run-sample
cat reports/forecast_examples/latest/metrics.json
```

---

## Credibility Boundary

This repo demonstrates **the Non-RT RIC rApp pattern for AI-for-RAN KPI forecasting on synthetic + small-public telemetry**. It does not claim:

- live RAN integration
- live Non-RT RIC integration (FlexRIC / OSC RIC / Nokia MantaRay / Ericsson IAP / Mavenir)
- E2 / A1 / O1 / R1 protocol implementations (the contracts are documented; the wire protocol is not)
- autonomous network control
- closed-loop policy enforcement
- production-grade lifecycle (Helm packaging, R1 service registration, KPI subscriptions over real ORAN endpoints)

These boundaries are intentional. The deliverable is the **pattern + the schemas + the measured forecasting evidence** — enough that a senior AI-RAN engineer reading the repo can confirm "this person understands how AI-for-RAN actually deploys", not enough to drop into a live RIC.

See [docs/AI_RAN_INTEGRATION.md](docs/AI_RAN_INTEGRATION.md) for the honest "pattern not deployment" boundary and the integration recipe with FlexRIC / OSC RIC.

---

## Prerequisites

- **Python 3.11 or newer** (see `pyproject.toml`)
- **git**, **make** (Linux / macOS) — Windows users see [Without make](#without-make-windows-or-direct-python) below

## Quickstart

```bash
git clone https://github.com/obiedeh/ai-ran-kpi-forecasting.git
cd ai-ran-kpi-forecasting
python -m venv .venv
source .venv/bin/activate
make install-dev
make run-sample
cat reports/forecast_examples/latest/metrics.json
```

Generate scenario evidence packs and the static portal:

```bash
make scenario-demo
make scenario-backhaul
make scenario-outage
make portal
make publish
```

Open `reports/index.html` for the reviewer-facing evidence portal.

### Without make (Windows or direct Python)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python ai-ran-kpi-forecasting.py run-sample --output-dir reports/forecast_examples/latest
pytest -q
```

After `pip install -e .`, the CLI is also available as `ai-ran-kpi-forecasting run-sample` via the entry point declared in `pyproject.toml`.

---

## CLI Usage

```bash
# Reproducible sample
python ai-ran-kpi-forecasting.py run-sample

# Generic CSV forecast
python ai-ran-kpi-forecasting.py forecast \
  --dataset-type generic \
  --data ./data/ran_kpi_sample.csv \
  --timestamp-col timestamp \
  --cell-id-col cell_id \
  --kpi-col prb_dl_util \
  --cell-id CELL_001 \
  --horizon 24 \
  --output-dir reports/forecast_examples/latest

# Generate synthetic telemetry
python ai-ran-kpi-forecasting.py generate-synthetic --output data/synthetic_ran_kpi.csv
```

---

## Data Contract

Generic CSV input requires `timestamp`, `cell_id`, and one or more numeric KPI columns — for example `prb_dl_util`, `throughput_dl_mbps`, `rrc_users`, `latency_ms`, `edge_gpu_util_pct`, `edge_memory_util_pct`.

See [DATA_CONTRACT.md](DATA_CONTRACT.md) for the full schema, column overrides, recommended KPI fields, and the Telecom Italia MI normalization map. The same contract is encoded as a JSON Schema in `schemas/kpm_input_v1.json` — that's the file the rApp manifest references as its R1 data-subscription contract.

---

## Make Targets

| Target | What it does |
|---|---|
| `make install-dev` | Install runtime + dev requirements |
| `make test` | Run pytest |
| `make lint` | Run ruff |
| `make run-sample` | Run the reproducible sample forecast |
| `make run-generic` | Run a generic CSV forecast against `data/ran_kpi_sample.csv` |
| `make run-telecom` | Run a forecast against the Telecom Italia MI dataset |
| `make synthetic` | Regenerate synthetic telemetry under `data/` |
| `make scenario-demo` / `scenario-backhaul` / `scenario-outage` | Generate scenario evidence packs |
| `make portal` / `publish` | Build the static evidence portal + publish page |
| `make model-comparison` | Run Ridge / GBR / MLP head-to-head on the same temporal split |
| `make verify` | Full pipeline regeneration + artifact validation |

---

## Repository Structure

```text
.
├── ai-ran-kpi-forecasting.py            # thin CLI wrapper
├── configs/sample_config.yaml
├── data/
│   ├── ran_kpi_sample.csv               # deterministic sample (committed)
│   └── synthetic_ran_kpi.csv            # generated by `make synthetic` (gitignored)
├── rapp_manifest.yaml                   # rApp identity + R1 / A1 contracts
├── schemas/
│   ├── kpm_input_v1.json                # KPM input schema (matches DATA_CONTRACT.md)
│   └── a1_policy_v1.json                # A1 policy output schema
├── docs/
│   ├── architecture.md                  # Pipeline + module responsibilities
│   ├── AI_RAN_INTEGRATION.md            # How this slots into FlexRIC / OSC RIC
│   └── diagrams/                        # Mermaid arch diagrams
├── reports/
│   ├── README.md
│   ├── sample_metrics_report.md
│   ├── forecast_examples/               # Sample + edge-AI runs
│   ├── model_comparison/                # Ridge vs GBR vs MLP (Phase 2)
│   ├── scenarios/                       # congestion / backhaul / outage packs
│   ├── publish/
│   └── index.html                       # Evidence portal
├── src/ai_ran_kpi_forecasting/
│   ├── data.py · features.py · models.py · forecast.py
│   ├── metrics.py · explainability.py · visualization.py · reports.py
│   ├── a1_policy.py                     # Forecast → A1 policy JSON candidate
│   └── cli.py
├── scripts/
│   ├── run_model_comparison.py          # Phase 2
│   └── simulate_r1_dataflow.py          # Phase 3 — KPM in → forecast → A1 out
├── tests/
│   ├── test_core_forecasting.py
│   └── test_pipeline.py
├── DATA_CONTRACT.md · PORTFOLIO_DELIVERABLES.md · PROJECT_STATUS.md · TECH_BRIEF.md
├── AGENTS.md · Makefile · pyproject.toml · LICENSE
```

---

## Deliverables and status

- [PORTFOLIO_DELIVERABLES.md](PORTFOLIO_DELIVERABLES.md) — reviewer-facing evidence map
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — honest implemented-vs-planned tracking
- [TECH_BRIEF.md](TECH_BRIEF.md) — one-page hiring-manager summary
- [docs/architecture.md](docs/architecture.md) — pipeline + module responsibilities
- [docs/AI_RAN_INTEGRATION.md](docs/AI_RAN_INTEGRATION.md) — rApp pattern integration recipe

## License

MIT.
