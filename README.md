# AI-RAN KPI Forecasting - Non-RT RIC rApp Pattern

## One-line summary

A reproducible Non-RT RIC rApp pattern for AI-for-RAN KPI forecasting: KPM-style telemetry in, forward-looking KPI forecasts out, advisory A1 policy candidates generated, and static evidence dashboards published for review.

This is not a claim of a deployed RIC application. It is a reproducible engineering pattern with typed contracts, no-leakage forecasting, scenario evidence, and explicit deployment boundaries.

> **▶ [Open the live evidence portal](https://obiedeh.github.io/ai-ran-kpi-forecasting/reports/index.html)** · [Dashboard](https://obiedeh.github.io/ai-ran-kpi-forecasting/reports/dashboard.html) · [Tech brief](TECH_BRIEF.md) · [Data contract](DATA_CONTRACT.md) · [Project status](PROJECT_STATUS.md)

## Why this exists

RAN operations cannot rely only on after-the-fact dashboards. The practical question is whether a non-real-time intelligence layer can see KPI degradation early enough to recommend a policy review before the cell is already in trouble.

From an engineering standpoint, I built this as a bridge between telecom KPI analytics and the way AI-for-RAN work is expected to be packaged: contracts, evidence, policy boundaries, and reproducible artifacts. A forecasting model alone is not enough signal.

The Non-RT RIC / rApp model is the relevant O-RAN architectural pattern for this class of non-real-time RAN intelligence: telemetry-driven analytics, forecasting, policy recommendation, and model lifecycle support. This repo implements that pattern on synthetic/sample telemetry; integration with FlexRIC, OSC RIC, or a vendor RIC is documented but not exercised.

## What problem it solves

This project asks a more operational question than “can a model forecast a KPI?” It asks whether a Non-RT intelligence layer can package forecast evidence, scenario impact, and an advisory A1 policy candidate in a way an engineer could inspect before taking action.

## The engineering pattern

```mermaid
flowchart LR
    A["KPM-style telemetry"] --> B["Schema and data contract"]
    B --> C["Temporal feature pipeline"]
    C --> D["Ridge / GBR / MLP model comparison"]
    D --> E["Forward KPI forecast"]
    E --> F["Advisory A1 policy candidate"]
    F --> G["Static dashboard and scenario evidence"]
    G --> H["Operator review"]
```

## Live evidence portal

- [Live evidence portal](https://obiedeh.github.io/ai-ran-kpi-forecasting/reports/index.html)
- [Live dashboard](https://obiedeh.github.io/ai-ran-kpi-forecasting/reports/dashboard.html)
- [Published local portal](reports/publish/latest/index.html)
- [Scenario dashboards](reports/scenarios/latest/)

GitHub shows committed HTML files as source code. Use the GitHub Pages links above to open the rendered pages.

## Headline evidence

| Signal | Value | Source |
|---|---|---|
| KPM-style input contract | shipped | `schemas/kpm_input_v1.json` |
| Advisory A1 output contract | shipped | `schemas/a1_policy_v1.json` |
| rApp packaging signal | shipped | `rapp_manifest.yaml` |
| Three-model comparison | Ridge 0.8368 RMSE, GBR 2.8755 RMSE, MLP 22.5926 RMSE | `reports/model_comparison/comparison_metrics.md` |
| Sample forecast metrics | RMSE 0.8368, MAE 0.6954, MAPE 0.8204% | `reports/forecast_examples/latest/metrics.json` |
| R1-style dataflow demo | KPM-style input to forecast to A1 candidate | `reports/r1_dataflow_demo/` |
| Scenario evidence | congestion, backhaul saturation, cell outage | `reports/scenarios/latest/` |
| Telecom Italia MI benchmark | Measured on the public Milan grid, 62 days, three squares, three models plus naive baselines; the naive baseline wins on two of three squares | `reports/forecast_examples/telecom_italia_mi/summary.json` |
| Reproducibility | `make verify` regenerates committed evidence artifacts | `Makefile` |

## What makes this more than a forecasting notebook

- Temporal evaluation: train/test splits preserve time order; no shuffled leakage.
- Operational feature design: lag features use only past KPI values available at inference time.
- Typed RAN telemetry contract: `schemas/kpm_input_v1.json` defines the KPM-style input boundary.
- Typed policy output contract: `schemas/a1_policy_v1.json` defines the advisory A1 candidate boundary.
- rApp packaging signal: `rapp_manifest.yaml` documents identity, inputs, outputs, and integration expectations.
- Scenario evidence: congestion, backhaul saturation, and cell outage reports connect forecasts to operator action.
- Reproducibility: `make verify` regenerates committed evidence artifacts.
- Honest boundary: no live RIC deployment, no autonomous control, no production policy enforcement.

## Architecture

The architecture is documented in [docs/architecture.md](docs/architecture.md), with Mermaid diagrams under [docs/diagrams/](docs/diagrams/).

The core runtime path is:

```text
KPM-style telemetry -> validation and feature generation -> temporal forecasting
-> advisory A1 policy candidate -> report artifacts -> operator review
```

## Dashboard and report artifacts

| Artifact | Purpose |
|---|---|
| `reports/index.html` | GitHub Pages evidence portal |
| `reports/dashboard.html` | top-level operational dashboard generated with the portal |
| `reports/forecast_examples/latest/metrics.json` | sample forecast metrics |
| `reports/model_comparison/comparison_metrics.md` | Ridge / GBR / MLP comparison |
| `reports/r1_dataflow_demo/a1_policy_candidate.json` | advisory A1 policy candidate |
| `reports/scenarios/latest/` | congestion, backhaul, and outage evidence packs |
| `reports/publish/latest/index.html` | release-friendly landing page |

## Measured results

The committed measured results use deterministic sample telemetry, not live operator data.

| Measurement | Result | Boundary |
|---|---:|---|
| Sample PRB forecast RMSE | 0.8368 | ridge baseline on `data/ran_kpi_sample.csv` |
| Sample PRB forecast MAE | 0.6954 | same sample, same temporal split |
| Sample PRB forecast MAPE | 0.8204% | small sample metric only |
| Gradient boosting RMSE | 2.8755 | weaker on current sample |
| MLP RMSE | 22.5926 | underfits current sample |

The small-data result is intentionally visible: Ridge wins here. The model is the least interesting part of the repo; the useful part is the engineering boundary around the model. The public Telecom Italia MI benchmark below tests whether that ranking survives a larger dataset; it does, and a naive baseline beats all three on two of three squares.

## Measured: Telecom Italia MI benchmark

Public dataset: Telecommunications - SMS, Call, Internet - MI,
[doi:10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV), Harvard Dataverse,
ODbL 1.0. 62 daily files, 2013-11-01 to 2014-01-01, 319,896,289 raw
rows, 20.8 GB, every file's MD5 verified against the Dataverse record
([`dataset.json`](reports/forecast_examples/telecom_italia_mi/dataset.json)). The files
are not committed; `make fetch-telecom EMAIL=<you>` downloads them (the
publisher's guestbook asks for an email) and `make benchmark-telecom` reproduces
everything below ([`summary.json`](reports/forecast_examples/telecom_italia_mi/summary.json),
generated 2026-09-09, host CPU).

Setup: hourly `internet_traffic` per square, summed over country codes. Three
squares chosen by a stated rule over the 62-day totals
([`cell_activity.csv`](reports/forecast_examples/telecom_italia_mi/cell_activity.csv)):
largest total, nearest the median, nearest the 10th percentile. Features are
calendar terms and lags 1, 2, 3, 6, 12, 24 of the target only. One time-ordered
split per cell: 1171 training hours, 293 test hours, test starting
2013-12-20 18:00 UTC, so the hold-out covers Christmas and New Year. One-step-ahead
hold-out error, with two naive baselines scored on the same rows.

RMSE, in the dataset's activity units:

| Cell | Square | 62-day total | Ridge | Gradient boosting | MLP | Naive last value | Seasonal naive 24 h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high | 5161 | 12,740,060 | [1697.5](reports/forecast_examples/telecom_italia_mi/5161/ridge_linear/metrics.json) | [2793.5](reports/forecast_examples/telecom_italia_mi/5161/gradient_boosting/metrics.json) | [7653.1](reports/forecast_examples/telecom_italia_mi/5161/mlp/metrics.json) | [2407.7](reports/forecast_examples/telecom_italia_mi/5161/baselines.json) | 4163.1 |
| mid | 3168 | 277,931 | [24.3](reports/forecast_examples/telecom_italia_mi/3168/ridge_linear/metrics.json) | [26.9](reports/forecast_examples/telecom_italia_mi/3168/gradient_boosting/metrics.json) | [245.8](reports/forecast_examples/telecom_italia_mi/3168/mlp/metrics.json) | [15.9](reports/forecast_examples/telecom_italia_mi/3168/baselines.json) | 43.8 |
| low | 9408 | 51,230 | [6.0](reports/forecast_examples/telecom_italia_mi/9408/ridge_linear/metrics.json) | [5.1](reports/forecast_examples/telecom_italia_mi/9408/gradient_boosting/metrics.json) | [79.8](reports/forecast_examples/telecom_italia_mi/9408/mlp/metrics.json) | [4.4](reports/forecast_examples/telecom_italia_mi/9408/baselines.json) | 7.8 |

MAPE, percent:

| Cell | Square | Ridge | Gradient boosting | MLP | Naive last value | Seasonal naive 24 h |
| --- | --- | --- | --- | --- | --- | --- |
| high | 5161 | 82.6 | 108.8 | 349.7 | 30.2 | 67.2 |
| mid | 3168 | 17.3 | 18.5 | 126.7 | 9.6 | 20.8 |
| low | 9408 | 16.1 | 12.3 | 161.7 | 11.6 | 18.3 |

What this says. On the median and low-activity squares the naive last-value
baseline beats all three models on every metric. On the busiest square Ridge
has the lowest RMSE but the naive baseline has lower MAE and far lower MAPE.
The MLP, unscaled and small by design, diverges on all three. The model ranking
from the 48-row sample (Ridge first, MLP last) holds, and the larger finding is
that none of the three earns its place over a one-line baseline on this hourly
task with these features. That is the result the repo now carries; the
engineering boundary around the model is unchanged.

## ONNX exports for edge inference benchmarks

The three sample-trained forecasters are exported to ONNX (opset 17) under
[`models/exports/`](models/exports/) by [`scripts/export_onnx.py`](scripts/export_onnx.py)
(`make export-onnx`, needs `pip install -r requirements-onnx.txt`). The
[manifest](models/exports/manifest.json) records the dataset hash, feature
names, input shape `(None, 16)`, per-file SHA-256, tool versions, and the
ONNX-versus-scikit-learn parity on the hold-out sample. Tests in
`tests/test_onnx_export.py` reproduce the export and check that the sample
metrics match `reports/model_comparison/`.

Purpose: run these models through the same edge inference harness used in
[jetson-edge-ai-security](https://github.com/obiedeh/jetson-edge-ai-security)
on Jetson AGX Thor. The measurement below is an inference-cost figure, not a
forecast-accuracy claim.

### Measured on Jetson AGX Thor

Run `3e55b967a7ea`, finished 2026-09-09T18:00:17Z, on Jetson AGX Thor
(tegra264, L4T R38.4.0, 120 W power mode, 122 GB), onnxruntime
`CPUExecutionProvider` (the CUDA provider has no kernels for this GPU in the
PyPI wheel), one intra-op and one inter-op thread, spinning disabled, batch 1,
synthetic Gaussian inputs of shape `(1, 16)`, paced open-loop load for 300 s
per tier at 10, 100 and 1000 events/s, tegrastats sampled at 1 Hz.
Artifact: [`reports/thor_benchmark/thor_benchmark.json`](reports/thor_benchmark/thor_benchmark.json)
([run log](reports/thor_benchmark/thor_benchmark_run.log),
[tegrastats samples](reports/thor_benchmark/thor_benchmark_tegrastats.jsonl)).
Idle board input power before load: 25,798 mW p50 over 60 s.

At 1000 events/s:

| Model | p50 ms | p95 ms | p99 ms | Achieved events/s | Deadline misses | Process RSS GB | VIN p50 mW | Tj peak C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ridge_linear` | 0.0112 | 0.0115 | 0.0118 | 1000.0 | 0 | 0.0694 | 24,170 | 41.562 |
| `gradient_boosting` | 0.015 | 0.0153 | 0.0155 | 1000.0 | 0 | 0.0729 | 24,112 | 40.25 |
| `mlp` | 0.0202 | 0.0206 | 0.0208 | 1000.0 | 2 | 0.0748 | 24,110 | 39.937 |

The harness's latency and throughput gates are defined for the security
repo's detector and forecaster and report `not measured` here; only the
memory gate (4 GB) evaluates, at 0.0802 GB peak RSS. No pass or fail
is claimed for the other three.

Thread-pool comparison. A second run with onnxruntime's default thread pool
and spinning enabled, 120 s per tier at 100 and 1000 events/s
([`default_threads.json`](reports/thor_benchmark/default_threads.json),
[log](reports/thor_benchmark/default_threads_run.log),
[tegrastats](reports/thor_benchmark/default_threads_tegrastats.jsonl)), compared
by the security repo's `compare_thread_runs.py` into
[`thread_comparison.json`](reports/thor_benchmark/thread_comparison.json). At
1000 events/s:

| Model | p95 ms, default pool | p95 ms, one thread no spin | Misses, default | Misses, one thread | VIN p50 mW, default | VIN p50 mW, one thread |
| --- | --- | --- | --- | --- | --- | --- |
| `ridge_linear` | 0.0115 | 0.0115 | 4 | 0 | 24,184 | 24,170 |
| `gradient_boosting` | 0.0222 | 0.0153 | 18,342 | 0 | 54,114 | 24,112 |
| `mlp` | 0.0083 | 0.0206 | 201 | 2 | 24,574 | 24,110 |

The gradient-boosting graph is the one that engages the default thread pool:
about 30 W of extra board power and 18,342 pacing misses, both removed by the
single-thread setting, reproducing the security repo's finding on a different
model. The linear model is unaffected. The MLP is faster under the default
pool but misses 201 deadlines there and 2 with one thread. Board power
includes unrelated host activity; each run's idle baseline is in its file.

An earlier attempt on 2026-09-08 was terminated before producing numbers;
its record is [`primary_failure.json`](reports/thor_benchmark/primary_failure.json).

## GitHub repo description

Recommended description:

> Non-RT RIC rApp pattern for AI-RAN KPI forecasting, scenario evidence, and advisory A1 policy generation.

## Credibility boundary

This repo demonstrates a Non-RT RIC rApp pattern for AI-for-RAN KPI forecasting on synthetic and small sample telemetry. The boundary is not a weakness; it is what keeps the evidence useful. It does not claim:

- live RAN integration
- live Non-RT RIC deployment
- operator validation
- E2, A1, O1, or R1 wire-protocol implementation
- autonomous network control
- closed-loop policy enforcement
- production rApp lifecycle or service registration

The deliverable is the pattern, contracts, model comparison, and evidence pack.

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

Windows / direct Python:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python ai-ran-kpi-forecasting.py run-sample --output-dir reports/forecast_examples/latest
python ai-ran-kpi-forecasting.py portal --output reports/index.html
```

## Reproduce all artifacts

```bash
make verify
```

`make verify` runs lint, tests, sample forecast, model comparison, R1 dataflow demo, scenario dashboards, portal generation, and publish-page generation.

## Repository map

```text
configs/                         sample run configuration
data/                            committed sample telemetry
docs/                            architecture and AI-RAN integration notes
reports/                         committed evidence artifacts and HTML pages
schemas/                         KPM input and advisory A1 policy contracts
scripts/                         model comparison and R1 dataflow demo scripts
src/ai_ran_kpi_forecasting/       package source
tests/                           unit and pipeline tests
rapp_manifest.yaml               rApp identity, inputs, outputs, and boundary
```

## Engineering signal

This project is designed around the operational shape of an AI-for-RAN workflow, not just model accuracy. The core signal is the boundary discipline around the model: typed telemetry input, time-ordered evaluation, scenario evidence, advisory policy output, and explicit limits around deployment.

- Temporal evaluation is used instead of shuffled train/test splits, so the forecast is tested closer to how it would behave in operation.
- KPM-style input and A1 advisory output are defined as typed contracts, making the system boundaries inspectable.
- Forecast outputs are connected to advisory policy candidates instead of being left as standalone charts.
- Weak model results remain visible in the evidence pack, because hiding them would make the evaluation less credible.
- The Telecom Italia MI benchmark is measured and the naive baseline's win on two of three squares is reported, not hidden.
- The HTML evidence pack is generated and GitHub Pages compatible, so results can be reviewed without cloning the repo.

## Next engineering steps

The next steps are intentionally narrow: improve evidence quality before adding integration complexity.

1. Run the Telecom Italia MI benchmark locally and publish the measured metrics.
2. Extend the workflow to multi-cell batch forecasting while preserving temporal evaluation and no-leakage feature rules.
3. Add drift and retraining-readiness reports so the project reflects an operational AI lifecycle.
4. Add a real Non-RT RIC adapter only when there is a target OSC RIC, FlexRIC, or vendor RIC environment to test against.

## License

MIT.
