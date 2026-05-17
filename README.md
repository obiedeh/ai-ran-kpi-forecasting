# AI-RAN KPI Forecasting

AI-native telecom forecasting and operational intelligence for AI-RAN and edge infrastructure.

This repository is a production-style engineering project for offline KPI forecasting: load RAN-style telemetry, build temporal features, train a deterministic baseline model, generate a forward forecast, and write evidence artifacts for review.

The scope is intentionally grounded. It does not provide live RAN integration, autonomous network control, or notebook-only workflows.

## Capabilities

- Generic RAN KPI CSV loading with timestamp and cell validation.
- Optional Telecom Italia MI dataset normalization.
- Per-cell KPI selection and temporal feature engineering.
- Lag features and forward-only train/test splitting.
- Deterministic ridge-regression forecasting baseline.
- Autoregressive forecast generation.
- Reproducible CSV, Markdown, JSON, and SVG report artifacts.
- Scenario demos for offline congestion, backhaul, and outage evidence packs.

## Repository Structure

```text
.
|-- ai-ran-kpi-forecasting.py          # thin compatibility wrapper
|-- configs/sample_config.yaml
|-- data/ran_kpi_sample.csv
|-- docs/ARCHITECTURE.md
|-- reports/
|   |-- README.md
|   |-- sample_metrics_report.md
|   `-- forecast_examples/
|-- src/ai_ran_kpi_forecasting/
|   |-- data.py
|   |-- features.py
|   |-- models.py
|   |-- forecast.py
|   |-- metrics.py
|   |-- visualization.py
|   `-- cli.py
|-- tests/
|-- Makefile
`-- pyproject.toml
```

`src/ai_ran_kpi_forecasting/forecasting.py` remains as a compatibility import for older callers.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
make test
make run-sample
```

The sample run writes:

- `reports/forecast_examples/latest/prb_dl_util_forecast.csv`
- `reports/forecast_examples/latest/prb_dl_util_holdout.csv`
- `reports/forecast_examples/latest/prb_dl_util_forecast.svg`
- `reports/forecast_examples/latest/metrics.json`
- `reports/forecast_examples/latest/metrics_summary.md`

For the reviewer-facing evidence map, see [PORTFOLIO_DELIVERABLES.md](PORTFOLIO_DELIVERABLES.md).

Open the generated static evidence portal:

```text
reports/index.html
reports/publish/latest/index.html
```

## CLI Usage

Run the reproducible sample:

```bash
python ai-ran-kpi-forecasting.py run-sample
```

Run a generic CSV forecast:

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

Generate synthetic telemetry:

```bash
python ai-ran-kpi-forecasting.py generate-synthetic --output data/synthetic_ran_kpi.csv
```

Generate scenario dashboards:

```bash
make scenario-demo
make scenario-backhaul
make scenario-outage
make portal
make publish
```

Scenario evidence is written under:

- `reports/scenarios/latest/congestion/`
- `reports/scenarios/latest/backhaul/`
- `reports/scenarios/latest/outage/`

## Make Targets

- `make install`: install runtime requirements.
- `make install-dev`: install development requirements.
- `make test`: run pytest.
- `make run-sample`: run the reproducible sample forecast.
- `make lint`: run ruff.

## Data Contract

Generic CSV input requires:

- `timestamp`: parseable timestamp.
- `cell_id`: cell or sector identifier.
- one or more numeric KPI columns, such as `prb_dl_util`, `throughput_dl_mbps`, `rrc_users`, or `latency_ms`.

See `DATA_CONTRACT.md` and `configs/sample_config.yaml` for details.
