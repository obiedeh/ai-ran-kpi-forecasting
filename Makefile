PYTHON ?= python
REPORT_DIR ?= reports/forecast_examples/latest
SCENARIO_DIR ?= reports/scenarios/latest

.PHONY: install install-dev test lint run-sample run-generic run-telecom synthetic forecast-edge-ai report scenario-demo scenario-backhaul scenario-outage portal publish model-comparison verify

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

run-sample:
	$(PYTHON) ai-ran-kpi-forecasting.py run-sample \
		--output-dir $(REPORT_DIR)

run-generic:
	$(PYTHON) ai-ran-kpi-forecasting.py forecast \
		--dataset-type generic \
		--data ./data/ran_kpi_sample.csv \
		--timestamp-col timestamp \
		--cell-id-col cell_id \
		--kpi-col prb_dl_util \
		--cell-id CELL_001 \
		--horizon 24 \
		--output-dir $(REPORT_DIR)

run-telecom:
	$(PYTHON) ai-ran-kpi-forecasting.py forecast \
		--dataset-type telecom-italia-mi \
		--data ./data/telecom_italia_mi \
		--aggregate hourly \
		--kpi-col internet_traffic \
		--horizon 24 \
		--output-dir $(REPORT_DIR)

synthetic:
	$(PYTHON) ai-ran-kpi-forecasting.py generate-synthetic \
		--output ./data/synthetic_ran_kpi.csv

forecast-edge-ai: synthetic
	$(PYTHON) ai-ran-kpi-forecasting.py forecast \
		--dataset-type generic \
		--data ./data/synthetic_ran_kpi.csv \
		--timestamp-col timestamp \
		--cell-id-col cell_id \
		--kpi-col edge_gpu_util_pct \
		--cell-id CELL_001 \
		--horizon 24 \
		--output-dir reports/forecast_examples/edge_ai/gpu_util
	$(PYTHON) ai-ran-kpi-forecasting.py forecast \
		--dataset-type generic \
		--data ./data/synthetic_ran_kpi.csv \
		--timestamp-col timestamp \
		--cell-id-col cell_id \
		--kpi-col edge_memory_util_pct \
		--cell-id CELL_001 \
		--horizon 24 \
		--output-dir reports/forecast_examples/edge_ai/memory_util

report: run-sample

scenario-demo:
	$(PYTHON) ai-ran-kpi-forecasting.py scenario-demo \
		--output-dir $(SCENARIO_DIR)/congestion

scenario-backhaul:
	$(PYTHON) ai-ran-kpi-forecasting.py scenario-demo \
		--scenario-type backhaul \
		--output-dir $(SCENARIO_DIR)/backhaul

scenario-outage:
	$(PYTHON) ai-ran-kpi-forecasting.py scenario-demo \
		--scenario-type outage \
		--output-dir $(SCENARIO_DIR)/outage

portal:
	$(PYTHON) ai-ran-kpi-forecasting.py portal \
		--output reports/index.html

publish:
	$(PYTHON) ai-ran-kpi-forecasting.py publish \
		--output-dir reports/publish/latest

model-comparison:
	$(PYTHON) scripts/run_model_comparison.py \
		--data data/ran_kpi_sample.csv \
		--output-dir reports/model_comparison

r1-dataflow-demo:
	$(PYTHON) scripts/simulate_r1_dataflow.py \
		--kpm-input data/ran_kpi_sample.csv \
		--cell-id CELL_001 \
		--kpi-col prb_dl_util \
		--threshold-pct 80 \
		--output-dir reports/r1_dataflow_demo

# Full verify: lint + tests + sample forecast + model comparison + scenario packs + portal.
# Validates that every committed report artifact regenerates from sources.
verify: lint test run-sample model-comparison r1-dataflow-demo scenario-demo scenario-backhaul scenario-outage portal publish
	@echo
	@echo "make verify complete. Inspect:"
	@echo "  reports/forecast_examples/latest/metrics.json"
	@echo "  reports/model_comparison/comparison_metrics.md"
	@echo "  reports/index.html"
