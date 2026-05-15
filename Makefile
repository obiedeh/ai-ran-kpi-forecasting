PYTHON ?= python
REPORT_DIR ?= reports/forecast_examples/latest
SCENARIO_DIR ?= reports/scenarios/latest

.PHONY: install install-dev test lint run-sample run-generic run-telecom synthetic report scenario-demo scenario-backhaul scenario-outage portal publish

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
