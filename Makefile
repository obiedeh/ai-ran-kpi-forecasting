PYTHON ?= python

.PHONY: install install-dev test lint run-generic run-telecom

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .

run-generic:
	$(PYTHON) ai-ran-kpi-forecasting.py \
		--dataset-type generic \
		--data ./data/ran_kpi_sample.csv \
		--timestamp-col timestamp \
		--cell-id-col cell_id \
		--kpi-col prb_dl_util \
		--cell-id CELL_001 \
		--horizon 24

run-telecom:
	$(PYTHON) ai-ran-kpi-forecasting.py \
		--dataset-type telecom-italia-mi \
		--data ./data/telecom_italia_mi \
		--aggregate hourly \
		--kpi-col internet_traffic \
		--horizon 24
