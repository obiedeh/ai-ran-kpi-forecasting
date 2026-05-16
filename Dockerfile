# Reproducible forecasting environment
#
# Build:
#   docker build -t ai-ran-kpi-forecasting:latest .
#
# Run sample forecast:
#   docker run --rm -v $(pwd)/reports:/app/reports ai-ran-kpi-forecasting:latest make run-sample
#
# Run tests:
#   docker run --rm ai-ran-kpi-forecasting:latest make test

FROM python:3.11-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/
COPY configs/ ./configs/
COPY Makefile ai-ran-kpi-forecasting.py ./

RUN pip install --no-cache-dir -e .

VOLUME ["/app/reports"]

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import ai_ran_kpi_forecasting" || exit 1

RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["make", "run-sample"]
