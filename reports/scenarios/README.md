# Scenario Dashboards

This directory stores telecom-style before/after scenario evidence.

The default demo creates:

- `latest/congestion/baseline/`: baseline telemetry and forecast evidence
- `latest/congestion/congestion/`: congestion-shock telemetry and forecast evidence
- `latest/congestion/dashboard/`: a combined HTML dashboard and summary
- `latest/backhaul/baseline/`: baseline telemetry and forecast evidence
- `latest/backhaul/backhaul/`: backhaul-shock telemetry and forecast evidence
- `latest/backhaul/dashboard/`: a combined HTML dashboard and summary
- `latest/outage/baseline/`: baseline telemetry and forecast evidence
- `latest/outage/outage/`: cell-outage telemetry and forecast evidence
- `latest/outage/dashboard/`: a combined HTML dashboard and summary
- `reports/index.html`: a portal page linking the evidence packs
- `reports/publish/latest/`: a release-friendly publish page and manifest

The goal is to resemble an actual wireless operations review pack:

- cell-level KPI traces
- congestion impact callouts
- throughput and latency behavior around the shock window
- forecast evidence in the same style as the baseline report bundle
