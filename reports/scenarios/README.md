# Scenario Dashboards

This directory stores telecom-style before/after scenario evidence.

The default demo creates:

- `latest/baseline/`: baseline telemetry and forecast evidence
- `latest/congestion/`: congestion-shock telemetry and forecast evidence
- `latest/dashboard/`: a combined HTML dashboard and summary
- `latest/scenario_metadata.json`: generation parameters and artifact paths

The goal is to resemble an actual wireless operations review pack:

- cell-level KPI traces
- congestion impact callouts
- throughput and latency behavior around the shock window
- forecast evidence in the same style as the baseline report bundle
