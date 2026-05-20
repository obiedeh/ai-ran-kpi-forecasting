# AGENTS.md

**AI-RAN KPI Forecasting — Non-RT RIC rApp pattern.** Operating instructions for AI coding agents working in this repository.

This is **AI-for-RAN at the operational layer** — the Non-RT RIC rApp pattern for KPI forecasting on RAN telemetry. Companion to [`wireless-link-intelligence-system`](https://github.com/obiedeh/wireless-link-intelligence-system), which targets AI-for-RAN at the PHY layer. Together they cover both tiers of AI-for-RAN.

**This repo is the rApp *pattern*, not a deployed rApp.** Every claim must be backed by a committed artifact: schemas, manifest, code, measured reports. See `docs/AI_RAN_INTEGRATION.md` for the explicit "pattern not deployment" boundary.

For shared engineering standards and skill definitions, read:

```text
https://github.com/obiedeh/obiedeh/tree/main/agent-skills
```

---

# Codex Role

Use Codex for:

- patches to `src/ai_ran_kpi_forecasting/` modules
- adding CLI commands following patterns in `cli.py`
- test generation for `tests/`
- updating report templates and artifact structures
- dependency and config changes

Do not use Codex for:

- changing forecast model architecture without Claude Code review
- adding live RAN integration or network control logic
- producing credibility claims not backed by evidence in `reports/`
- adding MCP servers or external tool integrations without explicit instruction

Default workflow:

```text
Claude Code = architecture review, skill selection, planning
Codex       = implement, patch, test
Claude Code = production-readiness check before merge
```

---

# Skill Selection

- `production-architecture-reviewer`: changes to pipeline structure, service boundaries, or module responsibilities
- `repo-hardening-refactor`: dead code, stale docs, duplicate helpers, unnecessary abstractions
- `runtime-stability-debugger`: forecast pipeline latency, memory pressure, long-running batch jobs
- `ai-ran-workflow-generator`: telecom KPI workflows, scenario generation, threshold logic, rollback, operational SOPs
- `observability-generator`: metrics, structured logging, report artifact coverage, alert thresholds
- `edge-ai-deployer`: Dockerfile, containerised CI runs, edge deployment of the forecasting CLI

---

# Project Structure

```text
src/ai_ran_kpi_forecasting/   # Core package — data, features, models, forecast, metrics, reports, viz
tests/                         # Pytest suite
data/                          # Sample and synthetic input CSVs
configs/                       # Sample YAML config
reports/                       # Generated forecast artifacts and scenario dashboards
docs/                          # Architecture and design docs
```

---

# Credibility Rules

This repo tracks its own credibility in `PROJECT_STATUS.md`. Codex must not:

- mark roadmap items complete without backing evidence in `reports/`
- add capability claims to `README.md` without a working code path
- generate report artifacts that are not reproducible from `make run-sample` or CI

---

# Anti-Bloat Rules

Do not create:

- new modules without a clear pipeline role
- duplicate feature engineering helpers
- speculative Phase 3/4 stubs before Phase 2 is fully complete
- notebook-only workflows

Every new file must justify at least one of:

- operational necessity
- reliability improvement
- observability improvement
- deployment-readiness improvement

---

# Output Format

At the end of each task, Codex should report:

1. Files changed and why
2. Tests run
3. Tests not run and why
4. Risks or follow-up work
5. Whether Claude Code review is needed
