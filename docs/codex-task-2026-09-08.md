# Codex task: first measurements for ai-ran-kpi-forecasting (2026-09-08)

You are working in `~/github/ai-ran-kpi-forecasting` (public, `main`, clean).
Read `AGENTS.md`, `docs/handoff-2026-09-08.md` and `docs/AI_RAN_INTEGRATION.md`
first. The Evidence Commit Rule and the claim boundary in `AGENTS.md` apply
to every step. Nothing is written into the README or the portal that does not
resolve to a committed artifact.

## Ground truth

- ONNX exports are done: `models/exports/{ridge_linear,gradient_boosting,mlp}_prb_dl_util.onnx`,
  input name `X`, shape `(None, 16)`, hashes and parity in
  `models/exports/manifest.json`. Regenerate with `make export-onnx` after
  `pip install -r requirements-onnx.txt`; the tests in
  `tests/test_onnx_export.py` must still pass.
- The benchmark harness lives in the security repo:
  `~/github/jetson-edge-ai-security/deploy/thor/run_benchmark.py`. Its
  Step 5b in that repo's `docs/codex-task-2026-09-08.md` adds a `--models-spec`
  option so it can take a JSON list of `{name, file, input_name, shape}`
  instead of its hardcoded pair. If that option does not exist yet when you
  reach Step 1, implement it there first (keep the output schema unchanged,
  add a test), commit it in that repo, then continue here.
- Jetson AGX Thor: `ssh jetsonthor`, venv at `~/edge-ids-bench/.venv`
  (numpy, psutil, onnxruntime-gpu 1.29.0). The CUDA provider fails on Thor
  (`cudaErrorNoKernelImageForDevice`); benchmark on the CPU provider and
  record it. Launch long runs with `setsid nohup ... < /dev/null`.
- Measured on the security repo's models today: default onnxruntime thread
  pool costs about 30 W of board power with pacing misses at 1000 events/s;
  one intra-op and one inter-op thread with spinning disabled removes both.
  Use the single-thread setting as the primary run here and the default as
  the comparison run.

## Work order

Commit each step with its evidence before starting the next. Report in the
format at the end.

### Step 1. Thor inference benchmark

- Copy the three ONNX files to Thor under `~/edge-ids-bench/models/airan/`.
  Write `deploy/thor/models_spec.json` in this repo listing them (name,
  file, `X`, `[1, 16]`), and commit it.
- Primary run: 300 s per tier at 10, 100, 1000 events/s,
  `--intra-op-threads 1 --inter-op-threads 1 --no-spin --provider cpu
  --idle-seconds 60`, output `reports/thor_benchmark/thor_benchmark.json`.
- Comparison run: 120 s at 100 and 1000 events/s with runtime defaults,
  output `reports/thor_benchmark/default_threads.json`; then run the
  security repo's `deploy/thor/compare_thread_runs.py` against the two and
  commit `reports/thor_benchmark/thread_comparison.json`.
- Copy back the `_tegrastats.jsonl` and `_run.log` sidecars (strip ANSI
  codes). Commit everything under `reports/thor_benchmark/`.
- README: add a "Measured on Jetson AGX Thor" subsection under the ONNX
  section with a table per model at 1000 events/s (p50, p95, p99, achieved
  rate, misses, RSS, VIN p50 vs idle), device and provider named, links to
  the files. State that this is inference cost only.

### Step 2. Telecom Italia MI benchmark (the accuracy gap)

- Download the "Telecommunications - SMS, Call, Internet - MI" dataset from
  Harvard Dataverse (verify the DOI and license on the landing page before
  downloading; record both in the artifact). Place the CSV or text files
  under `data/telecom_italia_mi/` (gitignored; add the ignore rule if
  missing). Record file names, sizes, SHA-256 and row counts in
  `reports/forecast_examples/telecom_italia_mi/dataset.json`.
- Confirm `load_telecom_italia_mi` parses the real files; fix the loader if
  the column layout differs from the fixture and add a test with a real
  three-line excerpt.
- Run `make run-telecom` (hourly aggregate, `internet_traffic`, horizon 24)
  for at least three cells of different activity levels. The split must be
  time-ordered; state the split point. Commit metrics, forecast CSVs and
  plots under `reports/forecast_examples/telecom_italia_mi/<cell>/`, and a
  `summary.json` with per-cell RMSE, MAE, MAPE for all three models plus a
  naive last-value baseline.
- README and portal: replace "Benchmark-ready: pending local public dataset
  files" with the measured table, dataset DOI, split rule, and the baseline.
  Keep the synthetic-sample table separate and labelled as such.

### Step 3. Portal and evidence page

- `make portal` and `make publish` after each step so `reports/index.html`
  and `reports/dashboard.html` carry the new sections. Every number links
  to its JSON or CSV. Add a "What is not established" list: no deployed
  rApp, no live RIC, no operator data, no GPU provider on this device.
- Add the ONNX-parity, Thor and Telecom Italia rows to
  `reports/README.md`.

### Step 4. Site case study section

- In `~/github/obiedeh.github.io`, the security case study page
  (`jetson-edge-ai-security.html`, created in that repo's Step 6) gets an
  "AI-RAN KPI forecasting" section. It carries: the pattern in two
  sentences, the Thor inference table from Step 1, the Telecom Italia
  result from Step 2 if it exists by then or "unmeasured" if not, and links
  to this repo's portal and artifacts. Do not import security-repo numbers
  into this section or vice versa.

### Step 5. CI

- Add `pip install -r requirements-onnx.txt` to the CI workflow so the
  ONNX parity tests run rather than skip. Keep the run under a few minutes.

## Checks before every commit

```bash
ruff check src tests scripts
python -m pytest -q
```

Citation check: every path linked from `README.md`, `docs/*.md`,
`reports/README.md` and the portal HTML must be tracked
(`git ls-files --error-unmatch <path>`). Exposure check on new files: no
home paths, LAN addresses or credentials.

## Boundaries

- Advisory outputs only; no RAN control, no live RIC connection.
- Never present sample-data metrics as forecast accuracy.
- Never pool Thor numbers from this repo with the security repo's in one
  figure without labelling each.
- If a run fails, commit the failure record and say so.
- Do not push; commit locally and report.

## Report format after each step

1. Files changed and why
2. Commands run and exit status
3. Artifacts committed, with paths and commit SHA
4. Numbers that changed in README or portal, old and new, with source path
5. Tests run, tests not run and why
6. Risks, open questions, what a claim-boundary review should look at
