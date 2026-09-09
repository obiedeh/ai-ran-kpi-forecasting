#!/usr/bin/env python3
"""Telecom Italia MI forecast benchmark: three cells, three models, two naive baselines.

Reads the raw Dataverse day files under ``--data`` (see
``scripts/fetch_telecom_italia_mi.py``), aggregates each square to hourly
``internet_traffic``, selects three cells by activity level with a stated rule,
and runs the repository's forecast pipeline per cell for ``ridge_linear``,
``gradient_boosting`` and ``mlp`` with the same time-ordered split. Two naive
baselines are scored on the identical hold-out rows: last value (t-1) and
seasonal last value (t-24 h).

Writes under ``--output-dir``:

* ``dataset.json``      files, bytes, SHA-256, Dataverse MD5, row counts, DOI, licence
* ``cell_activity.csv`` per-square total internet_traffic over the period
* ``<cell>/<model>/``   the standard report bundle (metrics.json, hold-out CSV, forecast, plots)
* ``<cell>/baselines.json``
* ``summary.json``      per-cell, per-model RMSE/MAE/MAPE, split points, selection rule, provenance

Only ``internet_traffic`` enters the feature pipeline (time features plus its own
lags); the other activity columns are not used as covariates, because they
would not be available ahead of time in a forecasting setting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_ran_kpi_forecasting.data import _read_telecom_italia_file, load_telecom_italia_mi  # noqa: E402
from ai_ran_kpi_forecasting.features import add_lag_features, build_feature_matrix, engineer_time_features  # noqa: E402
from ai_ran_kpi_forecasting.forecast import parse_lags, run_forecast_pipeline, temporal_train_test_split  # noqa: E402
from ai_ran_kpi_forecasting.metrics import regression_metrics  # noqa: E402
from ai_ran_kpi_forecasting.reports import write_report_bundle  # noqa: E402

DOI = "doi:10.7910/DVN/EGZHFV"
LICENSE = "ODbL 1.0 (Open Database License), per the Dataverse terms of use"
MODELS = ["ridge_linear", "gradient_boosting", "mlp"]
KPI = "internet_traffic"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            n += block.count(b"\n")
    return n


def dataverse_md5s() -> dict[str, str]:
    """Dataverse-published MD5 per filename, or an empty dict when offline."""
    try:
        import urllib.request

        url = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={DOI}"
        req = urllib.request.Request(url, headers={"User-Agent": "ai-ran-kpi-forecasting/benchmark"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            files = json.load(resp)["data"]["latestVersion"]["files"]
        return {f["dataFile"]["filename"]: f["dataFile"]["md5"] for f in files}
    except Exception:  # noqa: BLE001 - offline is acceptable; recorded as unverified
        return {}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe_dataset(files: list[Path]) -> dict:
    published = dataverse_md5s()
    records = []
    for f in files:
        md5 = md5_of(f)
        records.append({
            "filename": f.name,
            "bytes": f.stat().st_size,
            "sha256": sha256_of(f),
            "md5": md5,
            "md5_matches_dataverse": (published.get(f.name) == md5) if published else None,
            "rows": count_lines(f),
        })
    return {
        "doi": DOI,
        "landing_page": "https://doi.org/10.7910/DVN/EGZHFV",
        "title": "Telecommunications - SMS, Call, Internet - MI",
        "license": LICENSE,
        "guestbook": "Dataverse guestbook 'Privacy risk assessment' (email) answered per file by the fetch script",
        "format": "tab separated, no header: square_id, time_interval_ms, country_code, sms_in, sms_out, call_in, call_out, internet_traffic",
        "n_files": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
        "total_rows": sum(r["rows"] for r in records),
        "first_day": records[0]["filename"][-14:-4] if records else None,
        "last_day": records[-1]["filename"][-14:-4] if records else None,
        "dataverse_md5_checked": bool(published),
        "files": records,
    }


def cell_activity(files: list[Path]) -> pd.Series:
    totals: dict[int, float] = {}
    for f in files:
        day = _read_telecom_italia_file(f)
        s = day.groupby("cell_id")[KPI].sum()
        for cell, v in s.items():
            totals[int(cell)] = totals.get(int(cell), 0.0) + float(v)
    return pd.Series(totals, name=KPI).sort_values(ascending=False)


def select_cells(activity: pd.Series) -> dict[str, dict]:
    """High = largest total; mid = nearest the median; low = nearest the 10th percentile."""
    med, p10 = float(activity.median()), float(activity.quantile(0.10))
    high = int(activity.index[0])
    mid = int((activity - med).abs().idxmin())
    low = int((activity - p10).abs().idxmin())
    return {
        "high": {"cell_id": high, "total": float(activity[high]), "rule": "largest total internet_traffic"},
        "mid": {"cell_id": mid, "total": float(activity[mid]), "rule": f"total nearest the median ({med:.1f})"},
        "low": {"cell_id": low, "total": float(activity[low]), "rule": f"total nearest the 10th percentile ({p10:.1f})"},
    }


def naive_baselines(df_cell: pd.DataFrame, lags: list[int], test_size: float) -> dict:
    """Score last-value and seasonal (24 h) baselines on the pipeline's exact hold-out rows."""
    df_fe = engineer_time_features(df_cell, timestamp_col="timestamp")
    df_fe[f"{KPI}_lag_24"] = df_fe[KPI].shift(24)
    df_fe = add_lag_features(df_fe, target_col=KPI, lags=lags)
    df_fe = df_fe.dropna().reset_index(drop=True)
    X, y = build_feature_matrix(df_fe, target_col=KPI, exclude_cols=["timestamp", "cell_id"])
    _, X_test, _, y_test = temporal_train_test_split(X, y, test_size=test_size)
    last = X_test[f"{KPI}_lag_1"].to_numpy()
    seasonal = X_test[f"{KPI}_lag_24"].to_numpy()
    return {
        "naive_last_value": regression_metrics(y_test, last),
        "seasonal_naive_24h": regression_metrics(y_test, seasonal),
        "n_test": int(len(y_test)),
        "note": "Same hold-out rows as the models, after the 24 h seasonal lag is available.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/telecom_italia_mi"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/forecast_examples/telecom_italia_mi"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--lags", default="1,2,3,6,12,24")
    ap.add_argument("--cells", default="", help="comma list of square ids to force instead of the selection rule")
    ap.add_argument("--skip-dataset-json", action="store_true", help="reuse an existing dataset.json")
    ap.add_argument("--until", default="", help="drop hours at or after this UTC timestamp, e.g. 2013-12-20, so the hold-out ends before it")
    args = ap.parse_args()

    t0 = time.time()
    files = sorted(p for p in args.data.iterdir() if p.suffix == ".txt")
    if not files:
        print(f"no .txt files under {args.data}", file=sys.stderr)
        return 1
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lags = parse_lags(args.lags)

    dataset_path = out / "dataset.json"
    if args.skip_dataset_json and dataset_path.exists():
        dataset = json.loads(dataset_path.read_text())
    else:
        print(f"[benchmark] hashing and counting {len(files)} files", flush=True)
        dataset = describe_dataset(files)
        dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    print(f"[benchmark] dataset: {dataset['n_files']} files, {dataset['total_rows']:,} rows, "
          f"{dataset['total_bytes'] / 1e9:.2f} GB, {dataset['first_day']}..{dataset['last_day']}", flush=True)

    print("[benchmark] per-cell activity pass", flush=True)
    t_act = time.time()
    activity = cell_activity(files)
    activity.rename_axis("cell_id").reset_index().to_csv(out / "cell_activity.csv", index=False)
    if args.cells:
        forced = [int(c) for c in args.cells.split(",")]
        selected = {f"cell_{c}": {"cell_id": c, "total": float(activity.get(c, 0.0)), "rule": "forced by --cells"} for c in forced}
    else:
        selected = select_cells(activity)
    print(f"[benchmark] cells: {selected} ({time.time() - t_act:.0f}s)", flush=True)

    cell_ids = {v["cell_id"] for v in selected.values()}
    print(f"[benchmark] loading hourly series for {sorted(cell_ids)}", flush=True)
    hourly = load_telecom_italia_mi(args.data, aggregate="hourly", cell_ids=cell_ids)
    window: dict = {"until": None, "hours_dropped": 0}
    if args.until:
        cutoff = pd.Timestamp(args.until, tz="UTC")
        before = len(hourly)
        hourly = hourly[hourly["timestamp"] < cutoff].reset_index(drop=True)
        window = {"until": str(cutoff), "hours_dropped": int(before - len(hourly)),
                  "reason": "series truncated so the time-ordered hold-out ends before the cutoff"}
        print(f"[benchmark] window: hours before {cutoff} ({window['hours_dropped']} rows dropped)", flush=True)

    results: dict[str, dict] = {}
    for level, info in selected.items():
        cid = info["cell_id"]
        df_cell = hourly[hourly["cell_id"] == cid][["timestamp", "cell_id", KPI]].reset_index(drop=True)
        cell_dir = out / str(cid)
        cell_dir.mkdir(parents=True, exist_ok=True)
        csv_path = cell_dir / f"hourly_{KPI}.csv"
        df_cell.to_csv(csv_path, index=False)
        n = len(df_cell) - max(lags)  # rows after lag features
        split_idx = max(1, min(int(n * (1.0 - args.test_size)), n - 1))
        split_ts = df_cell["timestamp"].iloc[max(lags) + split_idx]
        entry: dict = {
            "level": level, "cell_id": cid, "selection": info,
            "hours": int(len(df_cell)), "first_hour": str(df_cell["timestamp"].iloc[0]),
            "last_hour": str(df_cell["timestamp"].iloc[-1]),
            "split": {"rule": f"time ordered, first {int((1 - args.test_size) * 100)}% train, last {int(args.test_size * 100)}% test, after lag features",
                      "n_train": int(split_idx), "n_test": int(n - split_idx), "first_test_hour": str(split_ts)},
            "models": {},
        }
        for model in MODELS:
            t_m = time.time()
            result = run_forecast_pipeline(
                data=str(csv_path), dataset_type="generic", cell_id=str(cid), kpi_col=KPI,
                test_size=args.test_size, horizon=args.horizon, lags=lags, model_name=model,
            )
            write_report_bundle(result, cell_dir / model)
            entry["models"][model] = {**result.metrics, "fit_and_eval_s": round(time.time() - t_m, 2)}
            print(f"[benchmark] {level} cell {cid} {model}: rmse {result.metrics['rmse']:.3f} "
                  f"mae {result.metrics['mae']:.3f} mape {result.metrics['mape']:.2f}% ({time.time() - t_m:.0f}s)", flush=True)
        entry["baselines"] = naive_baselines(df_cell, lags, args.test_size)
        (cell_dir / "baselines.json").write_text(json.dumps(entry["baselines"], indent=2) + "\n")
        results[str(cid)] = entry

    summary = {
        "schema": "telecom-italia-mi-benchmark-v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {k: dataset[k] for k in ("doi", "license", "n_files", "total_rows", "first_day", "last_day")},
        "kpi": KPI,
        "aggregate": "hourly, summed over country codes and the six 10-minute intervals",
        "features": "calendar/cyclic time features and lags of the target only",
        "lags": lags,
        "horizon": args.horizon,
        "test_size": args.test_size,
        "window": window,
        "models": MODELS,
        "cells": results,
        "provenance": {
            "host": platform.node(), "python": platform.python_version(),
            "pandas": pd.__version__, "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                      check=False, cwd=REPO_ROOT).stdout.strip() or None,
            "wall_clock_s": round(time.time() - t0, 1),
        },
        "note": "Forecast accuracy on public telecom data; per-cell hold-out metrics with the naive baselines "
                "beside them. Not an operator deployment result.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[benchmark] -> {out / 'summary.json'} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
