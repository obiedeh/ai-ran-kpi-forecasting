#!/usr/bin/env python3
"""Charts for the two measured benchmarks, drawn from their committed JSON.

* ``reports/forecast_examples/telecom_italia_mi/rmse_two_windows.svg``:
  hold-out RMSE per square for the three models and the naive baseline,
  holiday window beside the ordinary-weeks window, each square on its own
  axis because the squares differ by two orders of magnitude.
* ``reports/thor_benchmark/latency_power.svg``: p95 latency at 1000 events/s
  per model and board power, default thread pool beside one thread without
  spinning, from ``thread_comparison.json``.

Every value is read from the artifacts; nothing is typed here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TI = ROOT / "reports" / "forecast_examples" / "telecom_italia_mi"
TI_PRE = ROOT / "reports" / "forecast_examples" / "telecom_italia_mi_preholiday"
THOR = ROOT / "reports" / "thor_benchmark"

MODEL_LABELS = {"ridge_linear": "Ridge", "gradient_boosting": "Gradient boosting", "mlp": "MLP",
                "naive_last_value": "Naive last value", "seasonal_naive_24h": "Seasonal naive 24 h"}
COLORS = {"ridge_linear": "#38bdf8", "gradient_boosting": "#22c55e", "mlp": "#a78bfa",
          "naive_last_value": "#f59e0b", "seasonal_naive_24h": "#fbbf24"}


def _style(fig, axes):
    fig.patch.set_facecolor("#121a2c")
    for ax in axes:
        ax.set_facecolor("#121a2c")
        for spine in ax.spines.values():
            spine.set_color("#293653")
        ax.tick_params(colors="#9fb0ca", labelsize=8)
        ax.yaxis.label.set_color("#9fb0ca")
        ax.xaxis.label.set_color("#9fb0ca")
        ax.title.set_color("#eef4ff")
        ax.grid(axis="y", color="#293653", linewidth=0.5)
        ax.set_axisbelow(True)


def telecom_chart(out: Path) -> Path:
    full = json.loads((TI / "summary.json").read_text())
    pre = json.loads((TI_PRE / "summary.json").read_text())
    keys = ["ridge_linear", "gradient_boosting", "mlp", "naive_last_value"]
    cells = list(full["cells"].keys())
    fig, axes = plt.subplots(1, len(cells), figsize=(11, 3.8))
    for ax, cid in zip(axes, cells, strict=False):
        c_full, c_pre = full["cells"][cid], pre["cells"][cid]

        def val(c, k):
            return c["models"][k]["rmse"] if k in c["models"] else c["baselines"][k]["rmse"]

        width = 0.38
        xs = range(len(keys))
        ax.bar([x - width / 2 for x in xs], [val(c_full, k) for k in keys], width,
               color=[COLORS[k] for k in keys], alpha=0.45, label="holiday hold-out")
        ax.bar([x + width / 2 for x in xs], [val(c_pre, k) for k in keys], width,
               color=[COLORS[k] for k in keys], label="ordinary weeks")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([MODEL_LABELS[k].replace(" ", "\n") for k in keys], fontsize=7)
        ax.set_title(f"square {cid} ({c_full['level']})", fontsize=10)
        ax.set_ylabel("hold-out RMSE" if cid == cells[0] else "")
    _style(fig, axes)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#9fb0ca", alpha=0.45),
               plt.Rectangle((0, 0), 1, 1, color="#9fb0ca")]
    fig.legend(handles, [f"holiday hold-out (test from {full['cells'][cells[0]]['split']['first_test_hour'][:10]})",
                         f"ordinary weeks (series cut {pre['window']['until'][:10]})"],
               loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False,
               labelcolor="#9fb0ca")
    fig.suptitle("Telecom Italia MI, hourly internet_traffic: models against the naive baseline, two hold-out windows",
                 color="#eef4ff", fontsize=10, y=1.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def thor_chart(out: Path) -> Path:
    comp = json.loads((THOR / "thread_comparison.json").read_text())
    rows = [r for r in comp["rows"] if r["target_rps"] == 1000.0]
    models = [r["model"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    xs = range(len(models))
    w = 0.38
    ax1.bar([x - w / 2 for x in xs], [r["baseline"]["p95_ms"] for r in rows], w, color="#f59e0b", label="default thread pool")
    ax1.bar([x + w / 2 for x in xs], [r["variant"]["p95_ms"] for r in rows], w, color="#38bdf8", label="one thread, no spin")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylabel("p95 latency, ms, 1000 events/s")
    ax1.set_title("Inference latency", fontsize=10)
    ax2.bar([x - w / 2 for x in xs], [r["baseline"]["vin_p50_mw"] / 1000 for r in rows], w, color="#f59e0b", label="default thread pool")
    ax2.bar([x + w / 2 for x in xs], [r["variant"]["vin_p50_mw"] / 1000 for r in rows], w, color="#38bdf8", label="one thread, no spin")
    ax2.axhline(comp["baseline"]["idle_vin_p50_mw"] / 1000, color="#9fb0ca", linewidth=0.8, linestyle="--")
    ax2.text(len(models) - 0.5, comp["baseline"]["idle_vin_p50_mw"] / 1000 + 0.6, "idle", color="#9fb0ca", fontsize=7, ha="right")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(models, fontsize=8)
    ax2.set_ylabel("board input power VIN p50, W")
    ax2.set_title("Board power", fontsize=10)
    ax2.legend(fontsize=7, frameon=False, labelcolor="#9fb0ca")
    for ax, key in ((ax1, "deadline_misses"), ):
        for i, r in enumerate(rows):
            ax.text(i - w / 2, r["baseline"]["p95_ms"], f"{r['baseline'][key]:,} misses", color="#9fb0ca", fontsize=6, ha="center", va="bottom")
            ax.text(i + w / 2, r["variant"]["p95_ms"], f"{r['variant'][key]:,}", color="#9fb0ca", fontsize=6, ha="center", va="bottom")
    _style(fig, (ax1, ax2))
    fig.suptitle("Jetson AGX Thor, CPU execution provider, 2026-09-09: the three ONNX forecasters at 1000 events/s",
                 color="#eef4ff", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main() -> int:
    print(telecom_chart(TI / "rmse_two_windows.svg"))
    print(thor_chart(THOR / "latency_power.svg"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
