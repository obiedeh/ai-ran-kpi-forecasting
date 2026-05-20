"""Three-model head-to-head comparison on the sample KPI dataset.

Runs the same forecast pipeline three times — Ridge, GradientBoosting, MLP —
on the same KPI (``prb_dl_util``), the same cell (``CELL_001``), the same
non-shuffled temporal split, and the same horizon. Writes per-model
artifacts plus a side-by-side comparison table and overlay plot.

The point of the comparison is to surface "which family wins on this KPI
on this dataset" honestly rather than cherry-picking a single model. All
three are scikit-learn / NumPy — zero new dependencies.

Run::

    python scripts/run_model_comparison.py \\
        --data data/ran_kpi_sample.csv \\
        --output-dir reports/model_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ai_ran_kpi_forecasting.forecast import run_forecast_pipeline  # noqa: E402
from ai_ran_kpi_forecasting.models import MODEL_NAMES  # noqa: E402


def _run_one_model(
    model_name: str,
    data: Path,
    output_dir: Path,
    cell_id: str,
    kpi_col: str,
    horizon: int,
) -> dict[str, object]:
    """Run a single forecast pipeline and return a flat metrics row."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    result = run_forecast_pipeline(
        data=str(data),
        dataset_type="generic",
        cell_id=cell_id,
        kpi_col=kpi_col,
        horizon=horizon,
        model_name=model_name,
    )

    # Per-model artifacts.
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_name": result.model_name,
                "cell_id": result.cell_id,
                "target_col": result.target_col,
                **{k: float(v) for k, v in result.metrics.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result.holdout.to_csv(model_dir / "holdout.csv", index=False)
    result.forecast.to_csv(model_dir / "forecast.csv", index=False)

    row = {
        "model_name": result.model_name,
        "cell_id": result.cell_id,
        "target_col": result.target_col,
        "rmse": float(result.metrics["rmse"]),
        "mae": float(result.metrics["mae"]),
        "mape": float(result.metrics["mape"]),
    }
    return row


def _write_comparison_table(
    rows: list[dict[str, object]], output_dir: Path
) -> tuple[Path, Path]:
    csv_path = output_dir / "comparison_metrics.csv"
    md_path = output_dir / "comparison_metrics.md"
    fieldnames = ["model_name", "cell_id", "target_col", "rmse", "mae", "mape"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Model comparison — same KPI, same temporal split, same horizon",
        "",
        f"KPI: `{rows[0]['target_col']}` · Cell: `{rows[0]['cell_id']}`",
        "",
        "| Model | RMSE | MAE | MAPE (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['model_name']}` | "
            f"{float(r['rmse']):.4f} | "
            f"{float(r['mae']):.4f} | "
            f"{float(r['mape']):.4f} |"
        )
    lines.append("")
    lines.append(
        "All three trained on the same non-shuffled temporal split. "
        "Zero new dependencies — all sklearn / NumPy. "
        "Reproduce: `make model-comparison`."
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _write_overlay_plot(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Plot all three holdouts overlaid on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"ridge_linear": "#294c7a", "gradient_boosting": "#a66b00", "mlp": "#136f63"}
    actual_drawn = False
    for r in rows:
        name = str(r["model_name"])
        df_path = output_dir / name / "holdout.csv"
        with df_path.open("r", encoding="utf-8") as f:
            data_rows = list(csv.DictReader(f))
        actual = [float(d["actual"]) for d in data_rows]
        pred = [float(d["prediction"]) for d in data_rows]
        idx = list(range(len(actual)))
        if not actual_drawn:
            ax.plot(idx, actual, color="#172033", linewidth=2.0, label="actual")
            actual_drawn = True
        ax.plot(
            idx,
            pred,
            color=colors.get(name, "#888"),
            linewidth=1.4,
            linestyle="--",
            label=f"{name} (MAE={float(r['mae']):.3f})",
        )
    ax.set_xlabel("Holdout sample index (temporal order)")
    ax.set_ylabel(f"{rows[0]['target_col']}")
    ax.set_title("Three-model comparison on the same temporal holdout")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out_path = output_dir / "comparison_overlay.svg"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-model KPI forecast comparison.")
    parser.add_argument("--data", type=Path, default=Path("data/ran_kpi_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_comparison"))
    parser.add_argument("--cell-id", type=str, default="CELL_001")
    parser.add_argument("--kpi-col", type=str, default="prb_dl_util")
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        print(f"--- {model_name} ---")
        row = _run_one_model(
            model_name=model_name,
            data=args.data,
            output_dir=args.output_dir,
            cell_id=args.cell_id,
            kpi_col=args.kpi_col,
            horizon=args.horizon,
        )
        rows.append(row)
        print(
            f"  RMSE = {row['rmse']:.4f} | "
            f"MAE = {row['mae']:.4f} | "
            f"MAPE = {row['mape']:.4f}%"
        )

    csv_path, md_path = _write_comparison_table(rows, args.output_dir)
    svg_path = _write_overlay_plot(rows, args.output_dir)
    print(f"Wrote comparison CSV:    {csv_path}")
    print(f"Wrote comparison MD:     {md_path}")
    print(f"Wrote comparison plot:   {svg_path}")


if __name__ == "__main__":
    main()
