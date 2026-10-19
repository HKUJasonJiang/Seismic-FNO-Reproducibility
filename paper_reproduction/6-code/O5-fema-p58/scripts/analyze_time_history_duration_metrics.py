"""Compute time-history duration metrics for O5 downstream assessment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


G = 9.80665
REPO_ROOT = Path(__file__).resolve().parents[3]
O5_DIR = REPO_ROOT / "6-code" / "O5-fema-p58"
DEFAULT_DEMAND_DIR = O5_DIR / "results" / "multistructure" / "demands"
DEFAULT_OUTPUT_DIR = O5_DIR / "results" / "multistructure" / "duration_metrics"
DEFAULT_FIG_DIR = O5_DIR / "figures" / "multistructure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-dir", type=Path, default=DEFAULT_DEMAND_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--thresholds-g", nargs="+", type=float, default=[0.1, 0.2, 0.4, 1.0])
    parser.add_argument("--chunk-size", type=int, default=1024)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bracketed_duration(exceed: np.ndarray, dt_s: float) -> np.ndarray:
    any_exceed = exceed.any(axis=1)
    first = np.argmax(exceed, axis=1)
    last = exceed.shape[1] - 1 - np.argmax(exceed[:, ::-1], axis=1)
    out = np.zeros(exceed.shape[0], dtype=np.float64)
    out[any_exceed] = (last[any_exceed] - first[any_exceed]) * dt_s
    return out


def analyze_history(
    history_path: Path,
    thresholds_g: list[float],
    dt_s: float,
    chunk_size: int,
) -> tuple[list[dict], list[dict]]:
    history = np.load(history_path, mmap_mode="r")
    if history.ndim == 3 and history.shape[1] == 1:
        n_records, _, n_steps = history.shape
    elif history.ndim == 2:
        n_records, n_steps = history.shape
    else:
        raise ValueError(f"Expected [record,time] or [record,1,time], got {history.shape}")
    record_rows: list[dict] = []
    cav_values = np.empty(n_records, dtype=np.float64)
    arias_values = np.empty(n_records, dtype=np.float64)
    cum_by_threshold = {thr: np.empty(n_records, dtype=np.float64) for thr in thresholds_g}
    bracket_by_threshold = {thr: np.empty(n_records, dtype=np.float64) for thr in thresholds_g}
    for start in range(0, n_records, chunk_size):
        stop = min(start + chunk_size, n_records)
        chunk = np.asarray(history[start:stop], dtype=np.float64)
        if chunk.ndim == 3:
            chunk = chunk[:, 0, :]
        abs_g = np.abs(chunk) / G
        cav_values[start:stop] = np.sum(abs_g, axis=1) * dt_s
        arias_values[start:stop] = np.pi / (2.0 * G) * np.sum(chunk**2, axis=1) * dt_s
        for thr in thresholds_g:
            exceed = abs_g >= thr
            cum_by_threshold[thr][start:stop] = np.sum(exceed, axis=1) * dt_s
            bracket_by_threshold[thr][start:stop] = bracketed_duration(exceed, dt_s)
    for idx in range(n_records):
        base = {
            "record_index": idx,
            "floor_cav_g_s": float(cav_values[idx]),
            "floor_arias_intensity_m_s": float(arias_values[idx]),
        }
        for thr in thresholds_g:
            base[f"cumulative_duration_ge_{thr:g}g_s"] = float(cum_by_threshold[thr][idx])
            base[f"bracketed_duration_ge_{thr:g}g_s"] = float(bracket_by_threshold[thr][idx])
        record_rows.append(base)
    summary_rows = []
    summary_rows.append(
        {
            "metric": "floor_cav_g_s",
            "threshold_g": "",
            "mean": float(np.mean(cav_values)),
            "p50": float(np.quantile(cav_values, 0.50)),
            "p90": float(np.quantile(cav_values, 0.90)),
            "p95": float(np.quantile(cav_values, 0.95)),
        }
    )
    summary_rows.append(
        {
            "metric": "floor_arias_intensity_m_s",
            "threshold_g": "",
            "mean": float(np.mean(arias_values)),
            "p50": float(np.quantile(arias_values, 0.50)),
            "p90": float(np.quantile(arias_values, 0.90)),
            "p95": float(np.quantile(arias_values, 0.95)),
        }
    )
    for thr in thresholds_g:
        for metric, values in (
            ("cumulative_duration_s", cum_by_threshold[thr]),
            ("bracketed_duration_s", bracket_by_threshold[thr]),
        ):
            summary_rows.append(
                {
                    "metric": metric,
                    "threshold_g": thr,
                    "mean": float(np.mean(values)),
                    "p50": float(np.quantile(values, 0.50)),
                    "p90": float(np.quantile(values, 0.90)),
                    "p95": float(np.quantile(values, 0.95)),
                }
            )
    return record_rows, summary_rows


def plot_duration_summary(summary_rows: list[dict], fig_dir: Path) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row for row in summary_rows
        if row["metric"] == "cumulative_duration_s" and row["threshold_g"] != ""
    ]
    structures = sorted({row["paper_label"] for row in rows})
    thresholds = sorted({float(row["threshold_g"]) for row in rows})
    fig, axes = plt.subplots(1, len(structures), figsize=(6.5 * len(structures), 4.4), sharey=True)
    if len(structures) == 1:
        axes = [axes]
    colors = {"FEM": "#1f77b4", "FNO": "#ff3b20"}
    width = 0.35
    for ax, structure in zip(axes, structures):
        for offset, source in ((-width / 2, "FEM"), (width / 2, "FNO")):
            values = []
            for thr in thresholds:
                match = [
                    row for row in rows
                    if row["paper_label"] == structure and row["source"] == source and float(row["threshold_g"]) == thr
                ][0]
                values.append(match["p90"])
            x = np.arange(len(thresholds)) + offset
            ax.bar(x, values, width=width, label=source, color=colors[source], alpha=0.9)
        ax.set_xticks(np.arange(len(thresholds)))
        ax.set_xticklabels([f"{thr:g}g" for thr in thresholds])
        ax.set_xlabel("Floor acceleration threshold")
        ax.set_title(f"{structure}: p90 cumulative exceedance duration")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Duration (s)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    png = fig_dir / "fig_o5_time_history_duration_metrics.png"
    pdf = fig_dir / "fig_o5_time_history_duration_metrics.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png.resolve()), "pdf": str(pdf.resolve())}


def main() -> None:
    args = parse_args()
    summary = load_json(args.demand_dir / "o5_demand_export_summary.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summary_rows: list[dict] = []
    outputs = {}
    for item in summary["summaries"]:
        structure_id = item["structure_id"]
        paper_label = item["paper_label"]
        for source, key in (("FEM", "fem_history_path"), ("FNO", "fno_history_path")):
            record_rows, source_summary = analyze_history(
                args.demand_dir / item[key].replace("\\", "/").split("/")[-1],
                args.thresholds_g,
                float(item["dt_s"]),
                args.chunk_size,
            )
            for row in record_rows:
                row.update({"structure_id": structure_id, "paper_label": paper_label, "source": source})
            for row in source_summary:
                row.update({"structure_id": structure_id, "paper_label": paper_label, "source": source})
            record_path = args.output_dir / f"{structure_id}_{source.lower()}_time_history_duration_metrics.csv"
            write_csv(record_path, record_rows)
            all_summary_rows.extend(source_summary)
            outputs[f"{structure_id}_{source}_record_metrics"] = str(record_path.resolve())
    summary_path = args.output_dir / "o5_time_history_duration_summary.csv"
    write_csv(summary_path, all_summary_rows)
    fig_paths = plot_duration_summary(all_summary_rows, args.fig_dir)
    meta = {
        "thresholds_g": args.thresholds_g,
        "summary_csv": str(summary_path.resolve()),
        "figures": fig_paths,
        "outputs": outputs,
    }
    meta_path = args.output_dir / "o5_time_history_duration_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta_path)


if __name__ == "__main__":
    main()
