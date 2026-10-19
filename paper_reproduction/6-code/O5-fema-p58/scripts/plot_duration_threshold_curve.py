"""Plot mean cumulative acceleration-exceedance duration curves for O5."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


G = 9.80665
REPO_ROOT = Path(__file__).resolve().parents[3]
O5_DIR = REPO_ROOT / "6-code" / "O5-fema-p58"
DEFAULT_DEMAND_DIR = O5_DIR / "results" / "multistructure" / "demands"
DEFAULT_FIG_DIR = O5_DIR / "figures" / "multistructure_revision"
DEFAULT_MANUSCRIPT_FIG_DIR = O5_DIR / "figures" / "paper_exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-dir", type=Path, default=DEFAULT_DEMAND_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--manuscript-fig-dir", type=Path, default=DEFAULT_MANUSCRIPT_FIG_DIR)
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--threshold-min-g", type=float, default=0.02)
    parser.add_argument("--threshold-max-g", type=float, default=2.00)
    parser.add_argument("--threshold-step-g", type=float, default=0.02)
    return parser.parse_args()


def compute_mean_duration_curve(
    history_path: Path,
    thresholds_g: np.ndarray,
    dt_s: float,
    chunk_size: int,
) -> np.ndarray:
    history = np.load(history_path, mmap_mode="r")
    if history.ndim == 3 and history.shape[1] == 1:
        n_records = history.shape[0]
    elif history.ndim == 2:
        n_records = history.shape[0]
    else:
        raise ValueError(f"Expected [record,time] or [record,1,time], got {history.shape}")

    edges = np.concatenate(([0.0], thresholds_g, [np.inf]))
    exceed_counts = np.zeros(len(thresholds_g), dtype=np.float64)
    for start in range(0, n_records, chunk_size):
        stop = min(start + chunk_size, n_records)
        chunk = np.asarray(history[start:stop], dtype=np.float64)
        if chunk.ndim == 3:
            chunk = chunk[:, 0, :]
        hist, _ = np.histogram(np.abs(chunk).ravel() / G, bins=edges)
        reverse_cumulative = np.cumsum(hist[::-1])[::-1]
        exceed_counts += reverse_cumulative[1 : 1 + len(thresholds_g)]

    return exceed_counts * dt_s / n_records


def write_curve_csv(path: Path, thresholds_g: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold_g", *curves.keys()])
        for idx, threshold in enumerate(thresholds_g):
            writer.writerow([f"{threshold:.4f}", *[f"{values[idx]:.8f}" for values in curves.values()]])


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
        }
    )


def plot_curves(thresholds_g: np.ndarray, curves: dict[str, np.ndarray], fig_dir: Path) -> None:
    configure_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    colors = {"FEM": "#1f1f1f", "FNO": "#0072B2"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, structure in zip(axes, ("F-10", "FW-10")):
        for source, linestyle in (("FEM", "-"), ("FNO", "--")):
            key = f"{structure} {source}"
            ax.plot(
                thresholds_g,
                curves[key],
                color=colors[source],
                linestyle=linestyle,
                linewidth=1.35,
                label=source,
            )
        ax.set_title(structure)
        ax.set_xlabel(r"Threshold, $a_0$ (g)")
        ax.set_xlim(thresholds_g[0], thresholds_g[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, color="#d8d8d8", linewidth=0.45, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"Mean $\tau(a_0)$ (s)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.6)

    fig.savefig(fig_dir / "fig_o5_duration_threshold_curve.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_o5_duration_threshold_curve.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    thresholds_g = np.round(
        np.arange(
            args.threshold_min_g,
            args.threshold_max_g + args.threshold_step_g * 0.5,
            args.threshold_step_g,
        ),
        6,
    )
    cases = {
        "F-10 FEM": args.demand_dir / "F10_fem_roof_history_ms2.npy",
        "F-10 FNO": args.demand_dir / "F10_fno_roof_history_ms2.npy",
        "FW-10 FEM": args.demand_dir / "FW10_fem_roof_history_ms2.npy",
        "FW-10 FNO": args.demand_dir / "FW10_fno_roof_history_ms2.npy",
    }
    curves = {
        label: compute_mean_duration_curve(path, thresholds_g, args.dt_s, args.chunk_size)
        for label, path in cases.items()
    }
    csv_path = args.fig_dir / "fig_o5_duration_threshold_curve.csv"
    write_curve_csv(csv_path, thresholds_g, curves)
    plot_curves(thresholds_g, curves, args.fig_dir)

    args.manuscript_fig_dir.mkdir(parents=True, exist_ok=True)
    manuscript_pdf = args.manuscript_fig_dir / "ES_o5_duration_threshold_curve.pdf"
    manuscript_pdf.write_bytes((args.fig_dir / "fig_o5_duration_threshold_curve.pdf").read_bytes())
    print(manuscript_pdf)


if __name__ == "__main__":
    main()
