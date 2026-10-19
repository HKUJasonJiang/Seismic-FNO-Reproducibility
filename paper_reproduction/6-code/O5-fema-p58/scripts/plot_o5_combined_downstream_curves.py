"""Create the combined O5 downstream Fig. 15 with duration and FRS panels."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
O5_DIR = REPO_ROOT / "6-code" / "O5-fema-p58"
FIG_DIR = O5_DIR / "figures" / "multistructure_revision"
TABLE_DIR = O5_DIR / "results" / "multistructure" / "chapter6_revision"
MANUSCRIPT_FIG_DIR = O5_DIR / "figures" / "paper_exports"
DURATION_CSV = FIG_DIR / "fig_o5_duration_threshold_curve.csv"
FRS_CURVE_CSV = TABLE_DIR / "o5_period_exceedance_curve.csv"

STRUCTURE_COLORS = {"F-10": "#0072B2", "FW-10": "#D55E00"}
SOURCE_STYLES = {"FEM": "-", "FNO": "--"}
SOURCE_LABELS = {"FEM": "NTHA", "FNO": "FNO"}
BANDS = [
    (0.05, 0.20, "0.05-0.2s"),
    (0.50, 1.00, "0.5-1s"),
    (1.50, 3.00, "1.5-3s"),
]


def load_duration_curves() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with DURATION_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    thresholds = np.array([float(row["threshold_g"]) for row in rows], dtype=float)
    curves = {
        key: np.array([float(row[key]) for row in rows], dtype=float)
        for key in rows[0].keys()
        if key != "threshold_g"
    }
    return thresholds, curves


def load_frs_curves() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with FRS_CURVE_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            structure = row["paper_label"]
            source = row["source"]
            grouped.setdefault((structure, source), []).append(
                (float(row["period_s"]), float(row["exceedance_percent"]))
            )
    curves = {}
    for key, values in grouped.items():
        values = sorted(values)
        curves[key] = (
            np.array([v[0] for v in values], dtype=float),
            np.array([v[1] for v in values], dtype=float),
        )
    return curves


def set_tick_font(ax: plt.Axes, tick_font: FontProperties) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(tick_font)


def style_axes(ax: plt.Axes, tick_font: FontProperties) -> None:
    ax.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    set_tick_font(ax, tick_font)


def main() -> None:
    text_font = FontProperties(family="Georgia", size=11.0)
    title_font = FontProperties(family="Georgia", size=11.0)
    tick_font = FontProperties(family="Times New Roman", size=11.0)
    band_font = FontProperties(family="Georgia", size=11.0)
    legend_font = FontProperties(family="Georgia", size=11.0)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    thresholds, duration_curves = load_duration_curves()
    frs_curves = load_frs_curves()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.95))
    ax = axes[0]
    for structure in ("F-10", "FW-10"):
        for source in ("FEM", "FNO"):
            key = f"{structure} {source}"
            ax.plot(
                thresholds,
                duration_curves[key],
                color=STRUCTURE_COLORS[structure],
                linestyle=SOURCE_STYLES[source],
                linewidth=1.35,
                label=f"{structure} {SOURCE_LABELS[source]}",
            )
    ax.set_xlim(thresholds[0], 1.4)
    ax.set_ylim(0.0, 34.0)
    ax.set_xlabel(r"Threshold, $a_0$ (g)", fontproperties=text_font)
    ax.set_ylabel(r"Mean $\tau(a_0)$ (s)", fontproperties=text_font)
    style_axes(ax, tick_font)

    ax = axes[1]
    for xmin, xmax, label in BANDS:
        ax.axvspan(xmin, xmax, color="#DDDDDD", alpha=0.30, zorder=0)
        ax.text(
            0.5 * (xmin + xmax),
            96.0,
            label,
            ha="center",
            va="top",
            rotation=90 if xmax - xmin < 0.25 else 0,
            fontproperties=band_font,
        )
    for structure in ("F-10", "FW-10"):
        for source in ("FEM", "FNO"):
            periods, exceedance = frs_curves[(structure, source)]
            ax.plot(
                periods,
                exceedance,
                color=STRUCTURE_COLORS[structure],
                linestyle=SOURCE_STYLES[source],
                linewidth=1.35,
                label=f"{structure} {SOURCE_LABELS[source]}",
            )
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Equipment period T (s)", fontproperties=text_font)
    ax.set_ylabel("Samples exceeding 1g (%)", fontproperties=text_font)
    style_axes(ax, tick_font)

    axes[0].text(
        0.5,
        -0.31,
        "(a) Cumulative exceedance duration",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontproperties=title_font,
    )
    axes[1].text(
        0.5,
        -0.31,
        "(b) Equipment-period screening",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontproperties=title_font,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        prop=legend_font,
    )
    fig.tight_layout(w_pad=1.8, rect=[0.0, 0.08, 1.0, 0.94])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig_o5_downstream_duration_frs_combined.pdf"
    png = FIG_DIR / "fig_o5_downstream_duration_frs_combined.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    manuscript_pdf = MANUSCRIPT_FIG_DIR / "ES_o5_downstream_duration_frs_combined.pdf"
    manuscript_pdf.write_bytes(pdf.read_bytes())
    print(manuscript_pdf)


if __name__ == "__main__":
    main()
