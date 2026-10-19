"""Paper figure for Section 6.2.4: band-mean exceedance versus capacity threshold.

Reads the dense curve grid written by run_sensitivity.py; no statistics are recomputed here.
Run from any directory: E2/.venv/Scripts/python.exe E2/scripts/plot_paper_threshold_sensitivity.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
CURVES = HERE / "results" / "capacity_curve_plot_grid.csv"
OUT_DIR = HERE / "figures"
STEM = "R3_threshold_sensitivity"

STRUCTURES = [("F10", "F-10"), ("FW10", "FW-10")]
BANDS = [
    ("rigid_mep", "Rigid MEP, 0.05–0.20 s"),
    ("storage_rack_cross_aisle", "Rack cross-aisle, 0.50–1.00 s"),
    ("storage_rack_down_aisle", "Rack down-aisle, 1.50–3.00 s"),
]
CAPACITIES = [0.4, 1.0, 2.0]
X_MAX = 3.0


def load_curves():
    curves = {}
    with CURVES.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["structure"], row["band"])
            curves.setdefault(key, []).append(
                (float(row["capacity_g"]), float(row["reference_percent"]), float(row["fno_percent"]))
            )
    return {k: np.array(sorted(v)) for k, v in curves.items()}


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "pdf.fonttype": 42,
    })
    curves = load_curves()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True, sharey=True)
    letters = iter("abcdef")
    for row, (sid, sname) in enumerate(STRUCTURES):
        for col, (bid, bname) in enumerate(BANDS):
            ax = axes[row, col]
            data = curves[(sid, bid)]
            ax.plot(data[:, 0], data[:, 1], color="#0072B2", lw=1.6, label="NTHA")
            ax.plot(data[:, 0], data[:, 2], color="#D55E00", lw=1.4, ls="--", label="FNO")
            for c in CAPACITIES:
                ax.axvline(c, color="0.6", ls=":", lw=0.8)
            for y in (10, 50):
                ax.axhline(y, color="0.45", ls="-.", lw=0.7)
            ax.set_xlim(0, X_MAX)
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
            ax.set_title(f"({next(letters)}) {sname}: {bname}", fontsize=10.5)
            if row == 1:
                ax.set_xlabel("Capacity threshold, $c$ (g)")
            if col == 0:
                ax.set_ylabel("Exceedance (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{STEM}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / STEM}.pdf/.png")


if __name__ == "__main__":
    main()
