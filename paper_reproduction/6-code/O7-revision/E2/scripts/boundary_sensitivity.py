"""Sensitivity of the screening categories to the 10%/50% category boundaries.

For every pair of lower boundary l and upper boundary u, the NTHA- and FNO-based band-mean
exceedance percentages are classified as suitable (< l), conditional (l to < u), or
unsuitable (>= u), and the agreement between the two classifications is counted.
  (a) at the 1.0 g capacity used in Table 11: number of agreeing bands out of six;
  (b) over capacities 0.1-2.0 g (step 0.05 g): percentage of agreeing band-capacity cases.

Run from any directory: E2/.venv/Scripts/python.exe E2/scripts/boundary_sensitivity.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"
O5 = ROOT / "6-code" / "O5-fema-p58"
CFG = O5 / "configs" / "o5_equipment_period_bands_supported.json"
FIG_DIR = HERE / "figures"
STEM = "R3_boundary_sensitivity"

LOWER = np.arange(2, 26, 1)      # lower boundary l (%)
UPPER = np.arange(30, 81, 1)     # upper boundary u (%)
CAPACITIES = np.round(np.arange(0.1, 2.0001, 0.05), 4)
PAPER_CAPACITY = 1.0


def exceedance_percent(sorted_values, capacity):
    """Band-mean exceedance in percent: fraction of sample-period pairs with Sa >= capacity."""
    n = sorted_values.size
    return 100.0 * (n - np.searchsorted(sorted_values, capacity, side="left")) / n


def classify(q, lower, upper):
    return np.where(q < lower, 0, np.where(q < upper, 1, 2))


def load_exceedance():
    """Return {(structure, band): (q_ref, q_fno)} evaluated at CAPACITIES."""
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    result = {}
    for sid in ["F10", "FW10"]:
        src = np.load(O5 / "results" / "multistructure" / "frs_full" / f"{sid}_roof_frs_5pct.npz")
        periods = src["periods_s"].astype(np.float64)
        ref, pred = src["fem_sa_g"][:, 0, :], src["fno_sa_g"][:, 0, :]
        for bid, band in cfg["period_bands"].items():
            mask = (periods >= band["period_min_s"]) & (periods <= band["period_max_s"])
            sr = np.sort(ref[:, mask].astype(np.float64).ravel())
            sp = np.sort(pred[:, mask].astype(np.float64).ravel())
            result[(sid, bid)] = (exceedance_percent(sr, CAPACITIES), exceedance_percent(sp, CAPACITIES))
    return result


def agreement_maps(exceedance):
    paper_idx = int(np.argmin(np.abs(CAPACITIES - PAPER_CAPACITY)))
    at_paper = np.full((UPPER.size, LOWER.size), np.nan)
    overall = np.full((UPPER.size, LOWER.size), np.nan)
    for i, u in enumerate(UPPER):
        for j, l in enumerate(LOWER):
            if l >= u:
                continue
            agree_paper, agree_all, total_all = 0, 0, 0
            for q_ref, q_fno in exceedance.values():
                same = classify(q_ref, l, u) == classify(q_fno, l, u)
                agree_paper += int(same[paper_idx])
                agree_all += int(same.sum())
                total_all += same.size
            at_paper[i, j] = agree_paper
            overall[i, j] = 100.0 * agree_all / total_all
    return at_paper, overall


def write_rows(at_paper, overall):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "boundary_sensitivity_grid.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lower_percent", "upper_percent", "agreeing_bands_at_1g_of_6",
                    "agreement_percent_over_0p1_to_2g"])
        for i, u in enumerate(UPPER):
            for j, l in enumerate(LOWER):
                if l < u:
                    w.writerow([int(l), int(u), int(at_paper[i, j]), f"{overall[i, j]:.3f}"])


def plot(at_paper, overall):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    extent = (LOWER[0] - 0.5, LOWER[-1] + 0.5, UPPER[0] - 0.5, UPPER[-1] + 0.5)

    im0 = axes[0].imshow(at_paper, origin="lower", extent=extent, aspect="auto",
                         cmap=matplotlib.colors.ListedColormap(["#C6DBEF", "#6BAED6", "#08519C"]), vmin=3.5, vmax=6.5)
    cb0 = fig.colorbar(im0, ax=axes[0], ticks=[4, 5, 6])
    cb0.set_label("Agreeing bands (out of 6)")
    axes[0].set_title("(a) Capacity of 1.0 $g$", fontsize=11)

    im1 = axes[1].imshow(overall, origin="lower", extent=extent, aspect="auto",
                         cmap="Blues", vmin=90, vmax=100)
    cb1 = fig.colorbar(im1, ax=axes[1])
    cb1.set_label("Agreement (%)")
    axes[1].set_title("(b) Capacities of 0.1–2.0 $g$", fontsize=11)

    for ax in axes:
        ax.plot(10, 50, marker="*", ms=13, color="#D55E00", mec="white", mew=0.8, ls="none",
                label="Adopted boundaries (10%, 50%)")
        ax.set_xlabel("Lower boundary (%)")
        ax.set_ylabel("Upper boundary (%)")
    axes[0].legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{STEM}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    exceedance = load_exceedance()
    at_paper, overall = agreement_maps(exceedance)
    write_rows(at_paper, overall)
    plot(at_paper, overall)
    valid = ~np.isnan(overall)
    i10, j50 = int(np.where(UPPER == 50)[0][0]), int(np.where(LOWER == 10)[0][0])
    print(f"At (10%, 50%): {int(at_paper[i10, j50])}/6 bands agree at 1.0 g; "
          f"{overall[i10, j50]:.2f}% agreement over 0.1-2.0 g")
    print(f"Over all boundary pairs: 1.0 g agreeing bands min {int(np.nanmin(at_paper))}, "
          f"share of pairs with 6/6 = {100 * np.mean(at_paper[valid] == 6):.1f}%")
    print(f"Over all boundary pairs: overall agreement min {np.nanmin(overall):.2f}%, "
          f"max {np.nanmax(overall):.2f}%, median {np.nanmedian(overall):.2f}%")
    sub = overall[np.ix_((UPPER >= 40) & (UPPER <= 60), (LOWER >= 5) & (LOWER <= 15))]
    sub1 = at_paper[np.ix_((UPPER >= 40) & (UPPER <= 60), (LOWER >= 5) & (LOWER <= 15))]
    print(f"Lower 5-15%, upper 40-60%: overall agreement {np.nanmin(sub):.2f}-{np.nanmax(sub):.2f}%; "
          f"1.0 g agreeing bands {int(np.nanmin(sub1))}-{int(np.nanmax(sub1))}")
    print(f"Wrote {FIG_DIR / STEM}.pdf/.png and results/boundary_sensitivity_grid.csv")


if __name__ == "__main__":
    main()
