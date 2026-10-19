"""Generate revised Chapter 6 figures and tables for O5 downstream analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
O5_DIR = REPO_ROOT / "6-code" / "O5-fema-p58"
O3_SPECTRA_CSV = (
    REPO_ROOT
    / "6-code"
    / "O3-generalization"
    / "results"
    / "train_test_response_spectra"
    / "response_spectra_normalized_percentiles.csv"
)
REFINED_DIR = O5_DIR / "results" / "multistructure" / "refined_analysis"
DURATION_DIR = O5_DIR / "results" / "multistructure" / "duration_metrics"
FRS_FULL_DIR = O5_DIR / "results" / "multistructure" / "frs_full"
FIG_DIR = O5_DIR / "figures" / "multistructure_revision"
TABLE_DIR = O5_DIR / "results" / "multistructure" / "chapter6_revision"
PERIOD_CONFIG = O5_DIR / "configs" / "o5_equipment_period_bands_supported.json"
MANUSCRIPT_FIG_DIR = O5_DIR / "figures" / "paper_exports"

COLORS = {"FEM": "#0072B2", "FNO": "#D55E00", "train": "#4D4D4D", "test": "#009E73"}
STRUCTURES = {"F10": "F-10", "FW10": "FW-10"}
THRESHOLDS_G = [0.1, 0.2, 0.4]
FRS_CAPACITY_G = 1.0


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths[suffix] = str(path.resolve())
    plt.close(fig)
    return paths


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.24, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_period_bands() -> list[dict]:
    config = json.loads(PERIOD_CONFIG.read_text(encoding="utf-8"))
    bands = []
    for band_id, band in config["period_bands"].items():
        bands.append(
            {
                "band_id": band_id,
                "label": band["label"],
                "period_min_s": float(band["period_min_s"]),
                "period_max_s": float(band["period_max_s"]),
                "reference_basis": "; ".join(band.get("examples", [])),
            }
        )
    return bands


def plot_normalized_input_spectra() -> dict[str, str]:
    df = pd.read_csv(O3_SPECTRA_CSV)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    period = df["period_s"].to_numpy()
    ax.fill_between(
        period,
        df["train_p05_psa_g"],
        df["train_p95_psa_g"],
        color=COLORS["train"],
        alpha=0.16,
        label="Training 5-95%",
    )
    ax.fill_between(
        period,
        df["test_p05_psa_g"],
        df["test_p95_psa_g"],
        color=COLORS["test"],
        alpha=0.18,
        label="Testing 5-95%",
    )
    ax.plot(period, df["train_p50_psa_g"], color=COLORS["train"], linewidth=1.8, label="Training median")
    ax.plot(period, df["test_p50_psa_g"], color=COLORS["test"], linewidth=1.8, label="Testing median")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 6.0)
    ax.set_xlabel("Period T (s)")
    ax.set_ylabel("Normalized spectral acceleration, Sa/PGA")
    ax.set_title("Normalized response spectra of training and testing ground motions")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    return save_figure(fig, "fig_o5_normalized_input_spectra")


def duration_column(threshold_g: float) -> str:
    return f"cumulative_duration_ge_{threshold_g:g}g_s"


def load_duration_scale_rows() -> tuple[list[dict], list[dict]]:
    record_map = (
        pd.read_csv(REFINED_DIR / "o5_record_level_metrics.csv")
        .query("source == 'FEM'")
        [["structure_id", "record_index", "scale_index", "input_pga_g"]]
        .copy()
    )
    scale_rows: list[dict] = []
    sigma_rows: list[dict] = []
    for structure_id, paper_label in STRUCTURES.items():
        for source in ("FEM", "FNO"):
            duration_df = pd.read_csv(
                DURATION_DIR / f"{structure_id}_{source.lower()}_time_history_duration_metrics.csv"
            )
            merged = duration_df.merge(
                record_map[record_map["structure_id"] == structure_id],
                on="record_index",
                how="left",
                validate="one_to_one",
            )
            for threshold_g in THRESHOLDS_G:
                col = duration_column(threshold_g)
                values = merged[col].to_numpy(dtype=float)
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1))
                sigma_rows.append(
                    {
                        "structure_id": structure_id,
                        "paper_label": paper_label,
                        "source": source,
                        "threshold_g": threshold_g,
                        "mean_s": mean,
                        "std_s": std,
                        "mean_plus_1sigma_s": mean + std,
                        "mean_plus_2sigma_s": mean + 2.0 * std,
                        "mean_plus_3sigma_s": mean + 3.0 * std,
                    }
                )
                grouped = merged.groupby("scale_index", sort=True)
                for scale_index, sub in grouped:
                    sub_values = sub[col].to_numpy(dtype=float)
                    scale_rows.append(
                        {
                            "structure_id": structure_id,
                            "paper_label": paper_label,
                            "source": source,
                            "threshold_g": threshold_g,
                            "scale_index": int(scale_index),
                            "n": int(sub.shape[0]),
                            "input_pga_mean_g": float(sub["input_pga_g"].mean()),
                            "duration_mean_s": float(np.mean(sub_values)),
                            "duration_std_s": float(np.std(sub_values, ddof=1)),
                        }
                    )
    return scale_rows, sigma_rows


def plot_duration_by_scale(scale_df: pd.DataFrame, sigma_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    paths = {}
    for structure_id, paper_label in STRUCTURES.items():
        fig, axes = plt.subplots(1, len(THRESHOLDS_G), figsize=(13.2, 3.8), sharex=True)
        for ax, threshold_g in zip(axes, THRESHOLDS_G):
            for source in ("FEM", "FNO"):
                sub = scale_df[
                    (scale_df["structure_id"] == structure_id)
                    & (scale_df["source"] == source)
                    & (scale_df["threshold_g"] == threshold_g)
                ].sort_values("scale_index")
                x = sub["input_pga_mean_g"].to_numpy()
                mean = sub["duration_mean_s"].to_numpy()
                std = sub["duration_std_s"].to_numpy()
                ax.plot(x, mean, color=COLORS[source], linewidth=1.8, label=source)
                ax.fill_between(x, np.maximum(mean - std, 0.0), mean + std, color=COLORS[source], alpha=0.12)
            fem_sigma = sigma_df[
                (sigma_df["structure_id"] == structure_id)
                & (sigma_df["source"] == "FEM")
                & (sigma_df["threshold_g"] == threshold_g)
            ].iloc[0]
            for k, style in [(1, ":"), (2, "--"), (3, "-.")]:
                ax.axhline(
                    fem_sigma[f"mean_plus_{k}sigma_s"],
                    color="#6B6B6B",
                    linewidth=0.9,
                    linestyle=style,
                    alpha=0.7,
                )
            ax.set_title(f"{threshold_g:g}g threshold")
            ax.set_xlabel("Mean input PGA per scale (g)")
            style_axes(ax)
        axes[0].set_ylabel("Cumulative exceedance duration (s)")
        axes[0].legend(frameon=False)
        fig.suptitle(f"{paper_label}: duration demand over all 57 scale factors", y=1.05)
        fig.tight_layout()
        paths[structure_id] = save_figure(fig, f"fig_o5_duration_by_scale_{structure_id}")
    return paths


def load_frs_npz(structure_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(FRS_FULL_DIR / f"{structure_id}_roof_frs_5pct.npz")
    periods = data["periods_s"].astype(float)
    fem = data["fem_sa_g"].astype(float)[:, 0, :]
    fno = data["fno_sa_g"].astype(float)[:, 0, :]
    return periods, fem, fno


def plot_full_frs() -> dict[str, str]:
    bands = load_period_bands()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), sharey=True)
    for ax, (structure_id, paper_label) in zip(axes, STRUCTURES.items()):
        periods, fem, fno = load_frs_npz(structure_id)
        for band in bands:
            ax.axvspan(band["period_min_s"], band["period_max_s"], color="#DDDDDD", alpha=0.32)
        ax.plot(periods, np.mean(fem, axis=0), color=COLORS["FEM"], linewidth=1.9, label="FEM mean")
        ax.plot(periods, np.mean(fno, axis=0), color=COLORS["FNO"], linewidth=1.9, label="SeFNO mean")
        ax.set_title(paper_label)
        ax.set_xlabel("Equipment period T (s)")
        ax.set_xlim(0.0, 3.0)
        style_axes(ax)
    axes[0].set_ylabel("Roof spectral acceleration, Sa (g)")
    axes[0].legend(frameon=False)
    fig.suptitle("Mean roof floor response spectra from all 27,018 scaled samples", y=1.04)
    fig.tight_layout()
    return save_figure(fig, "fig_o5_full_roof_frs")


def frs_exceedance_rows() -> tuple[list[dict], list[dict]]:
    bands = load_period_bands()
    curve_rows: list[dict] = []
    band_rows: list[dict] = []
    for structure_id, paper_label in STRUCTURES.items():
        periods, fem, fno = load_frs_npz(structure_id)
        for source, spectra in (("FEM", fem), ("FNO", fno)):
            exceed_pct = np.mean(spectra >= FRS_CAPACITY_G, axis=0) * 100.0
            for period_s, pct in zip(periods, exceed_pct):
                if period_s <= 3.000001:
                    curve_rows.append(
                        {
                            "structure_id": structure_id,
                            "paper_label": paper_label,
                            "source": source,
                            "capacity_g": FRS_CAPACITY_G,
                            "period_s": float(period_s),
                            "exceedance_percent": float(pct),
                        }
                    )
            for band in bands:
                mask = (periods >= band["period_min_s"]) & (periods <= band["period_max_s"])
                band_rows.append(
                    {
                        "structure_id": structure_id,
                        "paper_label": paper_label,
                        "source": source,
                        "capacity_g": FRS_CAPACITY_G,
                        **band,
                        "mean_exceedance_percent": float(np.mean(exceed_pct[mask])),
                    }
                )
    return curve_rows, band_rows


def plot_period_exceedance(curve_df: pd.DataFrame, band_df: pd.DataFrame) -> dict[str, str]:
    bands = load_period_bands()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    plot_colors = {"FEM": "#1F1F1F", "FNO": "#0072B2"}
    for panel_label, ax, (structure_id, paper_label) in zip(("a", "b"), axes, STRUCTURES.items()):
        for band in bands:
            ax.axvspan(band["period_min_s"], band["period_max_s"], color="#DDDDDD", alpha=0.3)
            ax.text(
                0.5 * (band["period_min_s"] + band["period_max_s"]),
                96.0,
                f"{band['period_min_s']:.2g}-{band['period_max_s']:.2g}s",
                ha="center",
                va="top",
                fontsize=7,
                rotation=90 if band["period_max_s"] - band["period_min_s"] < 0.25 else 0,
            )
        for source in ("FEM", "FNO"):
            sub = curve_df[(curve_df["structure_id"] == structure_id) & (curve_df["source"] == source)]
            ax.plot(
                sub["period_s"],
                sub["exceedance_percent"],
                color=plot_colors[source],
                linewidth=1.45,
                linestyle="-" if source == "FEM" else "--",
                label=source,
            )
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 100.0)
        ax.set_xlabel("Equipment period T (s)")
        ax.set_title(f"({panel_label}) {paper_label}")
        style_axes(ax)
    axes[0].set_ylabel(f"Samples exceeding {FRS_CAPACITY_G:g}g (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2)
    fig.tight_layout(w_pad=1.8, rect=[0.0, 0.0, 1.0, 0.98])
    paths = save_figure(fig, "fig_o5_period_exceedance_combined")
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT_FIG_DIR / "ES_o5_period_exceedance_combined.pdf").write_bytes(
        (FIG_DIR / "fig_o5_period_exceedance_combined.pdf").read_bytes()
    )
    return paths


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    figures: dict[str, object] = {}

    figures["normalized_input_spectra"] = plot_normalized_input_spectra()

    scale_rows, sigma_rows = load_duration_scale_rows()
    write_csv(TABLE_DIR / "o5_duration_by_scale_summary.csv", scale_rows)
    write_csv(TABLE_DIR / "o5_duration_sigma_summary.csv", sigma_rows)
    scale_df = pd.DataFrame(scale_rows)
    sigma_df = pd.DataFrame(sigma_rows)
    figures["duration_by_scale"] = plot_duration_by_scale(scale_df, sigma_df)

    figures["full_roof_frs"] = plot_full_frs()
    curve_rows, band_rows = frs_exceedance_rows()
    write_csv(TABLE_DIR / "o5_period_exceedance_curve.csv", curve_rows)
    write_csv(TABLE_DIR / "o5_period_band_exceedance_summary.csv", band_rows)
    figures["period_exceedance"] = plot_period_exceedance(pd.DataFrame(curve_rows), pd.DataFrame(band_rows))

    meta_path = TABLE_DIR / "chapter6_revision_artifacts.json"
    meta_path.write_text(json.dumps(figures, indent=2), encoding="utf-8")
    print(meta_path)


if __name__ == "__main__":
    main()
