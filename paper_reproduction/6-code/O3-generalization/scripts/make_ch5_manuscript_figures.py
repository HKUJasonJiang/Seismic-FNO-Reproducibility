from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "Elsevier_EngineeringStructures_SeFNO" / "Figure"
GM_H5 = Path(r"D:\BaiduNetdiskDownload\SesimicTransformerData\MDOF\All_GMs\GMs_knet_3474_AF_57.h5")
TEST_SAMPLE_DIR = ROOT / "3-fig" / "test_sample"
CLASSICAL_PRED_DIR = ROOT / "6-code" / "O3-generalization" / "results" / "classical_gms_native_output_o1_full"

MODEL_COLORS = {
    "BiLSTM": "#5B8DB8",
    "Transformer": "#D9825B",
    "FNO-Large": "#4F8A72",
    "ResCNN1D": "#9B7BB9",
    "UNet1D": "#B0A06A",
    "MLP": "#7A7A7A",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Georgia",
            "mathtext.fontset": "stix",
            "mathtext.rm": "Times New Roman",
            "axes.linewidth": 0.75,
            "axes.edgecolor": "black",
            "axes.labelsize": 9.2,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axes(ax) -> None:
    ax.grid(True, color="#D7D7D7", linewidth=0.5, alpha=0.72)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")


def finish_fig7_axes(ax) -> None:
    ax.grid(True, color="#CFCFCF", linewidth=0.25, alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("black")
    ax.tick_params(width=0.5, which="major")
    ax.tick_params(width=0.4, which="minor")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        fontfamily="Georgia",
    )


def bottom_panel_title(ax, title: str) -> None:
    ax.text(
        0.5,
        -0.48,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        fontfamily="Times New Roman",
    )


def fig11_bottom_title(ax, title: str, y: float) -> None:
    ax.text(
        0.5,
        y,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.6,
        fontfamily="Times New Roman",
    )


def label_bars(ax, bars, values, fmt: str, *, log_scale: bool = False, dy: float = 3.0) -> None:
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        if log_scale:
            ax.annotate(
                fmt.format(value),
                xy=(x, y),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.0,
                fontfamily="Times New Roman",
            )
        else:
            ax.annotate(
                fmt.format(value),
                xy=(x, y),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.0,
                fontfamily="Times New Roman",
            )


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_two_column_text(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, skiprows=1)
    return data[:, 0], data[:, 1]


def top_response_panel(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    crop = image.crop((70, 15, width - 35, int(height * 0.465)))
    return np.asarray(crop)


def plot_classical_combined_examples() -> None:
    events = [
        {
            "title": "El Centro (1940), 50 Hz",
            "gm": "1940-ElCentro-60s-50Hz-ms2.txt",
            "response": "1940-ElCentro-50Hz-ms2_native_output.png",
        },
        {
            "title": "San Fernando (1971), 50 Hz",
            "gm": "1971-Sanfernando-60s-50Hz-ms2.txt",
            "response": "1971-Sanfernando-50Hz-ms2_native_output.png",
        },
        {
            "title": "Michoacan (1985), 200 Hz",
            "gm": "1985-Michoacan-60s-200Hz-ms2.txt",
            "response": "1985-Michoacan-200Hz-ms2_native_output.png",
        },
        {
            "title": "Kobe (1991), 100 Hz",
            "gm": "1991-Kobe-60s-100Hz-ms2.txt",
            "response": "1991-Kobe-100Hz-ms2_native_output.png",
        },
    ]

    fig = plt.figure(figsize=(7.35, 6.15), constrained_layout=True)
    gs = fig.add_gridspec(len(events), 2, width_ratios=[1.0, 2.0], hspace=0.18, wspace=0.05)

    for row, event in enumerate(events):
        time, acc = load_two_column_text(TEST_SAMPLE_DIR / event["gm"])
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(time, acc, color="#222222", lw=0.35)
        ax.set_xlim(0, 60)
        ax.set_title(event["title"], pad=2)
        ax.set_ylabel("Ground acc.\n(m/s$^2$)")
        if row == len(events) - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        finish_fig7_axes(ax)

        ax_img = fig.add_subplot(gs[row, 1])
        ax_img.imshow(top_response_panel(CLASSICAL_PRED_DIR / event["response"]), aspect="auto")
        ax_img.set_axis_off()

    save(fig, "ES_classical_native_combined_examples")


def plot_o1_screening() -> None:
    df = pd.read_csv(ROOT / "6-code" / "O1-baseline-comparison" / "results" / "report" / "o1_report_summary.csv")
    order = ["BiLSTM", "Transformer", "FNO-Large", "ResCNN1D", "UNet1D", "MLP"]
    df = df.set_index("model").loc[order].reset_index()
    x = np.arange(len(df))
    colors = [MODEL_COLORS[m] for m in df["model"]]

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.15), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(x, df["mse"], color=colors, edgecolor="black", linewidth=0.45, width=0.68)
    ax.set_yscale("log")
    ax.set_ylim(8e-3, 1.25)
    ax.set_ylabel("MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=32, ha="right")
    label_bars(ax, bars, df["mse"], "{:.3g}", log_scale=True, dy=1.8)
    ax2 = ax.twinx()
    ax2.plot(x, df["r2"], color="#111111", marker="o", markersize=3.6, linewidth=1.3, label="$R^2$")
    ax2.set_ylabel("$R^2$")
    ax2.set_ylim(0.0, 1.05)
    for label in ax2.get_yticklabels():
        label.set_fontfamily("Times New Roman")
    finish_axes(ax)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("black")
    ax2.grid(False)
    bottom_panel_title(ax, "(a) MSE and R²")

    ax = axes[1]
    bars = ax.bar(x, df["train_time_h"], color=colors, edgecolor="black", linewidth=0.45, width=0.68)
    ax.set_ylabel("Training time (h)")
    ax.set_ylim(0, 9)
    ax.set_yticks([0, 3, 6, 9])
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=32, ha="right")
    label_bars(ax, bars, df["train_time_h"], "{:.2f}", dy=2.0)
    finish_axes(ax)
    bottom_panel_title(ax, "(b) Training time")

    ax = axes[2]
    bars = ax.bar(x, df["peak_gpu_gb"], color=colors, edgecolor="black", linewidth=0.45, width=0.68)
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=32, ha="right")
    ax.set_ylim(0, max(df["peak_gpu_gb"]) * 1.12)
    label_bars(ax, bars, df["peak_gpu_gb"], "{:.2f}", dy=2.0)
    finish_axes(ax)
    bottom_panel_title(ax, "(c) Peak GPU memory")

    save(fig, "O1_fixed_screening_summary")


def load_gm_rows(row_ids: list[int]) -> np.ndarray:
    with h5py.File(GM_H5, "r") as h5:
        key = "Acc_GMs" if "Acc_GMs" in h5 else next(iter(h5.keys()))
        order = np.argsort(row_ids)
        sorted_ids = np.asarray(row_ids, dtype=int)[order]
        loaded = np.asarray(h5[key][sorted_ids, :], dtype=float)
        restored = np.empty_like(loaded)
        restored[order] = loaded
        return restored


def plot_pga_ood_examples() -> None:
    selected = [
        {"gm_id": 3393, "scale_id": 46, "label": "GM 3393, train upper"},
        {"gm_id": 3393, "scale_id": 56, "label": "GM 3393, high-PGA OOD"},
        {"gm_id": 635, "scale_id": 46, "label": "GM 635, train upper"},
        {"gm_id": 635, "scale_id": 56, "label": "GM 635, high-PGA OOD"},
    ]
    rows = [s["gm_id"] * 57 + s["scale_id"] for s in selected]
    data = load_gm_rows(rows)
    t = np.arange(data.shape[1]) / 50.0
    freqs = np.fft.rfftfreq(data.shape[1], d=1 / 50.0)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.25, 3.25),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [2.0, 1.0]},
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.96, bottom=0.17, wspace=0.25, hspace=0.82)
    pair_colors = {"train upper": "#4D4D4D", "high-PGA OOD": "#0072B2"}
    panel_titles = {
        (0, 0): "(a) GM 3393 input time history",
        (0, 1): "(b) GM 3393 Fourier spectrum",
        (1, 0): "(c) GM 635 input time history",
        (1, 1): "(d) GM 635 Fourier spectrum",
    }

    for r, gm_id in enumerate([3393, 635]):
        ax = axes[r, 0]
        for idx, item in enumerate([s for s in selected if s["gm_id"] == gm_id]):
            row_i = selected.index(item)
            key = "train upper" if item["scale_id"] == 46 else "high-PGA OOD"
            pga = np.max(np.abs(data[row_i]))
            ax.plot(t, data[row_i], color=pair_colors[key], lw=0.35, label=f"{key}, PGA={pga:.2f} m/s$^2$")
        ax.set_xlim(0, 60)
        ax.set_ylim(-22, 22)
        ax.set_yticks([-20, 0, 20])
        ax.set_ylabel("Acceleration (m/s$^2$)")
        if r == 1:
            ax.set_xlabel("Time (s)")
        finish_fig7_axes(ax)
        fig11_bottom_title(ax, panel_titles[(r, 0)], -0.32 if r == 0 else -0.46)
        legend = ax.legend(frameon=True, loc="upper right", fontsize=5.2, handlelength=1.5, framealpha=0.85)
        legend.get_frame().set_linewidth(0.4)
        for text in legend.get_texts():
            text.set_fontfamily("Georgia")

        ax = axes[r, 1]
        for item in [s for s in selected if s["gm_id"] == gm_id]:
            row_i = selected.index(item)
            key = "train upper" if item["scale_id"] == 46 else "high-PGA OOD"
            amp = np.abs(np.fft.rfft(data[row_i])) / data.shape[1]
            ax.plot(freqs, amp, color=pair_colors[key], lw=0.4, label=key)
        ax.set_xlim(0, 25)
        ax.set_ylabel("Fourier amplitude")
        if r == 1:
            ax.set_xlabel("Frequency (Hz)")
        finish_fig7_axes(ax)
        fig11_bottom_title(ax, panel_titles[(r, 1)], -0.32 if r == 0 else -0.46)
        legend = ax.legend(frameon=True, loc="upper right", fontsize=5.2, handlelength=1.5, framealpha=0.85)
        legend.get_frame().set_linewidth(0.4)
        for text in legend.get_texts():
            text.set_fontfamily("Georgia")
    save(fig, "ES_pga_ood_gm_examples")


def plot_pga_ood_overall() -> None:
    rows = [
        {"model": "BiLSTM", "mse": 0.01476, "r2": 0.9810, "train_time_s": 17620, "peak_gpu_gb": 15.61},
        {"model": "Transformer", "mse": 0.03190, "r2": 0.9590, "train_time_s": 7357, "peak_gpu_gb": 2.54},
        {"model": "FNO-Large", "mse": 0.06694, "r2": 0.9140, "train_time_s": 7271, "peak_gpu_gb": 3.30},
    ]
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    colors = [MODEL_COLORS[m] for m in df["model"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)

    ax = axes[0]
    ax.bar(x, df["mse"], color=colors, edgecolor="black", linewidth=0.55, width=0.64)
    ax.set_yscale("log")
    ax.set_ylabel("MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax2 = ax.twinx()
    ax2.plot(x, df["r2"], color="#111111", marker="o", linewidth=1.2, markersize=3.7)
    ax2.set_ylabel("$R^2$")
    ax2.set_ylim(0.86, 1.0)
    for label in ax2.get_yticklabels():
        label.set_fontfamily("Times New Roman")
    finish_axes(ax)
    ax2.grid(False)
    bottom_panel_title(ax, "(a) MSE and R²")

    ax = axes[1]
    width = 0.34
    ax.bar(x - width / 2, df["train_time_s"] / 3600, width=width, color="#8AA6C1", edgecolor="black", linewidth=0.5, label="Training time")
    ax.set_ylabel("Training time (h)")
    ax3 = ax.twinx()
    ax3.bar(x + width / 2, df["peak_gpu_gb"], width=width, color="#C9967A", edgecolor="black", linewidth=0.5, label="GPU memory")
    ax3.set_ylabel("Peak GPU memory (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    finish_axes(ax)
    for label in ax3.get_yticklabels():
        label.set_fontfamily("Times New Roman")
    ax3.grid(False)
    bottom_panel_title(ax, "(b) Training time and GPU memory")

    save(fig, "ES_pga_ood_overall_summary")


def plot_resolution_summary() -> None:
    native = pd.read_csv(
        ROOT / "6-code" / "O3-generalization" / "results" / "classical_gms_native_output_o1_full" / "classical_native_output_metrics.csv"
    )
    sampling = pd.read_csv(
        ROOT / "6-code" / "O3-generalization" / "results" / "sampling_ood" / "sampling_ood_overall_summary.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.65), constrained_layout=True)

    ax = axes[0]
    events = ["El Centro (1940)", "San Fernando (1971)", "Michoacan (1985)", "Kobe (1991)"]
    x = np.arange(len(events))
    width = 0.24
    for i, model in enumerate(["BiLSTM", "Transformer", "FNO-Large"]):
        vals = [native[(native["event"] == ev) & (native["model"] == model)]["r2_native_axis"].iloc[0] for ev in events]
        ax.bar(x + (i - 1) * width, vals, width=width, color=MODEL_COLORS[model], edgecolor="black", linewidth=0.45, label=model)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("$R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels(["El Centro\n50 Hz", "San Fernando\n50 Hz", "Michoacan\n200 Hz", "Kobe\n100 Hz"])
    ax.set_ylim(-1.4, 1.05)
    finish_axes(ax)
    ax.legend(frameon=False, loc="lower left", ncol=1)
    panel_label(ax, "A")

    ax = axes[1]
    for model in ["BiLSTM", "Transformer", "FNO-Large"]:
        part = sampling[sampling["model"] == model].sort_values("train_hz")
        ax.plot(part["train_hz"], part["r2"], marker="o", linewidth=1.35, markersize=4.0, color=MODEL_COLORS[model], label=model)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Training sampling rate (Hz)")
    ax.set_ylabel("50 Hz test $R^2$")
    ax.set_xticks([20, 25, 50])
    ax.set_ylim(-0.65, 1.05)
    finish_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "B")

    save(fig, "ES_resolution_ood_summary")


def plot_data_efficiency_summary() -> None:
    df = pd.read_csv(ROOT / "6-code" / "O2-data-efficiency" / "results" / "report" / "o2_report_summary.csv")
    axes_order = [("AC", "Amplitude coefficients"), ("GM", "Ground motions"), ("Joint", "Joint reduction")]
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55), constrained_layout=True, sharey=True)

    for ax, (axis_name, title) in zip(axes, axes_order):
        part = df[df["axis"] == axis_name]
        for model in [m for m in ["BiLSTM", "Transformer", "FNO-Large"] if m in set(part["model"])]:
            rows = part[part["model"] == model].sort_values("percent")
            ax.plot(
                rows["percent"],
                rows["r2"],
                marker="o",
                linewidth=1.35,
                markersize=3.8,
                color=MODEL_COLORS[model],
                label=model,
            )
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 40, 60, 80])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(title)
        ax.set_xlabel("Retained training data (%)")
        finish_axes(ax)
        ax.set_ylim(0.2, 1.01)
    axes[0].set_ylabel("Test $R^2$")
    axes[2].legend(frameon=False, loc="lower right")
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    panel_label(axes[2], "C")
    save(fig, "ES_data_efficiency_summary")


def main() -> None:
    setup_style()
    plot_o1_screening()
    if GM_H5.exists():
        plot_pga_ood_examples()
    else:
        print(f"Skipped PGA examples because {GM_H5} was not found")
    plot_classical_combined_examples()
    plot_resolution_summary()
    plot_data_efficiency_summary()


if __name__ == "__main__":
    main()
