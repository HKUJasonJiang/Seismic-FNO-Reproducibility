"""Plot manuscript Fig. 11 and Fig. 12 from exported inference arrays.

This script is intentionally lightweight: the remote server runs the models and
exports NPZ files, while the local machine handles final typography/layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
O3_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = O3_DIR.parent.parent
DEFAULT_DATA_DIR = O3_DIR / "results" / "fig11_fig13_rerun"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "Elsevier_EngineeringStructures_SeFNO" / "Figure"

MODEL_COLORS = {
    "Ground truth": "#111111",
    "Transformer": "#D73027",
    "FNO-Large": "#1F78B4",
}
FIG12_MODEL_COLORS = {
    "Transformer": "#D73027",
    "FNO-Large": "#1F78B4",
}
GT_LW = 0.58
MODEL_LW = 0.42
INPUT_LW = 0.42


def register_fonts() -> None:
    font_dirs = [
        O3_DIR / "assets" / "fonts",
        Path(r"C:\Windows\Fonts"),
    ]
    for font_dir in font_dirs:
        if not font_dir.exists():
            continue
        for pattern in ("georgia*.ttf", "times*.ttf"):
            for font_path in font_dir.glob(pattern):
                font_manager.fontManager.addfont(str(font_path))


def setup_style() -> None:
    register_fonts()
    plt.rcParams.update(
        {
            "font.family": "Georgia",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.linewidth": 0.75,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#CFCFCF", linewidth=0.35, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    ax.xaxis.get_offset_text().set_fontfamily("Times New Roman")
    ax.yaxis.get_offset_text().set_fontfamily("Times New Roman")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")


def panel_caption(ax: plt.Axes, label: str, text: str, y: float) -> None:
    ax.text(
        0.5,
        y,
        f"({label}) {text}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.0,
        fontfamily="Georgia",
    )


def panel_label(ax: plt.Axes, label: str, text: str | None = None) -> None:
    content = f"({label})" if text is None else f"({label}) {text}"
    ax.text(
        0.02,
        0.96,
        content,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontfamily="Georgia",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.6},
    )


def panel_title(ax: plt.Axes, label: str, text: str, fontsize: float = 10.0) -> None:
    ax.set_title(f"({label}) {text}", pad=3, fontsize=fontsize, fontfamily="Georgia")


def format_legend(legend) -> None:
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.82)
    legend.get_frame().set_linewidth(0.35)
    for text in legend.get_texts():
        text.set_fontfamily("Georgia")


def save_figure(fig: plt.Figure, output_dir: Path, figure_dir: Path | None, stem: str, aliases: tuple[str, ...] = ()) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir]
    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)
        targets.append(figure_dir)
    for target_dir in targets:
        for name in (stem, *aliases):
            fig.savefig(target_dir / f"{name}.pdf", bbox_inches="tight")
            fig.savefig(target_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def nice_symmetric_limit(*series: np.ndarray) -> float:
    max_abs = max(float(np.nanmax(np.abs(values))) for values in series if values.size)
    padded = max_abs * 1.06
    if padded <= 2.5:
        step = 0.5
    elif padded <= 6.0:
        step = 1.0
    else:
        step = 2.0
    return max(step, float(np.ceil(padded / step) * step))


def set_symmetric_ylim(ax: plt.Axes, *series: np.ndarray) -> None:
    limit = nice_symmetric_limit(*series)
    ax.set_ylim(-limit, limit)


def plot_fig11(data_dir: Path, figure_dir: Path | None) -> None:
    data = np.load(data_dir / "fig11_plot_data.npz")
    gm_ids = data["gm_id"]
    pgas = data["pga"]
    gt = data["gt"]
    transformer = data["transformer"]
    fno = data["fno"]
    r2_transformer = data["r2_transformer"]
    r2_fno = data["r2_fno"]
    sample_rate = float(data["sample_rate_hz"])
    t = np.arange(gt.shape[-1]) / sample_rate

    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.9), sharex=True, sharey=False)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.985, bottom=0.145, wspace=0.22, hspace=0.55)
    for i, ax in enumerate(axes.flat):
        ax.plot(t, gt[i], color=MODEL_COLORS["Ground truth"], lw=GT_LW, label="NTHA Result", alpha=0.88)
        ax.plot(
            t,
            transformer[i],
            color=MODEL_COLORS["Transformer"],
            lw=MODEL_LW,
            ls="--",
            label=f"Transformer $R^2={r2_transformer[i]:.3f}$",
            alpha=0.86,
        )
        ax.plot(
            t,
            fno[i],
            color=MODEL_COLORS["FNO-Large"],
            lw=MODEL_LW,
            ls="-.",
            label=f"FNO $R^2={r2_fno[i]:.3f}$",
            alpha=0.86,
        )
        ax.set_xlim(0, 60)
        set_symmetric_ylim(ax, gt[i], transformer[i], fno[i])
        ax.set_xlabel("Time (s)")
        ax.tick_params(axis="x", labelbottom=True)
        if i % 3 == 0:
            ax.set_ylabel("Roof acc.\n(m/s$^2$)")
        finish_axes(ax)
        panel_caption(ax, "abcdef"[i], f"GM ${int(gm_ids[i])}$, PGA=${pgas[i]:.1f}$ m/s$^2$", y=-0.34)
        legend = ax.legend(frameon=True, loc="upper right", handlelength=1.2, borderpad=0.25, labelspacing=0.2)
        format_legend(legend)
    save_figure(fig, data_dir, figure_dir, "ES_fig11_pga_extrapolation_time_histories")


def plot_fig12(data_dir: Path, figure_dir: Path | None) -> None:
    data = np.load(data_dir / "fig12_zero_shot_plot_data.npz")
    _plot_fig12_zero_shot(data, data_dir, figure_dir)


def _plot_fig12_zero_shot(data: np.lib.npyio.NpzFile, data_dir: Path, figure_dir: Path | None) -> None:
    n_events = int(data["n_events"])
    titles = data["title"]
    gm_hz = data["gm_hz"]
    r2_transformer = data["r2_transformer"]
    r2_fno = data["r2_fno"]

    fig = plt.figure(figsize=(7.35, 8.25))
    gs = fig.add_gridspec(
        n_events,
        2,
        width_ratios=[1.0, 2.0],
        left=0.085,
        right=0.995,
        top=0.985,
        bottom=0.075,
        wspace=0.25,
        hspace=0.92,
    )
    panel_letters = list("abcdefgh")
    for row in range(n_events):
        ax_in = fig.add_subplot(gs[row, 0])
        ax_out = fig.add_subplot(gs[row, 1], sharex=ax_in)
        time = data[f"gm_time_{row}"]
        gm_values = data[f"gm_{row}"]
        ax_in.plot(time, gm_values, color="#4C86B8", lw=INPUT_LW, alpha=0.9)
        ax_in.set_xlim(0, 60)
        set_symmetric_ylim(ax_in, gm_values)
        ax_in.set_ylabel("Input acc.\n(m/s$^2$)")
        ax_in.set_xlabel("Time (s)")
        ax_in.tick_params(axis="x", labelbottom=True)
        panel_caption(ax_in, panel_letters[row * 2], f"{titles[row]} input, ${gm_hz[row]:.0f}$ Hz", y=-0.42)
        finish_axes(ax_in)

        gt_values = data[f"gt_{row}"]
        transformer_values = data[f"transformer_{row}"]
        fno_values = data[f"fno_{row}"]
        ax_out.plot(time, gt_values, color=MODEL_COLORS["Ground truth"], lw=GT_LW, label="NTHA Result", alpha=0.88)
        ax_out.plot(
            time,
            transformer_values,
            color=FIG12_MODEL_COLORS["Transformer"],
            lw=MODEL_LW,
            ls="--",
            label=f"Transformer $R^2={r2_transformer[row]:.3f}$",
            alpha=0.86,
        )
        ax_out.plot(
            time,
            fno_values,
            color=FIG12_MODEL_COLORS["FNO-Large"],
            lw=MODEL_LW,
            ls="-.",
            label=f"FNO $R^2={r2_fno[row]:.3f}$",
            alpha=0.86,
        )
        ax_out.set_xlim(0, 60)
        set_symmetric_ylim(ax_out, gt_values, transformer_values, fno_values)
        ax_out.set_ylabel("Roof acc.\n(m/s$^2$)", labelpad=2)
        ax_out.set_xlabel("Time (s)")
        ax_out.tick_params(axis="x", labelbottom=True)
        panel_caption(ax_out, panel_letters[row * 2 + 1], f"{titles[row]} inference", y=-0.42)
        finish_axes(ax_out)
        legend = ax_out.legend(frameon=True, loc="upper right", ncol=3, handlelength=1.2, fontsize=7.6, borderpad=0.25, labelspacing=0.2)
        format_legend(legend)
    save_figure(
        fig,
        data_dir,
        figure_dir,
        "ES_fig13_zero_shot_o1_checkpoint_time_histories",
        aliases=("ES_fig12_zero_shot_o1_checkpoint_time_histories",),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure_dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()
    setup_style()
    plot_fig11(args.data_dir, args.figure_dir)
    plot_fig12(args.data_dir, args.figure_dir)


if __name__ == "__main__":
    main()
