"""Plot train-pool and fixed-test ground-motion response-spectrum distributions.

The canonical split contains 3000 training-pool ground motions and 474 fixed-test
ground motions. This script reads one representative scale of each K-NET record,
computes 5%-damped pseudo-acceleration response spectra, and overlays the two
sets as percentile bands.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete, lfilter, ss2tf


G = 9.80665
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GM_H5 = Path(
    r"D:\BaiduNetdiskDownload\SesimicTransformerData\MDOF\All_GMs\GMs_knet_3474_AF_57.h5"
)
DEFAULT_SPLIT = REPO_ROOT / "6-code" / "shared" / "split_indices" / "dataset_indices.pkl"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "6-code"
    / "O3-generalization"
    / "results"
    / "train_test_response_spectra"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gm-h5", type=Path, default=DEFAULT_GM_H5)
    parser.add_argument("--split-pkl", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scale-id", type=int, default=9, help="Representative scale index; O3 used 9 as about 1 m/s^2.")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--period-min", type=float, default=0.05)
    parser.add_argument("--period-max", type=float, default=6.0)
    parser.add_argument("--period-count", type=int, default=160)
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def oscillator_filter(period_s: float, damping: float, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi / period_s
    a_matrix = np.array([[0.0, 1.0], [-omega * omega, -2.0 * damping * omega]])
    b_matrix = np.array([[0.0], [-1.0]])
    c_matrix = np.array([[1.0, 0.0]])
    d_matrix = np.array([[0.0]])
    ad, bd, cd, dd, _ = cont2discrete((a_matrix, b_matrix, c_matrix, d_matrix), dt_s)
    b, a = ss2tf(ad, bd, cd, dd)
    return np.asarray(b[0], dtype=np.float64), np.asarray(a, dtype=np.float64)


def compute_psa(acc_ms2: np.ndarray, periods_s: np.ndarray, damping: float, dt_s: float, chunk_size: int) -> np.ndarray:
    spectra = np.empty((acc_ms2.shape[0], periods_s.size), dtype=np.float32)
    for p_idx, period_s in enumerate(periods_s):
        b, a = oscillator_filter(float(period_s), damping, dt_s)
        omega = 2.0 * np.pi / float(period_s)
        for start in range(0, acc_ms2.shape[0], chunk_size):
            stop = min(start + chunk_size, acc_ms2.shape[0])
            rel_disp = lfilter(b, a, np.asarray(acc_ms2[start:stop], dtype=np.float64), axis=-1)
            spectra[start:stop, p_idx] = (omega * omega * np.max(np.abs(rel_disp), axis=-1) / G).astype(np.float32)
        if (p_idx + 1) % 20 == 0 or p_idx == periods_s.size - 1:
            print(f"[spectra] {p_idx + 1}/{periods_s.size}: T={period_s:.3f}s")
    return spectra


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    train_pool = np.concatenate([np.asarray(data["train_gms"]), np.asarray(data["val_gms"])]).astype(np.int64)
    test = np.asarray(data["test_gms"]).astype(np.int64)
    return train_pool, test


def read_scaled_ground_motions(gm_h5: Path, scale_id: int) -> np.ndarray:
    with h5py.File(gm_h5, "r") as f:
        dset = f["Acc_GMs"]
        total_scaled, n_steps = dset.shape
        total_gms = total_scaled // 57
        row_indices = np.arange(total_gms, dtype=np.int64) * 57 + int(scale_id)
        print(f"[data] reading {total_gms} records x {n_steps} steps at scale_id={scale_id}")
        return np.asarray(dset[row_indices, :], dtype=np.float32)


def percentile_summary(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "p05": np.percentile(values, 5, axis=0),
        "p25": np.percentile(values, 25, axis=0),
        "p50": np.percentile(values, 50, axis=0),
        "p75": np.percentile(values, 75, axis=0),
        "p95": np.percentile(values, 95, axis=0),
    }


def write_summary_csv(path: Path, periods: np.ndarray, train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> None:
    fieldnames = ["period_s"]
    for prefix in ("train", "test"):
        for q in ("p05", "p25", "p50", "p75", "p95"):
            fieldnames.append(f"{prefix}_{q}_psa_g")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, period in enumerate(periods):
            row = {"period_s": float(period)}
            for q, arr in train.items():
                row[f"train_{q}_psa_g"] = float(arr[i])
            for q, arr in test.items():
                row[f"test_{q}_psa_g"] = float(arr[i])
            writer.writerow(row)


def plot_distribution(
    periods: np.ndarray,
    train_psa: np.ndarray,
    test_psa: np.ndarray,
    train_norm: np.ndarray,
    test_norm: np.ndarray,
    output_dir: Path,
    scale_id: int,
) -> Path:
    colors = {"train": "#0072B2", "test": "#D55E00"}
    train_sum = percentile_summary(train_psa)
    test_sum = percentile_summary(test_psa)
    train_norm_sum = percentile_summary(train_norm)
    test_norm_sum = percentile_summary(test_norm)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True)
    panels = [
        (axes[0], train_sum, test_sum, "Pseudo-spectral acceleration, PSa (g)", "Absolute spectra"),
        (axes[1], train_norm_sum, test_norm_sum, "PSa / PGA", "PGA-normalized spectral shape"),
    ]
    for ax, tr, te, ylabel, title in panels:
        ax.fill_between(periods, tr["p05"], tr["p95"], color=colors["train"], alpha=0.14, linewidth=0)
        ax.fill_between(periods, tr["p25"], tr["p75"], color=colors["train"], alpha=0.24, linewidth=0)
        ax.plot(periods, tr["p50"], color=colors["train"], linewidth=2.0, label="Train pool median")

        ax.fill_between(periods, te["p05"], te["p95"], color=colors["test"], alpha=0.14, linewidth=0)
        ax.fill_between(periods, te["p25"], te["p75"], color=colors["test"], alpha=0.24, linewidth=0)
        ax.plot(periods, te["p50"], color=colors["test"], linewidth=2.0, label="Fixed test median")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(periods.min(), periods.max())
        ax.set_xlabel("Period, T (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.22)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"5%-damped response-spectrum distribution, scale_id={scale_id}", y=1.02)
    fig.tight_layout()

    png_path = output_dir / "train_test_response_spectra_overlay.png"
    pdf_path = output_dir / "train_test_response_spectra_overlay.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    write_summary_csv(output_dir / "response_spectra_percentiles.csv", periods, train_sum, test_sum)
    write_summary_csv(output_dir / "response_spectra_normalized_percentiles.csv", periods, train_norm_sum, test_norm_sum)
    return png_path


def plot_curve_cloud(
    periods: np.ndarray,
    train_psa: np.ndarray,
    test_psa: np.ndarray,
    train_norm: np.ndarray,
    test_norm: np.ndarray,
    output_dir: Path,
    scale_id: int,
    seed: int = 42,
) -> Path:
    colors = {"train": "#0072B2", "test": "#D55E00"}
    rng = np.random.default_rng(seed)
    train_count = min(train_psa.shape[0], test_psa.shape[0])
    train_pick = rng.choice(train_psa.shape[0], size=train_count, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True)
    panels = [
        (axes[0], train_psa[train_pick], test_psa, "Pseudo-spectral acceleration, PSa (g)", "Absolute spectra"),
        (axes[1], train_norm[train_pick], test_norm, "PSa / PGA", "PGA-normalized spectral shape"),
    ]
    for ax, tr, te, ylabel, title in panels:
        ax.plot(periods, tr.T, color=colors["train"], alpha=0.035, linewidth=0.55)
        ax.plot(periods, te.T, color=colors["test"], alpha=0.050, linewidth=0.55)
        ax.plot(periods, np.median(tr, axis=0), color=colors["train"], linewidth=2.0, label=f"Train pool sample, n={train_count}")
        ax.plot(periods, np.median(te, axis=0), color=colors["test"], linewidth=2.0, label=f"Fixed test, n={test_psa.shape[0]}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(periods.min(), periods.max())
        ax.set_xlabel("Period, T (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.22)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"5%-damped response-spectrum curve cloud, scale_id={scale_id}", y=1.02)
    fig.tight_layout()

    png_path = output_dir / "train_test_response_spectra_curve_cloud.png"
    pdf_path = output_dir / "train_test_response_spectra_curve_cloud.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_pool_gms, test_gms = load_split(args.split_pkl)
    all_acc = read_scaled_ground_motions(args.gm_h5, args.scale_id)
    periods = np.geomspace(args.period_min, args.period_max, args.period_count)
    pga_g = np.max(np.abs(all_acc), axis=1) / G
    psa = compute_psa(all_acc, periods, args.damping, args.dt, args.chunk_size)

    train_psa = psa[train_pool_gms]
    test_psa = psa[test_gms]
    train_norm = train_psa / np.maximum(pga_g[train_pool_gms, None], 1e-12)
    test_norm = test_psa / np.maximum(pga_g[test_gms, None], 1e-12)
    fig_path = plot_distribution(periods, train_psa, test_psa, train_norm, test_norm, args.output_dir, args.scale_id)
    cloud_path = plot_curve_cloud(periods, train_psa, test_psa, train_norm, test_norm, args.output_dir, args.scale_id)

    np.savez_compressed(
        args.output_dir / "train_test_response_spectra_scale.npz",
        periods_s=periods.astype(np.float32),
        train_gms=train_pool_gms.astype(np.int32),
        test_gms=test_gms.astype(np.int32),
        train_psa_g=train_psa.astype(np.float32),
        test_psa_g=test_psa.astype(np.float32),
        train_psa_over_pga=train_norm.astype(np.float32),
        test_psa_over_pga=test_norm.astype(np.float32),
        train_pga_g=pga_g[train_pool_gms].astype(np.float32),
        test_pga_g=pga_g[test_gms].astype(np.float32),
    )
    summary = {
        "gm_h5": str(args.gm_h5),
        "split_pkl": str(args.split_pkl),
        "scale_id": int(args.scale_id),
        "train_pool_gm_count": int(train_pool_gms.size),
        "test_gm_count": int(test_gms.size),
        "dt_s": float(args.dt),
        "damping_ratio": float(args.damping),
        "period_min_s": float(periods.min()),
        "period_max_s": float(periods.max()),
        "period_count": int(periods.size),
        "train_pga_median_g": float(np.median(pga_g[train_pool_gms])),
        "test_pga_median_g": float(np.median(pga_g[test_gms])),
        "figure": str(fig_path.resolve()),
        "curve_cloud_figure": str(cloud_path.resolve()),
    }
    with (args.output_dir / "train_test_response_spectra_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
