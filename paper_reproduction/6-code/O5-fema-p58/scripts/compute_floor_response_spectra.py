"""Compute 5%-damped floor response spectra from exported roof histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import cont2discrete, lfilter, ss2tf


G = 9.80665
REPO_ROOT = Path(__file__).resolve().parents[3]
O5_DIR = REPO_ROOT / "6-code" / "O5-fema-p58"
DEFAULT_CASES = O5_DIR / "configs" / "o5_multistructure_cases.json"
DEFAULT_PERIODS = O5_DIR / "configs" / "o5_equipment_period_bands_supported.json"
DEFAULT_DEMAND_DIR = O5_DIR / "results" / "multistructure" / "demands"
DEFAULT_OUTPUT_DIR = O5_DIR / "results" / "multistructure" / "frs_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--period-config", type=Path, default=DEFAULT_PERIODS)
    parser.add_argument("--demand-dir", type=Path, default=DEFAULT_DEMAND_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--structures", nargs="+", default=None)
    parser.add_argument("--chunk-size", type=int, default=1536)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Optional evenly spaced subset size for FRS. Prefer this over max-records for formal subset runs.",
    )
    parser.add_argument("--period-count", type=int, default=90)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def period_grid(config: dict, override_count: int | None) -> np.ndarray:
    grid = config["period_grid_s"]
    count = override_count or int(grid["count"])
    return np.linspace(float(grid["min"]), float(grid["max"]), count, dtype=np.float64)


def oscillator_filter(period_s: float, damping: float, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi / period_s
    a_matrix = np.array([[0.0, 1.0], [-omega * omega, -2.0 * damping * omega]])
    b_matrix = np.array([[0.0], [-1.0]])
    c_matrix = np.array([[1.0, 0.0]])
    d_matrix = np.array([[0.0]])
    ad, bd, cd, dd, _ = cont2discrete((a_matrix, b_matrix, c_matrix, d_matrix), dt_s)
    b, a = ss2tf(ad, bd, cd, dd)
    return np.asarray(b[0], dtype=np.float64), np.asarray(a, dtype=np.float64)


def compute_sa(
    histories_ms2: np.ndarray,
    periods_s: np.ndarray,
    damping: float,
    dt_s: float,
    chunk_size: int,
) -> np.ndarray:
    n_records, n_floors, _n_steps = histories_ms2.shape
    spectra = np.empty((n_records, n_floors, periods_s.size), dtype=np.float32)
    for p_idx, period_s in enumerate(periods_s):
        b, a = oscillator_filter(float(period_s), damping, dt_s)
        omega = 2.0 * np.pi / float(period_s)
        for start in range(0, n_records, chunk_size):
            stop = min(start + chunk_size, n_records)
            acc = np.asarray(histories_ms2[start:stop], dtype=np.float64)
            rel_disp = lfilter(b, a, acc, axis=-1)
            sa_g = (omega * omega) * np.max(np.abs(rel_disp), axis=-1) / G
            spectra[start:stop, :, p_idx] = sa_g.astype(np.float32)
        if (p_idx + 1) % 10 == 0 or p_idx == periods_s.size - 1:
            print(f"[frs] period {p_idx + 1}/{periods_s.size}: T={period_s:.3f}s")
    return spectra


def compute_structure(
    structure_id: str,
    cases: dict,
    period_cfg: dict,
    periods_s: np.ndarray,
    demand_dir: Path,
    output_dir: Path,
    chunk_size: int,
    max_records: int | None,
    sample_count: int | None,
) -> dict:
    dt_s = float(cases["dt_s"])
    damping = float(period_cfg["damping_ratio"])
    structure = cases["structures"][structure_id]
    fem_path = demand_dir / f"{structure_id}_fem_roof_history_ms2.npy"
    fno_path = demand_dir / f"{structure_id}_fno_roof_history_ms2.npy"
    if not fem_path.exists():
        raise FileNotFoundError(fem_path)
    if not fno_path.exists():
        raise FileNotFoundError(fno_path)

    fem_history = np.load(fem_path, mmap_mode="r")
    fno_history = np.load(fno_path, mmap_mode="r")
    if max_records is not None:
        fem_history = fem_history[:max_records]
        fno_history = fno_history[:max_records]
    if sample_count is not None and sample_count < fem_history.shape[0]:
        sample_indices = np.linspace(0, fem_history.shape[0] - 1, sample_count)
        sample_indices = np.unique(np.round(sample_indices).astype(np.int64))
        fem_history = fem_history[sample_indices]
        fno_history = fno_history[sample_indices]
    else:
        sample_indices = np.arange(fem_history.shape[0], dtype=np.int64)
    print(f"[{structure_id}] FEM FRS from {fem_history.shape}")
    fem_sa_g = compute_sa(fem_history, periods_s, damping, dt_s, chunk_size)
    print(f"[{structure_id}] FNO FRS from {fno_history.shape}")
    fno_sa_g = compute_sa(fno_history, periods_s, damping, dt_s, chunk_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{structure_id}_roof_frs_5pct.npz"
    np.savez_compressed(
        out_path,
        periods_s=periods_s.astype(np.float32),
        fem_sa_g=fem_sa_g,
        fno_sa_g=fno_sa_g,
        floor_labels=np.array([structure.get("roof_floor_label", "roof")]),
        structure_id=np.array([structure_id]),
    )
    diff = fno_sa_g - fem_sa_g
    summary = {
        "structure_id": structure_id,
        "paper_label": structure.get("paper_label", structure_id),
        "n_records": int(fem_sa_g.shape[0]),
        "sample_indexing": "evenly_spaced_subset" if sample_count is not None else "all_available",
        "sample_indices_count": int(sample_indices.size),
        "n_floors": int(fem_sa_g.shape[1]),
        "n_periods": int(fem_sa_g.shape[2]),
        "damping_ratio": damping,
        "period_min_s": float(np.min(periods_s)),
        "period_max_s": float(np.max(periods_s)),
        "frs_npz": str(out_path.resolve()),
        "fem_sa_p90_max_g": float(np.max(np.quantile(fem_sa_g, 0.90, axis=0))),
        "fno_sa_p90_max_g": float(np.max(np.quantile(fno_sa_g, 0.90, axis=0))),
        "sa_mae_g": float(np.mean(np.abs(diff))),
        "sa_rmse_g": float(np.sqrt(np.mean(diff * diff))),
    }
    summary_path = output_dir / f"{structure_id}_frs_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    cases = load_json(args.cases)
    period_cfg = load_json(args.period_config)
    structures = args.structures or cases["main_cases"]
    periods_s = period_grid(period_cfg, args.period_count)
    output_dir = args.output_dir.resolve()
    summaries = []
    for structure_id in structures:
        summaries.append(
            compute_structure(
                structure_id,
                cases,
                period_cfg,
                periods_s,
                args.demand_dir.resolve(),
                output_dir,
                args.chunk_size,
                args.max_records,
                args.sample_count,
            )
        )
    with (output_dir / "o5_frs_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summaries": summaries}, f, indent=2)
    print(f"[summary] {output_dir / 'o5_frs_summary.json'}")


if __name__ == "__main__":
    main()
