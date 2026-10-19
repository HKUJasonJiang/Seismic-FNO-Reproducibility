"""Summarize equipment placement risk from floor response spectra."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
O5_DIR = REPO_ROOT / "6-code" / "O5-fema-p58"
DEFAULT_CASES = O5_DIR / "configs" / "o5_multistructure_cases.json"
DEFAULT_PERIOD_CONFIG = O5_DIR / "configs" / "o5_equipment_period_bands_supported.json"
DEFAULT_FRS_DIR = O5_DIR / "results" / "multistructure" / "frs_full"
DEFAULT_OUTPUT_DIR = O5_DIR / "results" / "multistructure" / "placement_supported"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--period-config", type=Path, default=DEFAULT_PERIOD_CONFIG)
    parser.add_argument("--frs-dir", type=Path, default=DEFAULT_FRS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--structures", nargs="+", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalized_period_config(config: dict) -> tuple[list[dict], dict, dict]:
    thresholds = config.get("exceedance_thresholds_g") or config.get("capacity_thresholds_g")
    if thresholds is None:
        raise KeyError("Expected exceedance_thresholds_g or capacity_thresholds_g")

    raw_limits = config.get("suitability_probability_limits") or config.get("suitability_thresholds")
    if raw_limits is None:
        raise KeyError("Expected suitability_probability_limits or suitability_thresholds")
    limits = {
        "suitable_max_exceedance_probability": raw_limits.get(
            "suitable_max_exceedance_probability",
            raw_limits.get("suitable_max_exceedance"),
        ),
        "avoid_min_exceedance_probability": raw_limits.get(
            "avoid_min_exceedance_probability",
            raw_limits.get("avoid_min_exceedance"),
        ),
    }

    if "equipment_period_bands" in config:
        bands = config["equipment_period_bands"]
    else:
        bands = []
        for band_id, band in config["period_bands"].items():
            bands.append(
                {
                    "id": band_id,
                    "label": band["label"],
                    "min_s": band["period_min_s"],
                    "max_s": band["period_max_s"],
                    "basis": "; ".join(band.get("examples", [])),
                }
            )
    return bands, thresholds, limits


def status_from_probability(probability: float, limits: dict) -> str:
    if probability < float(limits["suitable_max_exceedance_probability"]):
        return "suitable"
    if probability >= float(limits["avoid_min_exceedance_probability"]):
        return "avoid"
    return "conditional"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_frs(frs_dir: Path, structure_id: str) -> dict:
    path = frs_dir / f"{structure_id}_roof_frs_5pct.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {
        "path": path,
        "periods_s": data["periods_s"].astype(float),
        "fem_sa_g": data["fem_sa_g"].astype(float),
        "fno_sa_g": data["fno_sa_g"].astype(float),
        "floor_labels": [str(x) for x in data["floor_labels"]],
    }


def compute_structure(
    structure_id: str,
    cases: dict,
    period_config: dict,
    frs_dir: Path,
    output_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    structure = cases["structures"][structure_id]
    frs = load_frs(frs_dir, structure_id)
    periods = frs["periods_s"]
    bands, thresholds, limits = normalized_period_config(period_config)

    map_rows: list[dict] = []
    band_rows: list[dict] = []
    consistency_rows: list[dict] = []
    source_status_by_threshold = {}

    for source, spectra in (("FEM", frs["fem_sa_g"]), ("FNO", frs["fno_sa_g"])):
        for floor_idx, floor_label in enumerate(frs["floor_labels"]):
            for p_idx, period_s in enumerate(periods):
                demand = spectra[:, floor_idx, p_idx]
                row = {
                    "structure_id": structure_id,
                    "paper_label": structure.get("paper_label", structure_id),
                    "source": source,
                    "floor_index": floor_idx + 1,
                    "floor_label": floor_label,
                    "period_s": float(period_s),
                    "sa_p50_g": float(np.quantile(demand, 0.50)),
                    "sa_p90_g": float(np.quantile(demand, 0.90)),
                    "sa_p95_g": float(np.quantile(demand, 0.95)),
                }
                for label, threshold in thresholds.items():
                    row[f"prob_exceed_{label}"] = float(np.mean(demand >= float(threshold)))
                map_rows.append(row)

    for band in bands:
        min_t = float(band["min_s"])
        max_t = float(band["max_s"])
        mask = (periods >= min_t) & (periods <= max_t)
        if not np.any(mask):
            continue
        for source, spectra in (("FEM", frs["fem_sa_g"]), ("FNO", frs["fno_sa_g"])):
            for floor_idx, floor_label in enumerate(frs["floor_labels"]):
                band_demand = np.max(spectra[:, floor_idx, :][:, mask], axis=1)
                row = {
                    "structure_id": structure_id,
                    "paper_label": structure.get("paper_label", structure_id),
                    "source": source,
                    "floor_index": floor_idx + 1,
                    "floor_label": floor_label,
                    "band_id": band["id"],
                    "band_label": band["label"],
                    "min_period_s": min_t,
                    "max_period_s": max_t,
                    "reference_basis": band.get("basis", ""),
                    "sa_band_p50_g": float(np.quantile(band_demand, 0.50)),
                    "sa_band_p90_g": float(np.quantile(band_demand, 0.90)),
                    "sa_band_p95_g": float(np.quantile(band_demand, 0.95)),
                }
                for label, threshold in thresholds.items():
                    probability = float(np.mean(band_demand >= float(threshold)))
                    status = status_from_probability(probability, limits)
                    row[f"prob_exceed_{label}"] = probability
                    row[f"status_{label}"] = status
                    source_status_by_threshold[(source, floor_idx, band["id"], label)] = status
                band_rows.append(row)

    for band in bands:
        for floor_idx, floor_label in enumerate(frs["floor_labels"]):
            for label in thresholds:
                fem_status = source_status_by_threshold.get(("FEM", floor_idx, band["id"], label))
                fno_status = source_status_by_threshold.get(("FNO", floor_idx, band["id"], label))
                if fem_status is None or fno_status is None:
                    continue
                consistency_rows.append(
                    {
                        "structure_id": structure_id,
                        "paper_label": structure.get("paper_label", structure_id),
                        "floor_index": floor_idx + 1,
                        "floor_label": floor_label,
                        "band_id": band["id"],
                        "band_label": band["label"],
                        "threshold_label": label,
                        "fem_status": fem_status,
                        "fno_status": fno_status,
                        "is_consistent": int(fem_status == fno_status),
                    }
                )

    write_csv(output_dir / f"{structure_id}_placement_exceedance_map.csv", map_rows)
    write_csv(output_dir / f"{structure_id}_band_suitability.csv", band_rows)
    write_csv(output_dir / f"{structure_id}_placement_decision_consistency.csv", consistency_rows)
    return map_rows, band_rows, consistency_rows


def main() -> None:
    args = parse_args()
    cases = load_json(args.cases)
    period_config = load_json(args.period_config)
    structures = args.structures or cases["main_cases"]
    output_dir = args.output_dir.resolve()
    all_map_rows: list[dict] = []
    all_band_rows: list[dict] = []
    all_consistency_rows: list[dict] = []
    for structure_id in structures:
        map_rows, band_rows, consistency_rows = compute_structure(
            structure_id,
            cases,
            period_config,
            args.frs_dir.resolve(),
            output_dir,
        )
        all_map_rows.extend(map_rows)
        all_band_rows.extend(band_rows)
        all_consistency_rows.extend(consistency_rows)
        print(f"[{structure_id}] placement rows: {len(band_rows)}")
    write_csv(output_dir / "o5_placement_exceedance_map.csv", all_map_rows)
    write_csv(output_dir / "o5_placement_band_suitability.csv", all_band_rows)
    write_csv(output_dir / "o5_placement_decision_consistency.csv", all_consistency_rows)
    consistency = np.array([row["is_consistent"] for row in all_consistency_rows], dtype=float)
    summary = {
        "structures": structures,
        "n_band_rows": len(all_band_rows),
        "n_consistency_rows": len(all_consistency_rows),
        "decision_consistency_rate": float(np.mean(consistency)) if consistency.size else None,
    }
    with (output_dir / "o5_placement_risk_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {output_dir / 'o5_placement_risk_summary.json'}")


if __name__ == "__main__":
    main()
