"""Run P-58 acceleration-sensitive component probabilities for O5 cases."""

from __future__ import annotations

import argparse
import csv
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
O5_DIR = REPO_ROOT / "6-code" / "O5-fema-p58"
DEFAULT_CASES = O5_DIR / "configs" / "o5_multistructure_cases.json"
DEFAULT_COMPONENTS = O5_DIR / "configs" / "o5_component_scenarios.json"
DEFAULT_DEMAND_DIR = O5_DIR / "results" / "multistructure" / "demands"
DEFAULT_OUTPUT_DIR = O5_DIR / "results" / "multistructure" / "p58"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--demand-dir", type=Path, default=DEFAULT_DEMAND_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--structures", nargs="+", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_demands(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"No demand rows in {path}")
    return {
        "record_index": np.array([int(row["record_index"]) for row in rows], dtype=np.int64),
        "sample_index": np.array([int(row["sample_index"]) for row in rows], dtype=np.int64),
        "fem": np.array([float(row["fem_pfa_g"]) for row in rows], dtype=np.float64),
        "fno": np.array([float(row["fno_pfa_g"]) for row in rows], dtype=np.float64),
    }


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def exceedance_probability(demand_g: np.ndarray, theta_g: float, beta: float) -> np.ndarray:
    demand = np.maximum(np.asarray(demand_g, dtype=np.float64), 1.0e-12)
    return normal_cdf(np.log(demand / float(theta_g)) / float(beta))


def component_state_probabilities(component: dict, demand_g: np.ndarray) -> dict[str, np.ndarray]:
    mode = component["probability_mode"]
    states = component["damage_states"]
    if mode == "exclusive_sequential":
        exceed = [
            exceedance_probability(demand_g, state["theta_g"], state["beta"])
            for state in states
        ]
        out: dict[str, np.ndarray] = {"DS0": np.clip(1.0 - exceed[0], 0.0, 1.0)}
        for idx, state in enumerate(states):
            if idx < len(states) - 1:
                prob = exceed[idx] - exceed[idx + 1]
            else:
                prob = exceed[idx]
            out[state["state"]] = np.clip(prob, 0.0, 1.0)
        return out
    if mode == "mutually_exclusive_branch":
        out = {}
        branch_sum = np.zeros_like(demand_g, dtype=np.float64)
        for state in states:
            prob = exceedance_probability(demand_g, state["theta_g"], state["beta"])
            prob = prob * float(state.get("branch_probability", 1.0))
            out[state["state"]] = np.clip(prob, 0.0, 1.0)
            branch_sum += out[state["state"]]
        out["DS0"] = np.clip(1.0 - branch_sum, 0.0, 1.0)
        return out
    if mode == "simultaneous_occurrence":
        out = {}
        base_exceed = None
        for idx, state in enumerate(states):
            prob = exceedance_probability(demand_g, state["theta_g"], state["beta"])
            if idx == 0:
                base_exceed = prob
            prob = prob * float(state.get("branch_probability", 1.0))
            out[state["state"]] = np.clip(prob, 0.0, 1.0)
        if base_exceed is not None:
            out["DS0"] = np.clip(1.0 - base_exceed, 0.0, 1.0)
        return out
    raise ValueError(f"Unsupported probability_mode: {mode}")


def state_meta(component: dict, state_name: str) -> dict:
    if state_name == "DS0":
        return {"theta_g": "", "beta": "", "branch_probability": "", "probability_kind": "no_damage"}
    for state in component["damage_states"]:
        if state["state"] == state_name:
            kind = (
                "exclusive_state_probability"
                if component["probability_mode"] != "simultaneous_occurrence"
                else "occurrence_probability"
            )
            return {
                "theta_g": state["theta_g"],
                "beta": state["beta"],
                "branch_probability": state.get("branch_probability", 1.0),
                "probability_kind": kind,
            }
    raise KeyError(state_name)


def summarize_probability(values: np.ndarray) -> dict:
    return {
        "mean_probability": float(np.mean(values)),
        "p50_probability": float(np.quantile(values, 0.50)),
        "p90_probability": float(np.quantile(values, 0.90)),
        "p95_probability": float(np.quantile(values, 0.95)),
    }


def analyze_structure(
    structure_id: str,
    cases: dict,
    component_config: dict,
    demand_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    structure = cases["structures"][structure_id]
    scenario_id = structure.get("scenario_id") or component_config["scenario_assignments"][structure_id]
    scenario = component_config["scenarios"][scenario_id]
    demand = read_demands(demand_dir / f"{structure_id}_roof_pfa_demands.csv")
    component_rows: list[dict] = []
    error_rows: list[dict] = []
    scenario_rows: list[dict] = []

    expected_affected_by_source = {"FEM": np.zeros_like(demand["fem"]), "FNO": np.zeros_like(demand["fno"])}
    max_any_by_source = {"FEM": np.zeros_like(demand["fem"]), "FNO": np.zeros_like(demand["fno"])}

    for component in scenario["components"]:
        source_probs = {}
        for source, demand_key in (("FEM", "fem"), ("FNO", "fno")):
            probs = component_state_probabilities(component, demand[demand_key])
            source_probs[source] = probs
            any_damage = 1.0 - probs.get("DS0", np.ones_like(demand[demand_key]))
            expected_affected_by_source[source] += any_damage
            max_any_by_source[source] = np.maximum(max_any_by_source[source], any_damage)
            for state_name, values in probs.items():
                meta = state_meta(component, state_name)
                row = {
                    "structure_id": structure_id,
                    "paper_label": structure.get("paper_label", structure_id),
                    "scenario_id": scenario_id,
                    "scenario_label": scenario["label"],
                    "source": source,
                    "component_id": component["component_id"],
                    "component_label": component["label"],
                    "source_row": component["source_row"],
                    "hierarchy": component["hierarchy"],
                    "probability_mode": component["probability_mode"],
                    "state": state_name,
                    **meta,
                    **summarize_probability(values),
                }
                component_rows.append(row)

        all_states = sorted(set(source_probs["FEM"].keys()) | set(source_probs["FNO"].keys()))
        for state_name in all_states:
            if state_name not in source_probs["FEM"] or state_name not in source_probs["FNO"]:
                continue
            fem_values = source_probs["FEM"][state_name]
            fno_values = source_probs["FNO"][state_name]
            diff = fno_values - fem_values
            meta = state_meta(component, state_name)
            error_rows.append(
                {
                    "structure_id": structure_id,
                    "paper_label": structure.get("paper_label", structure_id),
                    "scenario_id": scenario_id,
                    "component_id": component["component_id"],
                    "component_label": component["label"],
                    "source_row": component["source_row"],
                    "hierarchy": component["hierarchy"],
                    "probability_mode": component["probability_mode"],
                    "state": state_name,
                    **meta,
                    "fem_mean_probability": float(np.mean(fem_values)),
                    "fno_mean_probability": float(np.mean(fno_values)),
                    "mean_probability_error": float(np.mean(diff)),
                    "mean_abs_probability_error": float(np.mean(np.abs(diff))),
                    "rmse_probability": float(np.sqrt(np.mean(diff * diff))),
                    "p95_abs_probability_error": float(np.quantile(np.abs(diff), 0.95)),
                }
            )

    for source in ("FEM", "FNO"):
        expected = expected_affected_by_source[source]
        max_any = max_any_by_source[source]
        scenario_rows.append(
            {
                "structure_id": structure_id,
                "paper_label": structure.get("paper_label", structure_id),
                "scenario_id": scenario_id,
                "scenario_label": scenario["label"],
                "source": source,
                "n_records": int(expected.size),
                "n_components": len(scenario["components"]),
                "expected_components_with_any_damage_mean": float(np.mean(expected)),
                "expected_components_with_any_damage_p50": float(np.quantile(expected, 0.50)),
                "expected_components_with_any_damage_p90": float(np.quantile(expected, 0.90)),
                "probability_at_least_one_component_damage_mean": float(np.mean(max_any)),
                "probability_at_least_one_component_damage_p90": float(np.quantile(max_any, 0.90)),
            }
        )
    return component_rows, error_rows, scenario_rows


def main() -> None:
    args = parse_args()
    cases = load_json(args.cases)
    component_config = load_json(args.components)
    structures = args.structures or cases["main_cases"]
    output_dir = args.output_dir.resolve()
    all_component_rows: list[dict] = []
    all_error_rows: list[dict] = []
    all_scenario_rows: list[dict] = []
    for structure_id in structures:
        component_rows, error_rows, scenario_rows = analyze_structure(
            structure_id,
            cases,
            component_config,
            args.demand_dir.resolve(),
        )
        all_component_rows.extend(component_rows)
        all_error_rows.extend(error_rows)
        all_scenario_rows.extend(scenario_rows)
        print(f"[{structure_id}] P-58 rows: {len(component_rows)}")
    write_csv(output_dir / "o5_multistructure_component_probabilities.csv", all_component_rows)
    write_csv(output_dir / "o5_multistructure_probability_errors.csv", all_error_rows)
    write_csv(output_dir / "o5_multistructure_scenario_summary.csv", all_scenario_rows)
    summary = {
        "structures": structures,
        "n_component_probability_rows": len(all_component_rows),
        "n_probability_error_rows": len(all_error_rows),
        "n_scenario_summary_rows": len(all_scenario_rows),
        "source_component_config": str(args.components.resolve()),
    }
    with (output_dir / "o5_multistructure_p58_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {output_dir / 'o5_multistructure_p58_summary.json'}")


if __name__ == "__main__":
    main()
