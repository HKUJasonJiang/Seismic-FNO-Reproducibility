"""Export FNO and FEM demand histories for the O5 FEMA P-58 workflow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


G = 9.80665
TOTAL_GMS = 3474
TOTAL_SCALES = 57
TEST_GMS = 474

REPO_ROOT = Path(__file__).resolve().parents[3]
O5_DIR = REPO_ROOT / "6-code" / "O5-fema-p58"
UPSTREAM_DIR = REPO_ROOT / "6-code" / "shared" / "upstream" / "Seismic-FNO-clean"
sys.path.insert(0, str(UPSTREAM_DIR))

DEFAULT_CONFIG = O5_DIR / "configs" / "o5_multistructure_cases.json"
DEFAULT_ARTIFACT_INDEX = O5_DIR / "input" / "o4_artifacts" / "artifact_index.json"
DEFAULT_OUTPUT_DIR = O5_DIR / "results" / "multistructure" / "demands"


class SingleBuildingDataset(Dataset):
    """Lazy HDF5 dataset for one structure and selected global sample indices."""

    def __init__(self, gm_h5: str | Path, building_h5: str | Path, indices: np.ndarray):
        self.gm_h5 = str(gm_h5)
        self.building_h5 = str(building_h5)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.gm_file = None
        self.building_file = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _open(self) -> None:
        self.gm_file = h5py.File(self.gm_h5, "r")
        self.building_file = h5py.File(self.building_h5, "r")

    def __getitem__(self, item: int):
        if self.gm_file is None or self.building_file is None:
            self._open()
        idx = int(self.indices[item])
        gm = torch.from_numpy(self.gm_file["Acc_GMs"][idx]).float().unsqueeze(0)
        response = torch.from_numpy(self.building_file["Acc_Floor_Response"][idx]).float().unsqueeze(0)
        return gm, response, idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-index", type=Path, default=DEFAULT_ARTIFACT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--structures", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the full O4 fixed test set.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_o4_test_indices(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    shuffled_gms = rng.permutation(TOTAL_GMS)
    test_gms = shuffled_gms[-TEST_GMS:]
    gm_grid, scale_grid = np.meshgrid(test_gms, np.arange(TOTAL_SCALES), indexing="ij")
    return (gm_grid * TOTAL_SCALES + scale_grid).reshape(-1).astype(np.int64)


def build_fno_large() -> torch.nn.Module:
    try:
        from neuralop.models import FNO
    except ModuleNotFoundError as exc:
        if exc.name == "tltorch":
            raise ModuleNotFoundError(
                "FNO-Large requires tensorly-torch (`tltorch`). "
                "Install it in the torch environment first."
            ) from exc
        raise
    return FNO(
        n_modes=(512,),
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        projection_channel_ratio=2,
        n_layers=8,
        domain_padding=0.1,
    )


def iter_batches(loader: DataLoader) -> Iterable[tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    for gm, response, indices in loader:
        yield gm, response, indices.numpy()


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    model = build_fno_large().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def pfa_g(history_ms2: np.ndarray) -> np.ndarray:
    return np.max(np.abs(history_ms2), axis=-1) / G


def export_structure(
    structure_id: str,
    config: dict,
    artifact_index: dict,
    indices: np.ndarray,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict:
    structure = config["structures"][structure_id]
    artifact = artifact_index["structures"][structure_id]
    building_h5 = Path(structure["local_hdf5_path"])
    gm_h5 = Path(config["gm_local_path"])
    checkpoint_path = Path(artifact["checkpoint_path"])
    if not building_h5.exists():
        raise FileNotFoundError(building_h5)
    if not gm_h5.exists():
        raise FileNotFoundError(gm_h5)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    dataset = SingleBuildingDataset(gm_h5, building_h5, indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    model = load_model(checkpoint_path, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(indices)
    with h5py.File(building_h5, "r") as f:
        response_shape = f["Acc_Floor_Response"].shape
        time_steps = int(response_shape[-1])
        if len(response_shape) != 2:
            raise ValueError(
                f"Expected roof-only response shape [sample, time], got {response_shape}"
            )

    fem_history_path = output_dir / f"{structure_id}_fem_roof_history_ms2.npy"
    fno_history_path = output_dir / f"{structure_id}_fno_roof_history_ms2.npy"
    fem_history = np.lib.format.open_memmap(
        fem_history_path, mode="w+", dtype="float32", shape=(n, 1, time_steps)
    )
    fno_history = np.lib.format.open_memmap(
        fno_history_path, mode="w+", dtype="float32", shape=(n, 1, time_steps)
    )

    pfa_rows = []
    cursor = 0
    with torch.no_grad():
        for gm, target, batch_indices in iter_batches(loader):
            gm = gm.to(device, non_blocking=True)
            pred = model(gm).detach().cpu().numpy().astype("float32")
            fem = target.numpy().astype("float32")
            batch_n = int(fem.shape[0])
            fem_history[cursor : cursor + batch_n] = fem
            fno_history[cursor : cursor + batch_n] = pred
            fem_pfa = pfa_g(fem[:, 0, :])
            fno_pfa = pfa_g(pred[:, 0, :])
            for local_i, global_idx in enumerate(batch_indices):
                global_idx = int(global_idx)
                pfa_rows.append(
                    {
                        "structure_id": structure_id,
                        "paper_label": structure.get("paper_label", structure_id),
                        "record_index": cursor + local_i,
                        "sample_index": global_idx,
                        "gm_index": global_idx // TOTAL_SCALES,
                        "scale_index": global_idx % TOTAL_SCALES,
                        "floor_index": structure.get("stories", 0) + 1,
                        "floor_label": structure.get("roof_floor_label", "roof"),
                        "is_roof": 1,
                        "fem_pfa_g": float(fem_pfa[local_i]),
                        "fno_pfa_g": float(fno_pfa[local_i]),
                        "pfa_error_g": float(fno_pfa[local_i] - fem_pfa[local_i]),
                        "pfa_abs_error_g": float(abs(fno_pfa[local_i] - fem_pfa[local_i])),
                    }
                )
            cursor += batch_n
            print(f"[{structure_id}] exported {cursor}/{n} histories")

    fem_history.flush()
    fno_history.flush()
    pfa_csv = output_dir / f"{structure_id}_roof_pfa_demands.csv"
    with pfa_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pfa_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pfa_rows)

    fem_pfas = np.array([row["fem_pfa_g"] for row in pfa_rows], dtype=float)
    fno_pfas = np.array([row["fno_pfa_g"] for row in pfa_rows], dtype=float)
    abs_err = np.abs(fno_pfas - fem_pfas)
    summary = {
        "structure_id": structure_id,
        "paper_label": structure.get("paper_label", structure_id),
        "n_records": int(n),
        "response_scope": config.get("response_scope", "roof_only"),
        "time_steps": time_steps,
        "dt_s": config["dt_s"],
        "fem_history_path": str(fem_history_path.resolve()),
        "fno_history_path": str(fno_history_path.resolve()),
        "pfa_csv": str(pfa_csv.resolve()),
        "fem_pfa_mean_g": float(np.mean(fem_pfas)),
        "fem_pfa_p50_g": float(np.quantile(fem_pfas, 0.50)),
        "fem_pfa_p90_g": float(np.quantile(fem_pfas, 0.90)),
        "fem_pfa_p95_g": float(np.quantile(fem_pfas, 0.95)),
        "fno_pfa_mean_g": float(np.mean(fno_pfas)),
        "fno_pfa_p50_g": float(np.quantile(fno_pfas, 0.50)),
        "fno_pfa_p90_g": float(np.quantile(fno_pfas, 0.90)),
        "fno_pfa_p95_g": float(np.quantile(fno_pfas, 0.95)),
        "pfa_mae_g": float(np.mean(abs_err)),
        "pfa_rmse_g": float(np.sqrt(np.mean((fno_pfas - fem_pfas) ** 2))),
        "pfa_error_p95_abs_g": float(np.quantile(abs_err, 0.95)),
    }
    summary_path = output_dir / f"{structure_id}_demand_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    artifact_index = load_json(args.artifact_index)
    structure_ids = args.structures or config["main_cases"]
    indices = make_o4_test_indices(args.seed)
    if args.max_records is not None:
        indices = indices[: args.max_records]
    output_dir = args.output_dir.resolve()
    summaries = []
    for structure_id in structure_ids:
        summaries.append(
            export_structure(
                structure_id=structure_id,
                config=config,
                artifact_index=artifact_index,
                indices=indices,
                output_dir=output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
            )
        )
    with (output_dir / "o5_demand_export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split": "O4 fixed test set",
                "test_gms": TEST_GMS,
                "total_scales": TOTAL_SCALES,
                "n_records_per_structure": int(len(indices)),
                "summaries": summaries,
            },
            f,
            indent=2,
        )
    print(f"[summary] {output_dir / 'o5_demand_export_summary.json'}")


if __name__ == "__main__":
    main()
