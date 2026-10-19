# Paper reproduction package

This directory organizes the experiments accompanying **Beyond Fixed-Grid Accuracy: Fourier Neural Operator Surrogates for Deployment-Oriented Seismic Response Assessment**. It was assembled against the manuscript sources recorded in `provenance/source_manifest.json` on 2026-10-19.

The original repository files remain in the parent directory. Their `neuralop/` and `module/` implementations are used by the experiments here. Do not substitute a separately installed `neuralop` version. This is a reviewable release candidate: [PAPER_MAP.md](PAPER_MAP.md) maps claims to evidence; [AUDIT_ZH.md](AUDIT_ZH.md) identifies remaining manuscript and artifact issues.

## Contents and evidence levels

| Location | Contents |
|---|---|
| `6-code/O1-baseline-comparison` | Six fixed-configuration models, training and evaluation |
| `6-code/O2-data-efficiency` | Ground-motion, amplitude-coefficient and joint scarcity |
| `6-code/O3-generalization` | PGA extrapolation, classical records, sampling-rate transfer |
| `6-code/O4-multi-structure` | Separate FNO training for six structural archetypes |
| `6-code/O5-fema-p58` | Roof PFA, fragility, duration, FRS and period-band screening |
| `6-code/O7-revision/E1` | Synchronized inference latency and memory benchmark |
| `6-code/O7-revision/E2` | Capacity and category-boundary sensitivity |
| `6-code/O7-revision/E3` | HDF5 integrity and intensity coverage audit |
| `6-code/O7-revision/E4` | Direct/wrapped sampling transfer and RoPE control |
| `6-code/O7-revision/E5*` | Nine candidates per architecture, validation selection, final full-data retraining |
| `recorded/` | Historical metrics, configurations, epoch logs, FRS arrays and audit evidence |
| `provenance/` | Original/adapted source hashes, manuscript hashes and pinned public asset inventory |
| `runs/` | Ignored workspace for newly generated results |

Recorded metrics are not regenerated model predictions. Most original O1/O3/O4 checkpoints and the HDF5 data are public; the new E4/E5 checkpoints have not yet been linked to a verified public artifact. The complete downstream FRS arrays (27,018 roof responses × 90 periods per structure) are included so the sensitivity claims can be recomputed without training or downloading the full database.

The six-family tuning extension is included for the author's requested release audit. The manuscript snapshot still describes only FNO, Transformer and BiLSTM tuning. Do not cite the additional three families as already described in that snapshot.

## Install

Use Python 3.12 and a CUDA-compatible PyTorch installation for GPU training. Install the remaining packages with:

```sh
python -m pip install -r paper_reproduction/requirements.txt
```

The requirements specify compatibility ranges, not a historical lockfile. Actual Python, PyTorch, CUDA and GPU versions are recorded in each trial's `config.json`. Different experiments used different GPU/runtime environments; bitwise equality across them is not claimed. Local validation details are in `VALIDATION.md`.

All commands below are run from the repository root. Use absolute paths for user-supplied data/cache arguments: subprocesses run inside a separate workspace. `run.py` supplies the repository imports and copies scripts/configurations into that workspace. Historical evidence stays in `recorded/`; newly produced output is under `runs/workspace/6-code/` by default.

## Quick checks without training

```sh
python paper_reproduction/verify.py
python paper_reproduction/run.py e2-capacity
python paper_reproduction/run.py e2-boundaries
python paper_reproduction/run.py o5-placement
python paper_reproduction/run.py o5-p58
```

`verify.py` checks source hashes, grouped splits, 54 complete screening runs, six complete final runs, validation-only selection and final summary arithmetic. It does not load the trained weights. The two E2 commands recompute the results from the included full FRS arrays; E2's historical `table11` filenames refer to the equipment screening table, currently Table 12 (`tab:o5_frs_matrix`).

## Download data and original weights

The asset downloader pins the public Hugging Face dataset to commit `df5ed30ac6766021cd2bd273dd2638403b70fdab` and verifies each downloaded SHA256. Review file sizes first:

```sh
python paper_reproduction/fetch_assets.py data --list
python paper_reproduction/fetch_assets.py data
python paper_reproduction/fetch_assets.py o1
python paper_reproduction/fetch_assets.py sampling
python paper_reproduction/fetch_assets.py pga
python paper_reproduction/fetch_assets.py o4
```

Data are stored under `paper_reproduction/data/`; weights are placed at the expected paths in `runs/workspace/`. `data/f6_only/` contains **only F-6**. This matters: the historical `DynamicDataset` loads every HDF5 response file in its supplied building directory. Passing the directory containing all six structures would change the benchmark.

## Reproduce the experiments

See [RECIPES.md](RECIPES.md) for training, evaluation, tuning and downstream commands, including exact batch sizes and sampling settings. The original fixed-configuration results and the tuned results are distinct experiments; never replace the manuscript's original tables with tuning metrics without changing their description.

Historical result JSON/CSV files retain their original machine paths as provenance. Those paths are not public download instructions or runnable defaults. The runner and asset downloader construct new paths. `provenance/historical_trainers/` preserves historical trainer versions for source-hash interpretation; it is not an execution interface. Server queues, SSH configuration, local credentials, exploratory GM-descriptor splits, test fixtures and unreported tuned scarcity/transfer runs are excluded.

Scientific plotting helpers preserve the original numerical workflow. The final manuscript PDFs also received manual typography/label edits; exact pixel identity with those PDFs is not claimed. Legacy `FEM`/`fem` field names mean the NTHA reference and are retained to keep saved arrays readable; paper-facing terminology is **NTHA Result**.

## Release status

Before calling this a complete public reproduction release, resolve the specific items in `AUDIT_ZH.md`: integrate the extra tuning results into the paper, link new trained weights and missing complete scarcity artifacts, settle parameter-count wording, and document the reference solver provenance and repository license. These items do not prevent checking the included numerical evidence today.
