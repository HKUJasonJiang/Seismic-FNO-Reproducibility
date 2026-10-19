# Reproduction recipes

Run from the repository root. In the examples, replace `/absolute/data` with the directory populated by `fetch_assets.py data`; replace `/absolute/cache` with a new float32 cache directory. `run.py` works on Windows and Linux; command continuations below are omitted for portability.

## Shared data protocol

The data contain 3,474 original GMs × 57 amplitude coefficients = 198,018 rows, each with 3,000 time samples at 50 Hz. Row index = `57 * gm_index + scale_index`. Split by GM before expanding the scale indices: 2,400 train GMs, 600 validation, 474 test, seed 42. Thus full split sizes are 136,800/34,200/27,018. This prevents scaled copies of a test GM entering training. Histories use m/s²; do not add undocumented normalization.

O1–O3 must use a response directory containing only `Blg_F6_18m_IM7_st0.h5`. The downloader creates `f6_only/` for this purpose. Keep the main six-structure files at `MDOF/knet-250/Data/fno/` for O4. Older local paths without the `fno` component are not the public data layout.

## Original fixed-configuration benchmark (Tables 3–4)

```sh
python paper_reproduction/run.py o1-train --models BiLSTM Transformer ResCNN1D UNet1D MLP --batch_size 64 --epochs 50 --base_data_dir /absolute/data --building_dir /absolute/data/f6_only
python paper_reproduction/run.py o1-fno --batch_size 64 --epochs 50 --base_data_dir /absolute/data --building_dir /absolute/data/f6_only
python paper_reproduction/run.py o1-evaluate --models BiLSTM Transformer ResCNN1D UNet1D MLP --fno_checkpoint /absolute/workspace/6-code/O1-baseline-comparison/results/runs/FNO-Large-bs64/checkpoints/FNO-Large-bs64_best.pt --base_data_dir /absolute/data --building_dir /absolute/data/f6_only
```

The default workspace is `paper_reproduction/runs/workspace`; substitute its absolute path in the checkpoint argument. Use `fetch_assets.py o1` for evaluation of the archived weights instead of new training. Epoch checkpoint selection is by validation MSE. FNO learning rate is 0.001; BiLSTM/Transformer are 0.0003. AdamW weight decay 0.0001; StepLR step 20, gamma 0.5. Preserve each historical trainer's own gradient-clipping behavior.

## E1: inference cost

```sh
python paper_reproduction/run.py e1 --out /absolute/benchmark_output
```

This launches isolated model/batch processes for six architectures and batch 1/8/64, FP32, TF32 off, CUDA synchronization, 20 warmups and 100 timed passes. Use a comparable GPU/runtime for meaningful comparison to the RTX 5090 results. The ~23× latency and ~12× allocated-memory ratios compare BiLSTM with FNO at batch 64. These measurements differ from the original training-time/peak-training-memory table.

## E5: equal candidate-count configuration checks

```sh
python paper_reproduction/run.py e5-cache --gm /absolute/data/MDOF/All_GMs/GMs_knet_3474_AF_57.h5 --building /absolute/data/f6_only/Blg_F6_18m_IM7_st0.h5 --output /absolute/cache
python paper_reproduction/tune.py --family FNO --stage dryrun --cache /absolute/cache
python paper_reproduction/tune.py --family FNO --stage screen --cache /absolute/cache
python paper_reproduction/tune.py --family FNO --stage select --cache /absolute/cache
python paper_reproduction/tune.py --family FNO --stage full --cache /absolute/cache
```

Repeat for `Transformer`, `BiLSTM`, `ResCNN1D`, `UNet1D`, `MLP`. The first three use E5; the remaining three use the additional-family source. Both use the same frozen split. The driver reproduces the **nine feasible candidate identities actually evaluated** per family, not all configurations enumerated in `search_space.json`. A GPU that cannot fit a selected candidate must be replaced with a sufficiently large GPU; lowering batch size changes the protocol. No automatic server scheduling is included.

Screen: 19 fixed, approximately evenly spaced amplitude indices; 45,600 train/11,400 validation rows; 20 epochs. Final: reset initialization and train the validation winner with all 57 levels for 50 epochs. Batch 64, seed 42, FP32, TF32 off, AdamW, gradient clip 1.0, StepLR gamma 0.5 with step 8/20 for screen/final. Final test uses all 27,018 held-out rows. `tune.py full` selects from **new** screening validation results; it does not silently use the archived winner or test metrics. For direct candidate execution, `run.py e5[-additional] --help` lists the interface. Truncated-batch runs are rejected by the packaged trainer.

Compare `cache/manifest.json` content hashes with `provenance/training_data_manifest.json` and `E5_additional_20261017/configs/expected_data_manifest.json` before treating a rerun as the same dataset. The runner preserves historical protocol JSON bytes for their hashes; historical E5 notes mention other stages, but only dryrun/screen/full are exposed in this release.

## PGA and sampling-rate extrapolation (Tables 5–7)

```sh
python paper_reproduction/run.py o3-pga --model FNO-Large --min_train_pga 1 --max_train_pga 10 --base_data_dir /absolute/data --building_dir /absolute/data/f6_only
python paper_reproduction/run.py o3-sampling --model FNO-Large --train_hz 20 --base_data_dir /absolute/data --building_dir /absolute/data/f6_only
```

Repeat PGA for Transformer, and sampling for both FNO-Large/Transformer at 20, 25 and 50 Hz. The PGA run selects amplitude levels using their median input PGA in m/s², retaining 37 levels. Sampling uses `scipy.signal.resample_poly`; 20 Hz is up=2/down=5, 25 Hz is up=1/down=2. The target evaluation grid is 50 Hz. FNO-Large here is the original architecture, not E5 FNO_07.

Classical records: `run.py o3-classical --classical_dir /absolute/classical_records`. Preserve native input sampling rates and the original 50 Hz reference responses, which are interpolated for the 100/200 Hz comparisons. This input artifact is not yet publicly indexed; see the audit.

## E4: RoPE control

```sh
python paper_reproduction/run.py e4 --mode train --gm /absolute/data/MDOF/All_GMs/GMs_knet_3474_AF_57.h5 --response /absolute/data/f6_only/Blg_F6_18m_IM7_st0.h5 --output /absolute/rope_output
python paper_reproduction/run.py e4 --mode rotary --gm /absolute/data/MDOF/All_GMs/GMs_knet_3474_AF_57.h5 --response /absolute/data/f6_only/Blg_F6_18m_IM7_st0.h5 --output /absolute/rope_output
python paper_reproduction/run.py e4 --mode original --gm /absolute/data/MDOF/All_GMs/GMs_knet_3474_AF_57.h5 --response /absolute/data/f6_only/Blg_F6_18m_IM7_st0.h5 --output /absolute/rope_output
```

First download the sampling checkpoints into the same workspace. RoPE trains at 20 Hz for 50 epochs. `rotary` also evaluates the original 20 Hz models as matched-grid references. The reported -0.4818 R² is `direct_index` at 50 Hz; `direct_time` and wrapped resampling are separate controls. Fixed patch length remains 15 points. Do not substitute wrapper accuracy for direct native-grid accuracy.

## Data scarcity (Table 8, Fig. 13)

```sh
python paper_reproduction/run.py o2-train --model FNO-Large --run_id E-AC05 --base_data_dir /absolute/data --building_dir /absolute/data/f6_only
```

For FNO-Large and Transformer, run `E-Base` and the `E-AC`, `E-GM`, `E-J` series with suffixes `80`, `60`, `40`, `20`, `10`, `05`. The AC counts are 46/34/23/11/6/3; GM totals are 2400/1800/1200/600/300/150 before train/validation subdivision. The held-out test GMs and all 57 test levels stay fixed. Use the complete `o2_f6_full_results_fno_transformer.csv` as the paper comparison target.

## Six archetypes and downstream checks (Tables 9–12)

```sh
python paper_reproduction/run.py o4-train --structures all --batch_size 1024 --epochs 50 --base_data_dir /absolute/data --data_dir /absolute/data/MDOF/knet-250/Data/fno
python paper_reproduction/run.py --data-root /absolute/data o5-export --structures F10 FW10
python paper_reproduction/run.py o5-frs --structures F10 FW10 --period-count 90
python paper_reproduction/run.py o5-p58
python paper_reproduction/run.py o5-placement
python paper_reproduction/run.py o5-duration
```

Actual O4 runs used **batch 1024**, whereas the old training script default was 1536. The packaged default is corrected to 1024; the command remains explicit. These are separate structure-specific models. `fetch_assets.py o4` can provide the original weights instead of retraining. O5 exports histories from those weights and the fixed test split. Its PFA and FRS arrays are roof-only; no floor-by-floor deployment map is implied.

FRS: 90 logarithmically spaced periods over 0.05–4 s, 5% damping, dt=0.02 s, all 27,018 records, no subsampling. The supported bands are 0.05–0.20, 0.50–1.00 and 1.50–3.00 s. Capacity exceedance is `Sa >= c`; categories are `<10%`, `10% <= q <50%`, and `q >=50%`. Supported band config is used by default. The historical JSON says 120 periods, but the archived analysis overrode it to 90; the public recipe and default use 90.

E2: capacity 0.1–2.0 g by 0.05 g (39 values), lower category boundary 2–25% by 1 pp (24 values), upper boundary 30–80% by 1 pp (51 values). Overall agreement aggregates 6 bands × 39 capacities per boundary pair. At 1.0 g, agreement can be 4/6, 5/6 or 6/6 depending on boundaries. Exact empirical capacity intervals and the discrete boundary grid answer different questions.

## E3: database audit

```sh
python paper_reproduction/run.py e3 --base /absolute/data/MDOF --out /absolute/database_audit
python paper_reproduction/run.py e3-intensity /absolute/database_audit
```

Inspect `audit_database.py --help` for the expected HDF5 layout. Scan finite/nonfinite/zero records and input intensity coverage. A finite stored response is not proof of historical solver convergence or absence of collapse; the public package does not claim to reconstruct that generation chain.
