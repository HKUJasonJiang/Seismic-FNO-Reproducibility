# Validation performed for this assembly

Validation date: 2026-10-19. These checks verify the curated package and saved evidence. GPU training was **not** rerun; the original metrics remain attributed to their historical runs.

| Check | Outcome |
|---|---|
| Python syntax for curated scientific scripts | Passed |
| CLI/import checks for 19 argument-based experiment entry points | Passed |
| Original six architectures: actual 3,000-point CPU forward, finite outputs, parameter counts | Passed; both element and real-scalar counts match the recorded O1 table |
| Six selected tuned architectures: actual 3,000-point CPU forward, finite outputs, real parameter counts | Passed |
| Regression/PFA metrics against a closed-form synthetic example | Passed |
| Recreate the seed-42 grouped split and compare with frozen indices | Exact match; no GM overlap between train/validation/test |
| 54 screening runs and six final runs | All complete, 20/50 contiguous epochs, finite training/validation losses, recorded configuration matches the frozen candidate, validation winner matches final run |
| Six-family final metrics and parameter counts | Match the six separate full-run summaries |
| Public GM/F6 files vs E5 historical source hashes | SHA256 matches for both files |
| E2 capacity calculation from full FRS arrays | Original 1.0 g table reproduced within 1e-10 percentage points; four comparison CSVs match |
| E2 boundary grid | CSV exactly matches the archived 1,224 boundary pairs; aggregate agreement 94.02%–98.29% |
| O5 P-58 recomputation | All three archived CSVs match within 1e-10 |
| O5 equipment screening with full 90-period FRS | Runs successfully; older placement outputs used a different grid and were excluded, rather than used as verification targets |
| Lightweight original O1/O3 evidence download | 86 CSV/JSON files retrieved at the pinned dataset revision |

Machine-readable reports are stored under `provenance/validation/`. `provenance/local_validation_environment.json` records the local package versions: Python 3.12.13, PyTorch 2.13.0+cu130 (CPU forwards), NumPy 2.5.1, SciPy 1.18.0 and pandas 3.0.5. These are validation-environment versions, not a replacement for the historical training environments in the archived configs.

Re-run the evidence and architecture checks with:

```sh
python paper_reproduction/verify.py
python paper_reproduction/smoke.py
```

See README and RECIPES for numerical postprocessing and training. Large public HDF5/checkpoint downloads, checkpoint-based full-test inference, six-family GPU retraining, original NTHA database generation, complete O2 checkpoint recovery and exact final-PDF typography were not performed during this assembly. They must not be inferred from passing these checks.
