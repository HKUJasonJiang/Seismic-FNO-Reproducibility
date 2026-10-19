# Manuscript and code consistency audit

Audit date: 2026-10-19.

This package was assembled from the original repository's `main` commit `e8063eb67a497b35a8f5b8f96a26a19457bb38a1` and checked against the latest historical artifact branch, `sync/paper-artifacts-20260615` (`df60ca2`). The master manuscript and section-file hashes are recorded in `provenance/source_manifest.json`. The consolidated package is maintained on `main` in `HKUJasonJiang/Seismic-FNO-Reproducibility`. The manuscript, response letter and original repository were not modified during this assembly.

## Collected and verified evidence

- Code and configurations for the original benchmark, data scarcity, PGA/sampling-rate extrapolation, six structural archetypes, component fragility probabilities, floor response spectra, exceedance duration and additional E1–E5 experiments.
- Fifty-four complete configuration-screening runs of 20 epochs each and six complete final runs of 50 epochs each. Configurations, epoch logs, results and source hashes are retained; server orchestration, duplicate mirrors and test fixtures are excluded.
- Frozen grouped splits and the complete held-out test set. Checks confirm no GM overlap among training, validation and test groups, validation-MSE candidate selection, and consistency between the aggregate metrics and six individual final runs.
- Original O1/O3 logs retrieved from a pinned public dataset revision, with data/checkpoint URLs, sizes and SHA256 hashes. Large weights are hosted separately from Git.
- E2 recomputed from the complete F10/FW10 FRS arrays, each covering 27,018 records and 90 periods. The 1.0 g table matches within 1e-10 percentage points; agreement across the scanned category boundaries is 94.02–98.29%.

## 1. Integrate all six configuration checks into the manuscript

The audited Section 5.1 paragraph describes FNO, Transformer and BiLSTM tuning. The complete comparison is:

| Architecture | Original R² | Tuned R² | Increase | Real-parameter count ratio |
|---|---:|---:|---:|---:|
| BiLSTM | 0.987250 | 0.987809 | 0.000559 | 1.749 |
| Transformer | 0.972503 | 0.975150 | 0.002646 | 1.397 |
| FNO | 0.942307 | 0.947302 | 0.004994 | 4.000 |
| ResCNN1D | 0.809627 | 0.870106 | 0.060479 | 1.500 |
| UNet1D | 0.722664 | 0.780096 | 0.057432 | 4.008 |
| MLP | 0.046200 | 0.387150 | 0.340949 | 1.777 |

The supported conclusion is that **the six-family accuracy ranking is unchanged**. Improvements are at most 0.005 for the leading three families. The other three improve more substantially but remain below FNO. The 0.005 bound must not be extended to all six families. Tuned metrics should also remain separate from the original configurations and their computational costs.

Suggested manuscript paragraph for author review:

> To assess the sensitivity of the comparison to configuration choice, nine feasible candidate configurations were evaluated for each of the six architectures. Candidates were screened for 20 epochs using the same 19 amplitude levels and grouped training/validation split, and the candidate with the lowest validation MSE in each family was then trained from initialization for 50 epochs using all 57 levels. The selected configurations retained the accuracy ordering BiLSTM, Transformer, FNO, ResCNN1D, UNet1D, and MLP. The test R² improvements for BiLSTM, Transformer, and FNO were 0.0006, 0.0026, and 0.0050, respectively. ResCNN1D, UNet1D, and MLP improved to R² values of 0.8701, 0.7801, and 0.3871, respectively, and remained below FNO. These results support the robustness of the observed accuracy ordering to the evaluated configuration choices. The original configurations are retained for the subsequent comparisons to preserve the benchmark protocol.

This check controls candidate count, epoch count, data splits and numerical precision. It does not match FLOPs or GPU hours or establish an exhaustive optimum. The additional three families are organized in the separate E5 extension directory with their manuscript-integration status stated explicitly.

## 2. State the parameter-count convention

Original Table 3 reports 8,505,793 FNO-Large parameter elements (`numel`), counting each complex element once. The equivalent real-scalar count is 16,927,169. The other architectures use real parameters. Both conventions are retained; E1/E5 expose `real_params`.

The table note should explain this convention. “Parameter-matched” should be qualified as a comparison of configurations of approximately similar size rather than a strict equal-real-parameter budget. Clarifying the counting rule does not require changing the models or recomputing their performance. The selected tuned FNO has 34,018,177 elements and 67,703,681 real scalars.

## 3. Align execution defaults with the archived experiments

- All six O4 configurations used **batch 1024**; the original script default was 1536. The packaged default and explicit recipe use 1024. This setting is separate from the O1 six-model benchmark's batch 64.
- The historical O5 period configuration says 120, but the complete FRS arrays used by the manuscript and E2 have **90 period points**. The public entry point and recipe use 90, while original configuration bytes and the adaptation record are retained.
- FRS and equipment-screening defaults use the three supported period bands reported in the manuscript. Additional proxy bands are excluded from the public recipes.
- F-6 data are supplied through a dedicated response directory to prevent accidental loading of all six structures into the single-structure benchmark.
- Older `placement_supported` outputs use a different period grid and are excluded from the curated evidence. Current equipment screening is recomputed from `frs_full`; the manuscript values are checked against `chapter6_revision` and E2. The three component-probability summary CSVs match the archived results.

These changes align execution defaults with the experiments actually performed. They do not alter saved numerical results or select new conclusions.

## 4. Distinguish sampling protocols and computational measurements

The RoPE paragraph matches E4: matched-grid R² = 0.9729 at 20 Hz and direct-index R² = −0.4818 at 50 Hz. A fixed 15-point patch represents different physical durations at different sampling rates; this comparison does not characterize every Transformer design.

E4 also retains input/output resampling-wrapper controls. The original 20 Hz Transformer reaches direct-50 R² = −0.5242 and wrapped-50 R² = 0.9613. These are different deployment procedures, so direct transfer should be identified consistently in the manuscript.

Table 4's Transformer R² = 0.9725 and Table 7's 50 Hz Transformer R² = 0.9727 come from different training runs. Their results and weights must not be interchanged.

The E1 latency and allocated-memory ratios of approximately 23× and 12× compare BiLSTM with FNO on an RTX 5090 at batch 64. Random initialization is used to measure architecture cost. These measurements are separate from training speed and from accuracy measured using trained weights.

## 5. Match sensitivity claims to the evaluated grid

Section 6.2 reports 94–98% agreement across band–capacity cases and gives the F-10 rigid-equipment disagreement at 0.4 g. Both have numerical support. State the grid increments explicitly: 0.05 g for capacity and one percentage point for each category boundary.

For each boundary pair, the agreement denominator is **six structure–period bands × 39 capacities = 234 cases**. This aggregate does not imply agreement in all six bands at every capacity: at 1.0 g, the minimum over the scanned boundary pairs is 4/6 agreeing bands. The maximum storage-rack exceedance difference is only 1.159 percentage points, but a sufficiently close category boundary can still change the assigned class. Stability is relative to the evaluated scan, rather than an invariance guarantee for arbitrary boundaries.

The implementation uses `Sa >= capacity` and category intervals `<10%`, `[10%, 50%)` and `>=50%`. Wording such as “exceeds 50%” should be aligned with the equality convention. Historical filenames containing `table11` refer to the equipment-screening table, now Table 12; use the stable label `tab:o5_frs_matrix` for mapping.

## 6. Describe the downstream engineering scope precisely

The downstream calculations evaluate component fragility probabilities from predicted peak acceleration demands, sums over selected component classes, acceleration-exceedance durations and equipment-period demand screening. Where documented, component fragility parameters are drawn from FEMA P-58 source materials.

Using those parameter sources does **not** constitute a complete FEMA P-58 performance or loss assessment. Repair cost, downtime and loss estimation are outside the implemented scope. A sum over selected class probabilities is neither a single damage probability nor a loss metric. Public descriptions should use component-level fragility probabilities and demand screening. Historical `p58` filenames and configuration keys are retained for reproducibility and source attribution.

## 7. Remaining release artifacts

| Item | Current evidence and limitation | Required action |
|---|---|---|
| Public E4 RoPE and six E5 final checkpoints | Training, logs and metrics are complete; the public inventory does not yet include these new weights | Archive each best checkpoint, record SHA256 and publish through model hosting or release assets |
| Complete O2 run-level logs/weights | Complete manuscript metrics are included; the public O2 archive contains an older partial E-AC80 subset | Establish the available per-run artifacts for Table 8/Fig. 13. The June 15 artifact index reports no final checkpoints found in its contemporary server search; checkpoint availability cannot be inferred from the complete table |
| Four classical GM inputs and corresponding NTHA histories | Native-resolution evaluation code and metrics are included, but exact inputs lack a public index | Archive the exact input/reference files, units, sampling rates and source records |
| Original Section 4 FNO model-to-figure index | Original code, summary tables and an older Hugging Face model repository exist | Map each Fig. 4–7 series to explicit configurations/checkpoints and hashes |
| Original NTHA generator and parameter chain | HDF5 checks establish stored finite values and coverage, not the historical solver version or convergence/instability criteria | Distinguish training reproduction on the archived database from regenerating that database. Unverified Fortran source is not presented as the original generator |
| Repository license and third-party notices | No root LICENSE is present | Select the project license and preserve applicable third-party notices |

Historical PFA/FRS modules use both 9.81 and 9.80665 m/s² for `g`. Each experiment's original convention is preserved to reproduce its archived values. R² is invariant to uniform rescaling, but errors and capacity thresholds expressed in g should follow the relevant module's convention.

## Excluded material

Descriptor-based GM-OOD experiments, unreported tuned scarcity/transfer runs, server queues/migrations/monitors, failed setup probes, synchronization-conflict files, caches, internal response drafts and unverified solver copies are excluded from the public experiment recipes. Historical source paths in archived logs remain provenance, not execution paths for a new machine.

Refresh the mapping and hashes against the final manuscript before release. This assembly provides a reviewable code and evidence package; it did not rerun the GPU training experiments.
