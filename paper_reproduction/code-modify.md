# Packaging change record

## 2026-10-19 — manuscript reproducibility collection

- Added an independent `paper_reproduction/` folder and a link in the repository README. Existing scientific source in the parent repository was retained.
- Selected paper-relevant O1–O5 and revision E1–E5 scripts/configs and complete numerical evidence. Kept fixed-configuration runs separate from newly tuned runs.
- Preserved original source and adapted source SHA256 values; preserved frozen protocol and split bytes. Historical results were not rewritten to match manuscript prose.
- Added an isolated workspace runner, pinned public-asset downloader, validation-based tuning driver, evidence checker, CPU model checks, paper mapping and author-facing audit.
- Adapted paths/import context; removed private delegated-server waits from executable trainers; restricted the public tuning interface to dryrun/screen/full and rejected partial-batch runs.
- Aligned O4 batch default with the recorded 1024 and O5 FRS default with the recorded 90 periods and supported bands. Resolved legacy history paths to local siblings and kept figure exports in the run workspace.
- Excluded the old placement outputs with a different period grid, partial O2 snapshots, unreported experiments, server orchestration and test fixtures. The archived full FRS and manuscript table summaries remain the authoritative downstream inputs.
- Added no new model training, changed no manuscript/response text, and did not push or publish.

Detailed results and remaining release gaps are in `VALIDATION.md` and `AUDIT_ZH.md`. No claim of complete checkpoint-level reproduction is made for artifacts still awaiting publication.

## 2026-10-19 — independent repository

- Created `HKUJasonJiang/Seismic-FNO-Reproducibility` as a private repository with one `main` branch, at the author's request to consolidate the machine-specific history.
- Verified all six original remote branch tips and confirmed `df60ca2` is the latest historical artifact branch. The final O2 summary matches that branch.
- Preserved exact source/evidence bytes in Git and scoped workspace ignore rules so historical `recorded/.../runs/` directories are included. All 436 source/evidence entries were checked against the staged Git blobs before upload.
- Retained the old repository unchanged; added repository history and provenance references instead of merging machine queues into the new release.
