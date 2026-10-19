# Seismic-FNO: paper reproducibility

Code, experiment configurations and numerical evidence for **Beyond Fixed-Grid Accuracy: Fourier Neural Operator Surrogates for Deployment-Oriented Seismic Response Assessment**.

This repository is the consolidated workspace for the revised paper. Its `main` branch is the single entry point for the curated code. The earlier development repository, [HKUJasonJiang/Seismic-FNO](https://github.com/HKUJasonJiang/Seismic-FNO), retains the historical machine-specific branches.

## Start here

- [Experiment package and quick checks](paper_reproduction/README.md)
- [Training and evaluation recipes](paper_reproduction/RECIPES.md)
- [Paper sections/tables/figures mapped to code and results](paper_reproduction/PAPER_MAP.md)
- [Author-facing consistency audit (中文)](paper_reproduction/AUDIT_ZH.md)
- [Validation already performed](paper_reproduction/VALIDATION.md)
- [Repository history and branch consolidation](docs/REPOSITORY_HISTORY.md)

`paper_reproduction/recorded/` includes the original fixed-configuration evidence, 54 configuration-screening records, six final tuned runs, full floor-response spectra and downstream numerical summaries. Reproduction scripts write new outputs into an ignored workspace, keeping the historical evidence intact.

The root-level FNO training scripts, `module/` and `neuralop/` retain the original implementation needed by these experiments. The historical usage notes are preserved in [LEGACY_README.md](LEGACY_README.md). The revised-paper recipes take precedence when reproducing the updated manuscript.

## Current status

This is a consolidated reproduction candidate, not a claim that every historical weight and input artifact is already publicly downloadable. Existing datasets and original checkpoints are indexed by pinned revision and SHA256. New E4/E5 checkpoints, complete historical O2 checkpoint coverage, and classical-record input/reference artifacts still require the actions documented in the audit. The manuscript's six-family tuning discussion and parameter-count wording also need alignment.

Use `main` for reviewed paper code. Before a public paper release, resolve the documented artifact gaps, choose the repository license with appropriate third-party notices, and tag a fixed paper version. New machine-specific working copies should synchronize with this repository instead of becoming separate paper entry points.
