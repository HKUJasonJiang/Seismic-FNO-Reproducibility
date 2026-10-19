# Repository history and consolidation

The original GitHub repository was checked on 2026-10-19. Its default branch was `main`, but its newest branch-tip commit was on `sync/paper-artifacts-20260615`. Dates below use Asia/Shanghai time.

| Original branch | Tip | Commit time | Role |
|---|---|---|---|
| `sync/paper-artifacts-20260615` | `df60ca2` | 2026-06-15 15:42 | Latest consolidated artifact index; includes the O2 final results and earlier machine merges |
| `sync/xgb1-o2-final` | `1fbde5f` | 2026-06-15 14:45 | Final O2 data-efficiency results |
| `revise` | `13ca29a` | 2026-06-09 15:11 | Earlier revision development line |
| `sync/xgb1-o4` | `1911060` | 2026-06-09 14:46 | Six-structure O4 work |
| `sync/xgb2-o1-o3-o2partial` | `df66472` | 2026-06-09 14:45 | O1/O3 and partial O2 work |
| `main` | `e8063eb` | 2026-03-05 19:00 | Original published code; not the latest experimental collection |

The Git history confirms that the latest artifact branch contains the earlier O4 and O1/O3 branch merges followed by the final O2 result commit. Choosing `main` merely because it is the default would miss these later additions.

This new repository consolidates the curated revised-paper package, including the October revision experiments. The complete FNO/Transformer O2 summary in the package was compared against `df60ca2` and matches. [The historical artifact index](legacy_artifacts_20260615.md) records that the final O2 collection was a report/results package: its contemporary server search did not find final checkpoints. Their availability must therefore not be inferred from the existence of the complete result table.

The original repository and its branches remain unchanged. This repository starts a clean history on `main` while retaining upstream commit IDs, source hashes and recorded experimental provenance. Current experiment coverage and limitations are in the paper package, rather than the historical artifact index.
