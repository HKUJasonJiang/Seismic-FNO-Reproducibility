# Seismic-FNO

## Beyond Fixed-Grid Accuracy: Fourier Neural Operator Surrogates for Deployment-Oriented Seismic Response Assessment

**Jun Jiang<sup>1</sup>, Chao Dong<sup>1</sup>, Yixuan Ma<sup>1</sup>, Chen Xiong<sup>2</sup>, Xuguang Wang<sup>1,*</sup>, M. Shahria Alam<sup>3</sup>**

<sup>1</sup> Department of Civil Engineering, The University of Hong Kong, Hong Kong, China<br>
<sup>2</sup> State Key Laboratory of Intelligent Construction and Healthy Operation and Maintenance of Deep Underground Engineering, College of Civil and Transportation Engineering, Shenzhen University, Shenzhen, China<br>
<sup>3</sup> School of Engineering, The University of British Columbia, Kelowna, BC, Canada<br>
<sup>*</sup> Corresponding author: Xuguang Wang

Code, experiment configurations, training records and numerical evidence accompanying the paper. This repository brings the original benchmark and the additional revision experiments into one reproducibility package.

**Keywords:** Fourier Neural Operator · Operator Learning · Dynamic Response Prediction · Seismic Assessment · Nonstructural Components

[Reproduction guide](paper_reproduction/README.md) · [Experiment recipes](paper_reproduction/RECIPES.md) · [Paper-to-code map](paper_reproduction/PAPER_MAP.md) · [Validation record](paper_reproduction/VALIDATION.md) · [Consistency audit](paper_reproduction/AUDIT.md)

## Research overview

Floor acceleration histories are needed to assess the seismic demands on acceleration-sensitive nonstructural components. They support calculations of peak floor acceleration, floor response spectra, fragility probabilities and the duration of acceleration exceedance. Repeated nonlinear time-history analysis (**NTHA**) can make these assessments computationally expensive when many ground motions, intensity levels or structural cases must be considered.

This study evaluates Fourier Neural Operators (**FNOs**) as surrogates that map a ground-motion acceleration history to the corresponding structural roof acceleration history. The central question is **when an FNO provides useful advantages for deployment**, considering prediction accuracy, computational cost, sampling-rate changes, limited training data and downstream engineering assessments.

The evaluation starts with an FNO configuration study and a six-architecture benchmark. It then examines PGA extrapolation, external earthquake records, direct transfer between temporal sampling grids, and reduced training-data coverage. Finally, the same FNO architecture is trained separately for six reinforced-concrete structural archetypes, and its predictions are propagated into component-level fragility and demand-screening calculations.

## Study design

### Database and evaluation protocol

| Item | Description |
|---|---|
| Ground motions | 3,474 K-NET records |
| Intensity variation | 57 amplitude coefficients per ground motion |
| Scaled inputs | 198,018 ground-motion–amplitude combinations per structure |
| Standard temporal grid | 50 Hz; 3,000 samples over a 60 s record |
| Surrogate input/output | Ground-motion acceleration → roof acceleration, in m/s² |
| Benchmark structure | F-6 reinforced-concrete frame |
| Benchmark split | 2,400 training, 600 validation and 474 test GMs; seed 42 |
| Full benchmark sample counts | 136,800 training, 34,200 validation and 27,018 test samples |

The benchmark split is made by **original ground-motion identity** before expanding the amplitude coefficients. Scaled versions of a held-out ground motion therefore remain in the held-out group.

### Models and configuration study

The FNO study varies the configured Fourier-mode parameter, hidden width and number of Fourier layers to examine the balance between waveform accuracy and computational cost. The reference configuration, **FNO-Large**, uses `n_modes=512`, 64 hidden channels and eight Fourier layers.

The comparative benchmark covers six model families: **FNO, BiLSTM, Transformer, ResCNN1D, UNet1D and MLP**. The original fixed-configuration results are preserved separately from the additional configuration-search results. The latter include nine screened candidates per family, validation-based selection and a full-data retraining of each selected configuration.

The six-archetype study covers **F-2, F-6, F-10, FW-10, FW-14 and FW-17**, comprising RC frame and frame–shear-wall structures. A separate surrogate is trained for each archetype; this experiment evaluates the reuse of the architecture and training approach across structures.

## Main findings

### 1. Fixed-grid accuracy establishes the baseline comparison

On the standard F-6 benchmark with both training and testing at 50 Hz, BiLSTM and Transformer achieve higher pointwise accuracy than FNO-Large. The following values are from the paper's **original fixed-configuration benchmark**, before the additional configuration searches.

| Model | Time-history MSE | Time-history R² | PFA R² |
|---|---:|---:|---:|
| BiLSTM | 0.0099 | 0.9872 | 0.9876 |
| Transformer | 0.0214 | 0.9725 | 0.9799 |
| FNO-Large | 0.0449 | 0.9423 | 0.9590 |
| ResCNN1D | 0.1481 | 0.8096 | 0.8874 |
| UNet1D | 0.2157 | 0.7227 | 0.5740 |
| MLP | 0.7420 | 0.0462 | −1.2870 |

MSE is computed on acceleration histories in m/s² and has units of (m/s²)². Full metrics, training costs and parameter-count conventions are available in the [original benchmark summary](paper_reproduction/recorded/6-code/O1-baseline-comparison/results/report/o1_report_summary.csv).

### 2. FNO maintains useful accuracy under direct sampling-rate transfer

The sampling-resolution experiment trains at 20 or 25 Hz and evaluates directly on the 50 Hz grid. FNO-Large retains high accuracy under these changes, while the evaluated patch-based Transformer degrades substantially.

| Training grid → evaluation grid | FNO-Large R² | Transformer R² |
|---|---:|---:|
| 20 Hz → 50 Hz | 0.9319 | −0.5242 |
| 25 Hz → 50 Hz | 0.9374 | −0.3486 |

These are **direct-transfer** results. Input/output resampling wrappers are evaluated separately in the supporting experiments. A Transformer control using rotary positional embeddings (RoPE) achieves R² = 0.9729 on its 20 Hz grid and R² = −0.4818 under direct-index evaluation at 50 Hz. Both Transformer variants retain a fixed 15-point patch length, so the physical duration represented by a patch changes with sampling rate. The conclusion concerns these evaluated implementations.

See the [sampling-resolution results](paper_reproduction/recorded/6-code/O3-generalization/results/sampling_ood/sampling_ood_overall_summary.csv) and [RoPE/control evidence](paper_reproduction/recorded/6-code/O7-revision/E4/results/).

### 3. Deployment assessment extends beyond a single accuracy metric

- **Data scarcity:** FNO-Large degrades more smoothly than Transformer under severe reductions in ground-motion diversity, amplitude-level coverage and their combination. The held-out test set stays fixed across these comparisons.
- **Inference cost:** on the recorded RTX 5090 benchmark at batch 64, BiLSTM has approximately 23 times the median inference latency and 12 times the peak allocated GPU memory of FNO-Large. These are architecture-cost measurements under the recorded benchmark settings.
- **Structural archetypes:** separately trained FNO-Large models achieve time-history R² values of approximately 0.864–0.926 and PFA R² values of approximately 0.929–0.953 across the six archetypes.
- **Engineering outputs:** predicted roof histories are used to calculate component fragility probabilities, cumulative acceleration-exceedance durations, and 5%-damped floor response spectra for equipment-period screening.

The downstream scope is component-level fragility and demand screening. Repair-cost, downtime and loss estimation are outside this study.

For the equipment-screening sensitivity study, capacities span 0.1–2.0 g in 0.05 g increments. Lower category boundaries span 2–25% and upper boundaries span 30–80%, both in one-percentage-point increments. Across this grid, NTHA- and FNO-based categories agree in **94.02–98.29%** of the aggregated band–capacity cases for each boundary pair. Short-period rigid-equipment cases close to a category boundary can differ and warrant reference NTHA verification.

Together, these experiments support FNO as a useful surrogate for preliminary response prediction and downstream screening when sampling-grid changes, limited data and computational resources matter.

## Additional experiments and saved evidence

The reproduction package includes the inference benchmark, capacity/category-boundary sensitivity, database integrity and intensity-coverage audit, RoPE/resampling controls, and the six-family configuration check.

For the configuration check, **54 screening runs** use 20 epochs and the same 19 amplitude levels. Each family's validation winner is then trained from initialization for **50 epochs using all 57 levels**. The six final runs preserve the accuracy ordering BiLSTM, Transformer, FNO, ResCNN1D, UNet1D and MLP. Their configurations, epoch logs and [final metrics](paper_reproduction/recorded/six_architecture_tuning.csv) are stored separately from the original paper benchmark.

The checked manuscript snapshot describes tuning for the first three families; the additional three-family extension is retained for the manuscript update. The [consistency audit](paper_reproduction/AUDIT.md) records this distinction and the remaining items to align before release.

## Repository layout

```text
Seismic-FNO-Reproducibility/
├── README.md
├── module/                         # Original data preparation and training utilities
├── neuralop/                       # FNO implementation used by the experiments
├── paper_reproduction/
│   ├── 6-code/
│   │   ├── O1-baseline-comparison/
│   │   ├── O2-data-efficiency/
│   │   ├── O3-generalization/
│   │   ├── O4-multi-structure/
│   │   ├── O5-fema-p58/
│   │   └── O7-revision/            # Additional E1–E5 experiments
│   ├── recorded/                  # Historical metrics, configurations, logs and FRS arrays
│   ├── provenance/                # File hashes, source versions and artifact inventories
│   ├── run.py                     # Experiment runner with an isolated output workspace
│   ├── fetch_assets.py            # Pinned data/checkpoint downloads with SHA256 checks
│   ├── tune.py                    # Candidate screening and validation-based selection
│   ├── verify.py                  # Recorded-evidence and split checks
│   └── smoke.py                   # CPU architecture and metric checks
└── docs/                           # Repository consolidation history
```

The root-level FNO training scripts retain the original hyperparameter-study workflow. Their historical instructions are in [LEGACY_README.md](LEGACY_README.md); use the [revised-paper recipes](paper_reproduction/RECIPES.md) for the experiments organized in this package.

## Getting started

Clone this repository and enter its root directory:

```sh
git clone https://github.com/HKUJasonJiang/Seismic-FNO-Reproducibility.git
cd Seismic-FNO-Reproducibility
```

Use Python 3.12 and install a PyTorch build appropriate for your hardware. GPU training requires a compatible CUDA environment. Then install the remaining dependencies:

```sh
python -m pip install -r paper_reproduction/requirements.txt
```

Verify the saved evidence and run the CPU architecture checks:

```sh
python paper_reproduction/verify.py
python paper_reproduction/smoke.py
```

The included full FRS arrays also support numerical reproduction without downloading the full training database:

```sh
python paper_reproduction/run.py e2-capacity
python paper_reproduction/run.py e2-boundaries
python paper_reproduction/run.py o5-p58
python paper_reproduction/run.py o5-placement
```

New outputs are written under `paper_reproduction/runs/`, separate from the saved evidence. For training, checkpoint evaluation, data layout and exact experimental settings, follow [RECIPES.md](paper_reproduction/RECIPES.md).

## Data, checkpoints and reproducibility status

The source data and available original checkpoints are hosted in the [JasonXF/SeFNO dataset repository](https://huggingface.co/datasets/JasonXF/SeFNO). This package provides a [pinned artifact inventory](paper_reproduction/provenance/public_assets.json) with file sizes and content hashes. Review the download plan before retrieving the full database:

```sh
python paper_reproduction/fetch_assets.py data --list
```

Saved evidence has been checked for complete epoch sequences, grouped-split integrity, validation-based candidate selection and agreement with the result summaries. CPU model checks and numerical downstream recomputations are documented in [VALIDATION.md](paper_reproduction/VALIDATION.md). These checks do not constitute a new GPU training run.

Publication links for the new RoPE/tuned checkpoints, complete historical O2 checkpoint coverage, and the classical-record input/reference artifacts remain to be completed. The [artifact and manuscript audit](paper_reproduction/AUDIT.md) lists their status. The final public release will also specify the repository license and retain the relevant third-party notices.

This repository's **`main` branch** is the unified entry point for the revised-paper code. The [earlier development repository](https://github.com/HKUJasonJiang/Seismic-FNO) retains the original machine-specific history; the [consolidation record](docs/REPOSITORY_HISTORY.md) explains how its branches relate to this package.
