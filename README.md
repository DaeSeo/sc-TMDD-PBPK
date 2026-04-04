# sc-TMDD-PBPK

A computational platform for **single-cell resolved Physiologically Based Pharmacokinetic (PBPK)** modelling with **target-mediated drug disposition (TMDD)** de-risking. The platform estimates cell-type-specific receptor concentrations (nM) by integrating single-cell RNA-seq data with bulk tissue proteomics through joint cross-tissue Bayesian MCMC inference, then simulates per-cell-type drug–target engagement dynamics across human tissues — and generates publication-quality figures.

---

## Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Workflow](#workflow)
  - [Step 1 — Bayesian Data Pipeline](#step-1--bayesian-data-pipeline-bayesian_pipelinepy)
  - [Step 2 — PBPK Simulation](#step-2--pbpk-simulation-run_simulationpy)
  - [Step 3 — Figure Generation](#step-3--figure-generation-generate_figurespy)
- [Key Modules](#key-modules)
- [Output Columns](#output-columns)
- [Figures Generated](#figures-generated)
- [Methodology](#methodology)
- [References](#references)

---

## Overview

Conventional PBPK–TMDD models parameterise receptor concentrations using bulk tissue averages, implicitly assuming homogeneous expression within each tissue. This platform resolves that limitation by:

1. **Bayesian deconvolution** — A joint hierarchical MCMC model decomposes bulk tissue proteomics (PaxDb, PPM) into cell-type-level protein concentrations using single-cell RNA-seq proportions (HPA, nCPM) with shared translation-efficiency parameters (θ) across tissues.
2. **Tissue-specific PPM → nM conversion** — Tissue-specific protein densities (ρ, mg/mL) and average proteome molecular weights convert PPM to molar concentrations: `C (nM) = PPM × ρ × 10³ / MW_avg`.
3. **Per-cell-type PBPK–TMDD** — A whole-body QSS-TMDD model (Gibiansky formulation) resolves receptor dynamics for every cell type in every tissue simultaneously, enabling direct comparison of single-cell (SC) vs. bulk TMDD risk (R₀ / K_SS).
4. **Publication-quality visualisation** — A dedicated visualiser generates global PK figures, TMDD index heatmaps, SC/Bulk fold-change panels, per-tissue receptor dynamics, NCA PK parameter tables, and full multi-dose comparison figures.

---

## Folder Structure

```
sc-TMDD-PBPK/
├── inference/
│   ├── __init__.py
│   ├── bayesian_ppm.py         # Joint cross-tissue Bayesian MCMC (PyMC/NUTS)
│   └── cell_volume.py          # BioNumbers-based cell volume & N_total database
├── pbpk/
│   ├── __init__.py
│   ├── config.py               # Drug/Target DB, tissue physiology, dosing regimens
│   ├── equations.py            # QSS-TMDD ODE system (Gibiansky total-drug)
│   └── simulator.py            # Radau ODE solver with event-driven multi-dose
├── .gitignore
├── HPA_Single_Cell.parquet     # Pre-downloaded HPA single-cell RNA-seq data
├── PaxDb_data.parquet          # Pre-downloaded PaxDb bulk tissue proteomics
├── LICENSE
├── bayesian_pipeline.py                     # Step 1: Data loading, merging & Bayesian inference
├── run_simulation.py                # Step 2: PBPK simulation (single-cell + bulk)
├── requirements.txt
├── README.md
└── generate_figures.py            # Step 3: High-resolution figure generation
```

---

## Installation

Requires **Python 3.11+**. A virtual environment is strongly recommended.

```bash
git clone https://github.com/[YOUR_USERNAME]/sc-TMDD-PBPK.git
cd sc-TMDD-PBPK

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `pymc` (v5+) | Bayesian MCMC sampling (NUTS) |
| `arviz` | Posterior diagnostics (R-hat, ESS, HDI) |
| `scipy` | Radau implicit ODE solver |
| `pandas` / `numpy` | Data manipulation |
| `pyarrow` | Parquet file I/O |
| `matplotlib` / `seaborn` | Publication-quality figure generation |

---

## Workflow

### Step 1 — Bayesian Data Pipeline (`bayesian_pipeline.py`)

Loads HPA single-cell and PaxDb bulk proteomics parquet files, merges them on tissue name, and runs joint cross-tissue MCMC inference to produce cell-type-resolved receptor concentrations.

```bash
# EGFR with default parquet paths
python bayesian_pipeline.py --target EGFR

# ERBB2 with custom MCMC settings
python bayesian_pipeline.py --target ERBB2 --draws 3000 --tune 3000

# Merge only — skip MCMC (useful for debugging)
python bayesian_pipeline.py --target EGFR --skip-inference

# Custom parquet paths
python bayesian_pipeline.py --target EGFR \
    --hpa-path data/HPA_Single_Cell.parquet \
    --pax-path data/PaxDb_data.parquet
```

**Output:** `data/Final_<TARGET>_Data.csv`

| Column | Description |
|--------|-------------|
| `local_ppm` | Bayesian posterior mean PPM per cell type |
| `local_ppm_hdi_low/high` | 94% HDI credible interval |
| `nM_concentration` | `local_ppm × K_organ` — single-cell tissue nM |
| `bulk_nM_concentration` | `bulk_ppm × K_organ` — tissue-average nM (PBPK R₀ input) |
| `actual_counts_per_cell` | `local_ppm × N_total × 1e-6` — molecules per cell |
| `volume_fraction` | Protein mass-weighted cell-type fraction |
| `converged` | MCMC convergence flag (R-hat < 1.01, ESS ≥ 400, 0 divergences) |

#### PaxDb Tissue Column Mapping

`bayesian_pipeline.py` maps PaxDb wide-format columns to canonical tissue names used throughout the pipeline:

| PaxDb Column | Canonical Name |
|---|---|
| `abundance_LUNG` | lung |
| `abundance_KIDNEY` | kidney |
| `abundance_PLASMA` | plasma |
| `abundance_SKIN` | skin |
| `abundance_BRAIN` | brain |
| `abundance_HEART` | heart |
| `abundance_LIVER` | liver |
| `abundance_PBMC` | blood |

---

### Step 2 — PBPK Simulation (`run_simulation.py`)

Runs the whole-body PBPK model with both single-cell and bulk receptor parameterisations in parallel.

#### Single-dose mode

```bash
# Default: cetuximab 400 mg/m², EGFR, 120 days
python run_simulation.py

# Custom drug / target / dose
python run_simulation.py --Drug cetuximab --Target EGFR --Dose 100 --Days 120

# Clinical dosing regimen
python run_simulation.py --Drug cetuximab --Target EGFR --Regimen loading_q1w --Days 84
```

#### Multi-dose mode

Each dose is simulated as an **independent single-dose experiment** (not repeated dosing).

```bash
python run_simulation.py --multi --doses 0.1,1,10,100 --Drug cetuximab --Target EGFR --Days 120
```

**Output:** `data/pbpk_<target>_<drug>_<dose>mg.csv` per dose — time-series of plasma concentration, tissue ISF concentration, free receptor [R], bound complex [RC], receptor occupancy (%), and TMDD index for every cell type in every tissue.

#### CLI Reference (`run_simulation.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--File` | `data/Final_EGFR_Data.csv` | Input CSV from Bayesian pipeline |
| `--Drug` | `Cetuximab` | Drug name (must exist in `config.py DRUG_DB`) |
| `--Target` | `EGFR` | Target name (must exist in `config.py TARGET_DB`) |
| `--Dose` | `None` | Custom dose mg/m² (overrides drug default) |
| `--Regimen` | `single` | Dosing regimen (`single`, `q1w`, `q2w`, `loading_q1w`, …) |
| `--Days` | `120` | Simulation duration (days) |
| `--multi` | `False` | Run independent single-dose simulations per dose |
| `--doses` | `0.1,1,10,100,500` | Comma-separated dose list for `--multi` (mg/m²) |

---

### Step 3 — Figure Generation (`generate_figures.py`)

Generates publication-quality figures (300 dpi, linear y-axes, 40-colour purple ramp palette).

#### Single-dose mode

```bash
# All tissues
python generate_figures.py --Target EGFR --Drug Cetuximab --Dose 400 --Tissue all

# Specific tissue
python generate_figures.py --Target EGFR --Drug Cetuximab --Dose 400 --Tissue Liver
```

#### Multi-dose comparison mode

```bash
python generate_figures.py --Target EGFR --Drug Cetuximab \
    --multi --doses 0.1,1,10,100 --Tissue all
```

#### CLI Reference (`generate_figures.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--Target` | `EGFR` | Target name |
| `--Drug` | `Cetuximab` | Drug name |
| `--Dose` | `400.0` | Dose (mg/m²) for single-dose mode |
| `--Tissue` | `all` | Tissue name or `all` |
| `--File` | auto-detect | Direct CSV path (overrides auto-detection) |
| `--multi` | `False` | Multi-dose comparison mode |
| `--doses` | `0.1,1,10,100,500` | Comma-separated doses for `--multi` |

---

## Key Modules

### `inference/bayesian_ppm.py`

Implements the joint cross-tissue hierarchical Bayesian model using PyMC (v5) with NUTS sampling.

**Statistical model:**

```
ppm_{i,t}   = alpha_t × scaled_nCPM_{i,t} × theta_i
bulk_pred_t = Σ_i (ppm_{i,t} × fraction_i)
obs_t       ~ Normal(bulk_pred_t, sigma = bulk_t × 0.30)
```

- **`alpha_t`** — tissue-level scaling factor (`HalfNormal` prior, σ = `bulk_ppm × 2`)
- **`theta_i`** — cell-type translation efficiency; non-centred: `z_i ~ Normal(0, 1)`, `theta_i = exp(0.55 × z_i)`
- Shared `theta` across tissues for cell types appearing in ≥ 2 tissues (cross-tissue information pooling)
- Convergence: R-hat < 1.01, ESS ≥ 400, zero divergent transitions

**K_ORGAN constants** (`PPM → nM`, `K = ρ × 10³ / MW_avg`):

| Tissue | ρ (mg/mL) | MW_avg (Da) | K_organ |
|--------|-----------|-------------|---------|
| Liver | 150 | 52,000 | 2.885 |
| Kidney | 120 | 50,000 | 2.400 |
| Heart | 130 | 50,000 | 2.600 |
| Lung | 100 | 50,000 | 2.000 |
| Brain | 100 | 50,000 | 2.000 |
| Skin | 80 | 50,000 | 1.600 |
| Breast | 60 | 50,000 | 1.200 |
| Blood / Plasma | 70 | 55,000 | 1.273 |

**Validation:** EGFR skin fibroblast: `local_ppm ≈ 22.7`, `N_total = 2×10⁹` → ~45,400 molecules/cell (literature: ~50,000 by flow cytometry).

---

### `inference/cell_volume.py`

Literature-based cell volume (µm³) and total protein molecule count (`N_total`) database used to convert local PPM to absolute molecule counts per cell.

**Formula:**
```
actual_counts_per_cell = local_ppm × N_total × 1e-6
```

| Cell class | Volume (µm³) | N_total (molecules) | Primary source |
|---|---|---|---|
| Adipocyte | 200,000 | 2.0 × 10⁹ | Snijder et al. 2009 |
| Cardiomyocyte | 20,000 | 4.0 × 10⁹ | Snijder et al. 2009 |
| Hepatocyte | 10,000 | 2.5 × 10⁹ | Wisniewski et al. 2014 |
| Fibroblast | 4,000 | 2.0 × 10⁹ | Schwanhäusser et al. 2011 |
| Epithelial | 2,000 | 1.8 × 10⁹ | Schwanhäusser et al. 2011 |
| Neuron (soma) | 2,000 | 3.5 × 10⁹ | Milo & Phillips 2015 |
| Large immune (Macrophage, DC) | 1,500 | 2.5 × 10⁹ | Milo & Phillips 2015 |
| Endothelial | 800 | 1.0 × 10⁹ | Snijder et al. 2009 |
| Mid immune (Neutrophil, Monocyte) | 350 | 8.0 × 10⁸ | Milo & Phillips 2015 |
| Lymphocyte (T/B/NK) | 200 | 2.0 × 10⁸ | Milo & Phillips 2015 |
| Erythrocyte | 90 | 3.0 × 10⁸ | BioNumbers |
| Platelet | 10 | 3.0 × 10⁷ | BioNumbers |
| Default | 1,000 | 2.0 × 10⁹ | Schwanhäusser et al. 2011 |

Cell-type matching uses case-insensitive keyword search with strict priority ordering (e.g. `hepatocyte` is matched before generic `epithelial`).

---

### `pbpk/config.py`

Master configuration holding:

- **`DRUG_DB`** — pharmacokinetic and binding parameters per drug (MW, dose, Kd, kon, koff, ke(RC), half-life, etc.)
- **`TARGET_DB`** — target biology per receptor (R₀ baseline, ksyn, kdeg, cellular localisation)
- **Tissue physiology** — organ volumes, blood flows, vascular fractions, reflection coefficients
- **Dosing regimens** — `single`, `q1w`, `q2w`, `loading_q1w`, and others

Extend by adding entries to `DRUG_DB` and `TARGET_DB`.

**Currently configured drug–target pairs:**

| Drug | Target | Type | K_d (nM) | K_SS (nM) |
|------|--------|------|----------|-----------|
| Cetuximab | EGFR | Membrane | 0.304 | 0.420 |
| Trastuzumab | ERBB2 | Membrane | 0.50 | 0.616 |
| Carlumab | CCL2 | Soluble | 0.2 | 0.203 |

---

### `pbpk/equations.py`

Implements the multi-cell-type QSS-TMDD ODE system per tissue compartment:

```
dC_plasma/dt  = (dose input) - CL·C - Σ_tissues Q·(C - C_v)
dC_ISF/dt     = (convection + diffusion) - Σ_cells RC binding
dR_i/dt       = ksyn_i - kdeg·R_i - kon·C_ISF·R_i + koff·RC_i
dRC_i/dt      = kon·C_ISF·R_i - (koff + ke)·RC_i
```

The Gibiansky total-drug QSS formulation is used for numerical robustness, with a rationalised quadratic for solving free drug concentration from total drug at near-saturation receptor occupancy.

---

### `pbpk/simulator.py`

- **Solver:** SciPy `Radau` (implicit Runge-Kutta order 5), `rtol = atol = 1e-8`
- **Dual-mode:** Runs SC and bulk parameterisations in a single call; output CSV contains a `mode` column (`single_cell` / `bulk`)
- **Soluble target auto-detection:** Cell types named `soluble_*` in the blood tissue are automatically routed to a dedicated plasma-phase TMDD compartment. Both SC and Bulk modes use the same R₀ (bulk plasma concentration) for these entries, ensuring identical RO curves — biologically correct since soluble receptors are a plasma protein pool, not cell-type specific. No CLI flag required.
- **Multi-dose:** Event-driven dosing via `solve_ivp` dense output and restart at each dose event
- **Output:** Time-series flattened to long format — one row per (time, tissue, cell_type, mode)

---

## Output Columns

The PBPK simulation output CSV (`data/pbpk_*.csv`) contains:

| Column | Description |
|--------|-------------|
| `time` | Simulation time (days) |
| `tissue` | Tissue compartment name |
| `cell_type` | Cell type within tissue |
| `mode` | `single_cell` or `bulk` |
| `plasma_conc_nM` | Free plasma drug concentration (nM) |
| `plasma_ctot_nM` | Total central compartment drug (free + blood-bound, nM) |
| `local_v_conc_nM` | Organ vascular concentration (nM) |
| `local_isf_conc_nM` | Interstitial fluid (ISF) concentration (nM) |
| `free_R_nM` | Free (unoccupied) receptor [R] (nM) |
| `bound_RC_nM` | Drug–receptor complex [RC] (nM) |
| `total_R_nM` | Total receptor pool Rtot = R + RC (nM) |
| `occupancy_pct` | Receptor occupancy = RC / Rtot × 100 (%) |
| `TMDD_index` | R₀ / K_SS — TMDD risk index (>1 = TMDD-relevant) |
| `R0_nM` | Baseline receptor density R₀ (nM) |

---

## Figures Generated

### Single-dose mode

Output directory: `plots_results/`

| Figure | Filename suffix | Description |
|--------|----------------|-------------|
| Fig-1 | `_fig1_pk.png` | Systemic PK: plasma free drug SC vs Bulk (+ C_tot optional) |
| Fig-2 | `_fig2_heatmap.png` | TMDD index heatmap: tissue × cell type (SC, t = 0) |
| Fig-3 | `_fig3_foldchange.png` | **KEY**: SC / Bulk TMDD index fold change (log₂ bar chart) |
| Fig-4 | `_fig4_occ_rc.png` | Per-tissue: receptor occupancy + [RC] + free [R] (3-panel) |
| Fig-5 | `_fig5_conc.png` | Per-tissue: concentration gradient Cc → Cv → Cisf |

### Multi-dose mode

Output directory: `plots_results/multi_<target>_<drug>/`

**Per-dose subfolders** (`dose_<X>mg/`): same Fig-1 through Fig-5 as above for each dose.

**Comparison subfolder** (`comparison/`):

| Figure | Filename suffix | Description |
|--------|----------------|-------------|
| Fig-C1 | `_compC1_serum.png` | Plasma free + total drug overlaid across all doses + NCA PK table |
| Fig-C2 | `_compC2_tissue_isf.png` | Tissue ISF / plasma concentration per tissue, all doses |
| Fig-C3 | `_compC3_bound_rc.png` | Top-N cell-type drug–receptor complex [RC] per tissue, all doses |
| Fig-C3c | `_compC3c_free_r.png` | Top-N cell-type free receptor [R] vs R₀ baseline, all doses |
| Fig-C4 | `_compC4_tmdd.png` | TMDD index: Bulk vs SC max vs SC mean — bar chart + heatmap |
| Fig-C5 | `_compC5_receptor_dynamics.png` | Full receptor state grid (Rtot / [RC] / [R]), top-N cells, all doses |

### Figure design rules

- All y-axes **linear**, starting at 0; actual concentration values shown (no log scale, no scientific notation)
- **40-colour purple ramp** palette for cell types; linestyle cycling after 40 types
- Multi-dose comparison uses **log-scaled purple colormap** interpolated by dose magnitude
- NCA PK parameters (C_max, t½, AUC₀₋∞, CL, Vss) computed analytically and printed to terminal for SC mode (Fig-C1)

---

## Methodology

### PPM → nM Conversion

```
C_i (nM) = PPM_i × ρ_tissue (mg/mL) × 10³ / MW_avg_proteome (Da)
```

Tissue-specific ρ values are derived from published tissue protein density measurements (Wisniewski et al. 2014). This approach is tissue-specific by construction and agrees with the independent empirical PaxDb-to-molarity correlation reported by Sepp & Muliaditan (2024).

### Bayesian MCMC Model

- **Non-centred parameterisation** — `z_i ~ Normal(0, 1)`, `θ_i = exp(σ_θ × z_i)` avoids Neal's funnel and improves NUTS geometry (Papaspiliopoulos et al. 2007; Betancourt & Girolami 2015)
- **σ_θ = 0.55** — calibrated from single-cell mRNA–protein correlation R² ≈ 0.50 (Schwanhäusser et al. 2011; Battich et al. 2015)
- **Observation model** — 30% coefficient of variation on bulk PPM, consistent with inter-laboratory PaxDb variability
- **Convergence criteria** — R-hat < 1.01, ESS ≥ 400, zero divergent transitions (Vehtari et al. 2021)
- **Sampler** — NUTS via PyMC v5; `target_accept = 0.95`, `max_treedepth = 12`

### PBPK–TMDD Model

- **Architecture** — Parallel organ layout (7 solid tissues + Rest-of-body + Blood), after Shah & Betts (2012)
- **Vascular–ISF transport** — Simplified two-pore theory: convection (reflection coefficient σ), passive diffusion, and lymphatic drainage
- **TMDD** — Gibiansky QSS total-drug formulation (Gibiansky et al. 2008); rationalised quadratic ensures numerical stability at near-saturation receptor occupancy
- **Soluble target handling** — Cell types prefixed `soluble_*` are separated from membrane-bound blood cell entries and assigned to a dedicated plasma-phase TMDD compartment. Binding occurs directly in plasma (not ISF); both SC and Bulk modes use identical R₀ = bulk plasma concentration, so SC and Bulk receptor occupancy curves are equivalent for soluble targets by construction. Lymph drainage in solid-organ ISF switches to total drug (`C_tot,isf`) when a soluble target is detected.
- **Solver** — SciPy `Radau` (implicit RK5), `rtol = atol = 1e-8`, well-suited for stiff TMDD ODE systems

---

## References

### Pharmacokinetics & PBPK–TMDD

- Mager DE, Jusko WJ (2001) General pharmacokinetic model for drugs exhibiting target-mediated drug disposition. *J Pharmacokinet Pharmacodyn* 28:507–532.
- Gibiansky L, Gibiansky E, Kakkar T, Ma P (2008) Approximations of the target-mediated drug disposition model and identifiability of model parameters. *J Pharmacokinet Pharmacodyn* 35:573–591.
- Shah DK, Betts AM (2012) Towards a platform PBPK model to characterize the plasma and tissue disposition of monoclonal antibodies in preclinical species and human. *J Pharmacokinet Pharmacodyn* 39:67–86.
- Sepp A, Muliaditan M (2024) Translational PBPK modelling of mAb disposition. *mAbs* 16:2324485.

### Bayesian Statistics & MCMC

- Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner P-C (2021) Rank-normalization, folding, and localization: an improved R̂ for assessing convergence of MCMC. *Bayesian Anal* 16:667–718.
- Betancourt M, Girolami M (2015) Hamiltonian Monte Carlo for hierarchical models. *Current Trends in Bayesian Methodology with Applications* 79:2–4.
- Papaspiliopoulos O, Roberts GO, Sköld M (2007) A general framework for the parametrization of hierarchical models. *Stat Sci* 22:59–73.
- Salvatier J, Wiecki TV, Fonnesbeck C (2016) Probabilistic programming in Python using PyMC3. *PeerJ Comput Sci* 2:e55.

### Proteomics & Single-Cell Biology

- Schwanhäusser B, Busse D, Li N, et al. (2011) Global quantification of mammalian gene expression control. *Nature* 473:337–342.
- Wisniewski JR, Hein MY, Cox J, Mann M (2014) A "proteomic ruler" for protein copy number and concentration estimation without spike-in standards. *Mol Cell Proteomics* 13:3497–3506.
- Snijder B, Sacher R, Rämö P, et al. (2009) Population context determines cell-to-cell variability in endocytosis and virus infection. *Nature* 461:520–523.
- Battich N, Stoeger T, Pelkmans L (2015) Control of transcript variability in single mammalian cells. *Cell* 163:1596–1610.
- Milo R, Phillips R (2015) *Cell Biology by Numbers*. Garland Science. (BioNumbers database: bionumbers.hms.harvard.edu)
- Alberts B, Johnson A, Lewis J, et al. (2014) *Molecular Biology of the Cell*, 6th ed. Garland Science.

### Data Sources

- Uhlén M et al. (2015) Tissue-based map of the human proteome. *Science* 347:1260419. (Human Protein Atlas; proteinatlas.org)
- Wang M et al. (2012) PaxDb, a database of protein abundance averages across all three domains of life. *Mol Cell Proteomics* 11:492–500. (pax-db.org)
- Desiere F, Deutsch EW, King NL, Nesvizhskii AI, Mallick P, Eng J, Chen S, Eddes J, Loevenich SN, Aebersold R (2006) The PeptideAtlas project. *Nucleic Acids Res* 34:D655–D658. (peptideatlas.org)

---

## License

See [LICENSE](LICENSE) for details.