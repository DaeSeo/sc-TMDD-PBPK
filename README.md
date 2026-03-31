# Single-Cell Resolution PBPK Modeling with Joint Cross-Tissue Bayesian MCMC Inference

A computational platform for **single-cell resolved Physiologically Based Pharmacokinetic (PBPK)** modeling with **target-mediated drug disposition (TMDD)** de-risking. The platform estimates cell-type-specific receptor concentrations (nM) by integrating single-cell RNA-seq data with bulk tissue proteomics through joint cross-tissue Bayesian MCMC inference, then simulates per-cell-type drug–target engagement dynamics across human tissues.

## Overview

Conventional PBPK-TMDD models parameterize receptor concentrations using bulk tissue averages, implicitly assuming homogeneous expression within each tissue. This platform resolves that limitation by:

1. **Bayesian deconvolution** — A joint hierarchical MCMC model decomposes bulk tissue proteomics (PaxDb, PPM) into cell-type-level protein concentrations using single-cell RNA-seq proportions (HPA, nCPM) with shared translation efficiency parameters across tissues.
2. **First-principles PPM → nM conversion** — Tissue-specific protein densities (ρ, mg/mL) and average proteome molecular weights convert PPM to molar concentrations: `C (nM) = PPM × ρ × 10³ / MW_avg`.
3. **Per-cell-type PBPK-TMDD** — A whole-body QSS-TMDD model (Gibiansky formulation) resolves receptor dynamics for every cell type in every tissue simultaneously, enabling direct comparison of single-cell vs. bulk TMDD risk (R₀/K_SS).

## Folder Structure

```
Single_Cell_Res_PBPK/
├── inference/
│   ├── __init__.py
│   ├── bayesian_ppm.py         # Joint cross-tissue Bayesian MCMC (PyMC/NUTS)
│   └── cell_volume.py          # BioNumbers-based cell volume database
├── pbpk/
│   ├── __init__.py
│   ├── config.py               # Drug/Target DB, tissue physiology, dosing regimens
│   ├── equations.py            # QSS-TMDD ODE system (Gibiansky total-drug formulation)
│   └── simulator.py            # Radau ODE solver with event-driven multi-dose support
├── HPA_Single_Cell.parquet     # Pre-downloaded HPA single-cell RNA-seq data
├── PaxDb_data.parquet          # Pre-downloaded PaxDb bulk tissue proteomics
├── main.py                     # Step 1: Data loading, merging & Bayesian inference
├── main_pbpk.py                # Step 2: PBPK simulation (single-cell + bulk)
├── requirements.txt            # Python dependencies
├── LICENSE
└── README.md
```

## Installation

Requires **Python 3.11+**. A virtual environment is recommended.

```bash
git clone https://github.com/[YOUR_USERNAME]/Single_Cell_Res_PBPK.git
cd Single_Cell_Res_PBPK

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `pymc` (v5+) | Bayesian MCMC sampling (NUTS) |
| `arviz` | Posterior diagnostics (R-hat, ESS, HDI) |
| `scipy` | Radau ODE solver |
| `pandas` / `numpy` | Data manipulation |
| `pyarrow` | Parquet file I/O |

## Quick Start

### Step 1 — Bayesian Inference

Load HPA and PaxDb parquet files, merge by target gene, and run joint cross-tissue MCMC to produce cell-type-resolved receptor concentrations.

```bash
# EGFR (default parquet paths)
python main.py --target EGFR

# ERBB2 with custom MCMC settings
python main.py --target ERBB2 --draws 3000 --tune 3000

# Merge only, skip MCMC (useful for debugging)
python main.py --target EGFR --skip-inference
```

**Output:** `data/Final_EGFR_Data.csv` — contains per-cell-type `nM_concentration`, `bulk_nM_concentration`, `volume_fraction`, and 94% HDI bounds.

### Step 2 — PBPK Simulation

Run the whole-body PBPK model with both single-cell and bulk receptor parameterizations.

```bash
# Single dose (default: cetuximab 400 mg/m², EGFR, 120 days)
python main_pbpk.py

# Custom drug/target/dose
python main_pbpk.py --Drug cetuximab --Target EGFR --Dose 100 --Days 120

# Multi-dose comparison
python main_pbpk.py --multi --doses 0.1,1,10,100 --Drug cetuximab --Target EGFR

# Clinical regimen
python main_pbpk.py --Drug cetuximab --Target EGFR --Regimen loading_q1w --Days 84
```

**Output:** `data/pbpk_egfr_cetuximab_100mg.csv` — time-series of plasma concentration, tissue ISF concentration, free receptor, bound RC, receptor occupancy, and TMDD index for every cell type.

## Supported Drug–Target Pairs

Configured in `pbpk/config.py`. Extend by adding entries to `DRUG_DB` and `TARGET_DB`.

| Drug | Target | Type | K_d (nM) | K_SS (nM) |
|------|--------|------|----------|-----------|
| Cetuximab | EGFR | Membrane | 0.304 | 0.420 |
| Trastuzumab | ERBB2 | Membrane | 0.50 | 0.616 |
| Carlumab | CCL2 | Soluble | 0.2 | 0.203 |

## Methodology

### PPM → nM Conversion

Cell-type-level protein concentrations are converted from PPM to nM using tissue-specific biophysical constants:

```
C_i (nM) = PPM_i × ρ_tissue (mg/mL) × 10³ / MW_avg_proteome (Da)
```

where ρ values are derived from published tissue protein density measurements (Wisniewski et al. 2014):

| Tissue | ρ (mg/mL) | MW_avg (Da) | K_organ |
|--------|-----------|-------------|---------|
| Liver | 150 | 52,000 | 2.885 |
| Kidney | 120 | 50,000 | 2.400 |
| Heart | 130 | 50,000 | 2.600 |
| Lung | 100 | 50,000 | 2.000 |
| Brain | 100 | 50,000 | 2.000 |
| Skin | 80 | 50,000 | 1.600 |
| Blood | 70 | 55,000 | 1.273 |

This first-principles approach is tissue-specific by construction and agrees with the independent empirical PaxDb-to-molarity correlation reported by Sepp & Muliaditan (2024, mAbs).

### Bayesian MCMC Model

- **Shared θ_i** — Translation efficiency parameters are shared across tissues for cell types appearing in ≥2 tissues, enabling cross-tissue information pooling.
- **Non-centered parameterization** — `z_i ~ Normal(0,1)`, `θ_i = exp(σ_θ × z_i)` avoids Neal's funnel.
- **σ_θ = 0.55** — Based on single-cell mRNA–protein R² ≈ 0.50 (Battich et al. 2015; Schwanhäusser et al. 2011).
- **Convergence criteria** — R-hat < 1.01, ESS ≥ 400, zero divergent transitions (Vehtari et al. 2021).

### PBPK-TMDD Model

- **Architecture** — Parallel organ layout (7 solid tissues + Rest + Blood), based on Shah & Betts (2012).
- **Transport** — Simplified two-pore theory (convection + diffusion + lymphatic drainage).
- **TMDD** — Gibiansky QSS total-drug formulation with rationalized quadratic for numerical stability.
- **Solver** — SciPy Radau (implicit Runge-Kutta order 5), rtol = atol = 10⁻⁸.

## References

- Shah DK, Betts AM (2012) *J Pharmacokinet Pharmacodyn* 39:67–86.
- Gibiansky L et al. (2008) *J Pharmacokinet Pharmacodyn* 35:573–591.
- Battich N et al. (2015) *Cell* 163:1596–1610.
- Wisniewski JR et al. (2014) *Mol Cell Proteomics* 13:3497–3506.
- Sepp A, Muliaditan M (2024) *mAbs* 16:2324485.
- Vehtari A et al. (2021) *Bayesian Anal* 16:667–718.
- Snijder B et al. (2009) *Nature* 461:520–523.

## License

See [LICENSE](LICENSE) for details.
