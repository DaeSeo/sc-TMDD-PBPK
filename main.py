"""
main.py
=======
PBPK Bayesian Data Pipeline — local parquet version.

Step 1 : Load HPA single-cell RNA-seq data (nCPM per cell type per tissue).
Step 2 : Load PaxDb bulk proteomics data (tissue-level PPM).
Step 3 : Merge on tissue name.
Step 4 : Run Bayesian MCMC inference → local PPM + nM per cell type.
Step 5 : Inject soluble target into blood compartment (PeptideAtlas MS).

Output columns (Final_<TARGET>_Data.csv)
────────────────────────────────────────
  local_ppm               Bayesian posterior mean PPM per cell type
  nM_concentration        local_ppm × K_organ  (single-cell tissue nM)
  bulk_nM_concentration   bulk_ppm  × K_organ  (tissue-average nM, PBPK R₀)
                          NOTE: for blood, includes soluble target if available
  actual_counts_per_cell  local_ppm × N_total × 1e-6  (molecules per cell)
  volume_fraction         protein mass-weighted cell-type fraction
  converged               MCMC convergence flag
  is_soluble              True for plasma soluble target row (blood only)

Usage
─────
  python main.py --target EGFR
  python main.py --target EGFR --hpa-path data/HPA_data.parquet \\
                               --pax-path data/PaxDb_data.parquet
  python main.py --target EGFR --skip-inference   # merge only, no MCMC
  python main.py --target EGFR --no-soluble       # skip soluble target
"""

import os
import argparse
import numpy as np
import pandas as pd
from inference import BayesianPPMCalculator


# PaxDb wide-format column → tissue name mapping
PAXDB_TISSUE_MAP = {
    "abundance_LUNG":   "lung",
    "abundance_KIDNEY": "kidney",
    "abundance_PLASMA": "plasma",
    "abundance_SKIN":   "skin",
    "abundance_BRAIN":  "brain",
    "abundance_HEART":  "heart",
    "abundance_LIVER":  "liver",
    "abundance_PBMC":   "blood",
}

# Below-detection-limit placeholder
EPSILON = 1e-6

# K_ORGAN for blood (must match bayesian_ppm.py)
# K = rho(mg/mL) × 1e3 / MW_avg(Da)  →  PPM → nM
K_BLOOD = 70.0 * 1e3 / 55_000   # ≈ 1.2727


# ══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_hpa(path: str, target_gene: str) -> pd.DataFrame:
    """Load HPA single-cell data for a specific gene from parquet."""
    df = pd.read_parquet(path)

    gene_col = "Gene name" if "Gene name" in df.columns else "Gene"
    mask     = df[gene_col].str.upper() == target_gene.upper()
    hpa_df   = df[mask].copy()

    if hpa_df.empty:
        available = sorted(df[gene_col].dropna().unique()[:20])
        raise ValueError(
            f"Gene '{target_gene}' not found in HPA data (column: {gene_col}).\n"
            f"  First 20 available: {available}"
        )

    hpa_df = hpa_df.rename(columns={
        "Gene name"    : "gene_name",
        "Gene"         : "gene",
        "Tissue"       : "tissue",
        "Cell type"    : "cell_type",
        "Total_Cells"  : "cells",
        "Weighted_nCPM": "nCPM",
    })

    required = ["tissue", "cell_type", "cells", "nCPM"]
    optional = ["gene_name", "gene"]
    hpa_df   = hpa_df[[c for c in required + optional if c in hpa_df.columns]].copy()

    hpa_df["nCPM"] = hpa_df["nCPM"].fillna(EPSILON).replace(0.0, EPSILON)

    print(f"  Found {len(hpa_df)} rows for {target_gene}")
    print(f"  Tissues: {sorted(hpa_df['tissue'].unique())}")
    return hpa_df


def load_paxdb(path: str, target_gene: str) -> pd.DataFrame:
    """Load PaxDb bulk proteomics data for a specific gene from parquet."""
    df = pd.read_parquet(path)

    mask    = df["gene_name"].str.upper() == target_gene.upper()
    pax_row = df[mask]
    if pax_row.empty:
        available = sorted(df["gene_name"].dropna().unique()[:20])
        raise ValueError(
            f"Gene '{target_gene}' not found in PaxDb data.\n"
            f"  First 20 available: {available}"
        )

    pax_row         = pax_row.iloc[0]
    records         = []
    epsilon_tissues = []

    for col, tissue_name in PAXDB_TISSUE_MAP.items():
        if col in df.columns:
            val = pd.to_numeric(pax_row.get(col), errors="coerce")
            if pd.notna(val) and val > 0:
                records.append({"tissue": tissue_name, "bulk_ppm": float(val)})
            else:
                records.append({"tissue": tissue_name, "bulk_ppm": EPSILON})
                epsilon_tissues.append(tissue_name)
        else:
            records.append({"tissue": tissue_name, "bulk_ppm": EPSILON})
            epsilon_tissues.append(tissue_name)

    real_tissues = [r["tissue"] for r in records if r["bulk_ppm"] > EPSILON]
    print(f"  Found bulk_ppm for {len(real_tissues)} tissues: {sorted(real_tissues)}")
    if epsilon_tissues:
        print(f"  [INFO] No PaxDb data for {epsilon_tissues} → using ε={EPSILON}")

    return pd.DataFrame(records)


def load_soluble_target(path: str, target_gene: str) -> tuple[float | None, str]:
    """
    Load soluble target concentration (nM) from PeptideAtlas MS parquet.

    The parquet has columns:
        gene_name      : str   (e.g. "EGFR", "ERBB2")
        estimated_nM   : float (MS-derived concentration × 110 correction)

    Returns
    -------
    (estimated_nM, status_msg)
        estimated_nM is None if target not found or file missing.
    """
    if not os.path.exists(path):
        return None, f"file not found: {path}"

    df = pd.read_parquet(path)

    # Flexible column matching
    gene_col = None
    for candidate in ["gene_name", "Gene name", "Gene", "gene"]:
        if candidate in df.columns:
            gene_col = candidate
            break
    if gene_col is None:
        return None, "no gene_name column found in soluble parquet"

    nm_col = None
    for candidate in ["estimated_nM", "Estimated_nM", "nM", "concentration_nM"]:
        if candidate in df.columns:
            nm_col = candidate
            break
    if nm_col is None:
        return None, "no estimated_nM column found in soluble parquet"

    mask  = df[gene_col].str.upper() == target_gene.upper()
    match = df[mask]

    if match.empty:
        available = sorted(df[gene_col].dropna().unique()[:20])
        return None, (f"'{target_gene}' not found. "
                      f"First 20 available: {available}")

    estimated_nM = float(match.iloc[0][nm_col])
    return estimated_nM, f"soluble {target_gene} = {estimated_nM:.4f} nM"


def inject_soluble_target(final_df: pd.DataFrame, target_gene: str,
                          soluble_nM: float) -> pd.DataFrame:
    """
    Inject soluble target into the blood compartment of the final dataset.

    The soluble row is treated as a first-class cell_type entry — identical
    schema to membrane-bound PBMC rows — so the PBPK simulator processes
    it without special-casing.  The only distinction is `is_soluble=True`,
    which tells the ODE to skip internalization/recycling.

    SC mode  : PBPK iterates cell_type rows → soluble is one more row,
               counted exactly once.  R₀_sc = Σ(all nM_concentration).
    Bulk mode: bulk_nM_concentration for ALL blood rows is updated to
               original + soluble_nM, so R₀_bulk reflects total target.
    """
    df = final_df.copy()

    # ── Add is_soluble flag ───────────────────────────────────────────────
    if "is_soluble" not in df.columns:
        df["is_soluble"] = False

    # ── Grab metadata from existing blood rows ────────────────────────────
    blood_mask = df["tissue"].str.lower() == "blood"
    blood_ref  = df.loc[blood_mask].iloc[0] if blood_mask.any() else None

    # ── Update bulk_nM for blood: add soluble to existing PBMC bulk ──────
    original_bulk = 0.0
    if "bulk_nM_concentration" in df.columns and blood_mask.any():
        original_bulk = float(df.loc[blood_mask, "bulk_nM_concentration"].iloc[0])
        df.loc[blood_mask, "bulk_nM_concentration"] = original_bulk + soluble_nM

    updated_bulk_nM = original_bulk + soluble_nM

    # ── Back-calculate local_ppm from nM ──────────────────────────────────
    soluble_ppm = soluble_nM / K_BLOOD if K_BLOOD > 0 else 0.0

    # ── Build soluble row — same schema as PBMC cell_type rows ────────────
    soluble_row = {}

    # 1) Inherit ALL columns from an existing blood row as defaults
    if blood_ref is not None:
        for col in df.columns:
            soluble_row[col] = blood_ref[col]

    # 2) Override with soluble-specific values
    soluble_row.update({
        "tissue":                 "blood",
        "cell_type":              f"soluble_{target_gene}",
        "cells":                  0,
        "nCPM":                   0.0,
        "bulk_ppm":               0.0,             # not part of PBMC bulk
        "local_ppm":              soluble_ppm,
        "nM_concentration":       soluble_nM,
        "bulk_nM_concentration":  updated_bulk_nM,  # matches other blood rows
        "actual_counts_per_cell": 0.0,              # no cells
        "volume_fraction":        0.0,              # no cell mass
        "is_soluble":             True,
        "converged":              True,              # directly measured
        "k_organ_used":           K_BLOOD,
    })

    # HDI not applicable (directly measured, no posterior)
    for hdi_col in ["local_ppm_hdi_low", "local_ppm_hdi_high"]:
        if hdi_col in df.columns:
            soluble_row[hdi_col] = np.nan

    soluble_df = pd.DataFrame([soluble_row])
    df = pd.concat([df, soluble_df], ignore_index=True)

    # Keep tissue grouping: blood rows together, brain together, etc.
    tissue_order = df["tissue"].drop_duplicates().tolist()
    df["_tissue_rank"] = df["tissue"].map(
        {t: i for i, t in enumerate(tissue_order)}
    )
    df = df.sort_values("_tissue_rank", kind="stable").drop(
        columns="_tissue_rank"
    ).reset_index(drop=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PBPK Bayesian Data Pipeline (local parquet)"
    )
    parser.add_argument("--target",         required=True,
                        help="Target gene name (e.g. EGFR, ERBB2)")
    parser.add_argument("--hpa-path",       default="HPA_Single_Cell.parquet",
                        help="Path to HPA parquet file")
    parser.add_argument("--pax-path",       default="PaxDb_data.parquet",
                        help="Path to PaxDb parquet file")
    parser.add_argument("--output-dir",     default="data",
                        help="Output directory for results")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip Bayesian MCMC (merge only)")
    parser.add_argument("--draws",          type=int, default=2000,
                        help="MCMC draws per chain")
    parser.add_argument("--tune",           type=int, default=2000,
                        help="MCMC tuning steps per chain")
    parser.add_argument("--soluble-path",   default="blood_soluble_protein.parquet",
                        help="Path to PeptideAtlas soluble target parquet")
    parser.add_argument("--no-soluble",     action="store_true",
                        help="Skip soluble target injection")
    args = parser.parse_args()

    target = args.target.upper()
    os.makedirs(args.output_dir, exist_ok=True)

    merged_path = os.path.join(args.output_dir, f"Merged_{target}.csv")
    final_path  = os.path.join(args.output_dir, f"Final_{target}_Data.csv")

    print(f"\n{'=' * 55}")
    print(f"  PBPK Bayesian Pipeline — {target}")
    print(f"{'=' * 55}")

    # ── Step 1: HPA ───────────────────────────────────────────────────────
    print(f"\n[Step 1] Loading HPA single-cell data...")
    hpa_df = load_hpa(args.hpa_path, target)

    # ── Step 2: PaxDb ─────────────────────────────────────────────────────
    print(f"\n[Step 2] Loading PaxDb bulk proteomics data...")
    pax_df = load_paxdb(args.pax_path, target)

    # ── Step 3: Merge ─────────────────────────────────────────────────────
    print(f"\n[Step 3] Merging datasets...")
    merged_df = pd.merge(
        hpa_df,
        pax_df[["tissue", "bulk_ppm"]].drop_duplicates(subset="tissue"),
        on="tissue",
        how="left",
    )

    still_missing = merged_df["bulk_ppm"].isna().sum()
    if still_missing:
        merged_df["bulk_ppm"] = merged_df["bulk_ppm"].fillna(EPSILON)
        print(f"  [INFO] {still_missing} rows had no bulk_ppm → ε={EPSILON}")

    matched = (merged_df["bulk_ppm"] > EPSILON).groupby(
                merged_df["tissue"]).any().sum()
    print(f"  Tissues with real bulk_ppm: {matched}")

    merged_df.to_csv(merged_path, index=False)
    print(f"  Saved: {merged_path}")

    # ── Step 4: Bayesian inference ────────────────────────────────────────
    if args.skip_inference:
        print(f"\n[Step 4] Skipped (--skip-inference)")
        final_df = merged_df
    else:
        print(f"\n[Step 4] Running Bayesian MCMC inference...")
        calculator = BayesianPPMCalculator(
            draws=args.draws,
            tune=args.tune,
        )
        final_df = calculator.calculate(merged_df)

    # ── Post-processing ───────────────────────────────────────────────────
    # Fill NaN with epsilon — column names match bayesian_ppm.py K_ORGAN output
    fill_cols = [
        "bulk_ppm",
        "local_ppm",
        "nM_concentration",
        "bulk_nM_concentration",
        "actual_counts_per_cell",
    ]
    for c in fill_cols:
        if c in final_df.columns:
            final_df[c] = final_df[c].fillna(EPSILON)

    # ── Step 5: Soluble target (PeptideAtlas MS) ──────────────────────────
    if not args.no_soluble:
        print(f"\n[Step 5] Loading soluble target data...")
        soluble_nM, status = load_soluble_target(args.soluble_path, target)
        if soluble_nM is not None and soluble_nM > 0:
            print(f"  {status}")

            # Capture original bulk_nM before injection
            blood_mask = final_df["tissue"].str.lower() == "blood"
            orig_bulk_nM = 0.0
            if blood_mask.any() and "bulk_nM_concentration" in final_df.columns:
                orig_bulk_nM = float(
                    final_df.loc[blood_mask, "bulk_nM_concentration"].iloc[0]
                )

            final_df = inject_soluble_target(final_df, target, soluble_nM)

            # Show blood compartment summary
            blood_rows = final_df[final_df["tissue"] == "blood"]
            sc_membrane = blood_rows.loc[
                ~blood_rows["is_soluble"], "nM_concentration"
            ].sum()
            total_sc_R0 = sc_membrane + soluble_nM
            updated_bulk = orig_bulk_nM + soluble_nM

            print(f"  Blood compartment R₀ breakdown:")
            print(f"    PBMC membrane-bound (Σ SC nM)  : {sc_membrane:.4f} nM")
            print(f"    Soluble target (PeptideAtlas)   : {soluble_nM:.4f} nM")
            print(f"    ─────────────────────────────────────────")
            print(f"    SC mode  total R₀  (Σ rows)    : {total_sc_R0:.4f} nM")
            print(f"    Bulk mode total R₀  (updated)  : {updated_bulk:.4f} nM")
            print(f"      (was {orig_bulk_nM:.4f} → +{soluble_nM:.4f} soluble)")
            pct_sol = soluble_nM / total_sc_R0 * 100 if total_sc_R0 > 0 else 0
            print(f"    Soluble fraction (SC basis)     : {pct_sol:.1f}%")
        else:
            print(f"  [INFO] Soluble target skipped: {status}")
            if "is_soluble" not in final_df.columns:
                final_df["is_soluble"] = False
    else:
        print(f"\n[Step 5] Skipped (--no-soluble)")
        if "is_soluble" not in final_df.columns:
            final_df["is_soluble"] = False

    final_df.to_csv(final_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Pipeline complete — {target}")
    print(f"  Output: {final_path}")
    print(f"{'=' * 55}")

    preview_cols = [
        "tissue", "cell_type", "is_soluble", "nCPM", "bulk_ppm",
        "local_ppm",
        "nM_concentration",        # local_ppm × K_organ  (SC R₀)
        "bulk_nM_concentration",   # bulk_ppm × K_organ + soluble (Bulk R₀)
        "actual_counts_per_cell",  # molecules per cell
    ]
    show_cols = [c for c in preview_cols if c in final_df.columns]
    print(f"\n[Preview — {target}]")
    print(final_df[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()