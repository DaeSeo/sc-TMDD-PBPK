"""
main.py
=======
PBPK Bayesian Data Pipeline — local parquet version.

Loads pre-downloaded HPA and PaxDb data from parquet files,
looks up the target gene, merges datasets, and runs Bayesian inference.

Usage
─────
  python main.py --target EGFR
  python main.py --target EGFR --hpa-path data/HPA_data.parquet --pax-path data/PaxDb_data.parquet
  python main.py --target EGFR --skip-inference   # merge only, no MCMC
"""

import os
import argparse
import pandas as pd
from inference import BayesianPPMCalculator


# PaxDb wide-format column → tissue name mapping
# PBMC maps to 'blood' to match HPA single-cell blood data.
# Plasma maps to 'plasma' (will naturally drop out if HPA has no 'plasma' cells).
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

# Biological epsilon: represents below-detection-limit expression.
# Used wherever a real measurement is absent, ensuring the HPA single-cell
# rows are retained and the ODE solver never encounters a true zero.
EPSILON = 1e-6


def load_hpa(path: str, target_gene: str) -> pd.DataFrame:
    """
    Load HPA single-cell data for a specific gene from parquet.
    """
    df = pd.read_parquet(path)

    # Case-insensitive gene lookup (try 'Gene name' first, fallback to 'Gene')
    gene_col = 'Gene name' if 'Gene name' in df.columns else 'Gene'
    mask = df[gene_col].str.upper() == target_gene.upper()
    hpa_df = df[mask].copy()

    if hpa_df.empty:
        available = sorted(df[gene_col].dropna().unique()[:20])
        raise ValueError(
            f"Gene '{target_gene}' not found in HPA data (column: {gene_col}).\n"
            f"  First 20 available: {available}"
        )

    # Standardise column names for downstream pipeline
    hpa_df = hpa_df.rename(columns={
        'Gene name': 'gene_name',
        'Gene': 'gene',
        'Tissue': 'tissue',
        'Cell type': 'cell_type',
        'Total_Cells': 'cells',
        'Weighted_nCPM': 'nCPM',
    })

    # Keep only required columns
    required = ['tissue', 'cell_type', 'cells', 'nCPM']
    optional = ['gene_name', 'gene']
    keep = [c for c in required + optional if c in hpa_df.columns]
    hpa_df = hpa_df[keep].copy()

    # Replace NaN and zero nCPM with epsilon to handle scRNA-seq dropout
    # and prevent zero-division in Bayesian model
    if 'nCPM' in hpa_df.columns:
        hpa_df['nCPM'] = hpa_df['nCPM'].fillna(EPSILON)
        hpa_df['nCPM'] = hpa_df['nCPM'].replace(0.0, EPSILON)

    print(f"  Found {len(hpa_df)} rows for {target_gene}")
    print(f"  Tissues: {sorted(hpa_df['tissue'].unique())}")

    return hpa_df


def load_paxdb(path: str, target_gene: str) -> pd.DataFrame:
    """
    Load PaxDb bulk proteomics data for a specific gene from parquet.
    Melts wide format into long format (tissue, bulk_ppm).

    Every tissue in PAXDB_TISSUE_MAP is always included in the output.
    Tissues with a valid positive measurement use the actual value.
    Tissues with missing or zero values receive EPSILON so that:
      - The HPA single-cell rows for that tissue are NOT dropped by the merge.
      - Bayesian inference can still run (prior-dominated posterior).
      - Downstream ODE solver never receives a true zero bulk_ppm.
    """
    df = pd.read_parquet(path)

    mask = df['gene_name'].str.upper() == target_gene.upper()
    pax_row = df[mask]

    if pax_row.empty:
        available = sorted(df['gene_name'].dropna().unique()[:20])
        raise ValueError(
            f"Gene '{target_gene}' not found in PaxDb data.\n"
            f"  First 20 available: {available}"
        )

    # Take first match (should be unique per gene)
    pax_row = pax_row.iloc[0]

    records = []
    epsilon_tissues = []

    for col, tissue_name in PAXDB_TISSUE_MAP.items():
        if col in df.columns:
            val = pd.to_numeric(pax_row.get(col), errors='coerce')
            if pd.notna(val) and val > 0:
                records.append({'tissue': tissue_name, 'bulk_ppm': float(val)})
            else:
                # No valid measurement → epsilon placeholder
                records.append({'tissue': tissue_name, 'bulk_ppm': EPSILON})
                epsilon_tissues.append(tissue_name)
        else:
            # Column absent from this PaxDb file → epsilon placeholder
            records.append({'tissue': tissue_name, 'bulk_ppm': EPSILON})
            epsilon_tissues.append(tissue_name)

    pax_df = pd.DataFrame(records)

    real_tissues = [r['tissue'] for r in records if r['bulk_ppm'] > EPSILON]
    print(f"  Found bulk_ppm for {len(real_tissues)} tissues: "
          f"{sorted(real_tissues)}")
    if epsilon_tissues:
        print(f"  [INFO] No PaxDb data for {epsilon_tissues} → using ε={EPSILON}")

    return pax_df


def main():
    parser = argparse.ArgumentParser(
        description="PBPK Bayesian Data Pipeline (local parquet)"
    )
    parser.add_argument("--target", required=True,
                        help="Target gene name (e.g. EGFR, ERBB2)")
    parser.add_argument("--hpa-path", default="HPA_Single_Cell.parquet",
                        help="Path to HPA parquet file")
    parser.add_argument("--pax-path", default="PaxDb_data.parquet",
                        help="Path to PaxDb parquet file")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory for results")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip Bayesian MCMC (merge only)")
    parser.add_argument("--draws", type=int, default=2000,
                        help="MCMC draws per chain")
    parser.add_argument("--tune", type=int, default=2000,
                        help="MCMC tuning steps per chain")
    args = parser.parse_args()

    target = args.target.upper()
    os.makedirs(args.output_dir, exist_ok=True)

    merged_path = os.path.join(args.output_dir, f"Merged_{target}.csv")
    final_path  = os.path.join(args.output_dir, f"Final_{target}_Data.csv")

    print(f"\n{'=' * 55}")
    print(f"  PBPK Bayesian Pipeline — {target}")
    print(f"{'=' * 55}")

    # ── Step 1: Load HPA data ─────────────────────────────────────────────
    print(f"\n[Step 1] Loading HPA single-cell data...")
    hpa_df = load_hpa(args.hpa_path, target)

    # ── Step 2: Load PaxDb data ───────────────────────────────────────────
    print(f"\n[Step 2] Loading PaxDb bulk proteomics data...")
    pax_df = load_paxdb(args.pax_path, target)

    # ── Step 3: Merge ─────────────────────────────────────────────────────
    print(f"\n[Step 3] Merging datasets...")

    merged_df = pd.merge(
        hpa_df,
        pax_df[['tissue', 'bulk_ppm']].drop_duplicates(subset='tissue'),
        on='tissue',
        how='left',
    )

    # Any tissue in HPA but not in PAXDB_TISSUE_MAP at all → epsilon
    # (e.g. a new tissue added to HPA after the map was defined)
    still_missing = merged_df['bulk_ppm'].isna().sum()
    if still_missing:
        merged_df['bulk_ppm'] = merged_df['bulk_ppm'].fillna(EPSILON)
        print(f"  [INFO] {still_missing} rows had no bulk_ppm after merge "
              f"→ filled with ε={EPSILON}")

    matched = (merged_df['bulk_ppm'] > EPSILON).groupby(
        merged_df['tissue']).any().sum()
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

    # Fill NaN in final output with epsilon.
    # Applies only to columns that may be absent for epsilon-bulk tissues
    # (prior-dominated posterior still produces local_ppm estimates).
    fill_cols = ['bulk_ppm', 'local_ppm', 'nM_concentration', 'bulk_nM_concentration']
    for c in fill_cols:
        if c in final_df.columns:
            final_df[c] = final_df[c].fillna(EPSILON)

    final_df.to_csv(final_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Pipeline complete — {target}")
    print(f"  Output: {final_path}")
    print(f"{'=' * 55}")

    cols = ['tissue', 'cell_type', 'nCPM', 'bulk_ppm',
            'local_ppm', 'nM_concentration']
    show_cols = [c for c in cols if c in final_df.columns]
    print(f"\n[Preview]")
    print(final_df[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()