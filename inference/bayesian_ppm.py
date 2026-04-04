"""
bayesian_ppm.py
===============
Bayesian inference of cell-type-level protein expression (local PPM → nM)
by combining HPA scRNA-seq (nCPM) and PaxDb bulk proteomics (tissue PPM).

Model: Joint cross-tissue hierarchical MCMC
--------------------------------------------
  ppm_i_t = alpha_t x scaled_nCPM_i_t x theta_i
  bulk_pred_t = sum_i(ppm_i_t x fraction_i)
  obs_t ~ Normal(bulk_pred_t, sigma=bulk_t x 0.30)

PPM to nM (K_ORGAN approach):
  K = rho_tissue (mg/mL) x 1e3 / MW_avg (Da)
  nM_concentration = local_ppm x K_organ
  bulk_nM_concentration = bulk_ppm x K_organ

actual_counts_per_cell:
  counts = local_ppm x N_total_per_cell x 1e-6
  Validated: EGFR skin fibroblast -> ~45,000 molecules/cell (lit: ~50k) OK

References:
  Schwanhäusser et al. (2011) Nature 473:337
  Wisniewski et al. (2014) Nat Methods 11:306
  Snijder et al. (2009) Nature 461:520
"""

import warnings
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from .cell_volume import get_volume_um3, get_counts_per_cell

warnings.filterwarnings("ignore")

# K_ORGAN: PPM -> nM  (K = rho mg/mL x 1e3 / MW_avg Da)
K_ORGAN = {
    "Liver":   150.0 * 1e3 / 52_000,
    "Kidney":  120.0 * 1e3 / 50_000,
    "Heart":   130.0 * 1e3 / 50_000,
    "Lung":    100.0 * 1e3 / 50_000,
    "Brain":   100.0 * 1e3 / 50_000,
    "Skin":     80.0 * 1e3 / 50_000,
    "Breast":   60.0 * 1e3 / 50_000,
    "Blood":    70.0 * 1e3 / 55_000,
    "Plasma":   70.0 * 1e3 / 55_000,
}

_TISSUE_KEYWORDS = [
    ("kidney", "Kidney"),
    ("liver",  "Liver"),
    ("heart",  "Heart"),
    ("breast", "Breast"),
    ("brain",  "Brain"),
    ("plasma", "Plasma"),
    ("blood",  "Blood"),
    ("lung",   "Lung"),
    ("skin",   "Skin", ["foreskin"]),
]

SIGMA_THETA      = 0.55
EPSILON_BULK     = 1e-3
EPSILON_NCPM_SUM = 1e-4


def _get_k_organ(tissue_name):
    tl = tissue_name.lower().strip()
    for key in K_ORGAN:
        if key.lower() == tl:
            return key, K_ORGAN[key]
    for entry in _TISSUE_KEYWORDS:
        keyword, key = entry[0], entry[1]
        blocked = entry[2] if len(entry) > 2 else []
        if keyword in tl and not any(b in tl for b in blocked):
            return key, K_ORGAN[key]
    return None, None


def _protein_mass_fractions(cell_types, cell_counts):
    vols = np.array([get_volume_um3(ct) for ct in cell_types], dtype=float)
    mass_proxy = cell_counts.astype(float) * vols
    total = mass_proxy.sum()
    if total > 0:
        return mass_proxy / total
    return np.ones(len(cell_counts)) / len(cell_counts)


class BayesianPPMCalculator:

    def __init__(self, draws=2000, tune=2000, chains=4,
                 target_accept=0.95, random_seed=42):
        self.draws         = draws
        self.tune          = tune
        self.chains        = chains
        self.target_accept = target_accept
        self.random_seed   = random_seed

    @staticmethod
    def _preprocess(df):
        df = df.copy()
        df["nCPM"]     = pd.to_numeric(df["nCPM"], errors="coerce").fillna(0.0)
        df["bulk_ppm"] = pd.to_numeric(df["bulk_ppm"], errors="coerce")
        df["cells"]    = pd.to_numeric(
            df["cells"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        ).fillna(1.0)
        df["cell_type"] = df["cell_type"].astype(str)
        for tissue in df["tissue"].unique():
            vals = df.loc[df["tissue"] == tissue, "bulk_ppm"].dropna().unique()
            if len(vals) > 1:
                med = float(np.nanmedian(vals))
                print(f"[WARNING] {tissue}: bulk_ppm mismatch -> median {med:.3f}")
                df.loc[df["tissue"] == tissue, "bulk_ppm"] = med
        return df

    def _run_joint_inference(self, df):
        tissues      = df["tissue"].unique().tolist()
        type_tc      = df.groupby("cell_type")["tissue"].nunique()
        shared_types = sorted(type_tc[type_tc >= 2].index.tolist())
        unique_types = sorted(type_tc[type_tc == 1].index.tolist())
        shared_idx   = {ct: i for i, ct in enumerate(shared_types)}
        unique_idx   = {ct: i for i, ct in enumerate(unique_types)}

        print(f"\n[Joint MCMC]")
        print(f"  Tissues: {len(tissues)}  Shared: {len(shared_types)}  Unique: {len(unique_types)}")

        tissue_data = {}
        for tissue in tissues:
            tdf         = df[df["tissue"] == tissue].copy()
            bulk_val    = float(tdf["bulk_ppm"].iloc[0])
            cell_types  = tdf["cell_type"].values
            cell_counts = tdf["cells"].values.astype(float)
            ncpm_vals   = tdf["nCPM"].values.astype(float)
            max_ncpm    = ncpm_vals.max()
            scaled_ncpm = ncpm_vals / (max_ncpm if max_ncpm > 0 else 1.0)
            fractions   = _protein_mass_fractions(cell_types, cell_counts)
            tissue_data[tissue] = {
                "bulk_val"   : bulk_val,
                "cell_types" : cell_types,
                "scaled_ncpm": scaled_ncpm.astype(np.float64),
                "fractions"  : fractions.astype(np.float64),
                "n_cells"    : len(cell_types),
            }

        with pm.Model():
            if shared_types:
                z_shared     = pm.Normal("z_shared", mu=0.0, sigma=1.0, shape=len(shared_types))
                theta_shared = pm.math.exp(SIGMA_THETA * z_shared)
            if unique_types:
                z_unique     = pm.Normal("z_unique", mu=0.0, sigma=1.0, shape=len(unique_types))
                theta_unique = pm.math.exp(SIGMA_THETA * z_unique)

            for tissue in tissues:
                td       = tissue_data[tissue]
                bulk_val = td["bulk_val"]
                alpha_t  = pm.HalfNormal(f"alpha_{tissue}", sigma=bulk_val * 2.0)
                ppm_list = []
                for i, ct in enumerate(td["cell_types"]):
                    s_ncpm = td["scaled_ncpm"][i]
                    theta  = (theta_shared[shared_idx[ct]]
                              if ct in shared_idx else theta_unique[unique_idx[ct]])
                    ppm_list.append(alpha_t * s_ncpm * theta)
                ppm_vec   = pm.math.stack(ppm_list)
                bulk_pred = pm.math.dot(ppm_vec, td["fractions"])
                pm.Normal(f"obs_{tissue}", mu=bulk_pred,
                          sigma=bulk_val * 0.3, observed=bulk_val)

            trace = pm.sample(
                draws=self.draws, tune=self.tune, chains=self.chains,
                target_accept=self.target_accept,
                nuts_sampler_kwargs={"max_treedepth": 12},
                progressbar=False, random_seed=self.random_seed,
            )

        divergences = int(trace.sample_stats.diverging.sum().item())
        try:
            rhat_max = float(np.nanmax(az.rhat(trace).to_array().values))
            ess_min  = float(np.nanmin(az.ess(trace).to_array().values))
        except Exception:
            rhat_max, ess_min = float("nan"), float("nan")

        converged = (divergences == 0 and rhat_max < 1.01
                     and (np.isnan(ess_min) or ess_min >= 400))
        flag = "OK" if converged else "WARN"
        print(f"  [{flag}] R-hat={rhat_max:.3f}  ESS={ess_min:.0f}  Div={divergences}")

        n_samples = trace.posterior.dims["chain"] * trace.posterior.dims["draw"]
        if shared_types:
            th_sh = np.exp(SIGMA_THETA *
                           trace.posterior["z_shared"].values.reshape(n_samples, len(shared_types)))
        if unique_types:
            th_un = np.exp(SIGMA_THETA *
                           trace.posterior["z_unique"].values.reshape(n_samples, len(unique_types)))

        tissue_results = {}
        for tissue in tissues:
            td            = tissue_data[tissue]
            alpha_samples = trace.posterior[f"alpha_{tissue}"].values.reshape(n_samples)
            ppm_samples   = np.zeros((n_samples, td["n_cells"]))
            for i, ct in enumerate(td["cell_types"]):
                s_ncpm  = td["scaled_ncpm"][i]
                theta_s = (th_sh[:, shared_idx[ct]] if ct in shared_idx
                           else th_un[:, unique_idx[ct]])
                ppm_samples[:, i] = alpha_samples * s_ncpm * theta_s
            ppm_mean = ppm_samples.mean(axis=0)
            hdi_low  = np.array([az.hdi(ppm_samples[:, i], hdi_prob=0.94)[0]
                                  for i in range(td["n_cells"])])
            hdi_high = np.array([az.hdi(ppm_samples[:, i], hdi_prob=0.94)[1]
                                  for i in range(td["n_cells"])])
            tissue_results[tissue] = {"ppm_mean": ppm_mean,
                                       "hdi_low": hdi_low, "hdi_high": hdi_high}

        return tissue_results, converged

    def calculate(self, df):
        """
        Run joint MCMC. Output columns:
          volume_fraction, local_ppm, local_ppm_hdi_low/high,
          nM_concentration      (local_ppm x K_organ),
          bulk_nM_concentration (bulk_ppm  x K_organ),
          actual_counts_per_cell (local_ppm x N_total x 1e-6),
          k_organ_used, converged
        """
        df = self._preprocess(df)

        # volume_fraction for ALL tissues (needed by PBPK even for skipped)
        for tissue in df["tissue"].unique():
            mask = df["tissue"] == tissue
            df.loc[mask, "volume_fraction"] = _protein_mass_fractions(
                df.loc[mask, "cell_type"].values,
                df.loc[mask, "cells"].values.astype(float)
            )

        for col in ["local_ppm", "local_ppm_hdi_low", "local_ppm_hdi_high",
                    "nM_concentration", "bulk_nM_concentration",
                    "actual_counts_per_cell", "k_organ_used"]:
            df[col] = np.nan
        df["converged"] = False

        mcmc_tissues, skipped_tissues = [], []
        for tissue in df["tissue"].unique():
            tdata    = df[df["tissue"] == tissue]
            bulk_val = tdata["bulk_ppm"].iloc[0]
            if pd.isna(bulk_val) or bulk_val <= EPSILON_BULK:
                skipped_tissues.append((tissue, "no real bulk_ppm"))
            elif tdata["nCPM"].sum() <= EPSILON_NCPM_SUM:
                skipped_tissues.append((tissue, "all nCPM = epsilon"))
            else:
                mcmc_tissues.append(tissue)

        if skipped_tissues:
            print(f"\n[MCMC skip] {len(skipped_tissues)} tissues:")
            for t, r in skipped_tissues:
                print(f"  {t}: {r}")

        print(f"\n[MCMC] {len(mcmc_tissues)} tissues: {sorted(mcmc_tissues)}")
        if not mcmc_tissues:
            return df

        try:
            tissue_results, converged = self._run_joint_inference(
                df[df["tissue"].isin(mcmc_tissues)].copy()
            )
        except Exception as exc:
            print(f"[ERROR] MCMC failed: {exc}")
            return df

        for tissue in mcmc_tissues:
            mask     = df["tissue"] == tissue
            res      = tissue_results[tissue]
            bulk_val = df.loc[mask, "bulk_ppm"].iloc[0]

            df.loc[mask, "local_ppm"]          = res["ppm_mean"]
            df.loc[mask, "local_ppm_hdi_low"]  = res["hdi_low"]
            df.loc[mask, "local_ppm_hdi_high"] = res["hdi_high"]
            df.loc[mask, "converged"]          = converged

            tissue_key, k = _get_k_organ(tissue)
            if k is None:
                print(f"  [WARN] {tissue} not in K_ORGAN — skipping nM conversion")
                continue

            df.loc[mask, "k_organ_used"]          = k
            df.loc[mask, "bulk_nM_concentration"] = bulk_val * k

            for ct, ppm_val in zip(df.loc[mask, "cell_type"].values, res["ppm_mean"]):
                row = mask & (df["cell_type"] == ct)
                df.loc[row, "nM_concentration"]      = ppm_val * k
                df.loc[row, "actual_counts_per_cell"] = get_counts_per_cell(ct, ppm_val)

            cnt_min = df.loc[mask, "actual_counts_per_cell"].min()
            cnt_max = df.loc[mask, "actual_counts_per_cell"].max()
            print(f"  [{tissue}] K={k:.3f}  bulk_nM={bulk_val*k:.1f}  "
                  f"counts/cell: {cnt_min:.0f}–{cnt_max:.0f}")

        return df