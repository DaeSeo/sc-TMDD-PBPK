"""
bayesian_ppm.py
===============
Bayesian inference of cell-type-level protein expression (local PPM → nM)
by combining HPA scRNA-seq (nCPM) and PaxDb bulk proteomics (tissue PPM).

Model: Joint cross-tissue hierarchical MCMC
────────────────────────────────────────────
  Core assumption:
    The same cell type (e.g. Endothelial cell) has similar RNA → protein
    translation efficiency (theta) regardless of which tissue it resides in.

  Generative model per tissue t, cell type i:
    ppm_i_t = alpha_t × scaled_nCPM_i_t × theta_i

    alpha_t  : tissue-specific absolute RNA→protein scale
               (absorbs differences in proteomics depth, cell density, etc.)
    theta_i  : cell-type-specific translation efficiency (shared across tissues)
               z_i ~ Normal(0, 1)                   [non-centered]
               theta_i = exp(SIGMA_THETA × z_i)     [Lognormal]
    scaled_nCPM : per-tissue max-normalised RNA expression ∈ (0, 1]

  Bulk prediction (protein mass-weighted sum over cell types):
    bulk_pred_t = Σ_i (ppm_i_t × fraction_i)
    fraction_i  = (n_cells_i × V_cell_i) / Σ_j(n_cells_j × V_cell_j)

  Likelihood (one proteomics observation per tissue):
    obs_t ~ Normal(bulk_pred_t, sigma = bulk_t × 0.30)
    observed value = bulk_t   [30% CV fixed; free sigma causes alpha-sigma funnel]

  Shared theta (cell types in ≥2 tissues):
    theta_i is constrained simultaneously by ALL tissues where cell type i appears.
    → information pooled across tissues; theta learned rather than prior-dominated.

  Unique theta (cell types in exactly 1 tissue):
    Independent theta constrained by that single tissue's bulk value.

Epsilon-bulk handling
─────────────────────
  Tissues whose bulk_ppm ≤ EPSILON_BULK (no real proteomics data) are excluded
  from MCMC to prevent:
    · alpha_t being pinned near zero (HalfNormal with σ ≈ 2ε would dominate)
    · likelihood variance ≈ 0 causing MCMC divergences
    · overriding real nCPM signal in shared theta estimation

  For these tissues, volume_fraction is still computed (needed by PBPK simulator).
  local_ppm is left as NaN; main.py fills it with epsilon in post-processing.

References
──────────
  Schwanhäusser et al. (2011) Nature 473:337     [sigma_theta=0.55 basis]
  Battich et al.       (2018) Cell Syst 7:458    [single-cell R²≈0.46-0.51]
  Wisniewski et al.    (2014) Mol Cell Proteomics [K_organ / rho values]
  Brown et al.         (1997) Toxicol Ind Health  [PBPK physiology]
  ICRP Pub. 89         (2002)                     [reference physiological params]
  Putnam               (1975) The Plasma Proteins  [plasma ~70 mg/mL]
  Vehtari et al.       (2021) Bayesian Anal. 16   [R-hat < 1.01]
  Snijder et al.       (2009) Nature 461:520      [protein ∝ cell volume]
"""

import warnings
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from .cell_volume import get_volume_um3

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# K_ORGAN — PPM → nM conversion constants
#
#   K = rho_tissue (mg/mL) × 1e3 / MW_avg_proteome (Da)
#
#   Derivation:
#     PPM_i = (mol_i / mol_total) × 1e6
#     C_i (nM) = PPM_i × rho (mg/mL) × 1e3 / MW_avg (Da)
# ══════════════════════════════════════════════════════════════════════════════

K_ORGAN = {
    "Liver":  150.0 * 1e3 / 52_000,   # 2.885
    "Kidney": 120.0 * 1e3 / 50_000,   # 2.400
    "Heart":  130.0 * 1e3 / 50_000,   # 2.600
    "Lung":   100.0 * 1e3 / 50_000,   # 2.000
    "Brain":  100.0 * 1e3 / 50_000,   # 2.000
    "Skin":    80.0 * 1e3 / 50_000,   # 1.600
    "Breast":  60.0 * 1e3 / 50_000,   # 1.200
    "Blood":   70.0 * 1e3 / 55_000,   # 1.273
    "Plasma":  70.0 * 1e3 / 55_000,   # 1.273
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

# RNA → protein translation efficiency variability (log-scale sigma)
# Basis: single-cell R² ≈ 0.50 (Battich 2018)
#        → Var(log theta) ≈ Var(log RNA) ≈ 0.55²
SIGMA_THETA = 0.55

# Minimum bulk_ppm to be treated as a real proteomics measurement.
# Values at or below this threshold are epsilon placeholders (no PaxDb data)
# and must NOT enter the MCMC likelihood — doing so pins alpha ≈ 0 and
# can cause divergences.
EPSILON_BULK = 1e-3

# Minimum summed nCPM for a tissue to be informative for MCMC.
# All-epsilon nCPM means scaled_ncpm = 1.0 for every cell type, erasing
# relative expression differences and making inference meaningless.
EPSILON_NCPM_SUM = 1e-4


# ── Helper functions ──────────────────────────────────────────────────────────

def _get_k_organ(tissue_name: str):
    """Map tissue name → (K_ORGAN key, K value). Returns (None, None) if not found."""
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


def _protein_mass_fractions(cell_types: np.ndarray,
                            cell_counts: np.ndarray) -> np.ndarray:
    """
    Compute protein mass fraction per cell type:
      f_i = (n_i × V_i) / Σ(n_j × V_j)

    Uses cell volume as a proxy for protein mass (Snijder 2009).
    """
    vols = np.array([get_volume_um3(ct) for ct in cell_types], dtype=float)
    mass_proxy = cell_counts.astype(float) * vols
    total = mass_proxy.sum()
    if total > 0:
        return mass_proxy / total
    return np.ones(len(cell_counts)) / len(cell_counts)


# ══════════════════════════════════════════════════════════════════════════════
# BayesianPPMCalculator
# ══════════════════════════════════════════════════════════════════════════════

class BayesianPPMCalculator:
    """
    Joint cross-tissue Bayesian MCMC inference.

    Calling .calculate(df) will:
      1. Compute volume_fraction for ALL tissues (PBPK simulator needs this)
      2. Filter to tissues with real bulk_ppm (> EPSILON_BULK) and real nCPM
      3. Run a single joint MCMC over those tissues
      4. Convert PPM → nM and return results
    """

    def __init__(self, draws=2000, tune=2000, chains=4,
                 target_accept=0.95, random_seed=42):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed

    # ── Preprocessing ─────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["nCPM"]     = pd.to_numeric(df["nCPM"], errors="coerce").fillna(0.0)
        df["bulk_ppm"] = pd.to_numeric(df["bulk_ppm"], errors="coerce")
        df["cells"]    = pd.to_numeric(
            df["cells"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        ).fillna(1.0)
        df["cell_type"] = df["cell_type"].astype(str)

        # Resolve intra-tissue bulk_ppm inconsistencies with median
        for tissue in df["tissue"].unique():
            vals = df.loc[df["tissue"] == tissue, "bulk_ppm"].dropna().unique()
            if len(vals) > 1:
                med = float(np.nanmedian(vals))
                print(f"[WARNING] {tissue}: bulk_ppm mismatch {np.round(vals, 2)}"
                      f" → using median {med:.3f}")
                df.loc[df["tissue"] == tissue, "bulk_ppm"] = med

        return df

    # ── Joint MCMC ────────────────────────────────────────────────────────

    def _run_joint_inference(self, df: pd.DataFrame):
        """
        Run joint cross-tissue MCMC.

        Design decisions
        ────────────────
        Non-centered parameterisation:
          z_i ~ Normal(0, 1)
          theta_i = exp(SIGMA_THETA × z_i)
          Avoids Neal's funnel geometry that causes divergences with
          centred Lognormal.

        Shared theta (cell type appears in ≥2 tissues):
          The SAME z_i index is referenced in the likelihood for every tissue
          where cell type i appears.  PyMC evaluates all tissue likelihoods
          jointly, so the posterior for z_i (→ theta_i) is shaped by ALL
          tissues simultaneously.  This is the "joint constraint" that makes
          theta estimable rather than prior-dominated.

        Unique theta (cell type in exactly 1 tissue):
          Independent z_i, constrained by that tissue's bulk only.

        Per-tissue alpha:
          Absorbs the absolute RNA→protein scale, proteomics depth differences,
          and any tissue-level batch effects.  HalfNormal prior centred near
          bulk_t; mode at 0 but most mass around bulk_t × 1.6.

        Likelihood:
          Single proteomics data point per tissue.  Fixed 30% CV avoids
          the alpha-sigma funnel that arises with a free sigma parameter.
        """
        tissues = df["tissue"].unique().tolist()

        # ── Classify cell types ───────────────────────────────────────────
        type_tissue_count = df.groupby("cell_type")["tissue"].nunique()
        shared_types = sorted(
            type_tissue_count[type_tissue_count >= 2].index.tolist()
        )
        unique_types = sorted(
            type_tissue_count[type_tissue_count == 1].index.tolist()
        )

        shared_idx = {ct: i for i, ct in enumerate(shared_types)}
        unique_idx = {ct: i for i, ct in enumerate(unique_types)}

        print(f"\n[Joint MCMC]")
        print(f"  Tissues          : {len(tissues)}")
        print(f"  Shared cell types: {len(shared_types)}  "
              f"(theta constrained across ≥2 tissues)")
        print(f"  Unique cell types: {len(unique_types)}  "
              f"(theta constrained by 1 tissue)")
        if shared_types:
            print(f"  Shared examples  : {shared_types[:8]}")

        # ── Per-tissue data ───────────────────────────────────────────────
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

        # ── PyMC model ────────────────────────────────────────────────────
        with pm.Model() as model:  # noqa: F841

            # ── theta: shared and unique ──────────────────────────────────
            # Non-centered parameterisation.
            # For shared types: z_shared is a vector of shape (n_shared,).
            # Each tissue likelihood that contains cell type i will reference
            # theta_shared[shared_idx[i]] — the SAME tensor node — so the
            # posterior of z_i is jointly constrained by all those tissues.
            if shared_types:
                z_shared    = pm.Normal("z_shared", mu=0.0, sigma=1.0,
                                        shape=len(shared_types))
                theta_shared = pm.math.exp(SIGMA_THETA * z_shared)

            if unique_types:
                z_unique    = pm.Normal("z_unique", mu=0.0, sigma=1.0,
                                        shape=len(unique_types))
                theta_unique = pm.math.exp(SIGMA_THETA * z_unique)

            # ── Per-tissue alpha + likelihood ─────────────────────────────
            for tissue in tissues:
                td       = tissue_data[tissue]
                bulk_val = td["bulk_val"]

                # alpha_t: tissue-level RNA→protein absolute scale.
                # HalfNormal(sigma=bulk_val*2): puts substantial prior mass
                # in the range [0, 2×bulk_val], centred ≈ bulk_val.
                alpha_t = pm.HalfNormal(f"alpha_{tissue}",
                                        sigma=bulk_val * 2.0)

                ppm_list = []
                for i, ct in enumerate(td["cell_types"]):
                    s_ncpm = td["scaled_ncpm"][i]
                    theta  = (theta_shared[shared_idx[ct]]
                              if ct in shared_idx
                              else theta_unique[unique_idx[ct]])
                    ppm_list.append(alpha_t * s_ncpm * theta)

                ppm_vec    = pm.math.stack(ppm_list)
                bulk_pred  = pm.math.dot(ppm_vec, td["fractions"])

                # Likelihood: one proteomics observation per tissue.
                # sigma = bulk_val × 0.30  (fixed 30% CV).
                # A free sigma parameter would create an alpha-sigma funnel,
                # causing divergences and poor mixing.
                pm.Normal(f"obs_{tissue}",
                          mu=bulk_pred,
                          sigma=bulk_val * 0.3,
                          observed=bulk_val)

            # ── Sampling ──────────────────────────────────────────────────
            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                nuts_sampler_kwargs={"max_treedepth": 12},
                progressbar=False,
                random_seed=self.random_seed,
            )

        # ── Convergence diagnostics ───────────────────────────────────────
        divergences = int(trace.sample_stats.diverging.sum().item())
        try:
            rhat_max = float(np.nanmax(az.rhat(trace).to_array().values))
            ess_min  = float(np.nanmin(az.ess(trace).to_array().values))
        except Exception:
            rhat_max, ess_min = float("nan"), float("nan")

        converged = (divergences == 0
                     and rhat_max < 1.01
                     and (np.isnan(ess_min) or ess_min >= 400))

        flag = "✓" if converged else "⚠"
        print(f"\n  {flag}  R-hat={rhat_max:.3f}  ESS_min={ess_min:.0f}  "
              f"Div={divergences}")
        if not converged:
            if divergences > 0:
                print(f"     Div={divergences} — consider increasing --tune")
            if rhat_max >= 1.01:
                print(f"     R-hat={rhat_max:.3f} — chains not mixing, "
                      "consider increasing --draws")
            if not np.isnan(ess_min) and ess_min < 400:
                print(f"     ESS_min={ess_min:.0f} — consider increasing --draws")

        # ── Extract posterior samples ─────────────────────────────────────
        n_samples = trace.posterior.dims["chain"] * trace.posterior.dims["draw"]

        if shared_types:
            th_sh = np.exp(
                SIGMA_THETA *
                trace.posterior["z_shared"].values.reshape(n_samples, len(shared_types))
            )
        if unique_types:
            th_un = np.exp(
                SIGMA_THETA *
                trace.posterior["z_unique"].values.reshape(n_samples, len(unique_types))
            )

        tissue_results = {}
        for tissue in tissues:
            td            = tissue_data[tissue]
            alpha_samples = trace.posterior[f"alpha_{tissue}"].values.reshape(n_samples)

            ppm_samples = np.zeros((n_samples, td["n_cells"]))
            for i, ct in enumerate(td["cell_types"]):
                s_ncpm         = td["scaled_ncpm"][i]
                theta_s        = (th_sh[:, shared_idx[ct]]
                                  if ct in shared_idx
                                  else th_un[:, unique_idx[ct]])
                ppm_samples[:, i] = alpha_samples * s_ncpm * theta_s

            ppm_mean = ppm_samples.mean(axis=0)
            hdi_low  = np.array([az.hdi(ppm_samples[:, i], hdi_prob=0.94)[0]
                                  for i in range(td["n_cells"])])
            hdi_high = np.array([az.hdi(ppm_samples[:, i], hdi_prob=0.94)[1]
                                  for i in range(td["n_cells"])])

            tissue_results[tissue] = {
                "ppm_mean": ppm_mean,
                "hdi_low" : hdi_low,
                "hdi_high": hdi_high,
            }

        return tissue_results, converged

    # ── Main pipeline ─────────────────────────────────────────────────────

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run joint MCMC and return results DataFrame.

        Processing order (important):
          1. Preprocess (type coercion, bulk_ppm median dedup)
          2. Compute volume_fraction for ALL tissues — must happen before any
             filtering because PBPK simulator needs this for every tissue
             including those without proteomics data.
          3. Classify tissues into mcmc_tissues vs skipped:
               mcmc_tissues : bulk_ppm > EPSILON_BULK  AND  nCPM sum > EPSILON_NCPM_SUM
               skipped      : epsilon-bulk placeholder OR all-epsilon nCPM
          4. Run joint MCMC over mcmc_tissues only
          5. Write posterior results; skipped tissues keep NaN (→ epsilon in main.py)

        Output columns
        ──────────────
          local_ppm             : posterior mean PPM per cell type
          local_ppm_hdi_low     : 94% HDI lower bound
          local_ppm_hdi_high    : 94% HDI upper bound
          nM_concentration      : local_ppm × K_organ
          bulk_nM_concentration : bulk_ppm × K_organ
          volume_fraction       : protein mass-weighted fraction (ALL tissues)
          k_organ_used          : K_organ value applied
          converged             : MCMC convergence flag
        """
        df = self._preprocess(df)

        # ── Step 1: Compute volume_fraction for ALL tissues ───────────────
        # This MUST run before any tissue filtering — the PBPK simulator
        # needs volume_fraction even for tissues that skip MCMC (e.g. blood
        # with no PaxDb proteomics data).
        for tissue in df["tissue"].unique():
            mask        = df["tissue"] == tissue
            tdata       = df[mask]
            fractions   = _protein_mass_fractions(
                tdata["cell_type"].values,
                tdata["cells"].values.astype(float)
            )
            df.loc[mask, "volume_fraction"] = fractions

        # ── Step 2: Initialise output columns ────────────────────────────
        for col in ["local_ppm", "local_ppm_hdi_low", "local_ppm_hdi_high",
                    "nM_concentration", "bulk_nM_concentration", "k_organ_used"]:
            df[col] = np.nan
        df["converged"] = False

        # ── Step 3: Classify tissues ──────────────────────────────────────
        mcmc_tissues    = []
        skipped_tissues = []

        for tissue in df["tissue"].unique():
            tdata    = df[df["tissue"] == tissue]
            bulk_val = tdata["bulk_ppm"].iloc[0]

            if pd.isna(bulk_val) or bulk_val <= EPSILON_BULK:
                skipped_tissues.append((tissue, "no real bulk_ppm (epsilon placeholder)"))
                continue

            if tdata["nCPM"].sum() <= EPSILON_NCPM_SUM:
                skipped_tissues.append((tissue, "all nCPM = epsilon (no RNA signal)"))
                continue

            mcmc_tissues.append(tissue)

        if skipped_tissues:
            print(f"\n[MCMC skip] {len(skipped_tissues)} tissues excluded:")
            for tissue, reason in skipped_tissues:
                print(f"  {tissue}: {reason}")

        print(f"\n[MCMC] Running on {len(mcmc_tissues)} tissues: "
              f"{sorted(mcmc_tissues)}")

        if not mcmc_tissues:
            print("[ERROR] No tissues with valid bulk_ppm and nCPM.")
            return df

        # ── Step 4: Joint MCMC ────────────────────────────────────────────
        df_valid = df[df["tissue"].isin(mcmc_tissues)].copy()
        try:
            tissue_results, converged = self._run_joint_inference(df_valid)
        except Exception as exc:
            print(f"[ERROR] Joint MCMC failed: {exc}")
            return df

        # ── Step 5: Write results ─────────────────────────────────────────
        for tissue in mcmc_tissues:
            mask     = df["tissue"] == tissue
            res      = tissue_results[tissue]
            bulk_val = df.loc[mask, "bulk_ppm"].iloc[0]

            df.loc[mask, "local_ppm"]          = res["ppm_mean"]
            df.loc[mask, "local_ppm_hdi_low"]  = res["hdi_low"]
            df.loc[mask, "local_ppm_hdi_high"] = res["hdi_high"]
            df.loc[mask, "converged"]          = converged

            tissue_key, k = _get_k_organ(tissue)
            if k is not None:
                df.loc[mask, "nM_concentration"]      = res["ppm_mean"] * k
                df.loc[mask, "bulk_nM_concentration"] = bulk_val * k
                df.loc[mask, "k_organ_used"]          = k
                print(f"  K_organ({tissue_key}) = {k:.4f}")
            else:
                print(f"  [WARNING] {tissue}: not in K_ORGAN dict — "
                      "add entry to bayesian_ppm.py K_ORGAN")

        return df