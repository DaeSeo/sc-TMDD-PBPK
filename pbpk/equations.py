"""
equations.py
============
Whole-Body PBPK + TMDD (QSS) ODE system for large-molecule mAbs (≥150 kDa).

Three governing principles
──────────────────────────

1. Two-Pore Transport (solid organs, simplified single-effective-pore)
   Antibodies cross the vascular endothelium via:
     · Convection  : Jconv  = L_lymph × Cv × (1 − σ)     [pressure-driven]
     · Diffusion   : PS_diff = PS × (Cv − Cisf/Kp)        [gradient-driven]
     · Lymph drain : Jlymph  = L_lymph × Cisf_drain        [ISF → central]
   σ (reflection coefficient) ≈ 0.85–0.99 for IgG.

   Lymph drainage note — target type matters:
     Membrane targets (EGFR, HER2, …): drug-RC is anchored to cell surface.
       → Only free drug drains:  Jlymph = L_lymph × Cisf_free
     Soluble targets (CCL2, VEGF, …): drug-RC is free in ISF.
       → Total drug drains:     Jlymph = L_lymph × Ctot_isf

2. QSS-TMDD with Gibiansky Total-Drug Formulation
   State variable is TOTAL drug per compartment (free + receptor-bound):
     Ctot = Cfree + Σ(RC_i × vf_i)
   Cfree recovered via the Gibiansky quadratic:
     Cfree = 0.5 × [−b + √(b² + 4·KSS·Ctot)]
     where b = KSS + Rtot_eff − Ctot.
   Reference: Gibiansky et al. (2008).

3. Parallel organ architecture — 6 solid organs + Rest + 1 blood (central)
   All organs in parallel to central compartment.
   See config.py for Lung f_q rationale.

State vector layout
───────────────────
  y[0]                            : Ctot_central
  Blood block (if present):
    y[1 : 1+n_blood]              : Rtot_i (blood cell receptors)
  Per solid organ (repeated):
    y[offset]                     : Cv
    y[offset+1]                   : Ctot_isf
    y[offset+2 : offset+2+n]      : Rtot_i (organ receptors)

  Without blood_data, y[0] = Cc_free (backward compatible).

Units: nM, L, L/day, day.

References
──────────
  Shah & Betts   (2012) J PK/PD 39:67
  Gibiansky et al.(2008) J PK/PD 35:573
  Grimm          (2009) J PK/PD 36:407
  Baxter et al.  (1994) Cancer Res 54:1517
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Gibiansky quadratic — scalar and vectorised versions
# ──────────────────────────────────────────────────────────────────────────────

def _recover_free_drug(Ctot: float, KSS: float, Rtot_eff: float) -> float:
    """
    Scalar Gibiansky quadratic.  Recovers Cfree from Ctot at one time point.

    Two numerical stabilisation fixes:

    Fix 1 — Rationalised form (CVODE / SimCYP standard):
      When b >> 0 (low drug, high receptor), naive −b + √(b²+ε) cancels
      two large numbers.  Rationalised form avoids this:
          Cfree = 2·KSS·Ctot / (b + √(b² + 4·KSS·Ctot))

    Fix 2 — No max(result, 0) clip (C1 continuity for Radau Jacobian):
      Hard clips create a kink at Ctot = 0, halving the numerical Jacobian
      and stalling Radau's Newton iterations.  Raw formula output is returned;
      downstream equations are self-correcting.
    """
    if Rtot_eff <= 0.0:
        return float(Ctot)
    b    = KSS + Rtot_eff - Ctot
    disc = b * b + 4.0 * KSS * Ctot
    if disc <= 0.0:
        return 0.0
    sqrt_disc = np.sqrt(disc)
    if b >= 0.0:
        return 2.0 * KSS * Ctot / (b + sqrt_disc)
    else:
        return 0.5 * (-b + sqrt_disc)


def _recover_free_drug_vec(Ctot_vec, KSS: float, Rtot_eff_vec):
    """
    Vectorised Gibiansky quadratic for post-processing (n_time,) arrays.
    Post-processing results clipped to ≥ 0 (visualisation only, not ODE).
    """
    Ctot_vec     = np.asarray(Ctot_vec, dtype=float)
    Rtot_eff_vec = np.maximum(np.asarray(Rtot_eff_vec, dtype=float), 0.0)
    b         = KSS + Rtot_eff_vec - Ctot_vec
    disc      = b * b + 4.0 * KSS * Ctot_vec
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    cfree_rat = np.where(
        disc > 0,
        2.0 * KSS * Ctot_vec / (b + sqrt_disc + 1e-300),
        0.0
    )
    cfree_std = 0.5 * (-b + sqrt_disc)
    Cfree = np.where(
        Rtot_eff_vec <= 0,
        Ctot_vec,
        np.where(b >= 0, cfree_rat, cfree_std)
    )
    return np.maximum(Cfree, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Main ODE
# ──────────────────────────────────────────────────────────────────────────────

def pbpk_qss_ode(t, y, params, system_data, blood_data=None, soluble_data=None):
    """
    ODE right-hand side: whole-body PBPK + TMDD (QSS, Gibiansky 2008).

    Parameters
    ----------
    t             : float         current time (days)
    y             : ndarray       state vector
    params        : Config        PK/PD parameters
    system_data   : list[dict]    solid organ specifications
    blood_data    : dict | None   blood cell membrane TMDD (membrane targets only)
    soluble_data  : dict | None   plasma-phase soluble target pool
                                  Mutually exclusive with blood_data:
                                  when soluble_target=True, blood_data=None.

    State vector layout
    ───────────────────
    Membrane target (soluble_data=None):
      y[0]              Ctot_central
      y[1:1+n_blood]    Rtot per blood cell type  (if blood_data)
      per solid organ:  [Cv, Ctot_isf, Rtot_1..n]

    Soluble target (blood_data=None):
      y[0]              Ctot_central
      y[1]              R_sol  (soluble receptor pool in plasma)
      per solid organ:  [Cv, Ctot_isf, Rtot_1..n]
    """
    KSS             = params.KSS
    all_derivatives = []
    offset          = 1

    # ──────────────────────────────────────────────────────────────────────
    # CENTRAL COMPARTMENT — recover Cc_free
    # ──────────────────────────────────────────────────────────────────────
    if blood_data is not None:
        n_blood      = blood_data['n_cells']
        vol_fracs_b  = blood_data['vol_fracs']
        ksyn_b       = blood_data['ksyn_vec']
        V_blood      = blood_data['V_blood']

        Ctot_central = y[0]
        Rtot_blood   = y[1: 1 + n_blood]

        Rtot_eff_b      = np.dot(np.maximum(Rtot_blood, 0.0), vol_fracs_b)
        Rtot_eff_scaled = Rtot_eff_b * V_blood / params.V_CENTRAL

        Cc_free = _recover_free_drug(Ctot_central, KSS, Rtot_eff_scaled)

        denom_b      = KSS + Cc_free + 1e-15
        RC_blood     = np.maximum(Rtot_blood, 0.0) * Cc_free / denom_b
        R_blood      = np.maximum(Rtot_blood, 0.0) * KSS    / denom_b
        RC_blood_eff = np.dot(RC_blood, vol_fracs_b)

        dRtot_blood_dt = ksyn_b - params.K_DEG * R_blood - params.K_INT * RC_blood
        all_derivatives.append(dRtot_blood_dt)
        offset += n_blood
    else:
        Cc_free      = _recover_free_drug(y[0], KSS, 0.0)
        RC_blood_eff = 0.0
        V_blood      = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # SOLUBLE TARGET — dedicated plasma-phase TMDD compartment
    #
    # Binding occurs directly in plasma (not ISF).
    # R_sol is the FREE soluble receptor pool in plasma (nM).
    # RC_sol = drug–receptor complex in plasma.
    #
    # SC and Bulk modes share the same R₀ (bulk plasma concentration)
    # → RO curves are identical for SC and Bulk (biologically correct).
    # ──────────────────────────────────────────────────────────────────────
    RC_sol_sink = 0.0
    if soluble_data is not None:
        R_sol     = y[offset]
        R_sol_pos = max(R_sol, 0.0)

        denom_sol = KSS + Cc_free + 1e-15
        RC_sol    = R_sol_pos * Cc_free / denom_sol
        R_sol_f   = R_sol_pos * KSS    / denom_sol

        dR_sol_dt   = soluble_data['ksyn'] - params.K_DEG * R_sol_f - params.K_INT * RC_sol
        RC_sol_sink = params.K_INT * RC_sol   # drug consumed by soluble target

        all_derivatives.append(np.array([dR_sol_dt]))
        offset += 1

    Cc = Cc_free

    # ──────────────────────────────────────────────────────────────────────
    # SOLID ORGAN LOOP
    # ──────────────────────────────────────────────────────────────────────
    venous_return = 0.0
    lymph_return  = 0.0

    for tissue in system_data:
        spec      = tissue['spec']
        n_cells   = tissue['n_cells']
        vol_fracs = tissue['vol_fracs']
        ksyn_vec  = tissue['ksyn_vec']

        Q_organ = params.Q_TOTAL * spec['f_q']
        V_v     = spec['V_total'] * spec['f_v']
        V_isf   = spec['V_total'] * spec['f_isf']

        if V_v <= 0:
            raise ValueError(
                f"V_v = 0 for '{tissue['name']}' — check f_v in TISSUE_SPECS.")
        if V_isf <= 0:
            raise ValueError(
                f"V_isf = 0 for '{tissue['name']}' — check f_isf in TISSUE_SPECS.")

        # ── State variables (no hard clips — C1 continuity for Jacobian) ──
        Cv       = y[offset]
        Ctot_isf = y[offset + 1]
        Rtot_vec = y[offset + 2: offset + 2 + n_cells]

        # Gibiansky quadratic → Cisf (free ISF drug)
        Rtot_eff = np.dot(np.maximum(Rtot_vec, 0.0), vol_fracs)
        Cisf     = _recover_free_drug(Ctot_isf, KSS, Rtot_eff)

        # QSS algebraic split
        Rtot_pos = np.maximum(Rtot_vec, 0.0)
        denom    = KSS + Cisf + 1e-15
        RC_vec   = Rtot_pos * Cisf / denom
        R_vec    = Rtot_pos * KSS  / denom

        # ── Two-Pore transport (operates on FREE drug only) ───────────
        L_lymph = spec['L_lymph']
        sigma   = spec['sigma']

        # No max(Cv, 0) clip — maintains C1 continuity for Radau Jacobian.
        # Cv is flow-driven (Q_organ >> L_lymph) so negative transients are
        # tiny and self-correcting: dCv/dt ∝ Q(Cc − Cv) restores Cv → Cc.
        Jconv   = L_lymph * Cv * (1.0 - sigma)
        PS_diff = spec['PS'] * (Cv - Cisf / spec['Kp'])

        # ── Lymph drainage: target-type dependent ─────────────────────
        # Membrane target: only free Cisf drains (drug-RC anchored to cell).
        # Soluble target (ISF):  total Ctot_isf drains (drug-RC free in ISF).
        # Note: primary soluble TMDD is now in plasma (soluble_data block above).
        #       Tissue ISF still uses membrane-style drain for any residual.
        if soluble_data is not None:
            Jlymph = L_lymph * Ctot_isf
        else:
            Jlymph = L_lymph * Cisf

        # TMDD sink: irreversible internalization of drug-receptor complex
        tmdd_sink = np.sum(params.K_INT * RC_vec * vol_fracs)

        # ── ODEs ──────────────────────────────────────────────────────
        dCv_dt = (
            (Q_organ / V_v) * (Cc - Cv)
            - PS_diff / V_v
            - Jconv   / V_v
        )
        dCtot_isf_dt = (
            + PS_diff / V_isf
            + Jconv   / V_isf
            - Jlymph  / V_isf
            - tmdd_sink
        )
        dRtot_dt = ksyn_vec - params.K_DEG * R_vec - params.K_INT * RC_vec

        venous_return += (Q_organ / params.V_CENTRAL) * (Cv - Cc)
        lymph_return  += Jlymph / params.V_CENTRAL

        all_derivatives.append(np.concatenate([[dCv_dt, dCtot_isf_dt], dRtot_dt]))
        offset += (2 + n_cells)

    # ──────────────────────────────────────────────────────────────────────
    # CENTRAL ODE
    # RC_sol_sink: drug consumed by plasma-phase soluble target internalization
    # ──────────────────────────────────────────────────────────────────────
    dCtot_central_dt = (
        venous_return
        + lymph_return
        - (params.CL_0 / params.V_CENTRAL) * Cc_free
        - params.K_INT * RC_blood_eff * V_blood / params.V_CENTRAL
        - RC_sol_sink   # soluble target plasma TMDD sink
    )

    return np.concatenate([[dCtot_central_dt], *all_derivatives])