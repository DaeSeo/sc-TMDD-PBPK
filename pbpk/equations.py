"""
equations.py
============
Whole-Body PBPK + TMDD (QSS) ODE system for large-molecule mAbs (≥150 kDa).

Three governing principles
──────────────────────────

1. Two-Pore Transport (solid organs)
   Antibodies cannot diffuse through tight junctions like small molecules.
   They cross the vascular endothelium only through large pores via:
     · Convection  : Jconv  = L_lymph × Cv × (1 − σ)     [pressure-driven]
     · Diffusion   : PS_diff = PS × (Cv − Cisf/Kp)        [concentration gradient]
     · Lymph drain : Jlymph  = L_lymph × Cisf             [ISF → central via lymph]
   σ (reflection coefficient) ≈ 0.85–0.99 for IgG.
   References: Shah & Betts (2012), Baxter et al. (1994).

2. QSS-TMDD with Gibiansky Total-Drug Formulation
   Kon >> all other PK rates → binding/unbinding at quasi-steady state.
   Rather than tracking free + bound separately (stiff ODE), the state
   variable is TOTAL drug per compartment (free + receptor-bound):

     Ctot = Cfree + Σ(RC_i × vf_i)

   At each ODE call, Cfree is recovered via the Gibiansky quadratic:

     Cfree² + (KSS + Rtot_eff − Ctot) × Cfree − KSS × Ctot = 0
     → Cfree = 0.5 × [−b + √(b² + 4·KSS·Ctot)]  where b = KSS + Rtot_eff − Ctot

   This implicitly captures the dRC/dt buffering term in dCtot/dt,
   preserving exact mass balance without ghost drug.
   Reference: Gibiansky et al. (2008).

3. Anatomical compartment split — 6 solid organs + 1 blood (central)

   Peripheral (solid organs): Brain, Heart, Kidney, Liver, Lung, Skin
     Drug path: Cc (plasma) → Cv (organ vascular) → Cisf (ISF) → receptor
     Two-Pore transport governs each step.
     State per organ: [Cv, Ctot_isf, Rtot_1, …, Rtot_n]

   Central (blood / PBMC):
     Blood cells (monocytes, T-cells, NK cells, platelets) float directly
     in plasma — no ISF barrier. They bind plasma drug (Cc_free) directly.
     The central compartment state variable is therefore:

       y[0] = Ctot_central = Cc_free + RC_blood_eff × (V_blood / V_CENTRAL)

     Cc_free is recovered from Ctot_central via the same Gibiansky quadratic.
     dRC/dt buffering is captured implicitly (identical to ISF approach).
     State for blood: [Rtot_blood_1, …, Rtot_blood_n]  (no separate Cv/Ctot_isf)

State vector layout
───────────────────
  y[0]                            : Ctot_central  — total central drug (nM)*
  Blood block (if present):
    y[1 : 1+n_blood]              : Rtot_i        — blood cell receptors (nM)
  Per solid organ (repeated):
    y[offset]                     : Cv            — organ vascular (nM)
    y[offset+1]                   : Ctot_isf      — total ISF drug (nM)
    y[offset+2 : offset+2+n]      : Rtot_i        — organ receptors (nM)

  * Without blood_data, y[0] = Cc_free (plain plasma; backward compatible).

Units: nM, L, L/day, day.

References
──────────
  Shah & Betts   (2012) J PK/PD 39:67         [two-pore physiology]
  Gibiansky et al.(2008) J PK/PD 35:573        [QSS total-drug]
  Grimm          (2009) J PK/PD 36:407         [QSS for mAb TMDD]
  Baxter et al.  (1994) Cancer Res 54:1517     [PS, σ for IgG]
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Gibiansky quadratic — scalar and vectorised versions
# ──────────────────────────────────────────────────────────────────────────────

def _recover_free_drug(Ctot: float, KSS: float, Rtot_eff: float) -> float:
    """
    Scalar Gibiansky quadratic.  Recovers Cfree from Ctot at one time point.

    Ctot = Cfree + Rtot_eff × Cfree / (KSS + Cfree)
    → Cfree² + (KSS + Rtot_eff − Ctot) × Cfree − KSS × Ctot = 0
    → positive root returned.
    """
    if Ctot <= 0.0:
        return 0.0
    if Rtot_eff <= 0.0:
        return float(Ctot)
    b = KSS + Rtot_eff - Ctot
    disc = b * b + 4.0 * KSS * Ctot
    return max(0.5 * (-b + np.sqrt(max(disc, 0.0))), 0.0)


def _recover_free_drug_vec(Ctot_vec, KSS: float, Rtot_eff_vec):
    """
    Vectorised Gibiansky quadratic.  Used in post-processing (n_time,) arrays.
    """
    Ctot_vec     = np.maximum(Ctot_vec, 0.0)
    Rtot_eff_vec = np.maximum(Rtot_eff_vec, 0.0)
    b    = KSS + Rtot_eff_vec - Ctot_vec
    disc = b * b + 4.0 * KSS * Ctot_vec
    Cfree = np.where(
        Rtot_eff_vec <= 0,
        Ctot_vec,
        np.where(Ctot_vec <= 0, 0.0,
                 0.5 * (-b + np.sqrt(np.maximum(disc, 0.0))))
    )
    return np.maximum(Cfree, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Main ODE
# ──────────────────────────────────────────────────────────────────────────────

def pbpk_qss_ode(t, y, params, system_data, blood_data=None):
    """
    ODE right-hand side: whole-body PBPK + TMDD (QSS, Gibiansky 2008).

    Parameters
    ----------
    t           : float         current time (days)
    y           : ndarray       state vector (see layout in module docstring)
    params      : Config        PK/PD parameters
    system_data : list[dict]    solid organ specifications
    blood_data  : dict | None   blood cell data (activates central TMDD)
    """
    KSS             = params.KSS
    all_derivatives = []
    offset          = 1

    # ──────────────────────────────────────────────────────────────────────
    # CENTRAL COMPARTMENT — recover Cc_free
    #
    # Principle 3 (blood / PBMC):
    #   y[0] = Ctot_central = Cc_free + RC_blood_eff_scaled
    #   Gibiansky quadratic recovers Cc_free from Ctot_central.
    #
    # Rtot_eff_scaled = Σ(Rtot_i × vf_i) × V_blood/V_CENTRAL
    #   [converts blood-volume nM to central-volume nM for the quadratic]
    #
    # dRtot_i/dt = ksyn_i − K_DEG × R_i − K_INT × RC_i
    #   [receptor synthesis, free-R degradation, RC internalization]
    # ──────────────────────────────────────────────────────────────────────
    if blood_data is not None:
        n_blood      = blood_data['n_cells']
        vol_fracs_b  = blood_data['vol_fracs']
        ksyn_b       = blood_data['ksyn_vec']
        V_blood      = blood_data['V_blood']

        Ctot_central = max(y[0], 0.0)
        Rtot_blood   = np.maximum(y[1: 1 + n_blood], 0.0)

        Rtot_eff_b      = np.dot(Rtot_blood, vol_fracs_b)               # nM (blood vol)
        Rtot_eff_scaled = Rtot_eff_b * V_blood / params.V_CENTRAL        # nM (central vol)

        Cc_free = _recover_free_drug(Ctot_central, KSS, Rtot_eff_scaled)

        # QSS decomposition — blood cells see Cc_free directly (no ISF)
        denom_b      = KSS + Cc_free + 1e-15
        RC_blood     = Rtot_blood * Cc_free / denom_b   # (n_blood,) nM
        R_blood      = Rtot_blood * KSS    / denom_b
        RC_blood_eff = np.dot(RC_blood, vol_fracs_b)    # weighted mean (blood vol)

        dRtot_blood_dt = ksyn_b - params.K_DEG * R_blood - params.K_INT * RC_blood
        all_derivatives.append(dRtot_blood_dt)
        offset += n_blood

    else:
        # No blood TMDD — y[0] is plain Cc_free (backward compatible)
        Cc_free      = max(y[0], 0.0)
        RC_blood_eff = 0.0
        V_blood      = 0.0

    Cc = Cc_free  # alias for solid organ loop

    # ──────────────────────────────────────────────────────────────────────
    # SOLID ORGAN LOOP — Principles 1 + 2
    #
    # Two-Pore transport (Principle 1):
    #   Jconv  = L_lymph × Cv × (1−σ)          [convection into ISF]
    #   Jlymph = L_lymph × Cisf                 [lymphatic drainage to central]
    #   PS_diff = PS × (Cv − Cisf/Kp)           [diffusion across endothelium]
    #
    # Gibiansky QSS state (Principle 2):
    #   Ctot_isf = total ISF drug (free + bound)
    #   Cisf = quadratic_recover(Ctot_isf, KSS, Rtot_eff)
    # ──────────────────────────────────────────────────────────────────────
    venous_return = 0.0
    lymph_return  = 0.0

    for tissue in system_data:
        spec      = tissue['spec']
        n_cells   = tissue['n_cells']
        vol_fracs = tissue['vol_fracs']
        ksyn_vec  = tissue['ksyn_vec']

        Q_organ  = params.Q_TOTAL * spec['f_q']
        V_v      = spec['V_total'] * spec['f_v']
        V_isf    = spec['V_total'] * spec['f_isf']

        if V_v <= 0:
            raise ValueError(
                f"V_v = 0 for '{tissue['name']}' — check f_v in TISSUE_SPECS."
            )
        if V_isf <= 0:
            raise ValueError(
                f"V_isf = 0 for '{tissue['name']}' — check f_isf in TISSUE_SPECS."
            )

        Cv       = max(y[offset],     0.0)
        Ctot_isf = max(y[offset + 1], 0.0)
        Rtot_vec = np.maximum(y[offset + 2: offset + 2 + n_cells], 0.0)

        # Gibiansky quadratic → Cisf (free ISF drug)
        Rtot_eff = np.dot(Rtot_vec, vol_fracs)
        Cisf     = _recover_free_drug(Ctot_isf, KSS, Rtot_eff)

        # QSS algebraic split
        denom  = KSS + Cisf + 1e-15
        RC_vec = Rtot_vec * Cisf / denom
        R_vec  = Rtot_vec * KSS  / denom

        # Two-Pore transport (operates on FREE drug only)
        L_lymph = spec['L_lymph']
        sigma   = spec['sigma']
        Jconv   = L_lymph * Cv   * (1.0 - sigma)          # convection into ISF
        Jlymph  = L_lymph * Cisf                           # free ISF → central (lymph)
        PS_diff = spec['PS'] * (Cv - Cisf / spec['Kp'])   # diffusion

        # TMDD sink: irreversible internalization of drug-receptor complex
        # RC_vec [nM], K_INT [day⁻¹] → tmdd_sink [nM/day] in ISF
        tmdd_sink = np.sum(params.K_INT * RC_vec * vol_fracs)

        # ODEs
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
    #
    # dCtot_central/dt:
    #   + venous_return   : blood flow returning from organs (uses Cc_free)
    #   + lymph_return    : lymphatic drainage (free ISF drug)
    #   − CL_0/V_c × Cc_free : linear non-specific clearance
    #   − K_INT × RC_blood_eff × V_blood/V_c : blood-cell internalization sink
    #
    # The dRC_blood/dt buffering term is NOT written explicitly.
    # It is captured IMPLICITLY: the integrator tracks Ctot_central, so when
    # Cc_free falls and RC decreases, drug is automatically released back into
    # the free pool — identical to how Ctot_isf works in solid organs.
    # This is the correct Gibiansky mass-balance approach.
    # ──────────────────────────────────────────────────────────────────────
    dCtot_central_dt = (
        venous_return
        + lymph_return
        - (params.CL_0 / params.V_CENTRAL) * Cc_free
        - params.K_INT * RC_blood_eff * V_blood / params.V_CENTRAL
    )

    return np.concatenate([[dCtot_central_dt], *all_derivatives])