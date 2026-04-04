"""
simulator.py
============
PBPKSimulator — builds the ODE system, integrates it, and unpacks results.

Architecture
────────────
  Peripheral compartments (solid organs — Two-Pore, parallel):
    Brain, Heart, Kidney, Liver, Lung, Skin, Breast, Rest
    State per organ: [Cv, Ctot_isf, Rtot_1, …, Rtot_n]

  Central compartment (blood / PBMC — intravascular TMDD):
    y[0] = Ctot_central = Cc_free + RC_blood_eff × (V_blood/V_CENTRAL)

Two simulation modes
────────────────────
  run_single_cell() : Bayesian-inferred per-cell-type nM
  run_bulk()        : tissue-average bulk nM

Column mapping for R0 (receptor initial concentration)
───────────────────────────────────────────────────────
  BULK mode:
    nM_col = 'bulk_nM_concentration'  (= bulk_nM_membrane_ISF)
    This is the tissue-average ISF receptor concentration,
    equivalent to Sepp (2024) mRint,organ.
    Consistent with Cisf (ISF drug conc.) in ODE units.

  SINGLE-CELL mode:
    nM_col = 'nM_concentration'  (= nM_local_ISF per cell type)
    The ODE computes Rtot_eff = Σ(vol_frac_i × R0_i).
    With R0_i = nM_local_ISF_i:
      Rtot_eff = Σ(vol_frac_i × nM_local_ISF_i) = bulk_nM_membrane_ISF  ✓
    Using contribution_to_bulk_ISF instead would apply vol_frac twice
    (double-weighting), giving Rtot_eff = Σ(nM_local_ISF_i × vol_frac_i²),
    which is INCORRECT.

  NOTE: nM_local_ISF and bulk_nM_membrane_ISF (ISF units) are the correct
  quantities for R0 because the ODE drug state Cisf is also in ISF-volume
  concentration units — unit consistency requires both in nM ISF.
  The '_tissue' variants (bulk_nM_membrane_tissue, contribution_to_bulk_tissue)
  are pre-f_ISF values used for Sepp comparison bookkeeping only.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from pbpk.equations import pbpk_qss_ode, _recover_free_drug_vec

EXCLUDED_TISSUES = {'plasma', 'blood'}

# Total blood cell volume (erythrocytes + leukocytes, excluding plasma).
# Shah & Betts (2012) Table 4: V_whole_blood ≈ 4.7 L, V_plasma ≈ 3.126 L
V_BLOOD_CELLS = 1.574  # L


class PBPKSimulator:

    def __init__(self, data_path: str, config):
        self.config = config
        self.df = pd.read_csv(data_path)

        if 'target' in self.df.columns:
            self.df = self.df[
                self.df['target'].str.lower() == config.target_name
            ].copy()

        print(f"\n[*] Loaded : {data_path}")
        print(f"    Rows   : {len(self.df)}")
        print(f"    Target : {config.target_name.upper()}")
        print(f"    Tissues: {sorted(self.df['tissue'].unique().tolist())}")

        self._validate_input()

    # ─────────────────────────────────────────────────────────────────────
    # Input validation
    # ─────────────────────────────────────────────────────────────────────

    def _validate_input(self):
        """Check required columns, tissue coverage, f_q mass balance, and nM sanity."""
        cfg = self.config
        df  = self.df

        # nM_concentration      = nM_local_ISF  (single-cell R0, ISF units)
        # bulk_nM_concentration = bulk_nM_membrane_ISF  (bulk R0, ISF units)
        REQUIRED = ['tissue', 'cell_type', 'nM_concentration',
                    'bulk_nM_concentration', 'volume_fraction', 'cells']

        missing_cols = [c for c in REQUIRED if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"[ERROR] Missing columns: {missing_cols}\n"
                f"  Available: {df.columns.tolist()}"
            )
        print("\n[✓] All required columns present.")

        # Tissue coverage check
        SPEC_EXCLUDED = {'plasma'}
        tissues_in_data = {str(t).lower() for t in df['tissue'].unique()} - SPEC_EXCLUDED
        tissues_in_spec = {str(t).lower() for t in cfg.TISSUE_SPECS.keys()} - SPEC_EXCLUDED
        missing_t = tissues_in_data - tissues_in_spec
        if missing_t:
            raise KeyError(
                f"[ERROR] Tissues not in TISSUE_SPECS: {missing_t}\n"
                f"  Add them to config.py TISSUE_SPECS."
            )
        print("[✓] All tissues have TISSUE_SPECS entries.")

        # f_q mass balance check
        solid_specs = {k: v for k, v in cfg.TISSUE_SPECS.items()
                       if k.lower() not in ('plasma', 'blood')}
        fq_sum = sum(v['f_q'] for v in solid_specs.values())
        if abs(fq_sum - 1.0) > 0.01:
            print(f"[WARNING] Σ f_q = {fq_sum:.4f} ≠ 1.0 — blood flow mass balance violated.")
        else:
            print(f"[✓] Σ f_q = {fq_sum:.4f} (mass balance OK)")

        # nM summary (single-cell R0 = nM_concentration = nM_local_ISF)
        SHOW = {str(t).lower() for t in df['tissue'].unique()} - {'plasma'}
        print(f"\n[*] nM concentration summary (single-cell, ISF units):")
        print(f"    {'Tissue':<12} {'Cell type':<40} {'R0 (nM)':>10}  {'R0/KSS':>8}")
        print(f"    {'-'*12} {'-'*40} {'-'*10}  {'-'*8}")
        for tissue_lc in sorted(SHOW):
            tdf = df[df['tissue'].str.lower() == tissue_lc]
            iv  = " [iv]" if tissue_lc == 'blood' else ""
            for _, row in tdf.iterrows():
                nm  = row['nM_concentration']   # nM_local_ISF
                idx = nm / cfg.KSS if cfg.KSS > 0 else float('nan')
                flag = "  ← TMDD RISK" if idx > 1.0 else ""
                print(f"    {row['tissue']:<12} {row['cell_type']:<40} "
                      f"{nm:>10.2f}  {idx:>8.2f}{flag}{iv}")

        print(f"\n[*] volume_fraction sum per tissue (should be ≈1.0):")
        for tissue_lc in sorted(SHOW):
            tdf    = df[df['tissue'].str.lower() == tissue_lc]
            vf_sum = tdf['volume_fraction'].sum()
            flag   = "  ← WARNING" if abs(vf_sum - 1.0) > 0.05 else ""
            label  = tdf['tissue'].iloc[0] if not tdf.empty else tissue_lc
            print(f"    {label:<12}  {vf_sum:.4f}{flag}")

        t_half_h = 0.693 / cfg.K_INT * 24
        print(f"\n[*] K_INT = {cfg.K_INT} day⁻¹  →  t½_int = {t_half_h:.1f} h")
        if t_half_h > 6:
            print("    [WARNING] t½_int > 6 h — verify K_INT.")
        print(f"[*] KSS = {cfg.KSS:.4f} nM   Kd = {cfg.Kd:.4f} nM")
        print(f"[*] Target type: {'soluble' if cfg.soluble_target else 'membrane-bound'}")

    # ─────────────────────────────────────────────────────────────────────
    # System builder
    # ─────────────────────────────────────────────────────────────────────

    def _build_system(self, use_bulk: bool):
        """
        Construct system_data (solid organs), blood_data, and initial y0.

        R0 column selection
        ───────────────────
          Bulk mode   → 'bulk_nM_concentration' (= bulk_nM_membrane_ISF)
            Single tissue-average ISF receptor concentration.
            ODE uses this directly as Rtot_eff.

          SC mode     → 'nM_concentration'  (= nM_local_ISF per cell type)
            ODE applies vol_frac weighting:
              Rtot_eff = Σ(vol_frac_i × nM_local_ISF_i) = bulk_nM_ISF  ✓
            This is mathematically equivalent to bulk mode for aggregate
            dynamics, but resolves per-cell-type occupancy individually.

        State vector layout
        ───────────────────
          y[0]                       Ctot_central
          y[1 : 1+n_blood]           Rtot per blood cell type
          per solid organ:
            y[off]                   Cv
            y[off+1]                 Ctot_isf
            y[off+2 : off+2+n]       Rtot per cell type
        """
        cfg    = self.config
        # ── R0 column: ISF concentration units in both modes ─────────────────
        # bulk_nM_concentration = bulk_nM_membrane_ISF  (tissue-level ISF nM)
        # nM_concentration      = nM_local_ISF          (per-cell ISF nM)
        # Both are in ISF-volume nM, consistent with Cisf in the ODE.
        nM_col = 'bulk_nM_concentration' if use_bulk else 'nM_concentration'
        label  = "BULK" if use_bulk else "SINGLE-CELL"
        print(f"\n[*] Building system — {label}  (R0 col: '{nM_col}')")

        # ── Dose at t=0 ──────────────────────────────────────────────────
        schedule    = sorted(cfg.DOSING_SCHEDULE, key=lambda x: x['time'])
        first_entry = schedule[0]
        if first_entry['time'] == 0:
            first_dose = first_entry['dose_mg_m2']
            total_nmol = first_dose * cfg.BSA / cfg.MW * 1e6
            Cc0        = total_nmol / cfg.V_CENTRAL
            print(f"\n[*] t=0 dose: {first_dose} mg/m²  →  {total_nmol:.1f} nmol"
                  f"  →  Cc0 = {Cc0:.2f} nM")
        else:
            Cc0 = 0.0
            print(f"\n[*] No dose at t=0 (first dose at day {first_entry['time']})")

        y0 = np.array([Cc0])

        # ── Blood compartment — auto-split by cell_type name ─────────────
        #
        # DEFAULT RULE (no CLI flag needed):
        #   cell_type starts with 'soluble_'  → plasma pool  (SC = Bulk, R0 always bulk)
        #   everything else                   → membrane TMDD on blood cells (SC ≠ Bulk OK)
        #
        # This covers mixed cases (e.g. EGFR data has both soluble_EGFR
        # and membrane-bound b-cells / t-cells in the same blood tissue).
        #
        # State vector ordering MUST match equations.py offset logic:
        #   y[0]                  Ctot_central
        #   y[1 : 1+n_blood]      Rtot membrane blood cells   (if blood_data)
        #   y[1+n_blood]          R_sol                        (if soluble_data)
        #   solid organs follow…
        # ─────────────────────────────────────────────────────────────────
        blood_data   = None
        soluble_data = None
        blood_rows   = self.df[self.df['tissue'].str.lower() == 'blood']

        if not blood_rows.empty:
            is_sol        = blood_rows['cell_type'].str.lower().str.startswith('soluble_')
            soluble_rows  = blood_rows[is_sol]
            membrane_rows = blood_rows[~is_sol]

            # ── 1) Membrane entries FIRST → y[1:1+n_blood] ───────────────
            if not membrane_rows.empty:
                n_blood  = len(membrane_rows)
                R0_blood = membrane_rows[nM_col].fillna(0.0).values.astype(float)

                vf_b = membrane_rows['volume_fraction'].fillna(
                    1.0 / n_blood).values.astype(float)
                vf_b = vf_b / vf_b.sum() if vf_b.sum() > 0 else vf_b

                Rtot_eff_scaled0 = np.dot(R0_blood, vf_b) * V_BLOOD_CELLS / cfg.V_CENTRAL
                RC0_scaled       = Rtot_eff_scaled0 * Cc0 / (cfg.KSS + Cc0 + 1e-15)
                y0[0]            = Cc0 + RC0_scaled

                blood_data = {
                    'name'     : 'blood',
                    'n_cells'  : n_blood,
                    'vol_fracs': vf_b,
                    'ksyn_vec' : cfg.K_DEG * R0_blood,
                    'V_blood'  : V_BLOOD_CELLS,
                    'tdf'      : membrane_rows,
                    'R0'       : R0_blood,
                }
                y0 = np.concatenate([y0, R0_blood])   # y[1:1+n_blood]

                print(f"\n    [blood — membrane] {n_blood} cell types")
                print(f"      R0 range   = {R0_blood.min():.2f}–{R0_blood.max():.2f} nM")
                tmdd_b = R0_blood / cfg.KSS
                risk_b = [(ct, f"{v:.1f}") for ct, v in
                          zip(membrane_rows['cell_type'].values, tmdd_b) if v > 1.0]
                if risk_b:
                    print(f"      TMDD risk: "
                          + ", ".join(f"{ct}={v}" for ct, v in risk_b))
                else:
                    print(f"      TMDD risk: none")
            else:
                print("\n    [blood — membrane] no membrane cell types in data")

            # ── 2) Soluble entries SECOND → y[1+n_blood] ─────────────────
            if not soluble_rows.empty:
                R0_sol = float(soluble_rows['bulk_nM_concentration'].fillna(0.0).mean())
                soluble_data = {
                    'R0'  : R0_sol,
                    'ksyn': cfg.K_DEG * R0_sol,
                    'tdf' : soluble_rows,
                }
                y0 = np.concatenate([y0, [R0_sol]])   # y[1+n_blood] or y[1] if no membrane

                print(f"\n    [soluble — plasma pool]  "
                      f"{soluble_rows['cell_type'].tolist()}")
                print(f"      R0 = {R0_sol:.4f} nM  (bulk_nM; SC = Bulk always)")
                print(f"      TMDD index R0/KSS = {R0_sol/cfg.KSS:.2f}"
                      + ("  ← TMDD RISK" if R0_sol / cfg.KSS > 1 else ""))
        else:
            print("\n    [blood] not in data — intravascular TMDD inactive")

        # ── Solid organ loop ──────────────────────────────────────────────
        tissues     = [t for t in self.df['tissue'].unique()
                       if str(t).lower() not in EXCLUDED_TISSUES]
        system_data = []

        for tissue in tissues:
            tdf     = self.df[self.df['tissue'] == tissue].copy()
            n_cells = len(tdf)
            R0      = tdf[nM_col].fillna(0.0).values.astype(float)

            vf = tdf['volume_fraction'].fillna(1.0 / n_cells).values.astype(float)
            vf = vf / vf.sum() if vf.sum() > 0 else vf

            spec_key = next(
                (k for k in cfg.TISSUE_SPECS if k.lower() == str(tissue).lower()),
                None
            )
            if spec_key is None:
                continue
            spec  = cfg.TISSUE_SPECS[spec_key]
            V_isf = spec['V_total'] * spec['f_isf']

            # Effective R0 check (SC: Rtot_eff = dot(vf, R0) = bulk_nM_ISF)
            Rtot_eff_check = float(np.dot(vf, R0))
            print(f"\n    [{tissue}] {n_cells} cell types")
            print(f"      V_isf      = {V_isf:.4f} L    f_q = {spec['f_q']:.4f}")
            print(f"      R0 range   = {R0.min():.2f}–{R0.max():.2f} nM")
            print(f"      Rtot_eff   = {Rtot_eff_check:.2f} nM  (= Σ vol_frac×R0)")

            tmdd_idx = R0 / cfg.KSS
            risk_t   = [(ct, f"{v:.1f}") for ct, v in
                        zip(tdf['cell_type'].values, tmdd_idx) if v > 1.0]
            if risk_t:
                print(f"      TMDD risk: "
                      + ", ".join(f"{ct}={v}" for ct, v in risk_t))
            else:
                print(f"      TMDD risk: none")

            system_data.append({
                'name'     : tissue,
                'n_cells'  : n_cells,
                'vol_fracs': vf,
                'ksyn_vec' : cfg.K_DEG * R0,
                'spec'     : spec,
                'tdf'      : tdf,
                'R0'       : R0,
            })
            y0 = np.concatenate([y0, [0.0, 0.0], R0])

        return y0, system_data, blood_data, soluble_data, tissues

    # ─────────────────────────────────────────────────────────────────────
    # ODE integrator with repeated dosing
    # ─────────────────────────────────────────────────────────────────────

    def _solve_with_dosing(self, y0, system_data, blood_data, soluble_data, days: int):
        """
        Integrate ODE with event-driven restart at each dose time.

        Numerical stability — per-variable atol (CVODE/SUNDIALS strategy):
          atol_i = max(|y0_i|, C_FLOOR) × RTOL_SCALE
        """
        cfg = self.config
        schedule = sorted(cfg.DOSING_SCHEDULE, key=lambda x: x['time'])

        dose_events = [(d['time'], d['dose_mg_m2'])
                       for d in schedule if 0 < d['time'] < days]
        dose_times  = {d[0] for d in dose_events}
        breakpoints = sorted({d[0] for d in dose_events} | {days})

        print(f"\n[*] Repeat dosing: {len(dose_events)} post-t0 doses over {days} days")
        for dt, dm in dose_events:
            dCc = dm * cfg.BSA / cfg.MW * 1e6 / cfg.V_CENTRAL
            print(f"    Day {dt:>4}: {dm} mg/m²  →  +{dCc:.2f} nM free Cc")

        # Tolerance setup
        C_FLOOR    = 1e-12
        RTOL_SCALE = 1e-4
        RTOL       = 1e-6

        atol_vec    = np.maximum(np.abs(y0), C_FLOOR) * RTOL_SCALE
        atol_vec[0] = max(atol_vec[0], C_FLOOR * 1e3)

        print(f"\n[*] Solver: Radau (implicit, L-stable)")
        print(f"    rtol       = {RTOL:.0e}")
        print(f"    atol range = {atol_vec.min():.2e} – {atol_vec.max():.2e}")
        print(f"    max_step   = 0.5 day")

        all_t, all_y = [], []
        y_cur  = y0.copy()
        t_cur  = 0.0

        for bp in breakpoints:
            if bp <= t_cur:
                continue

            t_eval = np.linspace(t_cur, bp, max(10, int((bp - t_cur) * 10)))

            atol_seg    = np.maximum(np.abs(y_cur), C_FLOOR) * RTOL_SCALE
            atol_seg[0] = max(atol_seg[0], C_FLOOR * 1e3)

            seg = solve_ivp(
                pbpk_qss_ode,
                t_span=(t_cur, bp),
                y0=y_cur,
                args=(cfg, system_data, blood_data, soluble_data),
                t_eval=t_eval,
                method='Radau',
                rtol=RTOL,
                atol=atol_seg,
                max_step=0.5,
            )
            if not seg.success:
                raise RuntimeError(
                    f"[ERROR] ODE solver failed at t={t_cur}–{bp}: {seg.message}"
                )
            all_t.append(seg.t)
            all_y.append(seg.y)

            y_cur = seg.y[:, -1].copy()

            if bp in dose_times:
                dose_total   = sum(dm for dt, dm in dose_events if dt == bp)
                delta_Cc_raw = dose_total * cfg.BSA / cfg.MW * 1e6 / cfg.V_CENTRAL

                if blood_data is not None:
                    from pbpk.equations import _recover_free_drug_vec
                    n_blood     = blood_data['n_cells']
                    vf_b        = blood_data['vol_fracs']
                    V_blood     = blood_data['V_blood']

                    Rtot_b      = np.maximum(y_cur[1: 1 + n_blood], 0.0)
                    Rtot_eff_sc = np.dot(Rtot_b, vf_b) * V_blood / cfg.V_CENTRAL

                    Ctot_pre    = max(y_cur[0], 0.0)
                    Cc_pre      = _recover_free_drug_vec(
                                      np.array([Ctot_pre]), cfg.KSS,
                                      np.array([Rtot_eff_sc]))[0]

                    Cc_post_free = Cc_pre + delta_Cc_raw
                    RC_post_sc   = Rtot_eff_sc * Cc_post_free / (cfg.KSS + Cc_post_free + 1e-15)
                    Ctot_post    = Cc_post_free + RC_post_sc

                    y_cur[0] = Ctot_post
                    print(f"    Day {bp}: +{delta_Cc_raw:.2f} nM free, "
                          f"Ctot {Ctot_pre:.2f}→{Ctot_post:.2f} nM")
                else:
                    y_cur[0] += delta_Cc_raw
                    print(f"    Day {bp}: Cc {seg.y[0,-1]:.2f}→{y_cur[0]:.2f} nM")

            t_cur = bp

        return np.concatenate(all_t), np.concatenate(all_y, axis=1)

    # ─────────────────────────────────────────────────────────────────────
    # Result unpacker
    # ─────────────────────────────────────────────────────────────────────

    def _unpack_results(self, t, y, system_data, blood_data, soluble_data,
                        label: str) -> pd.DataFrame:
        """Decompose ODE solution into per-cell-type time-series DataFrame."""
        cfg = self.config
        all_rows           = []
        Ctot_central_vals  = np.maximum(y[0], 0.0)
        offset             = 1

        # ── Blood membrane block (y[1:1+n_blood]) ────────────────────────
        # Must be read FIRST — matches _build_system and equations.py ordering.
        if blood_data is not None:
            n_blood = blood_data['n_cells']
            tdf_b   = blood_data['tdf']
            R0_b    = blood_data['R0']
            vf_b    = blood_data['vol_fracs']
            V_blood = blood_data['V_blood']

            Rtot_all_b      = np.maximum(y[offset: offset + n_blood], 0.0)
            Rtot_eff_b      = np.dot(vf_b, Rtot_all_b)
            Rtot_eff_scaled = Rtot_eff_b * V_blood / cfg.V_CENTRAL

            Cc_free_vals = _recover_free_drug_vec(
                Ctot_central_vals, cfg.KSS, Rtot_eff_scaled
            )
            denom_b = cfg.KSS + Cc_free_vals + 1e-15

            for i, row in enumerate(tdf_b.itertuples()):
                Rtot_vals = Rtot_all_b[i]
                R0_i      = R0_b[i]
                RC_vals   = Rtot_vals * Cc_free_vals / denom_b
                R_vals    = Rtot_vals * cfg.KSS      / denom_b
                occupancy = RC_vals / (Rtot_vals + 1e-12) * 100.0
                tmdd_idx  = R0_i / cfg.KSS if cfg.KSS > 0 else float('nan')

                all_rows.append(pd.DataFrame({
                    'time'              : t,
                    'mode'              : label,
                    'tissue'            : 'blood',
                    'cell_type'         : row.cell_type,
                    'plasma_conc_nM'    : Cc_free_vals,
                    'plasma_ctot_nM'    : Ctot_central_vals,
                    'local_v_conc_nM'   : Cc_free_vals,
                    'local_isf_conc_nM' : Cc_free_vals,
                    'local_isf_total_nM': Ctot_central_vals,
                    'free_R_nM'         : R_vals,
                    'bound_RC_nM'       : RC_vals,
                    'total_R_nM'        : Rtot_vals,
                    'occupancy_pct'     : occupancy,
                    'free_R_fraction'   : R_vals / (Rtot_vals + 1e-12),
                    'R0_nM'             : R0_i,
                    'KSS_nM'            : cfg.KSS,
                    'Kd_nM'             : cfg.Kd,
                    'TMDD_index'        : tmdd_idx,
                    'max_occupancy_pct' : occupancy.max(),
                }))
            offset += n_blood

        # ── Soluble plasma pool block (y[1+n_blood] or y[1] if no membrane) ──
        # Must be read SECOND — matches _build_system and equations.py ordering.
        # SC and Bulk output identical values here (R0 always bulk_nM).
        if soluble_data is not None:
            R_sol_vals  = np.maximum(y[offset], 0.0)
            # Recover Cc_free accounting for membrane blood RC (if any)
            if blood_data is not None:
                Cc_free_sol = Cc_free_vals   # already computed above
            else:
                Cc_free_sol = _recover_free_drug_vec(
                    Ctot_central_vals, cfg.KSS, np.zeros_like(Ctot_central_vals)
                )
            denom_sol   = cfg.KSS + Cc_free_sol + 1e-15
            RC_sol_vals = R_sol_vals * Cc_free_sol / denom_sol
            R_sol_free  = R_sol_vals * cfg.KSS    / denom_sol
            occ_sol     = RC_sol_vals / (R_sol_vals + 1e-12) * 100.0
            R0_sol      = soluble_data['R0']
            tmdd_sol    = R0_sol / cfg.KSS if cfg.KSS > 0 else float('nan')

            for _, srow in soluble_data['tdf'].iterrows():
                all_rows.append(pd.DataFrame({
                    'time'              : t,
                    'mode'              : label,
                    'tissue'            : 'blood',
                    'cell_type'         : srow['cell_type'],
                    'plasma_conc_nM'    : Cc_free_sol,
                    'plasma_ctot_nM'    : Ctot_central_vals,
                    'local_v_conc_nM'   : Cc_free_sol,
                    'local_isf_conc_nM' : Cc_free_sol,
                    'local_isf_total_nM': Ctot_central_vals,
                    'free_R_nM'         : R_sol_free,
                    'bound_RC_nM'       : RC_sol_vals,
                    'total_R_nM'        : R_sol_vals,
                    'occupancy_pct'     : occ_sol,
                    'free_R_fraction'   : R_sol_free / (R_sol_vals + 1e-12),
                    'R0_nM'             : R0_sol,
                    'KSS_nM'            : cfg.KSS,
                    'Kd_nM'             : cfg.Kd,
                    'TMDD_index'        : tmdd_sol,
                    'max_occupancy_pct' : occ_sol.max(),
                }))
            offset += 1

        # plasma_free_vals: used as Cc_free reference for solid organ output rows
        if blood_data is not None:
            plasma_free_vals = Cc_free_vals
        elif soluble_data is not None:
            plasma_free_vals = Cc_free_sol
        else:
            plasma_free_vals = Ctot_central_vals

        # ── Solid organ block ─────────────────────────────────────────────
        for tissue in system_data:
            n_cells       = tissue['n_cells']
            tdf           = tissue['tdf']
            R0_arr        = tissue['R0']
            vol_fracs     = tissue['vol_fracs']
            Cv_vals       = y[offset]
            Ctot_isf_vals = np.maximum(y[offset + 1], 0.0)

            Rtot_all = np.maximum(y[offset + 2: offset + 2 + n_cells], 0.0)
            Rtot_eff = np.dot(vol_fracs, Rtot_all)
            Cisf_vals = _recover_free_drug_vec(Ctot_isf_vals, cfg.KSS, Rtot_eff)

            for i, row in enumerate(tdf.itertuples()):
                Rtot_vals = Rtot_all[i]
                R0_i      = R0_arr[i]
                denom     = cfg.KSS + Cisf_vals + 1e-15
                RC_vals   = Rtot_vals * Cisf_vals / denom
                R_vals    = Rtot_vals * cfg.KSS   / denom
                occupancy = RC_vals / (Rtot_vals + 1e-12) * 100.0
                tmdd_idx  = R0_i / cfg.KSS if cfg.KSS > 0 else float('nan')

                all_rows.append(pd.DataFrame({
                    'time'              : t,
                    'mode'              : label,
                    'tissue'            : tissue['name'],
                    'cell_type'         : row.cell_type,
                    'plasma_conc_nM'    : plasma_free_vals,
                    'local_v_conc_nM'   : Cv_vals,
                    'local_isf_conc_nM' : Cisf_vals,
                    'local_isf_total_nM': Ctot_isf_vals,
                    'free_R_nM'         : R_vals,
                    'bound_RC_nM'       : RC_vals,
                    'total_R_nM'        : Rtot_vals,
                    'occupancy_pct'     : occupancy,
                    'free_R_fraction'   : R_vals / (Rtot_vals + 1e-12),
                    'R0_nM'             : R0_i,
                    'KSS_nM'            : cfg.KSS,
                    'Kd_nM'             : cfg.Kd,
                    'TMDD_index'        : tmdd_idx,
                    'max_occupancy_pct' : occupancy.max(),
                }))

            offset += (2 + n_cells)

        df_out = pd.concat(all_rows, ignore_index=True)

        # TMDD summary
        print(f"\n[*] TMDD Summary — {label}  (KSS={cfg.KSS:.4f} nM)")
        print(f"    {'Tissue':<12} {'Cell type':<40} {'R0':>8} {'R0/KSS':>8} {'MaxOcc%':>8}")
        print(f"    {'-'*12} {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
        for _, row in df_out.drop_duplicates(['tissue','cell_type']).iterrows():
            flag = " ← TMDD" if row['TMDD_index'] > 1.0 else ""
            iv   = " [iv]"   if str(row['tissue']).lower() == 'blood' else ""
            print(f"    {row['tissue']:<12} {row['cell_type']:<40} "
                  f"{row['R0_nM']:>8.2f} {row['TMDD_index']:>8.2f} "
                  f"{row['max_occupancy_pct']:>8.1f}{flag}{iv}")

        return df_out

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def run_single_cell(self, days: int = 120) -> pd.DataFrame:
        """PBPK with Bayesian single-cell nM concentrations (nM_local_ISF as R0)."""
        print(f"\n{'─'*60}")
        print(f"  SINGLE-CELL PBPK — QSS  ({days} days)")
        print(f"{'─'*60}")
        y0, system_data, blood_data, soluble_data, _ = self._build_system(use_bulk=False)
        t, y = self._solve_with_dosing(y0, system_data, blood_data, soluble_data, days)
        return self._unpack_results(t, y, system_data, blood_data, soluble_data, "single_cell")

    def run_bulk(self, days: int = 120) -> pd.DataFrame:
        """PBPK with bulk tissue-average nM concentrations (bulk_nM_membrane_ISF as R0)."""
        print(f"\n{'─'*60}")
        print(f"  BULK PBPK — QSS  ({days} days)")
        print(f"{'─'*60}")
        y0, system_data, blood_data, soluble_data, _ = self._build_system(use_bulk=True)
        t, y = self._solve_with_dosing(y0, system_data, blood_data, soluble_data, days)
        return self._unpack_results(t, y, system_data, blood_data, soluble_data, "bulk")

    def run_all_and_save(self, output_filename: str = "pbpk_results.csv",
                         days: int = 120):
        """Run single-cell + bulk, concatenate, save, and print comparison."""
        df_sc   = self.run_single_cell(days=days)
        df_bulk = self.run_bulk(days=days)
        df_all  = pd.concat([df_sc, df_bulk], ignore_index=True)

        os.makedirs("data", exist_ok=True)
        out_path = os.path.join("data", output_filename)
        df_all.to_csv(out_path, index=False)
        print(f"\n[✓] Saved → {out_path}  ({len(df_all):,} rows)")

        # SC vs Bulk TMDD comparison
        print(f"\n{'='*60}")
        print(f"  BULK vs SINGLE-CELL TMDD  (KSS={self.config.KSS:.4f} nM)")
        print(f"{'='*60}")
        print(f"  {'Tissue':<12} {'Cell type':<35} "
              f"{'Bulk R0/KSS':>12} {'SC R0/KSS':>10} {'Δ':>8}")
        print(f"  {'-'*12} {'-'*35} {'-'*12} {'-'*10} {'-'*8}")

        t0      = df_sc['time'].min()
        sc_t0   = df_sc[df_sc['time'] == t0]
        bulk_t0 = df_bulk[df_bulk['time'] == t0]

        for tissue in sorted(df_sc['tissue'].unique()):
            if str(tissue).lower() == 'plasma':
                continue
            iv = "  [iv]" if str(tissue).lower() == 'blood' else ""

            merged = (
                sc_t0[sc_t0['tissue'] == tissue][['cell_type', 'TMDD_index']]
                .merge(
                    bulk_t0[bulk_t0['tissue'] == tissue][['cell_type', 'TMDD_index']],
                    on='cell_type', suffixes=('_sc', '_bulk')
                )
            )
            for _, row in merged.iterrows():
                delta = row['TMDD_index_sc'] - row['TMDD_index_bulk']
                if   row['TMDD_index_sc'] > 1.0 and row['TMDD_index_bulk'] <= 1.0:
                    flag = "  ← SC DETECTS, BULK MISSES"
                elif row['TMDD_index_sc'] <= 1.0 and row['TMDD_index_bulk'] > 1.0:
                    flag = "  ← BULK OVER-PREDICTS"
                elif abs(delta) > 0.5:
                    flag = "  ← SIGNIFICANT DIFF"
                else:
                    flag = ""
                print(f"  {tissue:<12} {row['cell_type']:<35} "
                      f"{row['TMDD_index_bulk']:>12.2f} "
                      f"{row['TMDD_index_sc']:>10.2f} "
                      f"{delta:>+8.2f}{flag}{iv}")

        return df_all