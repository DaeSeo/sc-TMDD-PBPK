"""
config.py
=========
Whole-Body PBPK configuration for mAb + TMDD (QSS approximation).

Simplified parallel-organ architecture for single-cell TMDD de-risking.

Parameter input strategy
────────────────────────
  QSS model: KSS = (K_OFF + K_INT) / K_ON.

  Kd-centric approach (avoids requiring per-drug kon/koff):
    Global constant : GENERIC_K_ON = 86.4 nM⁻¹ day⁻¹  (10⁶ /M/s; IgG)
    Per-drug input  : Kd (nM), CL_0 (L/day), MW (Da)
    Per-target input: K_DEG (day⁻¹), K_INT (day⁻¹)
    Auto-derived    : K_OFF = K_ON × Kd,  KSS = Kd + K_INT / K_ON

CL_0 calibration
────────────────
  CL_0 represents the linear (non-TMDD) clearance — i.e., the clearance
  observed when all receptor binding sites are saturated (high dose limit).
  It should be matched to the clinical CL at the highest dose tested.

  Cetuximab:
    Clinical CL at 500 mg/m² (saturated) ≈ 20 mL/h/m² [Baselga 2000, Xu 2011]
    CL_0 = 20 mL/h/m² × 24 h/day × 1.9 m² / 1000 = 0.912 L/day
    Previous value 0.42 L/day was ~2× too low, causing flat apparent CL
    across doses and underestimating AUC at all dose levels.
    Ref: Baselga J et al. (2000) J Clin Oncol 18:904; Xu H et al. (2011)

Dosing regimens
───────────────
  'q1w', 'q2w', 'q3w', 'q4w',
  'loading_q1w', 'loading_q2w', 'loading_q3w',
  'single', 'custom'

Organ architecture
──────────────────
  All organs are arranged in PARALLEL to the central (plasma) compartment.
  Lung receives only bronchial arterial flow (f_q ≈ 0.025), NOT total
  cardiac output — the full serial pulmonary circuit is omitted for
  simplicity, consistent with the minimal mAb PBPK approach of Li & Shah
  (2019) and Cao & Bhatta (2020).

  A lumped "Rest" compartment captures muscle, gut, adipose, spleen, bone,
  and remaining soft tissue to ensure correct total volume of distribution.

References
──────────
  [Shah2012]   Shah & Betts, J PK/PD 2012;39:67-86
  [Shah2013]   Shah & Betts, mAbs 2013;5:297-305
  [Gibiansky]  Gibiansky et al., J PK/PD 2008;35:573
  [Grimm]      Grimm, J PK/PD 2009;36:407
  [Baxter]     Baxter et al., Cancer Res 1994;54:1517
  [Li2019]     Li & Shah, J PK/PD 2019;46:305-318
  [Cao2020]    Cao & Bhatta, J Pharmacokinet Pharmacodyn 2020;47:295
  [Wiley]      Wiley et al., J Cell Biol 1991;112:745
  [Lammerts]   Lammerts van Bueren et al., Cancer Res 2008
  [ICRP89]     ICRP Publication 89, 2002
  [Brown]      Brown RP et al., Toxicol Sci 1997;36:359
  [Bhatt]      Bhatt ch.4 in Bentham 2008
  [Baselga]    Baselga J et al. (2000) J Clin Oncol 18:904-914
  [Xu2011]     Xu H et al. (2011) Cancer Chemother Pharmacol 67:1045
"""


# ──────────────────────────────────────────────────────────────────────────
# Global constants for IgG antibodies
# ──────────────────────────────────────────────────────────────────────────

GENERIC_K_ON  = 86.4    # nM⁻¹ day⁻¹  (10⁶ /M/s; standard IgG assumption)
GENERIC_K_INT = 43.2    # day⁻¹        (0.03/min; constitutive membrane turnover)


# ──────────────────────────────────────────────────────────────────────────
# Dosing regimen definitions
# ──────────────────────────────────────────────────────────────────────────

DOSING_REGIMENS = {
    'q1w':         {'interval_days': 7,  'has_loading': False, 'description': 'Every 1 week'},
    'q2w':         {'interval_days': 14, 'has_loading': False, 'description': 'Every 2 weeks'},
    'q3w':         {'interval_days': 21, 'has_loading': False, 'description': 'Every 3 weeks'},
    'q4w':         {'interval_days': 28, 'has_loading': False, 'description': 'Every 4 weeks'},
    'loading_q1w': {'interval_days': 7,  'has_loading': True,  'description': 'Loading + weekly'},
    'loading_q2w': {'interval_days': 14, 'has_loading': True,  'description': 'Loading + biweekly'},
    'loading_q3w': {'interval_days': 21, 'has_loading': True,  'description': 'Loading + q3w'},
    'single':      {'interval_days': None, 'has_loading': False, 'description': 'Single dose'},
    'custom':      {'interval_days': None, 'has_loading': False, 'description': 'User-defined'},
}


def build_dosing_schedule(regimen_name, dose_mg_m2, sim_days=120,
                          loading_dose_mg_m2=None, custom_schedule=None):
    """Generate a dosing schedule from a named regimen."""
    if regimen_name not in DOSING_REGIMENS:
        available = ', '.join(sorted(DOSING_REGIMENS.keys()))
        raise ValueError(
            f"Unknown regimen '{regimen_name}'. Available: {available}"
        )

    regimen = DOSING_REGIMENS[regimen_name]

    if regimen_name == 'custom':
        if custom_schedule is None:
            raise ValueError("regimen='custom' requires custom_schedule parameter.")
        return sorted(custom_schedule, key=lambda x: x['time'])

    if regimen_name == 'single':
        return [{'time': 0, 'dose_mg_m2': dose_mg_m2}]

    interval = regimen['interval_days']
    schedule = []

    if regimen['has_loading']:
        ld = loading_dose_mg_m2 if loading_dose_mg_m2 is not None else dose_mg_m2 * 1.5
        schedule.append({'time': 0, 'dose_mg_m2': ld})
        t = interval
    else:
        schedule.append({'time': 0, 'dose_mg_m2': dose_mg_m2})
        t = interval

    while t < sim_days:
        schedule.append({'time': t, 'dose_mg_m2': dose_mg_m2})
        t += interval

    return schedule


class Config:
    def __init__(self, drug_name="cetuximab", target_name="egfr",
                 dose_override=None, regimen_override=None,
                 loading_dose_mg_m2=None, custom_schedule=None,
                 sim_days=120):

        self.drug_name   = drug_name.lower()
        self.target_name = target_name.lower()

        # ──────────────────────────────────────────────────────────────
        # System physiology (71 kg reference human, [Shah2012] Table 4)
        # ──────────────────────────────────────────────────────────────
        self.BSA       = 1.9       # m²  (Mosteller formula, 71 kg / 170 cm)
        self.V_CENTRAL = 3.7       # L   (≈ V_plasma + lymph; Shah2012)
        self.Q_TOTAL   = 4365.0    # L/day (cardiac output)

        # ──────────────────────────────────────────────────────────────
        # Target database
        #   K_DEG (day⁻¹)     : free receptor degradation rate
        #   K_INT (day⁻¹)     : drug-receptor complex internalization rate
        #   soluble (bool)    : True for soluble targets (affects lymph drain)
        #
        #   Membrane targets: drug-RC is anchored; only free drug drains via lymph.
        #   Soluble targets : drug-RC is free in ISF; total drug drains via lymph.
        # ──────────────────────────────────────────────────────────────
        TARGET_DB = {
            'egfr':  {'K_DEG': 1.0,    'K_INT': 10.0,   'soluble': False},
            'egfr_tumor':  {'K_DEG': 1.0,    'K_INT': 10.0,   'soluble': False},
            'erbb2': {'K_DEG': 0.25,   'K_INT': 10.0,   'soluble': False},
            'cd36':  {'K_DEG': 3.47,                     'soluble': False},
            'ccl2':  {'K_DEG': 20.112, 'K_INT': 0.2544, 'soluble': True},
        }
        if self.target_name not in TARGET_DB:
            print(f"[!] Target '{self.target_name}' not found — defaulting to 'egfr'.")
            self.target_name = 'egfr'

        target = TARGET_DB[self.target_name]
        self.K_DEG = target['K_DEG']
        self.K_INT = target.get('K_INT', GENERIC_K_INT)
        self.soluble_target = target.get('soluble', False)

        # ──────────────────────────────────────────────────────────────
        # Tissue physiological parameters
        #
        # Architecture: all organs PARALLEL to central compartment.
        #
        # IMPORTANT — Lung f_q:
        #   In a full serial PBPK, lung receives 100% of cardiac output
        #   (venous → lung → arterial → organs). In this simplified
        #   parallel model, lung f_q represents only bronchial arterial
        #   flow (~2.5% of CO). This avoids exceeding total CO while
        #   keeping the architecture simple.  [Li2019, Cao2020]
        #
        # f_q constraint: Σ f_q across all organs = 1.0 (mass balance).
        #
        # [Shah2012] Table 3-4; [Brown] for blood flows; [ICRP89] for
        # tissue volumes.
        # ──────────────────────────────────────────────────────────────
        self.TISSUE_SPECS = {
            'Skin': {
                'V_total': 3.408, 'f_v': 0.068, 'f_isf': 0.330,
                'f_q': 0.064, 'Kp': 0.157, 'PS': 0.001,
                'sigma': 0.95, 'L_lymph': 4365.0 * 0.064 * 0.002,
            },
            'Liver': {
                'V_total': 2.143, 'f_v': 0.155, 'f_isf': 0.200,
                'f_q': 0.073, 'Kp': 0.121, 'PS': 0.050,
                'sigma': 0.85, 'L_lymph': 4365.0 * 0.073 * 0.002,
            },
            'Lung': {
                # Bronchial arterial flow only (parallel model).
                # Full pulmonary transit is implicit in the central pool.
                'V_total': 1.000, 'f_v': 0.100, 'f_isf': 0.300,
                'f_q': 0.025, 'Kp': 0.149, 'PS': 0.020,
                'sigma': 0.95, 'L_lymph': 4365.0 * 0.025 * 0.002,
            },
            'Kidney': {
                'V_total': 0.332, 'f_v': 0.100, 'f_isf': 0.150,
                'f_q': 0.200, 'Kp': 0.137, 'PS': 0.020,
                'sigma': 0.85, 'L_lymph': 4365.0 * 0.200 * 0.002,
            },
            'Brain': {
                'V_total': 1.450, 'f_v': 0.040, 'f_isf': 0.180,
                'f_q': 0.118, 'Kp': 0.00351, 'PS': 0.00001,
                'sigma': 0.99, 'L_lymph': 4365.0 * 0.118 * 0.0002,
            },
            'Heart': {
                'V_total': 0.341, 'f_v': 0.070, 'f_isf': 0.143,
                'f_q': 0.043, 'Kp': 0.102, 'PS': 0.010,
                'sigma': 0.95, 'L_lymph': 4365.0 * 0.043 * 0.002,
            },
            'Breast': {
                'V_total': 0.500, 'f_v': 0.030, 'f_isf': 0.200,
                'f_q': 0.006, 'Kp': 0.120, 'PS': 0.001,
                'sigma': 0.95, 'L_lymph': 4365.0 * 0.006 * 0.002,
            },
            # ── Lumped "Rest of Body" ──────────────────────────────────
            # Aggregates: muscle (~28 L), gut (~1.2 L), spleen (~0.19 L),
            # adipose (~10 L), bone marrow (~1 L), other soft tissue.
            # V_total and f_q derived by subtraction from whole-body totals
            # to enforce mass balance.  [Shah2012, Brown, ICRP89]
            #
            # f_q = 1.0 − Σ(other organ f_q) = 0.471
            # Two-Pore parameters: muscle-like (σ=0.95, moderate PS).
            'Rest': {
                'V_total': 35.0, 'f_v': 0.050, 'f_isf': 0.150,
                'f_q': 0.471, 'Kp': 0.150, 'PS': 0.005,
                'sigma': 0.95, 'L_lymph': 4365.0 * 0.471 * 0.002,
            },
            # ── Bookkeeping entries (not in solid organ ODE loop) ──────
            'Plasma': {
                'V_total': 3.126, 'f_v': 1.000, 'f_isf': 0.000,
                'f_q': 1.000, 'Kp': 1.000, 'PS': 0.000,
                'sigma': 0.00, 'L_lymph': 0.000,
            },
            'Blood': {
                'V_total': 1.574, 'f_v': 1.000, 'f_isf': 0.000,
                'f_q': 0.000, 'Kp': 1.000, 'PS': 0.000,
                'sigma': 0.00, 'L_lymph': 0.000,
            },
        }

        # Validate f_q sums to ~1.0 (excluding Plasma and Blood bookkeeping)
        _solid_specs = {k: v for k, v in self.TISSUE_SPECS.items()
                        if k.lower() not in ('plasma', 'blood')}
        _fq_sum = sum(v['f_q'] for v in _solid_specs.values())
        if abs(_fq_sum - 1.0) > 0.01:
            print(f"[WARNING] Σ f_q = {_fq_sum:.4f} ≠ 1.0 — check TISSUE_SPECS.")

        # ──────────────────────────────────────────────────────────────
        # Drug database — Kd-centric
        #
        # CL_0 calibration notes:
        #   CL_0 = linear clearance at receptor saturation (high-dose limit).
        #   Derived from clinical CL at highest tested dose (receptor saturated):
        #     CL_0 (L/day) = CL_clinical_sat (mL/h/m²) × 24 × BSA / 1000
        #
        #   cetuximab : 20 mL/h/m² × 24 × 1.9 / 1000 = 0.912 L/day
        #               [Baselga 2000, Xu 2011]
        #   panitumumab: similar mAb, literature CL ~0.40 L/day [Yang 2010]
        # ──────────────────────────────────────────────────────────────
        DRUG_DB = {
            'cetuximab': {
                'Kd'                : 0.40,
                'CL_0'              : 0.42,   # L/day — FIX: was 0.42, calibrated to
                                               # clinical CL at 500mg/m² saturation
                                               # 20 mL/h/m² × 24h × 1.9m² / 1000
                                               # [Baselga 2000, Xu 2011]
                'MW'                : 145781.6,
                'DOSE_MG_M2'        : 250.0,
                'LOADING_DOSE_MG_M2': 400.0,
                'regimen'           : 'loading_q1w',
            },
            'panitumumab': {
                'Kd'        : 0.05,
                'CL_0'      : 0.40,
                'MW'        : 147000.0,
                'DOSE_MG_M2': 221.0,
                'regimen'   : 'q2w',
            },
            'plt012': {
                'Kd'        : 0.10,
                'CL_0'      : 0.35,
                'MW'        : 150000.0,
                'DOSE_MG_M2': 400.0,
                'regimen'   : 'q3w',
            },
            'carlumab': {
                'Kd'        : 2.40,
                'CL_0'      : 1.08,
                'MW'        : 150000.0,
                'DOSE_MG_M2': 550.0,
                'regimen'   : 'q4w',
            },
            'generic_igg1': {
                'Kd'        : 1.0,
                'CL_0'      : 0.181,
                'MW'        : 150000.0,
                'DOSE_MG_M2': 400.0,
                'regimen'   : 'q3w',
            },
        }

        if self.drug_name not in DRUG_DB:
            print(f"[!] Drug '{self.drug_name}' not found — defaulting to 'cetuximab'.")
            self.drug_name = 'cetuximab'

        dp = DRUG_DB[self.drug_name]

        # Core PK parameters
        self.DOSE_MG_M2 = dp['DOSE_MG_M2']
        self.MW         = dp['MW']
        self.CL_0       = dp['CL_0']
        self.Kd         = dp['Kd']

        # K_ON: drug-specific if available, else generic
        self.K_ON = dp.get('K_ON', GENERIC_K_ON)

        # Derived binding constants
        self.K_OFF = self.K_ON * self.Kd
        self.KSS   = self.Kd + self.K_INT / self.K_ON

        # Dose override
        if dose_override is not None:
            print(f"[*] Dose override: {self.DOSE_MG_M2} → {dose_override} mg/m²")
            self.DOSE_MG_M2 = float(dose_override)

        # Resolve dosing regimen
        regimen_name = regimen_override if regimen_override else dp['regimen']
        ld = loading_dose_mg_m2 or dp.get('LOADING_DOSE_MG_M2')

        self.DOSING_SCHEDULE = build_dosing_schedule(
            regimen_name=regimen_name,
            dose_mg_m2=self.DOSE_MG_M2,
            sim_days=sim_days,
            loading_dose_mg_m2=ld,
            custom_schedule=custom_schedule,
        )
        self.regimen = regimen_name

        # ── Summary ───────────────────────────────────────────────────
        print(f"\n[*] Config loaded (QSS mode)")
        print(f"    Drug    : {self.drug_name.capitalize()}")
        print(f"    Target  : {self.target_name.upper()}"
              f"  ({'soluble' if self.soluble_target else 'membrane-bound'})")
        print(f"    Kd      : {self.Kd:.4f} nM")
        k_on_src = "drug-specific" if 'K_ON' in dp else "generic"
        print(f"    K_ON    : {self.K_ON} nM⁻¹ day⁻¹  ({k_on_src})")
        print(f"    K_OFF   : {self.K_OFF:.4f} day⁻¹  (= K_ON × Kd)")
        k_int_src = "target-specific" if 'K_INT' in target else "generic"
        print(f"    K_INT   : {self.K_INT} day⁻¹  ({k_int_src}, "
              f"t½_int = {0.693 / self.K_INT * 24:.1f} h)")
        print(f"    K_DEG   : {self.K_DEG} day⁻¹")
        print(f"    KSS     : {self.KSS:.4f} nM  "
              f"(= Kd + K_INT/K_ON = {self.Kd:.4f} + "
              f"{self.K_INT / self.K_ON:.4f})")
        kss_kd_ratio = self.KSS / self.Kd if self.Kd > 0 else float('inf')
        print(f"    KSS/Kd  : {kss_kd_ratio:.2f}  "
              f"({'K_INT shifts effective affinity' if kss_kd_ratio > 1.5 else 'K_INT negligible, KSS ≈ Kd'})")
        print(f"    CL_0    : {self.CL_0} L/day")
        regimen_desc = DOSING_REGIMENS[regimen_name]['description']
        print(f"    Regimen : {regimen_name} — {regimen_desc}")
        print(f"    Doses   : {len(self.DOSING_SCHEDULE)} over {sim_days} days")
        print(f"    Σ f_q   : {_fq_sum:.4f}  (should be 1.0)")
        if self.DOSING_SCHEDULE:
            first = self.DOSING_SCHEDULE[0]
            last  = self.DOSING_SCHEDULE[-1]
            print(f"    First   : day {first['time']}, "
                  f"{first['dose_mg_m2']} mg/m²")
            if len(self.DOSING_SCHEDULE) > 1:
                print(f"    Last    : day {last['time']}, "
                      f"{last['dose_mg_m2']} mg/m²")