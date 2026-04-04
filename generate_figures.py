"""
visualisation.py
================
PBPK Result Visualiser — publication-quality figures.

Figure overview
───────────────
  Single-dose mode (default):
    Global (--Tissue all):
      Fig-1  *_fig1_pk.png           Systemic PK: plasma free drug SC vs Bulk
      Fig-2  *_fig2_heatmap.png      TMDD Index heatmap: tissue × cell type (SC)
      Fig-3  *_fig3_foldchange.png   SC / Bulk fold change  ← KEY FIGURE

    Per tissue:
      Fig-4  *_fig4_occ_rc.png       Receptor occupancy + Bound RC per cell type
      Fig-5  *_fig5_conc.png         Concentration gradient: Cc → Cv → Cisf

  Multi-dose mode (--multi):
    Per-dose subfolder: same Fig1–5 as above for each dose.

    Comparison subfolder (plots_results/multi_{tgt}_{drug}/comparison/):
      Fig-C1  comp_serum.png         Plasma Cc (SC) overlaid across all doses
      Fig-C2  comp_tissue_isf.png    Tissue ISF/plasma conc per tissue, all doses
      Fig-C3  comp_{tissue}_cells.png  Top-N cell bound RC per tissue, all doses

Usage — single dose
───────────────────
  python visualisation.py --Target EGFR --Drug Cetuximab --Dose 400 --Tissue all
  python visualisation.py --Target EGFR --Drug Cetuximab --Dose 400 --Tissue Liver

Usage — multi-dose
──────────────────
  python visualisation.py --Target EGFR --Drug Cetuximab \\
      --multi --doses 0.1,1,10,100 --Tissue all

Design rules
────────────
  · All y-axes are linear, starting at 0.
  · Actual concentration values shown (no log / scientific notation).
  · All cell types shown at equal line weight.
  · 40-color palette (purple ramp) for cell types.
  · Multi-dose comparison uses custom purple colormap (log-scaled by dose).
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.cm as _mcm
import seaborn as sns

warnings.filterwarnings("ignore")


def _get_cmap(name):
    """Matplotlib version-safe colormap loader."""
    try:
        import matplotlib
        if matplotlib.__version__ >= '3.7':
            import matplotlib.colormaps as _cms
            return _cms[name]
        elif matplotlib.__version__ >= '3.5':
            import matplotlib.colormaps as _cms
            return _cms.get_cmap(name)
        else:
            return _mcm.get_cmap(name)
    except Exception:
        return _mcm.get_cmap(name)


# ══════════════════════════════════════════════════════════════════════════════
# Global publication style
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.labelsize'    : 11,
    'axes.titlesize'    : 12,
    'axes.titleweight'  : 'bold',
    'axes.labelweight'  : 'bold',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.linewidth'    : 0.8,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'xtick.major.width' : 0.8,
    'ytick.major.width' : 0.8,
    'legend.fontsize'   : 8,
    'legend.framealpha' : 0.9,
    'legend.edgecolor'  : '0.75',
    'grid.linewidth'    : 0.4,
    'grid.alpha'        : 0.45,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'lines.linewidth'   : 1.4,
})

# ── Custom purple palette ─────────────────────────────────────────────────────
PAL_DARK    = '#6B3F69'
PAL_MID     = '#8D5F8C'
PAL_LIGHT   = '#A376A2'
PAL_PALE    = '#DDC3C3'

C_SC        = PAL_DARK
C_BULK      = PAL_PALE
C_CTOT      = PAL_MID
C_CV        = PAL_LIGHT
C_CISF      = PAL_DARK
C_REF       = '#9E8E8E'

TMDD_THRESH = 1.0
LW          = 1.4
LW_THIN     = 0.9

# ── Cell-type palette: 40 purple-toned perceptually distinct colors ───────────
_PURPLE_RAMP = [
    '#3D1F3C', '#4E2D4D', '#5C3859', '#6B3F69', '#7A4E78',
    '#8D5F8C', '#9B6F9A', '#A376A2', '#B08BAF', '#BD9CBC',
    '#C9AECA', '#D4BED5', '#DDC3C3', '#C4A4A4', '#AB8686',
    '#926868', '#7A4E4E', '#623737', '#8B6B6B', '#A38989',
    '#4A2849', '#5D3B5C', '#704E6F', '#836182', '#967495',
    '#A987A8', '#BC9ABB', '#CFADCE', '#E2C0E1', '#C0A0BF',
    '#9E809D', '#7C607B', '#6B5070', '#5A4065', '#49305A',
    '#6E4F6D', '#8E6F8D', '#AE8FAD', '#CEAFCD', '#DDC3DD',
]

_POOL    = _PURPLE_RAMP
_LSTYLES = ['-', '--', '-.', ':']


def _cell_palette(cell_types_sorted: list) -> dict:
    return {ct: _POOL[i % 40] for i, ct in enumerate(cell_types_sorted)}


def _cell_ls(rank: int) -> str:
    return _LSTYLES[(rank // 40) % 4]


# ══════════════════════════════════════════════════════════════════════════════
# Multi-dose color utilities
# ══════════════════════════════════════════════════════════════════════════════

def _dose_colors(doses: list) -> dict:
    _DOSE_RAMP = ['#DDC3C3', '#A376A2', '#8D5F8C', '#6B3F69',
                  '#4E2D4D', '#3D1F3C', '#2E1530']
    doses_sorted = sorted(doses)
    n = len(doses_sorted)
    if n == 1:
        return {doses_sorted[0]: '#6B3F69'}

    log_vals = np.log10([max(d, 1e-12) for d in doses_sorted])
    lo, hi   = log_vals.min(), log_vals.max()
    if hi == lo:
        norms = [0.5] * n
    else:
        norms = [(v - lo) / (hi - lo) for v in log_vals]

    ramp_len = len(_DOSE_RAMP) - 1
    result = {}
    for d, norm in zip(doses_sorted, norms):
        idx  = norm * ramp_len
        lo_i = int(idx)
        hi_i = min(lo_i + 1, ramp_len)
        frac = idx - lo_i
        c_lo = np.array(mcolors.to_rgb(_DOSE_RAMP[lo_i]))
        c_hi = np.array(mcolors.to_rgb(_DOSE_RAMP[hi_i]))
        c    = c_lo + frac * (c_hi - c_lo)
        result[d] = mcolors.to_hex(c)
    return result


def _dose_label(dose: float) -> str:
    if dose < 0.01:
        return f"{dose:.3g} mg/m²"
    if dose < 1.0:
        return f"{dose:.2g} mg/m²"
    if dose < 10.0:
        return f"{dose:.1f} mg/m²"
    return f"{dose:.0f} mg/m²"


# ══════════════════════════════════════════════════════════════════════════════
# IO helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {path}")


def _linear_yax(ax, data_vals=None) -> None:
    """
    Linear y-axis starting at 0.
    Actual concentration values shown — no log scale, no scientific notation.
    """
    ax.set_yscale('linear')
    if data_vals is not None and len(data_vals) > 0:
        finite_vals = [v for v in data_vals if np.isfinite(v) and v >= 0]
        top = max(finite_vals) * 1.10 if finite_vals else 1.0
    else:
        top = 1.0
    ax.set_ylim(0, top)
    # Show plain decimal numbers (e.g. 0.0001 instead of 1e-4 or 10^-4)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')


def _mode_split(df: pd.DataFrame):
    return df[df['mode'] == 'single_cell'], df[df['mode'] == 'bulk']


def _sort_by_peak_rc(sc_df: pd.DataFrame, cell_types: list) -> list:
    peak = sc_df.groupby('cell_type')['bound_RC_nM'].max()
    ordered = peak.reindex(cell_types).fillna(0).sort_values(ascending=False)
    return ordered.index.tolist()


def _try_paths(tgt: str, drug: str, dose: float) -> str | None:
    candidates = [
        f"data/pbpk_{tgt.lower()}_{drug.lower()}_{dose:.1f}mg.csv",
        f"data/pbpk_{tgt.lower()}_{drug.lower()}_{dose:.0f}mg.csv",
        f"data/pbpk_{tgt.lower()}_{drug.lower()}_{dose}mg.csv",
        f"data/pbpk_{tgt.lower()}_{drug.lower()}_{int(dose)}mg.csv"
        if dose == int(dose) else None,
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
# NCA PK parameter calculator
# ══════════════════════════════════════════════════════════════════════════════

def _compute_nca_params(time: np.ndarray, conc: np.ndarray,
                        dose_mg_m2: float, bsa: float = 1.9,
                        mw_da: float = 145_500.0) -> dict:
    mask = np.isfinite(conc) & (conc >= 0)
    t, c = time[mask], conc[mask]
    if len(t) < 3:
        return {}

    idx_max  = np.argmax(c)
    cmax_nM  = float(c[idx_max])
    tmax_d   = float(t[idx_max])
    cmax_ug  = cmax_nM * mw_da / 1e6

    auc_nM_day = float(np.trapz(c, t))
    auc_ug_h   = auc_nM_day * (mw_da / 1e6) * 24

    term_mask = (c > cmax_nM * 0.01) & (t > t[idx_max])
    t_term, c_term = t[term_mask], c[term_mask]
    t_half_d = np.nan
    if len(t_term) >= 3:
        n_term = max(3, len(t_term) // 3)
        t_fit  = t_term[-n_term:]
        c_fit  = c_term[-n_term:]
        pos    = c_fit > 0
        if pos.sum() >= 2:
            slope, _ = np.polyfit(t_fit[pos], np.log(c_fit[pos]), 1)
            if slope < 0:
                t_half_d = float(0.693 / abs(slope))

    dose_mg    = dose_mg_m2 * bsa
    cl_mL_h    = (dose_mg * 1e3) / auc_ug_h if auc_ug_h > 0 else np.nan
    cl_L_day   = cl_mL_h / 1000 * 24 if np.isfinite(cl_mL_h) else np.nan
    cl_mL_h_m2 = cl_mL_h / bsa if np.isfinite(cl_mL_h) else np.nan
    vss_L      = (cl_L_day * t_half_d / 0.693
                  if (np.isfinite(cl_L_day) and np.isfinite(t_half_d))
                  else np.nan)

    return {
        'Cmax_nM'          : cmax_nM,
        'Cmax_ug_mL'       : cmax_ug,
        'Tmax_days'        : tmax_d,
        'AUC_nM_day'       : auc_nM_day,
        'AUC_ug_mL_h'      : auc_ug_h,
        't_half_days'      : t_half_d,
        't_half_hours'     : t_half_d * 24 if np.isfinite(t_half_d) else np.nan,
        'CL_apparent_L_day': cl_L_day,
        'CL_mL_h_m2'       : cl_mL_h_m2,
        'Vss_L'            : vss_L,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Fig-1  Systemic PK
# ══════════════════════════════════════════════════════════════════════════════

def fig1_pk_profile(df: pd.DataFrame, target_name: str, drug_name: str,
                    save_path: str = None) -> None:
    """Plasma free drug concentration (nM) over time: SC vs Bulk. Linear y-axis."""
    sc, bulk = _mode_split(df)
    has_ctot = 'plasma_ctot_nM' in df.columns

    fig, ax = plt.subplots(figsize=(8, 5))

    def _pk_series(mode_df, col):
        if mode_df.empty or col not in mode_df.columns:
            return None
        return mode_df.groupby('time')[col].first().reset_index()

    pk_sc   = _pk_series(sc,   'plasma_conc_nM')
    pk_bulk = _pk_series(bulk, 'plasma_conc_nM')

    all_vals = []

    if pk_sc is not None:
        ax.plot(pk_sc['time'], pk_sc['plasma_conc_nM'],
                color=C_SC, lw=LW, label='Single-cell  $C_{c,\\ free}$', zorder=4)
        all_vals.extend(pk_sc['plasma_conc_nM'].dropna().tolist())
    if pk_bulk is not None:
        ax.plot(pk_bulk['time'], pk_bulk['plasma_conc_nM'],
                color=C_BULK, lw=LW, ls='--', label='Bulk  $C_{c,\\ free}$', zorder=4)
        all_vals.extend(pk_bulk['plasma_conc_nM'].dropna().tolist())

    if has_ctot:
        ctot_sc = _pk_series(sc, 'plasma_ctot_nM')
        if ctot_sc is not None and ctot_sc['plasma_ctot_nM'].max() > 0:
            ax.plot(ctot_sc['time'], ctot_sc['plasma_ctot_nM'],
                    color=C_CTOT, lw=LW_THIN, ls=':',
                    label='SC  $C_{tot,\\ central}$  (free + blood-bound)',
                    zorder=3, alpha=0.85)
            all_vals.extend(ctot_sc['plasma_ctot_nM'].dropna().tolist())

    _linear_yax(ax, all_vals)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Plasma concentration (nM)')
    ax.set_title(f'{target_name.upper()} / {drug_name.capitalize()}  —  Systemic PK')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-2  TMDD Index Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig2_tmdd_heatmap(df: pd.DataFrame, target_name: str,
                      save_path: str = None) -> None:
    """TMDD index (R₀/KSS) heatmap: rows = cell types, columns = tissues."""
    sc = df[df['mode'] == 'single_cell']
    t0 = sc['time'].min()
    sc_t0 = sc[sc['time'] == t0][['tissue', 'cell_type', 'TMDD_index']].copy()

    pivot = sc_t0.pivot_table(
        index='cell_type', columns='tissue',
        values='TMDD_index', aggfunc='first'
    ).fillna(0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    n_rows, n_cols = pivot.shape
    fig_h = max(6, n_rows * 0.27)
    fig_w = max(7, n_cols * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    pos_vals = pivot.values[pivot.values > 0]
    vmax = float(np.nanpercentile(pos_vals, 95)) if len(pos_vals) > 0 else 5.0
    vmax = max(vmax, 5.0)

    sns.heatmap(
        pivot, ax=ax,
        cmap='PuRd', vmin=0, vmax=vmax,
        linewidths=0.3, linecolor='white',
        annot=True, fmt='.0f', annot_kws={'size': 6.5},
        cbar_kws={'label': 'TMDD Index  (R₀ / KSS)', 'shrink': 0.55},
    )
    ax.set_xlabel('Tissue', labelpad=6)
    ax.set_ylabel('')
    ax.set_title(f'{target_name.upper()}  —  Single-Cell TMDD Index\n')
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=7.5)
    plt.tight_layout()
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-3  Fold Change
# ══════════════════════════════════════════════════════════════════════════════

def fig3_fold_change(df: pd.DataFrame, target_name: str,
                     save_path: str = None) -> None:
    """SC / Bulk TMDD index fold change (KEY figure)."""
    sc   = df[df['mode'] == 'single_cell']
    bulk = df[df['mode'] == 'bulk']
    t0   = df['time'].min()

    sc_t0   = (sc[sc['time'] == t0][['tissue', 'cell_type', 'TMDD_index']]
               .rename(columns={'TMDD_index': 'sc_idx'}))
    bulk_t0 = (bulk[bulk['time'] == t0][['tissue', 'cell_type', 'TMDD_index']]
               .rename(columns={'TMDD_index': 'bulk_idx'}))
    merged  = sc_t0.merge(bulk_t0, on=['tissue', 'cell_type'])

    merged = merged[
        (merged['sc_idx'] > TMDD_THRESH) | (merged['bulk_idx'] > TMDD_THRESH)
    ].copy()

    if merged.empty:
        print("   [WARNING] No TMDD-relevant cells found.")
        return

    merged['fold']  = (merged['sc_idx'] + 1e-9) / (merged['bulk_idx'] + 1e-9)
    merged['label'] = (merged['tissue'].str.capitalize() + '\n' + merged['cell_type'])
    merged = merged.sort_values('fold', ascending=False)

    if len(merged) > 60:
        plot_df = pd.concat([merged.head(30), merged.tail(30)]).drop_duplicates()
    else:
        plot_df = merged.copy()
    plot_df = plot_df.sort_values('fold', ascending=True)

    log2_fold  = np.log2(plot_df['fold'].clip(lower=1/64, upper=256))
    bar_colors = [C_SC if f >= 1.0 else C_BULK for f in plot_df['fold']]

    fig_h = max(8, len(plot_df) * 0.34)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    ax.barh(range(len(plot_df)), log2_fold,
            color=bar_colors, edgecolor='none', height=0.72)
    ax.axvline(0,           color='black', lw=1.2, ls='--', zorder=5, label='SC = Bulk')
    ax.axvline( np.log2(2), color=C_REF,   lw=0.8, ls=':',  zorder=4, alpha=0.7)
    ax.axvline(-np.log2(2), color=C_REF,   lw=0.8, ls=':',  zorder=4, alpha=0.7)

    tick_folds  = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    tick_log2   = [np.log2(v) for v in tick_folds]
    tick_labels = ['÷16','÷8','÷4','÷2','1','×2','×4','×8','×16','×32','×64','×128','×256']
    data_range  = (log2_fold.min() - 0.5, log2_fold.max() + 0.5)
    valid = [(v, l) for v, l in zip(tick_log2, tick_labels)
             if data_range[0] <= v <= data_range[1]]
    ax.set_xticks([v for v, _ in valid])
    ax.set_xticklabels([l for _, l in valid], fontsize=9)
    ax.set_xlim(data_range)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['label'], fontsize=7.5)

    for i, (fold, log2_f) in enumerate(zip(plot_df['fold'], log2_fold)):
        if fold >= 4.0:
            ax.text(log2_f + 0.08, i, f'×{fold:.0f}',
                    va='center', ha='left', fontsize=7, color='#6B3F69', fontweight='bold')
        elif fold <= 0.25:
            ax.text(log2_f - 0.08, i, f'÷{1/fold:.0f}',
                    va='center', ha='right', fontsize=7, color='#DDC3C3', fontweight='bold')

    ax.set_xlabel('Fold change  (SC TMDD index / Bulk TMDD index)  [log₂ scale]')
    ax.set_title(
        f'{target_name.upper()}  —  Single-Cell vs Bulk: TMDD Index\n'
        f'Blue (>1): Bulk under-predicts  ·  Red (<1): Bulk over-predicts'
    )
    n_under = (plot_df['fold'] > 2).sum()
    n_over  = (plot_df['fold'] < 0.5).sum()
    ax.text(0.99, 0.01,
            f'Bulk under-predicts (>×2): {n_under} cell types\n'
            f'Bulk over-predicts  (>÷2): {n_over} cell types',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.75', lw=0.8))

    patch_sc   = mpatches.Patch(color=C_SC,   label='SC > Bulk  (Bulk misses sink)')
    patch_bulk = mpatches.Patch(color=C_BULK, label='SC < Bulk  (Bulk false sink)')
    ax.legend(handles=[patch_sc, patch_bulk], loc='lower right', fontsize=9)
    ax.grid(True, axis='x', lw=0.4, alpha=0.5)
    plt.tight_layout()
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-4  Per-tissue: Receptor Occupancy + Bound RC
# ══════════════════════════════════════════════════════════════════════════════

def fig4_occupancy_bound(tissue_df: pd.DataFrame, tissue_name: str,
                         target_name: str, save_path: str = None) -> None:
    """Three-panel receptor dynamics figure per tissue. All y-axes linear."""
    sc, bulk   = _mode_split(tissue_df)
    is_blood   = tissue_name.lower() == 'blood'
    all_cts    = sorted(tissue_df['cell_type'].unique())
    sorted_cts = _sort_by_peak_rc(sc, all_cts)
    ct_color   = _cell_palette(sorted_cts)
    n_ct       = len(sorted_cts)

    fig, (ax_occ, ax_rc, ax_fr) = plt.subplots(
        1, 3, figsize=(24, 6), constrained_layout=True
    )
    fig.suptitle(
        f'{tissue_name.capitalize()}  ·  {target_name.upper()}'
        + ('  [intravascular]' if is_blood else ''),
        fontweight='bold'
    )

    # ── Panel 1: Receptor occupancy ───────────────────────────────────
    for rank, ct in enumerate(sorted_cts):
        c      = ct_color[ct]
        ls     = _cell_ls(rank)
        sc_ct  = sc[sc['cell_type'] == ct]
        bk_ct  = bulk[bulk['cell_type'] == ct]
        if not sc_ct.empty:
            ax_occ.plot(sc_ct['time'], sc_ct['occupancy_pct'],
                        color=c, lw=LW, ls=ls, label=ct, zorder=3)
        if not bk_ct.empty:
            ax_occ.plot(bk_ct['time'], bk_ct['occupancy_pct'],
                        color=c, lw=LW_THIN, ls='--', alpha=0.45, zorder=2)

    ax_occ.axhline(50, color=C_REF, lw=LW_THIN, ls=':',  alpha=0.8, label='50 %')
    ax_occ.axhline(90, color=C_REF, lw=LW_THIN, ls='-.', alpha=0.8, label='90 %')
    ax_occ.set_ylim(0, 105)
    ax_occ.set_xlabel('Time (days)')
    ax_occ.set_ylabel('Receptor occupancy (%)')
    ax_occ.set_title('Receptor occupancy per cell type\n'
                     '(solid = SC  |  dashed = Bulk)')
    ax_occ.grid(True, axis='y')
    ax_occ.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_occ.ticklabel_format(style='plain', axis='y')

    n_cols_leg = max(1, n_ct // 14 + 1)
    if n_ct > 8:
        ax_occ.legend(fontsize=6.5, ncol=n_cols_leg,
                      loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      handlelength=1.4, framealpha=0.92, borderaxespad=0,
                      title='Cell type  (solid=SC, dash=Bulk)',
                      title_fontsize=6)
    else:
        ax_occ.legend(fontsize=7, ncol=n_cols_leg, loc='lower right',
                      handlelength=1.5, framealpha=0.92,
                      title='solid=SC  |  dashed=Bulk', title_fontsize=6.5)

    # ── Panel 2: Bound RC ─────────────────────────────────────────────
    rc_vals = []
    for rank, ct in enumerate(sorted_cts):
        c       = ct_color[ct]
        ls      = _cell_ls(rank)
        sc_ct   = sc[sc['cell_type'] == ct]
        bulk_ct = bulk[bulk['cell_type'] == ct]
        if not sc_ct.empty:
            ax_rc.plot(sc_ct['time'], sc_ct['bound_RC_nM'],
                       color=c, lw=LW, ls=ls, label=ct, zorder=3)
            rc_vals.extend(sc_ct['bound_RC_nM'].dropna().tolist())
        if not bulk_ct.empty:
            ax_rc.plot(bulk_ct['time'], bulk_ct['bound_RC_nM'],
                       color=c, lw=LW_THIN, ls='--', alpha=0.45, zorder=2)

    _linear_yax(ax_rc, rc_vals)
    ax_rc.set_xlabel('Time (days)')
    ax_rc.set_ylabel('[Drug–Receptor]  RC (nM)')
    ax_rc.set_title('Drug–receptor complex [RC] per cell type\n'
                    '(solid = SC  |  dashed = Bulk)')
    ax_rc.grid(True, axis='y')
    ax_rc.legend(fontsize=7, ncol=n_cols_leg, loc='upper right',
                 handlelength=1.5, framealpha=0.9)

    # ── Panel 3: Free receptor R ──────────────────────────────────────
    has_free_r = 'free_R_nM'  in tissue_df.columns
    has_rtot   = 'total_R_nM' in tissue_df.columns

    if has_free_r:
        fr_vals = []
        for rank, ct in enumerate(sorted_cts):
            c       = ct_color[ct]
            ls      = _cell_ls(rank)
            sc_ct   = sc[sc['cell_type'] == ct]
            bulk_ct = bulk[bulk['cell_type'] == ct]
            if not sc_ct.empty:
                ax_fr.plot(sc_ct['time'], sc_ct['free_R_nM'],
                           color=c, lw=LW, ls=ls, label=ct, zorder=3)
                fr_vals.extend(sc_ct['free_R_nM'].dropna().tolist())
            if not bulk_ct.empty:
                ax_fr.plot(bulk_ct['time'], bulk_ct['free_R_nM'],
                           color=c, lw=LW_THIN, ls='--', alpha=0.45, zorder=2)

        _linear_yax(ax_fr, fr_vals)
        ax_fr.set_xlabel('Time (days)')
        ax_fr.set_ylabel('Free receptor  [R] (nM)')
        ax_fr.set_title('Unoccupied receptor [R] per cell type\n'
                        '(solid = SC  |  dashed = Bulk  |  R = Rtot − RC)')
        ax_fr.grid(True, axis='y')
        ax_fr.legend(fontsize=7, ncol=n_cols_leg, loc='upper right',
                     handlelength=1.5, framealpha=0.9)

        if has_rtot:
            inset = ax_fr.inset_axes([0.03, 0.55, 0.37, 0.37])
            rtot_vals = []
            for rank, ct in enumerate(sorted_cts[:6]):
                c  = ct_color[ct]
                ls = _cell_ls(rank)
                sc_ct = sc[sc['cell_type'] == ct]
                if not sc_ct.empty:
                    inset.plot(sc_ct['time'], sc_ct['total_R_nM'],
                               color=c, lw=0.9, ls=ls)
                    rtot_vals.extend(sc_ct['total_R_nM'].dropna().tolist())
            inset.set_ylim(bottom=0)
            if rtot_vals:
                inset.set_ylim(0, max(rtot_vals) * 1.1)
            inset.yaxis.set_major_formatter(mticker.ScalarFormatter())
            inset.ticklabel_format(style='plain', axis='y')
            inset.set_xlabel('days', fontsize=6)
            inset.set_ylabel('Rtot (nM)', fontsize=6)
            inset.set_title('Total receptor Rtot', fontsize=6.5)
            inset.tick_params(labelsize=5.5)
            inset.grid(True, axis='y', alpha=0.4)
    else:
        ax_fr.text(0.5, 0.5, 'free_R_nM not in data',
                   transform=ax_fr.transAxes, ha='center', va='center',
                   fontsize=10, color='#888888')
        ax_fr.set_title('Unoccupied receptor [R]\n(data unavailable)')

    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-5  Per-tissue: Concentration Gradient
# ══════════════════════════════════════════════════════════════════════════════

def fig5_conc_gradient(tissue_df: pd.DataFrame, tissue_name: str,
                       target_name: str, save_path: str = None) -> None:
    """Cc → Cv → Cisf concentration gradient. Linear y-axis with actual values."""
    sc, bulk  = _mode_split(tissue_df)
    is_blood  = tissue_name.lower() == 'blood'
    has_ctot  = 'plasma_ctot_nM' in tissue_df.columns

    fig, ax = plt.subplots(figsize=(8, 5))

    def _ts(mode_df, col):
        if col not in mode_df.columns or mode_df.empty:
            return None, None
        s = mode_df.groupby('time')[col].first().reset_index()
        return s['time'].values, s[col].values

    all_vals = []

    if is_blood:
        t, v = _ts(sc, 'plasma_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_SC, lw=LW, label='$C_{c,\\ free}$ SC')
            all_vals.extend(v[v > 0].tolist())
        if has_ctot:
            t, v = _ts(sc, 'plasma_ctot_nM')
            if t is not None and np.max(v) > 0:
                ax.plot(t, v, color=C_CTOT, lw=LW_THIN, ls=':',
                        label='$C_{tot,\\ central}$ SC', alpha=0.85)
                all_vals.extend(v[v > 0].tolist())
        t, v = _ts(bulk, 'plasma_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_BULK, lw=LW_THIN, ls='--',
                    label='$C_{c,\\ free}$ Bulk', alpha=0.7)
        ax.set_title(
            f'{tissue_name.capitalize()}  ·  {target_name.upper()}  —  '
            f'Intravascular concentration'
        )
    else:
        t, v = _ts(sc, 'plasma_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_SC, lw=LW, label='Plasma $C_c$ (SC)')
            all_vals.extend(v[v > 0].tolist())
        t, v = _ts(sc, 'local_v_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_CV, lw=LW, ls='--', label='Organ vascular $C_v$ (SC)')
            all_vals.extend(v[v > 0].tolist())
        t, v = _ts(sc, 'local_isf_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_CISF, lw=LW, ls='-.', label='ISF $C_{isf}$ (SC)')
            all_vals.extend(v[v > 0].tolist())
        t, v = _ts(bulk, 'plasma_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_SC, lw=LW_THIN, ls=':', alpha=0.55,
                    label='Plasma $C_c$ (Bulk)')
        t, v = _ts(bulk, 'local_isf_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_CISF, lw=LW_THIN, ls=':', alpha=0.55,
                    label='ISF $C_{isf}$ (Bulk)')
        ax.set_title(
            f'{tissue_name.capitalize()}  ·  {target_name.upper()}  —  '
            f'Concentration gradient\n$C_c$ → $C_v$ → $C_{{isf}}$'
        )

    _linear_yax(ax, all_vals)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Concentration (nM)')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y')
    plt.tight_layout()
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-C1  Multi-dose: Serum concentration comparison
# ══════════════════════════════════════════════════════════════════════════════

def figC1_multi_serum(dose_dfs: dict, target_name: str, drug_name: str,
                      save_path: str = None,
                      mw_da: float = 145_500.0, bsa: float = 1.9) -> None:
    """Plasma free drug (SC) overlaid across all doses. Linear y-axis."""
    doses  = sorted(dose_dfs.keys())
    colors = _dose_colors(doses)

    pk_params = {}
    for dose in doses:
        df = dose_dfs[dose]
        sc = df[df['mode'] == 'single_cell']
        if sc.empty:
            continue
        pk = sc.groupby('time')['plasma_conc_nM'].first().reset_index()
        if pk.empty:
            continue
        params = _compute_nca_params(
            pk['time'].values, pk['plasma_conc_nM'].values,
            dose_mg_m2=dose, bsa=bsa, mw_da=mw_da
        )
        if params:
            pk_params[dose] = params

    has_ctot = any('plasma_ctot_nM' in df.columns for df in dose_dfs.values())

    if pk_params:
        fig = plt.figure(figsize=(18, 7))
        gs  = fig.add_gridspec(1, 3, width_ratios=[3, 2, 3] if has_ctot else [3, 3, 0.01])
        ax_free = fig.add_subplot(gs[0, 0])
        ax_tab  = fig.add_subplot(gs[0, 1])
        ax_tot  = fig.add_subplot(gs[0, 2]) if has_ctot else None
    else:
        n_panels = 2 if has_ctot else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5), squeeze=False)
        ax_free = axes[0, 0]
        ax_tab  = None
        ax_tot  = axes[0, 1] if has_ctot else None

    all_free, all_tot = [], []

    for dose in doses:
        df = dose_dfs[dose]
        sc = df[df['mode'] == 'single_cell']
        if sc.empty:
            continue
        c     = colors[dose]
        label = _dose_label(dose)

        pk = sc.groupby('time')['plasma_conc_nM'].first().reset_index()
        if not pk.empty:
            ax_free.plot(pk['time'], pk['plasma_conc_nM'],
                         color=c, lw=LW, label=label)
            all_free.extend(pk['plasma_conc_nM'].dropna().tolist())

        if has_ctot and ax_tot is not None and 'plasma_ctot_nM' in sc.columns:
            ct = sc.groupby('time')['plasma_ctot_nM'].first().reset_index()
            if not ct.empty and ct['plasma_ctot_nM'].max() > 0:
                ax_tot.plot(ct['time'], ct['plasma_ctot_nM'],
                            color=c, lw=LW, label=label)
                all_tot.extend(ct['plasma_ctot_nM'].dropna().tolist())

    _linear_yax(ax_free, all_free)
    ax_free.set_xlabel('Time (days)')
    ax_free.set_ylabel('Plasma free drug  $C_{c,\\ free}$ (nM)')
    ax_free.set_title('Systemic PK — Plasma Free Drug (SC)\nDose comparison')
    ax_free.legend(title='Single dose (SC)', fontsize=8, ncol=1)
    ax_free.grid(True, axis='y')

    if ax_tot is not None:
        _linear_yax(ax_tot, all_tot)
        ax_tot.set_xlabel('Time (days)')
        ax_tot.set_ylabel('$C_{tot,\\ central}$ (nM)')
        ax_tot.set_title('Central Total Drug  $C_{tot,\\ central}$ (SC)\n(free + blood-bound)')
        ax_tot.legend(title='Single dose (SC)', fontsize=8, ncol=1)
        ax_tot.grid(True, axis='y')

    # ── PK Parameter Table ────────────────────────────────────────────
    if ax_tab is not None and pk_params:
        ax_tab.axis('off')
        ax_tab.set_title('NCA PK Parameters (SC mode)', fontsize=11,
                         fontweight='bold', pad=12)

        col_labels = ['Dose\n(mg/m²)', 'Cmax\n(nM)', 'Cmax\n(µg/mL)',
                      't½\n(h)', 'AUC₀₋∞\n(µg·h/mL)', 'CL\n(mL/h/m²)',
                      'Vss\n(L)']
        table_data = []
        cell_colors_list = []

        for dose in sorted(pk_params.keys()):
            p = pk_params[dose]
            row = [
                f"{dose:.4g}",
                f"{p['Cmax_nM']:.1f}",
                f"{p['Cmax_ug_mL']:.1f}",
                f"{p['t_half_hours']:.1f}" if np.isfinite(p['t_half_hours']) else '—',
                f"{p['AUC_ug_mL_h']:.0f}" if np.isfinite(p['AUC_ug_mL_h']) else '—',
                f"{p['CL_mL_h_m2']:.1f}"  if np.isfinite(p['CL_mL_h_m2']) else '—',
                f"{p['Vss_L']:.2f}"        if np.isfinite(p['Vss_L']) else '—',
            ]
            table_data.append(row)
            c_dose = colors.get(dose, '#FFFFFF')
            rgba   = mcolors.to_rgba(c_dose, alpha=0.12)
            cell_colors_list.append([rgba] * len(col_labels))

        table = ax_tab.table(
            cellText=table_data,
            colLabels=col_labels,
            cellColours=cell_colors_list,
            colColours=[('#DDC3C3')] * len(col_labels),
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.6)

        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor('#6B3F69')
            cell.set_text_props(color='white', fontweight='bold', fontsize=7.5)

        print(f"\n  ── NCA PK Parameters (SC mode, {target_name}/{drug_name}) ──")
        print(f"  {'Dose':>10} {'Cmax(nM)':>10} {'Cmax(µg/mL)':>12} "
              f"{'t½(h)':>8} {'AUC(µg·h/mL)':>14} {'CL(mL/h/m²)':>12} {'Vss(L)':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*8} {'-'*14} {'-'*12} {'-'*8}")
        for dose in sorted(pk_params.keys()):
            p = pk_params[dose]
            t_h = f"{p['t_half_hours']:.1f}" if np.isfinite(p['t_half_hours']) else '—'
            auc = f"{p['AUC_ug_mL_h']:.0f}" if np.isfinite(p['AUC_ug_mL_h']) else '—'
            cl  = f"{p['CL_mL_h_m2']:.1f}"  if np.isfinite(p['CL_mL_h_m2']) else '—'
            vss = f"{p['Vss_L']:.2f}"        if np.isfinite(p['Vss_L']) else '—'
            print(f"  {dose:>10.4g} {p['Cmax_nM']:>10.1f} {p['Cmax_ug_mL']:>12.1f} "
                  f"{t_h:>8} {auc:>14} {cl:>12} {vss:>8}")

    fig.suptitle(
        f'{target_name.upper()} / {drug_name.capitalize()}  —  '
        f'Serum Concentration + PK Parameters: {len(doses)} doses',
        fontweight='bold', y=1.02
    )
    plt.tight_layout()
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-C3  Multi-dose: Cell-type bound RC comparison (per tissue)
# ══════════════════════════════════════════════════════════════════════════════

def figC3_multi_cell_bound_rc(dose_dfs: dict, tissue_name: str,
                               target_name: str, drug_name: str,
                               save_path: str = None,
                               top_n: int = 9) -> None:
    """Per-tissue bound RC per cell type across doses. Linear y-axis, shared."""
    doses  = sorted(dose_dfs.keys())
    colors = _dose_colors(doses)

    max_dose = max(doses)
    max_df   = dose_dfs[max_dose]
    sc_max   = max_df[
        (max_df['mode'] == 'single_cell') & (max_df['tissue'] == tissue_name)
    ]
    if sc_max.empty:
        print(f"   [SKIP] figC3: tissue '{tissue_name}' not in highest dose data.")
        return

    peak_rc = sc_max.groupby('cell_type')['bound_RC_nM'].max()
    top_cts = peak_rc.sort_values(ascending=False).head(top_n).index.tolist()
    n_ct    = len(top_cts)
    if n_ct == 0:
        return

    y_top = float(peak_rc[top_cts[0]]) * 1.10
    y_top = max(y_top, 1e-6)

    n_cols = min(3, n_ct)
    n_rows = (n_ct + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, ct in enumerate(top_cts):
        ax = axes_flat[idx]

        for dose in doses:
            df = dose_dfs[dose]
            sc = df[
                (df['mode'] == 'single_cell') &
                (df['tissue'] == tissue_name) &
                (df['cell_type'] == ct)
            ]
            if sc.empty:
                continue
            ax.plot(sc['time'], sc['bound_RC_nM'],
                    color=colors[dose], lw=LW, label=_dose_label(dose))

        ax.set_ylim(0, y_top)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        ax.set_title(ct, fontsize=8.5, fontweight='bold')
        ax.set_xlabel('Time (days)', fontsize=8)
        ax.set_ylabel('[Drug–Receptor] (nM)', fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, axis='y', alpha=0.45)

    for idx in range(n_ct, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = [
        Line2D([0], [0], color=colors[d], lw=LW, label=_dose_label(d))
        for d in doses
    ]
    fig.legend(handles=handles, title='Single dose (SC)',
               fontsize=9, loc='lower center',
               ncol=min(len(doses), 6),
               bbox_to_anchor=(0.5, -0.01),
               framealpha=0.95)

    fig.suptitle(
        f'{tissue_name.capitalize()}  ·  {target_name.upper()}  —  '
        f'Drug–Receptor Complex per Cell Type\n'
        f'(SC mode  ·  top {n_ct} cell types  ·  shared y-axis: '
        f'max = {y_top:.4g} nM)',
        fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-C3c  Multi-dose: Free receptor [R] per cell type (per tissue)
# ══════════════════════════════════════════════════════════════════════════════

def figC3c_multi_cell_free_r(dose_dfs: dict, tissue_name: str,
                              target_name: str, drug_name: str,
                              save_path: str = None,
                              top_n: int = 9) -> None:
    """Per-tissue free receptor [R] per cell type across doses. Linear y-axis."""
    doses  = sorted(dose_dfs.keys())
    colors = _dose_colors(doses)

    max_dose = max(doses)
    max_df   = dose_dfs[max_dose]
    sc_max   = max_df[
        (max_df['mode'] == 'single_cell') & (max_df['tissue'] == tissue_name)
    ]
    if sc_max.empty or 'free_R_nM' not in max_df.columns:
        print(f"   [SKIP] figC3c: tissue '{tissue_name}' missing or no free_R_nM.")
        return

    peak_rc = sc_max.groupby('cell_type')['bound_RC_nM'].max()
    top_cts = peak_rc.sort_values(ascending=False).head(top_n).index.tolist()
    n_ct    = len(top_cts)
    if n_ct == 0:
        return

    t0_rows = sc_max[sc_max['cell_type'] == top_cts[0]]
    r0_top  = float(t0_rows['R0_nM'].iloc[0]) if not t0_rows.empty else 1.0
    y_top   = r0_top * 1.10

    n_cols = min(3, n_ct)
    n_rows = (n_ct + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, ct in enumerate(top_cts):
        ax = axes_flat[idx]

        ct_row = sc_max[sc_max['cell_type'] == ct]
        r0_ct  = float(ct_row['R0_nM'].iloc[0]) if not ct_row.empty else 0.0

        ax.axhline(r0_ct, color=C_REF, lw=0.9, ls=':', alpha=0.7,
                   label=f'R₀ = {r0_ct:.3g} nM')

        for dose in doses:
            df  = dose_dfs[dose]
            sc  = df[
                (df['mode'] == 'single_cell') &
                (df['tissue'] == tissue_name) &
                (df['cell_type'] == ct)
            ]
            if sc.empty or 'free_R_nM' not in sc.columns:
                continue
            ax.plot(sc['time'], sc['free_R_nM'],
                    color=colors[dose], lw=LW, label=_dose_label(dose))

        ax.set_ylim(0, y_top)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        ax.set_title(ct, fontsize=8.5, fontweight='bold')
        ax.set_xlabel('Time (days)', fontsize=8)
        ax.set_ylabel('Free receptor  [R] (nM)', fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, axis='y', alpha=0.45)
        ax.legend(fontsize=6.5, loc='lower right', framealpha=0.85,
                  handlelength=1.2)

    for idx in range(n_ct, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = [
        Line2D([0], [0], color=colors[d], lw=LW, label=_dose_label(d))
        for d in doses
    ]
    handles += [Line2D([0], [0], color=C_REF, lw=0.9, ls=':', label='R₀ baseline')]
    fig.legend(handles=handles, title='Single dose (SC)',
               fontsize=9, loc='lower center',
               ncol=min(len(doses) + 1, 7),
               bbox_to_anchor=(0.5, -0.01),
               framealpha=0.95)

    fig.suptitle(
        f'{tissue_name.capitalize()}  ·  {target_name.upper()}  —  '
        f'Free (Unoccupied) Receptor [R] per Cell Type\n'
        f'(SC mode  ·  top {n_ct} cell types  ·  shared y-axis anchored to R₀)',
        fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-C5  Multi-dose: Full receptor dynamics per tissue (Rtot / RC / R)
# ══════════════════════════════════════════════════════════════════════════════

def figC5_receptor_dynamics(dose_dfs: dict, tissue_name: str,
                             target_name: str, drug_name: str,
                             save_path: str = None,
                             top_n: int = 6) -> None:
    """
    Per-tissue: full receptor state across doses.
    All panels use linear y-axes with actual concentration values.

    Grid: rows = 3 receptor species, columns = top-N cell types.
      Row 1  Rtot (nM)  — total receptor pool
      Row 2  [RC] (nM)  — drug-bound receptor
      Row 3  [R]  (nM)  — free/unoccupied receptor
    """
    doses  = sorted(dose_dfs.keys())
    colors = _dose_colors(doses)

    max_dose = max(doses)
    max_df   = dose_dfs[max_dose]
    sc_max   = max_df[
        (max_df['mode'] == 'single_cell') & (max_df['tissue'] == tissue_name)
    ]
    if sc_max.empty:
        print(f"   [SKIP] figC5: tissue '{tissue_name}' not in highest dose data.")
        return

    has_free_r = 'free_R_nM'   in max_df.columns
    has_rtot   = 'total_R_nM'  in max_df.columns
    has_rc     = 'bound_RC_nM' in max_df.columns
    missing    = [c for c, ok in [('free_R_nM', has_free_r),
                                   ('total_R_nM', has_rtot),
                                   ('bound_RC_nM', has_rc)] if not ok]
    if missing:
        print(f"   [SKIP] figC5: missing columns {missing}.")
        return

    peak_rc = sc_max.groupby('cell_type')['bound_RC_nM'].max()
    top_cts = peak_rc.sort_values(ascending=False).head(top_n).index.tolist()
    n_ct    = len(top_cts)
    if n_ct == 0:
        return

    # Species: (column, row-label, y-label, row-color-accent)
    SPECIES = [
        ('total_R_nM',  'Rtot — Total receptor',  'Rtot (nM)',   '#2E4057'),
        ('bound_RC_nM', '[RC] — Drug–receptor',    '[RC] (nM)',   PAL_MID),
        ('free_R_nM',   '[R] — Free receptor',     '[R] (nM)',    PAL_DARK),
    ]
    n_rows = len(SPECIES)

    fig, axes = plt.subplots(
        n_rows, n_ct,
        figsize=(4.5 * n_ct, 3.8 * n_rows),
        squeeze=False
    )

    # Pre-compute shared y-limits per row (linear: 0 to max*1.1)
    row_vals = {r: [] for r in range(n_rows)}
    for ci, ct in enumerate(top_cts):
        for dose in doses:
            df = dose_dfs[dose]
            sc = df[
                (df['mode'] == 'single_cell') &
                (df['tissue'] == tissue_name) &
                (df['cell_type'] == ct)
            ]
            if sc.empty:
                continue
            for ri, (col, _, _, _) in enumerate(SPECIES):
                if col in sc.columns:
                    vals = sc[col].dropna().tolist()
                    row_vals[ri].extend(vals)

    row_ylims = {}
    for ri in range(n_rows):
        vals = [v for v in row_vals[ri] if np.isfinite(v) and v >= 0]
        if not vals:
            row_ylims[ri] = (0, 1.0)
        else:
            row_ylims[ri] = (0, max(vals) * 1.10)

    # ── Plot ──────────────────────────────────────────────────────────────
    for ri, (col, row_label, ylabel, accent) in enumerate(SPECIES):
        for ci, ct in enumerate(top_cts):
            ax = axes[ri][ci]

            for dose in doses:
                df = dose_dfs[dose]
                sc = df[
                    (df['mode'] == 'single_cell') &
                    (df['tissue'] == tissue_name) &
                    (df['cell_type'] == ct)
                ]
                if sc.empty or col not in sc.columns:
                    continue
                ts = sc.groupby('time')[col].first()
                ax.plot(ts.index, ts.values,
                        color=colors[dose], lw=LW, alpha=0.92)

            # R₀ baseline for [R] row
            if col == 'free_R_nM':
                ct_row = sc_max[sc_max['cell_type'] == ct]
                if not ct_row.empty and 'R0_nM' in ct_row.columns:
                    r0 = float(ct_row['R0_nM'].iloc[0])
                    ax.axhline(r0, color=C_REF, lw=0.9, ls=':',
                               alpha=0.8, label=f'R₀={r0:.3g}')

            # Linear y-axis with actual values
            ax.set_yscale('linear')
            ax.set_ylim(*row_ylims[ri])
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='y')

            ax.set_xlabel('Time (days)', fontsize=8)
            ax.tick_params(labelsize=7.5)
            ax.grid(True, axis='y', alpha=0.35)

            if ri == 0:
                ax.set_title(ct, fontsize=8.5, fontweight='bold')

            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=8.5, color=accent,
                              fontweight='bold')
            else:
                ax.set_ylabel('')

            ax.spines['left'].set_color(accent)
            ax.spines['left'].set_linewidth(2.0)

    # ── Shared legend ─────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], color=colors[d], lw=LW, label=_dose_label(d))
        for d in doses
    ]
    handles += [Line2D([0], [0], color=C_REF, lw=0.9, ls=':',
                       label='R₀ baseline ([R] row)')]
    fig.legend(handles=handles, fontsize=9,
               loc='lower center', ncol=len(doses) + 1,
               bbox_to_anchor=(0.5, -0.02),
               framealpha=0.95, title='Single dose (SC)')

    fig.suptitle(
        f'{tissue_name.capitalize()}  ·  {target_name.upper()}  —  '
        f'Receptor State Dynamics\n'
        f'Rows: Rtot (total) · [RC] (drug-bound) · [R] (free)    '
        f'Columns: top {n_ct} cell types by peak RC',
        fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-C2 (figC3b)  Multi-dose: Tissue ISF concentration summary
# ══════════════════════════════════════════════════════════════════════════════

def figC3b_multi_tissue_isf_summary(dose_dfs: dict, target_name: str,
                                     drug_name: str,
                                     save_path: str = None,
                                     max_tissues: int = 9) -> None:
    """
    Drug concentration at tissue-exposure site across all tissues and doses.
    Linear y-axis — actual concentration values.
    """
    doses  = sorted(dose_dfs.keys())
    colors = _dose_colors(doses)

    all_tissues = sorted(set(
        t for df in dose_dfs.values() for t in df['tissue'].unique()
    ))
    if len(all_tissues) > max_tissues:
        all_tissues = all_tissues[:max_tissues]

    n_t    = len(all_tissues)
    n_cols = min(3, n_t)
    n_rows = (n_t + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, tissue in enumerate(all_tissues):
        ax       = axes_flat[idx]
        is_blood = tissue.lower() == 'blood'
        conc_col = 'plasma_conc_nM' if is_blood else 'local_isf_conc_nM'
        all_vals = []

        for dose in doses:
            df = dose_dfs[dose]
            sc = df[(df['mode'] == 'single_cell') & (df['tissue'] == tissue)]
            if sc.empty or conc_col not in sc.columns:
                continue
            ts = sc.groupby('time')[conc_col].first().reset_index()
            if not ts.empty:
                ax.plot(ts['time'], ts[conc_col],
                        color=colors[dose], lw=LW, label=_dose_label(dose))
                all_vals.extend(ts[conc_col].dropna().tolist())

        _linear_yax(ax, all_vals)

        if is_blood:
            ylabel = '$C_{p,free}$ (nM)'
            ax.set_title(f'{tissue.capitalize()}  [iv]',
                         fontsize=10, fontweight='bold')
        else:
            ylabel = '$C_{ISF,free}$ (nM)'
            ax.set_title(tissue.capitalize(), fontsize=10, fontweight='bold')

        ax.set_xlabel('Time (days)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis='y', alpha=0.4)

    for idx in range(n_t, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = [
        Line2D([0], [0], color=colors[d], lw=LW, label=_dose_label(d))
        for d in doses
    ]
    fig.legend(handles=handles, title='Single dose (SC)',
               fontsize=9, loc='lower center',
               ncol=min(len(doses), 6),
               bbox_to_anchor=(0.5, -0.02),
               framealpha=0.95)

    fig.suptitle(
        f'{target_name.upper()} / {drug_name.capitalize()}  —  '
        f'Tissue Drug Exposure across Doses\n'
        f'Solid organs: $C_{{ISF,free}}$  ·  Blood: $C_{{p,free}}$  '
        f'(SC mode, single dose)',
        fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-C4  Multi-dose: Bulk TMDD Index comparison
# ══════════════════════════════════════════════════════════════════════════════

def figC4_bulk_tmdd_index(dose_dfs: dict, target_name: str, drug_name: str,
                           save_path: str = None) -> None:
    """Bulk-mode TMDD index (R₀/KSS) across tissues. Bar chart + heatmap."""
    ref_dose = max(dose_dfs.keys())
    df_ref   = dose_dfs[ref_dose]
    t0       = df_ref['time'].min()
    df_t0    = df_ref[df_ref['time'] == t0]

    bulk_t0 = df_t0[df_t0['mode'] == 'bulk']
    sc_t0   = df_t0[df_t0['mode'] == 'single_cell']

    if bulk_t0.empty and sc_t0.empty:
        print("   [SKIP] figC4: no data at t=0.")
        return

    tissues_all = sorted(df_t0['tissue'].unique())
    records = []
    for tissue in tissues_all:
        b_tdf = bulk_t0[bulk_t0['tissue'] == tissue]
        s_tdf = sc_t0[sc_t0['tissue'] == tissue]

        bulk_val    = float(b_tdf['TMDD_index'].iloc[0]) if not b_tdf.empty else 0.0
        sc_max_val  = float(s_tdf['TMDD_index'].max())   if not s_tdf.empty else 0.0
        sc_mean_val = float(s_tdf['TMDD_index'].mean())  if not s_tdf.empty else 0.0

        records.append({
            'tissue'  : tissue,
            'Bulk'    : bulk_val,
            'SC_max'  : sc_max_val,
            'SC_mean' : sc_mean_val,
        })

    rec_df = pd.DataFrame(records).set_index('tissue')
    rec_df = rec_df.loc[rec_df['SC_max'].sort_values(ascending=False).index]
    tissues_sorted = rec_df.index.tolist()
    n_tissues      = len(tissues_sorted)

    fig_w = max(11, n_tissues * 1.4 + 4)
    fig_h = max(5,  n_tissues * 0.55 + 2)
    fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    COL_BULK    = PAL_PALE
    COL_SC_MAX  = PAL_DARK
    COL_SC_MEAN = PAL_MID

    x     = np.arange(n_tissues)
    width = 0.25

    ax_bar.bar(x - width, rec_df['Bulk'],    width=width * 0.90,
               color=COL_BULK,    label='Bulk',     edgecolor='white', lw=0.4)
    ax_bar.bar(x,          rec_df['SC_max'],  width=width * 0.90,
               color=COL_SC_MAX,  label='SC  max',  edgecolor='white', lw=0.4)
    ax_bar.bar(x + width,  rec_df['SC_mean'], width=width * 0.90,
               color=COL_SC_MEAN, label='SC  mean', edgecolor='white', lw=0.4)

    ax_bar.axhline(1.0, color=PAL_DARK, lw=1.4, ls='--',
                   label='TMDD threshold  (R₀/KSS = 1)', zorder=5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [t.capitalize() for t in tissues_sorted],
        rotation=30, ha='right', fontsize=9
    )
    ax_bar.set_ylim(bottom=0)
    ax_bar.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_bar.ticklabel_format(style='plain', axis='y')
    ax_bar.set_xlabel('Tissue', labelpad=5)
    ax_bar.set_ylabel('TMDD Index  (R₀ / KSS)')
    ax_bar.set_title(
        'TMDD Index: Bulk vs Single-Cell\n'
        '(R₀/KSS > 1 = target-mediated clearance risk)'
    )
    ax_bar.legend(fontsize=8, loc='upper right', framealpha=0.95)
    ax_bar.grid(True, axis='y', lw=0.4, alpha=0.5)

    heat_df = rec_df[['Bulk', 'SC_max', 'SC_mean']].copy()
    heat_df.columns = ['Bulk', 'SC  max', 'SC  mean']
    heat_df.index   = [t.capitalize() for t in heat_df.index]

    pos_vals = heat_df.values[heat_df.values > 0]
    vmax_h   = float(np.nanpercentile(pos_vals, 95)) if len(pos_vals) else 5.0
    vmax_h   = max(vmax_h, 2.0)

    sns.heatmap(
        heat_df, ax=ax_heat,
        cmap='PuRd', vmin=0, vmax=vmax_h,
        linewidths=0.4, linecolor='white',
        annot=True, fmt='.1f', annot_kws={'size': 8.5},
        cbar_kws={'label': 'TMDD Index  (R₀ / KSS)', 'shrink': 0.65},
    )
    ax_heat.set_xlabel('')
    ax_heat.set_ylabel('')
    ax_heat.set_title(
        'TMDD Index Heatmap\n'
        '(Bulk  ·  SC max  ·  SC mean per tissue)'
    )
    ax_heat.tick_params(axis='x', rotation=0, labelsize=9)
    ax_heat.tick_params(axis='y', rotation=0, labelsize=9)

    fig.suptitle(
        f'{target_name.upper()} / {drug_name.capitalize()}  —  '
        f'TMDD Index  (R₀ / KSS): Bulk vs Single-Cell\n'
        f'Dose-independent  ·  t = 0  ·  '
        f'SC max = highest-expressing cell type per tissue',
        fontweight='bold', y=1.02
    )
    plt.tight_layout()
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PBPK Visualiser')
    parser.add_argument('--Target', type=str,   default='EGFR')
    parser.add_argument('--Drug',   type=str,   default='Cetuximab')
    parser.add_argument('--Dose',   type=float, default=400.0)
    parser.add_argument('--Tissue', type=str,   default='all',
                        help="Tissue name or 'all'")
    parser.add_argument('--File',   type=str,   default=None,
                        help='Direct CSV path (overrides auto-detection)')
    parser.add_argument('--multi',  action='store_true',
                        help='Multi-dose comparison mode')
    parser.add_argument('--doses',  type=str,   default='0.1,1,10,100,500',
                        help='Comma-separated doses for --multi (mg/m²)')
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  PBPK Visualiser  —  {args.Target} / {args.Drug}")
    print(f"{'='*55}")

    tgt  = args.Target
    drug = args.Drug

    # ── MULTI-DOSE MODE ────────────────────────────────────────────────────
    if args.multi:
        doses = [float(d.strip()) for d in args.doses.split(',')]
        print(f"\n[*] Multi-dose mode  —  {len(doses)} doses: {doses} mg/m²")

        dose_dfs: dict[float, pd.DataFrame] = {}
        for dose in doses:
            path = args.File if args.File else _try_paths(tgt, drug, dose)
            if path and os.path.exists(str(path)):
                dose_dfs[dose] = pd.read_csv(path)
                print(f"  [✓] {dose:>8.2f} mg/m²  →  {path}  "
                      f"({len(dose_dfs[dose]):,} rows)")
            else:
                print(f"  [!] {dose:>8.2f} mg/m²  →  CSV not found, skipping")

        if not dose_dfs:
            print("[ERROR] No dose CSVs found. Run main_pbpk.py --multi first.")
            return

        loaded_doses = sorted(dose_dfs.keys())
        out_root = os.path.join(
            'plots_results',
            f'multi_{tgt.lower()}_{drug.lower()}'
        )
        os.makedirs(out_root, exist_ok=True)
        print(f"\n[*] Output root: {out_root}/")

        run_global = args.Tissue.lower() == 'all'

        for dose, df in dose_dfs.items():
            dose_dir = os.path.join(out_root, f'dose_{dose}mg')
            os.makedirs(dose_dir, exist_ok=True)
            base     = os.path.join(dose_dir,
                                    f"{tgt.lower()}_{drug.lower()}_{dose}mg")

            print(f"\n  ── Dose {dose} mg/m²  ({dose_dir}) ──")
            tissues  = sorted(df['tissue'].unique())
            selected = (tissues if run_global
                        else [t for t in tissues
                              if t.lower() == args.Tissue.lower()])

            required = ['tissue', 'cell_type', 'time', 'mode',
                        'occupancy_pct', 'plasma_conc_nM',
                        'bound_RC_nM', 'local_isf_conc_nM',
                        'TMDD_index', 'R0_nM']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"  [ERROR] Missing columns {missing} — skipping dose {dose}")
                continue

            if run_global:
                print('  [Global figures]')
                fig1_pk_profile(df, tgt, drug,
                                save_path=f"{base}_fig1_pk.png")
                fig2_tmdd_heatmap(df, tgt,
                                  save_path=f"{base}_fig2_heatmap.png")
                fig3_fold_change(df, tgt,
                                 save_path=f"{base}_fig3_foldchange.png")

            for tissue in selected:
                print(f"  [{tissue}]")
                tdf   = df[df['tissue'] == tissue]
                tbase = os.path.join(
                    dose_dir,
                    f"{tgt.lower()}_{drug.lower()}_{tissue.lower()}_{dose}mg"
                )
                fig4_occupancy_bound(tdf, tissue, tgt,
                                     save_path=f"{tbase}_fig4_occ_rc.png")
                fig5_conc_gradient(tdf, tissue, tgt,
                                   save_path=f"{tbase}_fig5_conc.png")

        comp_dir  = os.path.join(out_root, 'comparison')
        os.makedirs(comp_dir, exist_ok=True)
        comp_base = os.path.join(comp_dir, f"{tgt.lower()}_{drug.lower()}")

        print(f"\n  ── Comparison figures  ({comp_dir}) ──")

        figC1_multi_serum(
            dose_dfs, tgt, drug,
            save_path=f"{comp_base}_compC1_serum.png"
        )
        figC3b_multi_tissue_isf_summary(
            dose_dfs, tgt, drug,
            save_path=f"{comp_base}_compC2_tissue_isf.png"
        )

        all_tissues = sorted(set(
            t for df in dose_dfs.values() for t in df['tissue'].unique()
        ))
        tissues_to_plot = all_tissues if run_global else [
            t for t in all_tissues if t.lower() == args.Tissue.lower()
        ]
        for tissue in tissues_to_plot:
            figC3_multi_cell_bound_rc(
                dose_dfs, tissue, tgt, drug,
                save_path=(f"{comp_base}_{tissue.lower()}"
                           f"_compC3_bound_rc.png")
            )
            figC3c_multi_cell_free_r(
                dose_dfs, tissue, tgt, drug,
                save_path=(f"{comp_base}_{tissue.lower()}"
                           f"_compC3c_free_r.png")
            )
            figC5_receptor_dynamics(
                dose_dfs, tissue, tgt, drug,
                save_path=(f"{comp_base}_{tissue.lower()}"
                           f"_compC5_receptor_dynamics.png")
            )

        figC4_bulk_tmdd_index(
            dose_dfs, tgt, drug,
            save_path=f"{comp_base}_compC4_tmdd.png"
        )

        per_dose_figs = (
            (3 if run_global else 0) + len(tissues_to_plot) * 2
        ) * len(dose_dfs)
        comp_figs = 3 + len(tissues_to_plot) * 3
        total = per_dose_figs + comp_figs

        print(f"\n[✓] {total} figure(s) saved")
        print(f"    Per-dose  : {per_dose_figs}  "
              f"(in {len(dose_dfs)} dose subfolder(s))")
        print(f"    Comparison: {comp_figs}  (in {comp_dir})")

    # ── SINGLE-DOSE MODE ───────────────────────────────────────────────────
    else:
        file_path = args.File or _try_paths(tgt, drug, args.Dose)
        if file_path is None or not os.path.exists(str(file_path)):
            print(f"[ERROR] CSV not found. Tried patterns like "
                  f"data/pbpk_{tgt.lower()}_{drug.lower()}_*.csv")
            return

        df = pd.read_csv(file_path)
        print(f"[*] Loaded {len(df):,} rows  |  "
              f"Modes: {df['mode'].unique().tolist()}  |  "
              f"Tissues: {sorted(df['tissue'].unique().tolist())}")

        required = ['tissue', 'cell_type', 'time', 'mode', 'occupancy_pct',
                    'plasma_conc_nM', 'bound_RC_nM', 'local_isf_conc_nM',
                    'TMDD_index', 'R0_nM']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[ERROR] Missing columns: {missing}")
            return

        out_dir = 'plots_results'
        os.makedirs(out_dir, exist_ok=True)

        tissues    = sorted(df['tissue'].unique())
        dose       = args.Dose
        run_global = args.Tissue.lower() == 'all'
        selected   = (tissues if run_global
                      else [t for t in tissues
                            if t.lower() == args.Tissue.lower()])
        if not selected:
            print(f"[ERROR] Tissue '{args.Tissue}' not found. "
                  f"Available: {tissues}")
            return

        base = os.path.join(out_dir,
                            f"{tgt.lower()}_{drug.lower()}_{dose:.0f}mg")

        if run_global:
            print('\n  [Global figures]')
            fig1_pk_profile(df, tgt, drug,
                            save_path=f"{base}_fig1_pk.png")
            fig2_tmdd_heatmap(df, tgt,
                              save_path=f"{base}_fig2_heatmap.png")
            fig3_fold_change(df, tgt,
                             save_path=f"{base}_fig3_foldchange.png")

        for tissue in selected:
            print(f"  [{tissue}]")
            tdf   = df[df['tissue'] == tissue]
            tbase = os.path.join(
                out_dir,
                f"{tgt.lower()}_{drug.lower()}_{tissue.lower()}_{dose:.0f}mg"
            )
            fig4_occupancy_bound(tdf, tissue, tgt,
                                 save_path=f"{tbase}_fig4_occ_rc.png")
            fig5_conc_gradient(tdf, tissue, tgt,
                               save_path=f"{tbase}_fig5_conc.png")

        n_figs = (3 if run_global else 0) + len(selected) * 2
        print(f"\n[✓] {n_figs} figure(s) saved to '{out_dir}/'")
        if run_global:
            print(f"    Key: {base}_fig3_foldchange.png")


if __name__ == '__main__':
    main()