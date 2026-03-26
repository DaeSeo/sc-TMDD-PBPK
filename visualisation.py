"""
visualisation.py
================
PBPK Result Visualiser — publication-quality figures.

Figure overview
───────────────
  Global (--Tissue all):
    Fig-1  *_fig1_pk.png           Systemic PK: plasma free drug SC vs Bulk
    Fig-2  *_fig2_heatmap.png      TMDD Index heatmap: tissue × cell type (SC)
    Fig-3  *_fig3_foldchange.png   SC / Bulk fold change  ← KEY FIGURE

  Per tissue:
    Fig-4  *_fig4_occ_rc.png       Receptor occupancy + Bound RC per cell type
    Fig-5  *_fig5_conc.png         Concentration gradient: Cc → Cv → Cisf

Design rules
────────────
  · All linear y-axes start at 0 (ylim bottom=0).
  · Log y-axes start at max(data_min × 0.5, 1e-4).
  · All cell types shown at equal line weight — no "top-5 bold" convention.
  · 40-color palette (tab20 + tab20b) for cell types; line styles cycle for >40.
  · Cell types sorted by peak bound RC (highest first) so primary drug sinks
    receive the most visually distinct colors.
  · Fold-change x-axis in log2 scale so ×2 and ÷2 are symmetric.

Usage
─────
  python visualisation.py --Target EGFR --Drug Cetuximab --Dose 400 --Tissue all
  python visualisation.py --Target EGFR --Drug Cetuximab --Dose 400 --Tissue Liver
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
from matplotlib.lines import Line2D
import matplotlib.cm as _mcm
import seaborn as sns

def _get_cmap(name):
    """
    matplotlib version-safe cmap loader.
      < 3.5 : matplotlib.cm.get_cmap  (old API)
      3.5–3.6: matplotlib.colormaps.get_cmap
      >= 3.7 : matplotlib.colormaps[name]  (get_cmap deprecated)
    """
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

warnings.filterwarnings("ignore")


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

# ── Two-condition colors (SC vs Bulk) ─────────────────────────────────────────
C_SC        = '#1B4F8A'   # deep navy blue
C_BULK      = '#A93226'   # deep crimson
C_CTOT      = '#7D3C98'   # purple — Ctot_central (blood TMDD)
C_CV        = '#1F618D'   # steel blue — organ vascular
C_CISF      = '#148F77'   # teal — ISF
C_REF       = '#5D6D7E'   # grey — reference lines

TMDD_THRESH = 1.0
LW          = 1.4         # main line weight
LW_THIN     = 0.9         # secondary / Bulk lines

# ── Cell-type palette: 40 perceptually distinct colors ───────────────────────
# tab20 (dark+light pairs) + tab20b (earth tones) = 40 unique colors
_POOL = (
    list(_get_cmap('tab20').colors)
    + list(_get_cmap('tab20b').colors)
)   # length 40

_LSTYLES = ['-', '--', '-.', ':']   # cycle when n_cell_types > 40


def _cell_palette(cell_types_sorted: list) -> dict:
    """
    Return {cell_type: color} mapping.
    Cell types should be pre-sorted by biological priority (e.g. peak bound RC desc).
    """
    return {ct: _POOL[i % 40] for i, ct in enumerate(cell_types_sorted)}


def _cell_ls(rank: int) -> str:
    """Line style for rank-th cell type (cycle after 40)."""
    return _LSTYLES[(rank // 40) % 4]


# ══════════════════════════════════════════════════════════════════════════════
# IO helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {path}")


def _log_yax(ax, data_vals=None) -> None:
    """
    Apply log y-scale.
    Bottom set to max(data_min × 0.5, 1e-4) if data_vals provided, else 1e-4.
    """
    ax.set_yscale('log')
    if data_vals is not None and len(data_vals) > 0:
        pos = [v for v in data_vals if v > 0]
        bottom = max(min(pos) * 0.5, 1e-4) if pos else 1e-4
    else:
        bottom = 1e-4
    ax.set_ylim(bottom=bottom)


def _mode_split(df: pd.DataFrame):
    return df[df['mode'] == 'single_cell'], df[df['mode'] == 'bulk']


def _sort_by_peak_rc(sc_df: pd.DataFrame, cell_types: list) -> list:
    """Sort cell types by peak SC bound_RC_nM descending (primary drug sinks first)."""
    peak = sc_df.groupby('cell_type')['bound_RC_nM'].max()
    ordered = peak.reindex(cell_types).fillna(0).sort_values(ascending=False)
    return ordered.index.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-1  Systemic PK
# ══════════════════════════════════════════════════════════════════════════════

def fig1_pk_profile(df: pd.DataFrame, target_name: str, drug_name: str,
                    save_path: str = None) -> None:
    """
    Plasma free drug concentration (nM) over time: SC vs Bulk.

    If blood TMDD is active (plasma_ctot_nM column present), also plots
    Ctot_central (dotted purple) to show how much drug is sequestered
    intravascularly on blood cell receptors.

    Y-axis: log scale, bottom = max(data_min × 0.5, 1e-4).
    """
    sc, bulk = _mode_split(df)
    has_ctot = 'plasma_ctot_nM' in df.columns

    fig, ax = plt.subplots(figsize=(8, 5))

    def _pk_series(mode_df, col):
        if mode_df.empty or col not in mode_df.columns:
            return None
        return mode_df.groupby('time')[col].first().reset_index()

    pk_sc   = _pk_series(sc,   'plasma_conc_nM')
    pk_bulk = _pk_series(bulk, 'plasma_conc_nM')

    if pk_sc is not None:
        ax.plot(pk_sc['time'], pk_sc['plasma_conc_nM'],
                color=C_SC, lw=LW, label='Single-cell  $C_{c,\\ free}$', zorder=4)

    if pk_bulk is not None:
        ax.plot(pk_bulk['time'], pk_bulk['plasma_conc_nM'],
                color=C_BULK, lw=LW, ls='--', label='Bulk  $C_{c,\\ free}$', zorder=4)

    if has_ctot:
        ctot_sc = _pk_series(sc, 'plasma_ctot_nM')
        if ctot_sc is not None and ctot_sc['plasma_ctot_nM'].max() > 0:
            ax.plot(ctot_sc['time'], ctot_sc['plasma_ctot_nM'],
                    color=C_CTOT, lw=LW_THIN, ls=':',
                    label='SC  $C_{tot,\\ central}$  (free + blood-bound)',
                    zorder=3, alpha=0.85)

    # Collect all positive values for log bottom
    all_vals = []
    for d in [pk_sc, pk_bulk]:
        if d is not None:
            all_vals.extend(d.iloc[:, 1].dropna().tolist())
    _log_yax(ax, all_vals)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Plasma concentration (nM)')
    ax.set_title(f'{target_name.upper()} / {drug_name.capitalize()}  —  Systemic PK')
    ax.legend()
    ax.grid(True, which='both', axis='y')

    plt.tight_layout()
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-2  TMDD Index Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig2_tmdd_heatmap(df: pd.DataFrame, target_name: str,
                      save_path: str = None) -> None:
    """
    TMDD index (R₀/KSS) heatmap: rows = cell types, columns = tissues.
    Single-cell results only.  Blood column included.

    Color scale: RdYlGn_r, green = low risk, red = high.
    Cells annotated with numeric values.
    Rows sorted by max TMDD index descending so primary drug sinks appear first.
    """
    sc = df[df['mode'] == 'single_cell']
    t0 = sc['time'].min()
    sc_t0 = sc[sc['time'] == t0][['tissue', 'cell_type', 'TMDD_index']].copy()

    pivot = sc_t0.pivot_table(
        index='cell_type', columns='tissue',
        values='TMDD_index', aggfunc='first'
    ).fillna(0)

    # Sort rows: highest max TMDD first
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
        cmap='RdYlGn_r', vmin=0, vmax=vmax,
        linewidths=0.3, linecolor='white',
        annot=True, fmt='.0f', annot_kws={'size': 6.5},
        cbar_kws={'label': 'TMDD Index  (R₀ / KSS)', 'shrink': 0.55},
    )

    ax.set_xlabel('Tissue', labelpad=6)
    ax.set_ylabel('')
    ax.set_title(
        f'{target_name.upper()}  —  Single-Cell TMDD Index\n'
        f'R₀/KSS > 1 = TMDD risk  |  0 = not expressed  |  blood = intravascular'
    )
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=7.5)

    plt.tight_layout()
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-3  Fold Change  (KEY FIGURE)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_fold_change(df: pd.DataFrame, target_name: str,
                     save_path: str = None) -> None:
    """
    SC / Bulk TMDD index fold change.  KEY publication figure.

    X-axis: log2 scale (symmetric around fold=1).
      × 2  =  SC has double the TMDD risk → Bulk under-predicts
      ÷ 2  =  Bulk has double the TMDD risk → Bulk over-predicts

    Blue bars: SC > Bulk  (Bulk misses real drug sink)
    Red  bars: SC < Bulk  (Bulk invents a false drug sink)

    All cells with TMDD > 1 in either mode are shown at equal bar width.
    If >60 cells, shows top-30 over-predicted and top-30 under-predicted.
    """
    sc   = df[df['mode'] == 'single_cell']
    bulk = df[df['mode'] == 'bulk']
    t0   = df['time'].min()

    sc_t0   = (sc[sc['time'] == t0][['tissue','cell_type','TMDD_index']]
               .rename(columns={'TMDD_index': 'sc_idx'}))
    bulk_t0 = (bulk[bulk['time'] == t0][['tissue','cell_type','TMDD_index']]
               .rename(columns={'TMDD_index': 'bulk_idx'}))
    merged  = sc_t0.merge(bulk_t0, on=['tissue','cell_type'])

    # Filter to TMDD-relevant cells
    merged = merged[
        (merged['sc_idx'] > TMDD_THRESH) | (merged['bulk_idx'] > TMDD_THRESH)
    ].copy()

    if merged.empty:
        print("   [WARNING] No TMDD-relevant cells found.")
        return

    # Fold change; floor both at a small epsilon to avoid ÷0 and log(0)
    merged['fold']  = (merged['sc_idx'] + 1e-9) / (merged['bulk_idx'] + 1e-9)
    merged['label'] = (merged['tissue'].str.capitalize()
                       + '\n' + merged['cell_type'])
    merged = merged.sort_values('fold', ascending=False)

    # Keep balanced top-30 over and top-30 under if total > 60
    if len(merged) > 60:
        plot_df = pd.concat(
            [merged.head(30), merged.tail(30)]
        ).drop_duplicates()
    else:
        plot_df = merged.copy()
    plot_df = plot_df.sort_values('fold', ascending=True)

    log2_fold  = np.log2(plot_df['fold'].clip(lower=1/64, upper=256))
    bar_colors = [C_SC if f >= 1.0 else C_BULK for f in plot_df['fold']]

    fig_h = max(8, len(plot_df) * 0.34)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    ax.barh(range(len(plot_df)), log2_fold,
            color=bar_colors, edgecolor='none', height=0.72)

    # Reference lines
    ax.axvline(0,            color='black',  lw=1.2,    ls='--', zorder=5, label='SC = Bulk')
    ax.axvline( np.log2(2),  color=C_REF,    lw=0.8,    ls=':',  zorder=4, alpha=0.7)
    ax.axvline(-np.log2(2),  color=C_REF,    lw=0.8,    ls=':',  zorder=4, alpha=0.7)

    # X-axis: fold-change ticks in human-readable form
    tick_folds = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    tick_log2  = [np.log2(v) for v in tick_folds]
    tick_labels = ['÷16','÷8','÷4','÷2','1','×2','×4','×8','×16','×32','×64','×128','×256']
    # Clip to data range
    data_range = (log2_fold.min() - 0.5, log2_fold.max() + 0.5)
    valid = [(v, l) for v, l in zip(tick_log2, tick_labels)
             if data_range[0] <= v <= data_range[1]]
    ax.set_xticks([v for v, _ in valid])
    ax.set_xticklabels([l for _, l in valid], fontsize=9)
    ax.set_xlim(data_range)

    # Y-axis: cell type labels
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['label'], fontsize=7.5)

    # Annotate extreme values (fold ≥ ×4 or ≤ ÷4)
    for i, (fold, log2_f) in enumerate(zip(plot_df['fold'], log2_fold)):
        if fold >= 4.0:
            ax.text(log2_f + 0.08, i, f'×{fold:.0f}',
                    va='center', ha='left', fontsize=7,
                    color='#1B4F8A', fontweight='bold')
        elif fold <= 0.25:
            ax.text(log2_f - 0.08, i, f'÷{1/fold:.0f}',
                    va='center', ha='right', fontsize=7,
                    color='#A93226', fontweight='bold')

    ax.set_xlabel('Fold change  (SC TMDD index / Bulk TMDD index)  [log₂ scale]')
    ax.set_title(
        f'{target_name.upper()}  —  Single-Cell vs Bulk: TMDD Index\n'
        f'Blue (>1): Bulk under-predicts  ·  Red (<1): Bulk over-predicts'
    )

    # Summary stats
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
    """
    Left panel  — Receptor occupancy (%) over time.
      Within a tissue all cells share the same Cisf (QSS), so occupancy =
      Cisf / (KSS + Cisf) × 100 is identical across cell types.
      Shows SC (solid) vs Bulk (dashed) as one representative line each.
      Y: 0 → 105 %.

    Right panel — Bound drug RC (nM) per cell type over time.
      RC_i = Rtot_i × Cisf / (KSS + Cisf).  Even though occupancy% is
      the same, cells with higher R₀ sequester more drug in absolute terms.
      ALL cell types shown at equal line weight.
      Cell types sorted by peak SC bound RC (primary drug sinks first → best colors).
      Y: from 0.
    """
    sc, bulk = _mode_split(tissue_df)
    is_blood  = tissue_name.lower() == 'blood'
    all_cts   = sorted(tissue_df['cell_type'].unique())

    # Sort by peak SC bound RC so biggest drug sinks get most distinct colors
    sorted_cts = _sort_by_peak_rc(sc, all_cts)
    ct_color   = _cell_palette(sorted_cts)

    fig, (ax_occ, ax_rc) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        f'{tissue_name.capitalize()}  ·  {target_name.upper()}'
        + ('  [intravascular]' if is_blood else ''),
        fontweight='bold', y=1.01
    )

    # ── Left: Occupancy ────────────────────────────────────────────────────
    # All cells share the same occupancy curve within a tissue.
    # Use the first available cell type as representative.
    rep_ct = sorted_cts[0] if sorted_cts else all_cts[0]

    sc_rep   = sc[sc['cell_type'] == rep_ct]
    bulk_rep = bulk[bulk['cell_type'] == rep_ct]

    if not sc_rep.empty:
        ax_occ.plot(sc_rep['time'], sc_rep['occupancy_pct'],
                    color=C_SC, lw=LW, label='Single-cell')
    if not bulk_rep.empty:
        ax_occ.plot(bulk_rep['time'], bulk_rep['occupancy_pct'],
                    color=C_BULK, lw=LW, ls='--', label='Bulk')

    ax_occ.axhline(50, color=C_REF, lw=LW_THIN, ls=':',  alpha=0.7, label='50 %')
    ax_occ.axhline(90, color=C_REF, lw=LW_THIN, ls='-.', alpha=0.7, label='90 %')
    ax_occ.set_ylim(0, 105)   # y from 0
    ax_occ.set_xlabel('Time (days)')
    ax_occ.set_ylabel('Receptor occupancy (%)')
    ax_occ.set_title('Receptor occupancy\n(all cells share ISF → identical curve)')
    ax_occ.legend(fontsize=8)
    ax_occ.grid(True, axis='y')
    ax_occ.text(0.97, 0.50,
                'Occupancy = Cisf/(KSS + Cisf)\n'
                'does NOT depend on R₀.\n'
                'See right panel for R₀ effect.',
                transform=ax_occ.transAxes, ha='right', va='center',
                fontsize=7.5, color='#5D6D7E',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.8', lw=0.6))

    # ── Right: Bound RC per cell type ──────────────────────────────────────
    for rank, ct in enumerate(sorted_cts):
        c  = ct_color[ct]
        ls = _cell_ls(rank)

        sc_ct   = sc[sc['cell_type'] == ct]
        bulk_ct = bulk[bulk['cell_type'] == ct]

        if not sc_ct.empty:
            ax_rc.plot(sc_ct['time'], sc_ct['bound_RC_nM'],
                       color=c, lw=LW, ls=ls, label=ct, zorder=3)
        if not bulk_ct.empty:
            ax_rc.plot(bulk_ct['time'], bulk_ct['bound_RC_nM'],
                       color=c, lw=LW_THIN, ls='--', alpha=0.45, zorder=2)

    ax_rc.set_ylim(bottom=0)   # y from 0
    ax_rc.set_xlabel('Time (days)')
    ax_rc.set_ylabel('Bound drug  RC (nM)')
    ax_rc.set_title('Drug–receptor complex per cell type\n(solid = SC  |  dashed = Bulk)')
    ax_rc.grid(True, axis='y')

    # Legend: arranged in columns to avoid overflow
    n_cols_leg = max(1, len(sorted_cts) // 14 + 1)
    ax_rc.legend(fontsize=7, ncol=n_cols_leg, loc='upper right',
                 handlelength=1.5, framealpha=0.9)

    plt.tight_layout()
    _save(fig, save_path) if save_path else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig-5  Per-tissue: Concentration Gradient
# ══════════════════════════════════════════════════════════════════════════════

def fig5_conc_gradient(tissue_df: pd.DataFrame, tissue_name: str,
                       target_name: str, save_path: str = None) -> None:
    """
    Two-pore transport concentration gradient for a solid organ:
      Cc (plasma free) → Cv (organ vascular) → Cisf (ISF)

    SC: solid lines.  Bulk: dotted lines (same color, dimmed).
    Y-axis: log scale, bottom = max(data_min × 0.5, 1e-4).

    For blood (intravascular compartment):
      No Cv or Cisf — shows only Cc_free (SC) and Ctot_central (SC dotted).
    """
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
        # Blood: Cc_free and optionally Ctot_central
        t, v = _ts(sc, 'plasma_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_SC, lw=LW, label='$C_{c,\\ free}$ SC')
            all_vals.extend(v[v > 0].tolist())

        if has_ctot:
            t, v = _ts(sc, 'plasma_ctot_nM')
            if t is not None and np.max(v) > 0:
                ax.plot(t, v, color=C_CTOT, lw=LW_THIN, ls=':',
                        label='$C_{tot,\\ central}$ SC  (free + blood-bound)',
                        alpha=0.85)
                all_vals.extend(v[v > 0].tolist())

        t, v = _ts(bulk, 'plasma_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_BULK, lw=LW_THIN, ls='--',
                    label='$C_{c,\\ free}$ Bulk', alpha=0.7)
            all_vals.extend(v[v > 0].tolist())

        ax.set_title(
            f'{tissue_name.capitalize()}  ·  {target_name.upper()}  —  '
            f'Intravascular concentration\n(no ISF; blood cells contact plasma directly)'
        )

    else:
        # Solid organ: Cc → Cv → Cisf
        t, v = _ts(sc, 'plasma_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_SC, lw=LW, label='Plasma $C_c$ (SC)')
            all_vals.extend(v[v > 0].tolist())

        t, v = _ts(sc, 'local_v_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_CV, lw=LW, ls='--',
                    label='Organ vascular $C_v$ (SC)')
            all_vals.extend(v[v > 0].tolist())

        t, v = _ts(sc, 'local_isf_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_CISF, lw=LW, ls='-.',
                    label='ISF $C_{isf}$ (SC)')
            all_vals.extend(v[v > 0].tolist())

        # Bulk: same colors, dotted + dimmed
        t, v = _ts(bulk, 'plasma_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_SC, lw=LW_THIN, ls=':',
                    alpha=0.55, label='Plasma $C_c$ (Bulk)')
        t, v = _ts(bulk, 'local_isf_conc_nM')
        if t is not None:
            ax.plot(t, v, color=C_CISF, lw=LW_THIN, ls=':',
                    alpha=0.55, label='ISF $C_{isf}$ (Bulk)')

        ax.set_title(
            f'{tissue_name.capitalize()}  ·  {target_name.upper()}  —  '
            f'Concentration gradient\n$C_c$ → $C_v$ → $C_{{isf}}$  (Two-Pore transport)'
        )

    _log_yax(ax, all_vals)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Concentration (nM)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', axis='y')

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
    parser.add_argument('--Tissue', type=str,   required=True,
                        help="Tissue name or 'all'")
    parser.add_argument('--File',   type=str,   default=None)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  PBPK Visualiser  —  {args.Target} / {args.Drug}")
    print(f"{'='*55}")

    # Locate CSV (try both 400.0mg and 400mg filename formats)
    def _try_paths(tgt, drug, dose):
        candidates = [
            f"data/pbpk_{tgt.lower()}_{drug.lower()}_{dose:.1f}mg.csv",
            f"data/pbpk_{tgt.lower()}_{drug.lower()}_{dose:.0f}mg.csv",
            f"data/pbpk_{tgt.lower()}_{drug.lower()}_{dose}mg.csv",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    file_path = args.File or _try_paths(args.Target, args.Drug, args.Dose)
    if file_path is None or not os.path.exists(str(file_path)):
        print(f"[ERROR] CSV not found. Tried patterns like "
              f"data/pbpk_{args.Target.lower()}_{args.Drug.lower()}_*.csv")
        return

    df = pd.read_csv(file_path)
    print(f"[*] Loaded {len(df):,} rows  |  "
          f"Modes: {df['mode'].unique().tolist()}  |  "
          f"Tissues: {sorted(df['tissue'].unique().tolist())}")

    required = ['tissue','cell_type','time','mode','occupancy_pct',
                'plasma_conc_nM','bound_RC_nM','local_isf_conc_nM',
                'TMDD_index','R0_nM']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return

    out_dir = 'plots_results'
    os.makedirs(out_dir, exist_ok=True)

    tissues    = sorted(df['tissue'].unique())
    tgt        = args.Target
    drug       = args.Drug
    dose       = args.Dose
    run_global = args.Tissue.lower() == 'all'

    selected = tissues if run_global else [
        t for t in tissues if t.lower() == args.Tissue.lower()
    ]
    if not selected:
        print(f"[ERROR] Tissue '{args.Tissue}' not found. Available: {tissues}")
        return

    base = os.path.join(out_dir, f"{tgt.lower()}_{drug.lower()}_{dose:.0f}mg")

    # ── Global figures ────────────────────────────────────────────────────
    if run_global:
        print('\n  [Global figures]')
        fig1_pk_profile(df, tgt, drug,
                        save_path=f"{base}_fig1_pk.png")
        fig2_tmdd_heatmap(df, tgt,
                          save_path=f"{base}_fig2_heatmap.png")
        fig3_fold_change(df, tgt,
                         save_path=f"{base}_fig3_foldchange.png")

    # ── Per-tissue figures ────────────────────────────────────────────────
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