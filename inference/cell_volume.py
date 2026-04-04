"""
cell_volume.py
==============
Literature-based cell volume constants and lookup utilities.

Units : µm³  (cubic micrometers),  1 µm³ = 1e-15 L

Sources
-------
BioNumbers (bionumbers.hms.harvard.edu)
Alberts et al. "Molecular Biology of the Cell", 6th ed.
Snijder et al. (2009) Nature 461:520  — cell-size / protein-content scaling
Schwanhäusser et al. (2011) Nature 473:337  — N_total_protein per cell
Wisniewski et al. (2014) Nat Methods 11:306  — N_total_protein hepatocyte
Milo & Phillips "Cell Biology by Numbers" (2015) — immune cell protein counts
"""

N_A = 6.022e23   # Avogadro constant (mol⁻¹)

# ── Volume constants (µm³) ─────────────────────────────────────────────────
VOL_ADIPOCYTE      = 200_000   # Lipid-laden; enormous cytoplasm
VOL_CARDIOMYOCYTE  =  20_000   # Striated muscle; high cytoplasmic volume
VOL_HEPATOCYTE     =  10_000   # Liver parenchymal; ~15 µm edge cube
VOL_FIBROBLAST     =   4_000   # Stromal / connective tissue
VOL_EPITHELIAL     =   2_000   # Epithelial, tubular, secretory
VOL_NEURON         =   2_000   # Cortical neuron soma (excludes axon/dendrites)
VOL_LARGE_IMMUNE   =   1_500   # Macrophage, Kupffer cell, Dendritic cell
VOL_ENDOTHELIAL    =     800   # Vascular lining; thin, elongated
VOL_MID_IMMUNE     =     350   # Neutrophil, Monocyte, Eosinophil, Basophil
VOL_LYMPHOCYTE     =     200   # T-cell, B-cell, NK cell
VOL_ERYTHROCYTE    =      90   # RBC; biconcave disc, no nucleus
VOL_PLATELET       =      10   # Megakaryocyte fragment; no nucleus
VOL_DEFAULT        =   1_000   # Generic fallback when name not recognised


# ── Total protein molecules per cell (N_total) ────────────────────────────
# Used to convert PPM → absolute molecule count per single cell:
#   actual_counts = local_ppm × N_total × 1e-6
#
# Sources:
#   fibroblast / default : Schwanhäusser et al. (2011) Nature 473:337
#   hepatocyte           : Wisniewski et al. (2014) Nat Methods 11:306
#   immune cells         : Milo & Phillips "Cell Biology by Numbers" (2015)
#   others               : scaled from above by volume ratio (Snijder 2009)

N_TOTAL_PROTEIN = {
    "adipocyte":     2.0e9,
    "cardiomyocyte": 4.0e9,
    "hepatocyte":    2.5e9,   # Wisniewski 2014
    "fibroblast":    2.0e9,   # Schwanhäusser 2011
    "epithelial":    1.8e9,
    "neuron":        3.5e9,
    "large_immune":  2.5e9,
    "endothelial":   1.0e9,
    "mid_immune":    8.0e8,
    "lymphocyte":    2.0e8,
    "erythrocyte":   3.0e8,
    "platelet":      3.0e7,
    "default":       2.0e9,
}


def get_volume_um3(cell_name: str) -> float:
    """
    Return literature-based cell volume (µm³) from a cell-type name string.
    Matching is case-insensitive keyword search.
    Returns VOL_DEFAULT if no keyword matches.
    """
    name = cell_name.lower()

    # 1. Giant / specialised cells — check first (most specific)
    if "adipocyte" in name or "fat cell" in name:
        return VOL_ADIPOCYTE
    if "cardiomyocyte" in name:
        return VOL_CARDIOMYOCYTE
    if "hepatocyte" in name:
        return VOL_HEPATOCYTE

    # 2. Neurons — before endothelial/glia to avoid false match
    if any(x in name for x in [
        "neuron", "excitatory", "inhibitory", "pyramidal", "purkinje",
        "granule cell", "interneuron", "glutamatergic", "gabaergic",
        "dopaminergic", "cholinergic", "serotonergic",
    ]):
        return VOL_NEURON

    # 3. Stromal / connective tissue
    if any(x in name for x in ["fibroblast", "stellate", "stromal", "mesenchymal"]):
        return VOL_FIBROBLAST

    # 4. Epithelial / secretory
    if any(x in name for x in [
        "epithelial", "keratinocyte", "secretory",
        "proximal tubule", "alveolar", "podocyte", "ciliated",
    ]):
        return VOL_EPITHELIAL

    # 5. Endothelial / neuro-support
    if any(x in name for x in ["endothelial", "pericyte", "astrocyte", "glia"]):
        return VOL_ENDOTHELIAL

    # 6. Large immune / phagocytes
    if any(x in name for x in ["macrophage", "kupffer", "dendritic", "cdc", "pdc", "mast"]):
        return VOL_LARGE_IMMUNE

    # 7. Medium immune
    if any(x in name for x in ["neutrophil", "monocyte", "eosinophil", "basophil"]):
        return VOL_MID_IMMUNE

    # 8. Lymphocytes
    if any(x in name for x in ["t-cell", "t cell", "b-cell", "b cell", " nk ", "lymphocyte"]):
        return VOL_LYMPHOCYTE

    # 9. Blood components
    if "erythrocyte" in name or "red blood" in name:
        return VOL_ERYTHROCYTE
    if "platelet" in name:
        return VOL_PLATELET

    return VOL_DEFAULT


def get_n_total(cell_name: str) -> float:
    """
    Return N_total_protein (molecules per cell) for a given cell-type name.
    Used to compute actual_counts_per_cell from local PPM.
    Sources: Schwanhäusser (2011), Wisniewski (2014), Milo & Phillips (2015)
    """
    name = cell_name.lower()

    if "adipocyte" in name or "fat cell" in name:
        return N_TOTAL_PROTEIN["adipocyte"]
    if "cardiomyocyte" in name:
        return N_TOTAL_PROTEIN["cardiomyocyte"]
    if "hepatocyte" in name:
        return N_TOTAL_PROTEIN["hepatocyte"]
    if any(x in name for x in [
        "neuron", "excitatory", "inhibitory", "pyramidal", "purkinje",
        "granule cell", "interneuron", "glutamatergic", "gabaergic",
    ]):
        return N_TOTAL_PROTEIN["neuron"]
    if any(x in name for x in ["fibroblast", "stellate", "stromal", "mesenchymal"]):
        return N_TOTAL_PROTEIN["fibroblast"]
    if any(x in name for x in [
        "keratinocyte", "epithelial", "alveolar", "podocyte",
        "secretory", "ciliated", "proximal tubule",
    ]):
        return N_TOTAL_PROTEIN["epithelial"]
    if any(x in name for x in ["endothelial", "pericyte", "astrocyte", "glia"]):
        return N_TOTAL_PROTEIN["endothelial"]
    if any(x in name for x in ["macrophage", "kupffer", "dendritic", "cdc", "pdc", "mast"]):
        return N_TOTAL_PROTEIN["large_immune"]
    if any(x in name for x in ["neutrophil", "monocyte", "eosinophil", "basophil"]):
        return N_TOTAL_PROTEIN["mid_immune"]
    if any(x in name for x in ["t-cell", "t cell", "b-cell", "b cell", " nk ", "lymphocyte"]):
        return N_TOTAL_PROTEIN["lymphocyte"]
    if "erythrocyte" in name or "red blood" in name:
        return N_TOTAL_PROTEIN["erythrocyte"]
    if "platelet" in name:
        return N_TOTAL_PROTEIN["platelet"]
    return N_TOTAL_PROTEIN["default"]


def get_counts_per_cell(cell_name: str, local_ppm: float) -> float:
    """
    Absolute number of target protein molecules per single cell.

    Derivation (PaxDb molecule-count PPM convention):
        ppm       = N_protein / N_total × 1e6
        N_protein = ppm × N_total × 1e-6

    Validation — EGFR skin fibroblast:
        local_ppm ≈ 22.7,  N_total = 2×10⁹
        → counts ≈ 45,400 molecules/cell  ✓
        (consistent with flow-cytometry EGFR counts on stromal fibroblasts)

    Sources: Schwanhäusser (2011), Wisniewski (2014), Milo & Phillips (2015)
    """
    return local_ppm * get_n_total(cell_name) * 1e-6


def get_total_volume_L(cell_type: str, cell_count: float) -> float:
    """
    Total volume (litres) occupied by `cell_count` cells of `cell_type`.
    Formula : V_total(L) = volume_per_cell(µm³) × n_cells × 1e-15
    """
    return get_volume_um3(cell_type) * cell_count * 1e-15