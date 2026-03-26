"""
cell_volume.py
==============
Literature-based cell volume constants and lookup utilities.

Units : µm³  (cubic micrometers)
        1 µm³ = 1e-15 L

Sources
-------
BioNumbers (bionumbers.hms.harvard.edu)
Alberts et al. "Molecular Biology of the Cell", 6th ed.
Snijder et al. (2009) Nature 461:520  — cell-size / protein-content scaling
"""

# ── Volume constants (µm³) ─────────────────────────────────────────────────
VOL_ADIPOCYTE      = 200_000   # Lipid-laden; enormous cytoplasm
VOL_CARDIOMYOCYTE  =  20_000   # Striated muscle; high cytoplasmic volume
VOL_HEPATOCYTE     =  10_000   # Liver parenchymal; ~15 µm edge cube
VOL_FIBROBLAST     =   4_000   # Stromal / connective tissue
VOL_EPITHELIAL     =   2_000   # Epithelial, tubular, secretory
VOL_LARGE_IMMUNE   =   1_500   # Macrophage, Kupffer cell, Dendritic cell
VOL_ENDOTHELIAL    =     800   # Vascular lining; thin, elongated
VOL_MID_IMMUNE     =     350   # Neutrophil, Monocyte, Eosinophil, Basophil
VOL_LYMPHOCYTE     =     200   # T-cell, B-cell, NK cell
VOL_ERYTHROCYTE    =      90   # RBC; biconcave disc, no nucleus
VOL_PLATELET       =      10   # Megakaryocyte fragment; no nucleus
VOL_DEFAULT        =   1_000   # Generic fallback when name not recognised


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

    # 2. Stromal / connective tissue
    if any(x in name for x in ["fibroblast", "stellate", "stromal", "mesenchymal"]):
        return VOL_FIBROBLAST

    # 3. Epithelial / secretory
    if any(x in name for x in [
        "epithelial", "keratinocyte", "secretory",
        "proximal tubule", "alveolar", "podocyte", "ciliated",
    ]):
        return VOL_EPITHELIAL

    # 4. Endothelial / neuro-support
    if any(x in name for x in ["endothelial", "pericyte", "astrocyte", "glia"]):
        return VOL_ENDOTHELIAL

    # 5. Large immune / phagocytes
    if any(x in name for x in ["macrophage", "kupffer", "dendritic", "cdc", "pdc", "mast"]):
        return VOL_LARGE_IMMUNE

    # 6. Medium immune
    if any(x in name for x in ["neutrophil", "monocyte", "eosinophil", "basophil"]):
        return VOL_MID_IMMUNE

    # 7. Lymphocytes
    if any(x in name for x in ["t-cell", "t cell", "b-cell", "b cell", " nk ", "lymphocyte"]):
        return VOL_LYMPHOCYTE

    # 8. Blood components
    if "erythrocyte" in name or "red blood" in name:
        return VOL_ERYTHROCYTE
    if "platelet" in name:
        return VOL_PLATELET

    return VOL_DEFAULT


def get_total_volume_L(cell_type: str, cell_count: float) -> float:
    """
    Total volume (litres) occupied by `cell_count` cells of `cell_type`.

    Formula : V_total(L) = volume_per_cell(µm³) × n_cells × 1e-15
    """
    return get_volume_um3(cell_type) * cell_count * 1e-15