"""
Bayesian Inference & Scaling Package
Converts bulk proteomics/transcriptomics into single-cell spatial concentrations.
"""

from .cell_volume import get_volume_um3, get_total_volume_L
from .bayesian_ppm import BayesianPPMCalculator

__all__ = ['get_volume_um3', 'get_total_volume_L', 'BayesianPPMCalculator']