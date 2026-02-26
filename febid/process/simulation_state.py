"""
Simulation State Container

Pure data container holding all simulation arrays, parameters, and constants.
No computation logic - just state storage.
"""

import numpy as np

from febid.Structure import Structure
from febid.continuum_model_base import ContinuumModel


class SimulationState:
    """
    Container for all simulation data arrays and parameters.

    This is a data container with no computation logic.
    All arrays are references to Structure arrays or locally managed arrays.
    """

    def __init__(self, structure: Structure, model: ContinuumModel,
                 heat_cond: float, room_temp: float = 294):
        """Initialize simulation state with structure-backed arrays."""
        # Core references
        self.structure = structure
        self.model = model
        self.beam = model.beam
        self.precursor = model.precursor

        # Physical constants
        self.kb = 0.00008617  # Boltzmann constant
        self.heat_cond = heat_cond
        self.room_temp = room_temp

        # Cell geometry
        self.cell_size = structure.cell_size
        self.cell_V = structure.cell_size ** 3

        # Structural metadata
        self.substrate_height = structure.substrate_height  # Thickness of the substrate
        self.n_substrate_cells = structure.deposit[:structure.substrate_height].size  # the number of the cells in the substrate
        self.max_neib = 0  # Maximum number of surface nearest neighbors
        self.max_z = 0  # Maximum height of deposited structure (in cells)

        # Additional arrays (managed locally, not in Structure)
        self.beam_matrix = np.zeros_like(structure.deposit, dtype=np.int32)
        self.beam_matrix_surface = np.zeros_like(structure.deposit, dtype=np.int32)
        self.surface_temp = np.zeros_like(structure.temperature)
        self.D_temp = np.zeros_like(structure.precursor)  # Temperature-dependent diffusion coefficients
        self.tau_temp = np.zeros_like(structure.precursor)  # Temperature-dependent residence times
        self.surface_all = np.zeros_like(structure.deposit, dtype=bool)  # All surface cells (including semi-surface)

        # Scaling factor
        self.deposition_scaling = 1.0  # multiplier of the deposit increment; used to speed up the process

        # Temperature tracking flag
        self.temperature_tracking = False
