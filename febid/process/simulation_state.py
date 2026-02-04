"""
Simulation State Container

Pure data container holding all simulation arrays, parameters, and constants.
No computation logic - just state storage.
"""

import numpy as np
from febid.Structure import Structure
from febid.continuum_model_base import BeamSettings, PrecursorParams, ContinuumModel


class SimulationState:
    """
    Container for all simulation data arrays and parameters.

    This is a "dumb" data container with no computation logic.
    All arrays are references to Structure arrays or locally managed arrays.
    """

    def __init__(self, structure: Structure, model: ContinuumModel,
                 heat_cond: float, room_temp: float = 294):
        """
        Initialize simulation state.

        Parameters
        ----------
        structure : Structure
            The 3D structure object containing main arrays
        model : ContinuumModel
            Physics model with beam and precursor parameters
        heat_cond : float
            Thermal conductivity
        room_temp : float, optional
            Room temperature in Kelvin (default: 294)
        """
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
        self.substrate_height = structure.substrate_height
        self.n_substrate_cells = structure.deposit[:structure.substrate_height].size
        self.max_neib = 0  # Maximum number of surface nearest neighbors
        self.max_z = 0  # Maximum height of deposited structure (in cells)

        # Additional arrays (managed locally, not in Structure)
        self.beam_matrix = np.zeros_like(structure.deposit, dtype=np.int32)
        self.surface_temp = np.zeros_like(structure.temperature)
        self.D_temp = np.zeros_like(structure.precursor)  # Temperature-dependent diffusion coefficients
        self.tau_temp = np.zeros_like(structure.precursor)  # Temperature-dependent residence times

        # Scaling factor
        self.deposition_scaling = 1.0  # Will be set during Process initialization

        # Temperature tracking flag
        self.temperature_tracking = False  # Will be set during Process initialization