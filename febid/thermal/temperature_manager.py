"""
Temperature Manager

Manages temperature tracking and temperature-dependent parameters for FEBID simulation.

This component:
- Calculates and updates temperature fields
- Provides temperature-dependent diffusion coefficients D(T)
- Provides temperature-dependent residence times τ(T)
- Manages recalculation triggers based on volume thresholds
- Provides unified data access (no branching in caller code)

IMPORTANT: TemperatureManager is acceleration-agnostic. It provides raw data only
(scalars or full arrays). DataViewManager is responsible for slicing and selecting
appropriate forms (1D vs 2D) based on acceleration mode.
"""

import numpy as np
from typing import Union
from febid.process.simulation_state import SimulationState
from febid.process.data_view_manager import DataViewManager
import febid.thermal.heat_transfer as heat_transfer
from febid.libraries.rolling.roll import surface_temp_av
from febid.slice_trics import get_3d_slice
from febid.logging_config import setup_logger

logger = setup_logger(__name__)


class TemperatureManager:
    """
    Manages temperature tracking and temperature-dependent parameters.

    IMPORTANT: Acceleration-agnostic - provides raw data only, no shape selection.
    """

    def __init__(self, state: SimulationState, view_manager: DataViewManager, step_volume=10000.0) -> None:
        """
        Initialize TemperatureManager.

        Parameters
        ----------
        state : SimulationState
            Simulation state containing structure and arrays
        view_manager : DataViewManager
            View manager for spatial slices and indices
        step_volume : float, optional
            Temperature recalculation interval based on deposited volume
        """
        self.state = state
        self.view_manager = view_manager

        # Configuration
        self.enabled = state.temperature_tracking
        self._temp_step = 10000.0  # nm³ volume threshold
        self._temp_step_cells = self._temp_step / state.cell_V  # temperature recalculation interval normalized by volume
        self._solution_accuracy = 0.01  # Heat solver accuracy

        # Tracking
        self._calc_count = 0
        self._recalc_requested = False
        self._max_temperature = state.room_temp

        # Cached indices
        self._solid_index = None  # Cached solid cell indices (deposit < 0)

        # NO coefficient caching - DataViewManager generates forms on-demand

    # ===== Main Update Methods (Two-Phase System) =====

    def update_after_cell_filling(self) -> None:
        """
        Phase 1: Update coefficient arrays for NEW topology with CURRENT temperatures.

        Called after cell_filled_routine() when surface topology changes.
        Ensures coefficient array shapes match new surface_all before physics resumes.
        This prevents shape mismatches when surface_all changes but temperature hasn't
        been recalculated yet.
        """

        # Recalculate D and tau for NEW surface_all using CURRENT temperatures
        # This ensures arrays match new topology even before temperature recalculation
        self._update_surface_temperatures()
        self._update_diffusion_coefficients()
        self._update_residence_times()

    def update_temperature_field(self, heating: np.ndarray) -> None:
        """
        Phase 2: Recalculate temperature field and update coefficients with new temperatures.

        Called after MC simulation when heating data available.
        Full temperature solve + coefficient updates with fresh temperature values.

        Parameters
        ----------
        heating : np.ndarray
            Volumetric heat source array from MC simulation
        """
        self._update_solid_index()  # Pre-cache indices for solver

        # Full recalculation requested
        self._solve_heat_equation(heating)

        # Update derived quantities with new temperatures
        self.update_after_cell_filling()

        logger.info(f'Current max. temperature: {self._max_temperature} K')
        # Update tracking
        self._calc_count += 1
        self._recalc_requested = False

    def check_and_request_recalculation(self, filled_cells: int) -> None:
        """
        Check if temperature recalculation is needed based on volume threshold.

        Called from cell_filled_routine() to set recalculation flag.

        Parameters
        ----------
        filled_cells : int
            Total number of filled cells so far
        """
        # Trigger recalc if more cells filled than threshold
        # Structure should be at least 5 cells high to avoid early recalcs during initial growth phase
        structure_min_volume_condition = self.state.max_z - self.state.substrate_height - 3 > 2
        # Use normalized cell count to trigger recalculation at consistent volume intervals
        step_volume_condition = filled_cells > self._calc_count * self._temp_step_cells
        self._recalc_requested = structure_min_volume_condition and step_volume_condition


    def initialize_cell_temperature(self, cell: tuple, temp_array: np.ndarray) -> None:
        """
        Initialize temperature of newly filled cell by averaging surroundings.

        Parameters
        ----------
        cell : tuple
            (z, y, x) cell coordinates
        temp_array : np.ndarray
            Temperature array view (from SurfaceUpdateView)
        """
        if not self.enabled:
            return

        from febid.slice_trics import get_3d_slice

        temp_slice, _ = get_3d_slice(cell, temp_array.shape, 2)
        temp_kern = temp_array[temp_slice]
        condition = (temp_kern > self.state.room_temp)

        if np.any(condition):
            temp_array[cell] = temp_kern[condition].sum() / np.count_nonzero(condition)

    # ===== Coefficient Access (Raw Data - No Slicing, No Acceleration Logic) =====

    def get_D(self) -> Union[float, np.ndarray]:
        """
        Get diffusion coefficient: scalar or full array (no slicing).

        Returns
        -------
        Union[float, np.ndarray]
            If temp OFF: Scalar constant precursor.D
            If temp ON: Full array state.D_temp (DataViewManager will slice)
        """
        if self.enabled:
            return self.state.D_temp  # Full array
        else:
            return self.state.precursor.D  # Scalar

    def get_tau(self) -> Union[float, np.ndarray]:
        """
        Get residence time: scalar or full array (no slicing).

        Returns
        -------
        Union[float, np.ndarray]
            If temp OFF: Scalar constant precursor.tau
            If temp ON: Full array state.tau_temp (DataViewManager will slice)
        """
        if self.enabled:
            return self.state.tau_temp  # Full array
        else:
            return self.state.precursor.tau  # Scalar

    # ===== Properties =====

    @property
    def max_temperature(self) -> float:
        """Current maximum temperature in structure."""
        if self.enabled:
            return self.state.surface_temp.max()
        else:
            return self.state.room_temp

    @property
    def requires_recalculation(self) -> bool:
        """Check if temperature recalculation has been requested."""
        return self._recalc_requested

    @property
    def calculation_count(self) -> int:
        """Number of times temperature has been recalculated."""
        return self._calc_count

    # ===== Private Implementation Methods =====

    def _solve_heat_equation(self, heating: np.ndarray) -> None:
        """
        Run steady-state heat transfer solver.

        Parameters
        ----------
        heating : np.ndarray
            Volumetric heat source array
        """
        from timeit import default_timer as df

        view = self.view_manager.get_temperature_recalc_view()
        slice_no_sub = view.slice_no_sub

        # Extract heating for irradiated region
        if isinstance(heating, np.ndarray):
            heat = heating[slice_no_sub]
        else:
            heat = heating

        # Log heating power
        logger.info(f'Total heating power: {heat.sum() / 1e6:.3f} W/nm/K/1e6')

        # Solve heat equation (modifies temperature in-place)
        start = df()
        heat_transfer.heat_transfer_steady_sor(
            view.temp,
            self.state.heat_cond,
            self.state.cell_size,
            heat,
            self._solution_accuracy
        )
        logger.info(f'Temperature recalculation took {df() - start:.4f} s')

    def _update_surface_temperatures(self) -> None:
        """Calculate surface temperatures by averaging neighboring solid cells."""
        surface_index = self.view_manager._index_surface_2d
        semi_surface_index = self.view_manager._index_semi_surface_2d
        slice_2d = self.view_manager._slice_irradiated_2d

        surface_temp = self.state.surface_temp[slice_2d]
        temp = self.state.structure.temperature[slice_2d]

        surface_temp[...] = 0  # Reset
        surface_temp_av(surface_temp, temp, *surface_index)  # Average for surface
        surface_temp_av(surface_temp, surface_temp, *semi_surface_index)  # Semi-surface

        # Update max temperature tracking
        self._max_temperature = self.state.surface_temp.max()

    def _update_diffusion_coefficients(self) -> None:
        """
        Calculate temperature-dependent diffusion coefficients.

        Updates state.D_temp for ALL surface cells (surface + semi_surface).
        No caching - DataViewManager generates appropriate forms on-demand.
        """
        # Get surface_all index (includes surface AND semi_surface)
        surface_all = self.view_manager._index_surface_all_2d
        slice_2d = self.view_manager._slice_irradiated_2d

        # Get surface temperatures
        temp_2d = self.state.surface_temp[slice_2d]

        # Calculate D(T) for ALL surface cells (surface + semi_surface)
        D_2d = np.zeros_like(temp_2d)
        D_2d[surface_all] = self.state.precursor.diffusion_coefficient_at_T(
            temp_2d[surface_all]
        )

        # Store in state array (source of truth)
        self.state.D_temp[slice_2d] = D_2d

    def _update_residence_times(self) -> None:
        """
        Calculate temperature-dependent residence times.

        Updates state.tau_temp for ALL surface cells (surface + semi_surface).
        No caching - DataViewManager generates appropriate forms on-demand.
        """
        # Get surface_all index (includes surface AND semi_surface)
        surface_all = self.view_manager._index_surface_all_2d
        slice_2d = self.view_manager._slice_irradiated_2d

        # Get surface temperatures
        temp_2d = self.state.surface_temp[slice_2d]

        # Calculate tau(T) for ALL surface cells (surface + semi_surface)
        tau_2d = np.zeros_like(temp_2d)
        tau_2d[surface_all] = self.state.precursor.residence_time_at_T(
            temp_2d[surface_all]
        )

        # Store in state array (source of truth - 2D only)
        self.state.tau_temp[slice_2d] = tau_2d

        # NO caching of 1D flattened form - DataViewManager generates when needed

    def _update_solid_index(self) -> None:
        """Generate and cache indices of solid cells for heat solver."""
        view = self.view_manager.get_temperature_recalc_view()
        deposit = view.deposit
        index = (deposit < 0).nonzero()
        self._solid_index = (np.intc(index[0]), np.intc(index[1]), np.intc(index[2]))
