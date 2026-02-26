import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple

from febid.slice_trics import index_where, concat_index, cast_index_to_int
from febid.process.simulation_state import SimulationState


@dataclass
class DepositionView:
    """
    Everything needed for deposition calculation.
    
    Key attributes:
    * ``deposit`` (3D array view (restricted if acceleration on, full if off))
    * ``precursor`` (3D array view (restricted if acceleration on, full if off))
    * ``beam_matrix`` (1D flattened array (acceleration on) or full 3D array (acceleration off))
    * ``index``: Fancy index tuple (acceleration on) or np.s_[:] (acceleration off)
    * ``acceleration_enabled`` (Flag indicating which mode is active)
    """
    deposit: np.ndarray
    precursor: np.ndarray
    beam_matrix: np.ndarray
    index: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], slice]
    acceleration_enabled: bool


@dataclass
class PrecursorDensityView:
    """
    Everything needed for precursor density calculation.
    
    Key attributes:
    * ``precursor`` (2D precursor array for the irradiated region)
    * ``surface`` (surface-only boolean mask)
    * ``semi_surface_index`` (index tuple for semi-surface cells)
    * ``beam_matrix`` (flattened beam flux for surface cells)
    * ``tau`` (residence-time coefficients, array or scalar)
    * ``D`` (diffusion coefficients, view or scalar)
    * ``acceleration_enabled`` (flag indicating active mode)
    """
    precursor: np.ndarray
    surface: np.ndarray
    semi_surface_index: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], None]
    beam_matrix: np.ndarray
    tau: Union[np.ndarray, float]
    D: Union[np.ndarray, float]
    acceleration_enabled: bool


@dataclass
class DiffusionView:
    """
    Everything needed for diffusion calculation.
    
    Key attributes:
    * ``surface_all`` (combined surface + semi-surface boolean array)
    * ``surface_all_index`` (fancy index tuple for diffusion-active cells)
    * ``D`` (diffusion coefficients, view or scalar)
    * ``acceleration_enabled`` (flag indicating active mode)
    """
    surface_all: np.ndarray
    surface_all_index: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], slice]
    D: Union[np.ndarray, float]
    acceleration_enabled: bool


@dataclass
class SurfaceUpdateView:
    """
    Everything needed for cell filling updates.
    
    This view always uses basic slicing (not fancy indexing) because cell filling
    operations need full neighborhood access.
    
    Key attributes:
    * ``deposit`` (3D view around irradiated area)
    * ``precursor`` (3D view around irradiated area)
    * ``surface`` (3D view around irradiated area)
    * ``semi_surface`` (3D view around irradiated area)
    * ``ghosts`` (3D view around irradiated area)
    * ``temp`` (3D view around irradiated area (if temperature tracking enabled))
    * ``surface_neighbors`` (3D view around irradiated area)
    * ``irradiated_area_3d`` (The slice object for converting local to global indices)
    """
    deposit: np.ndarray
    precursor: np.ndarray
    surface: np.ndarray
    semi_surface: np.ndarray
    ghosts: np.ndarray
    temp: Union[np.ndarray, None]
    surface_neighbors: np.ndarray
    irradiated_area_3d: slice

@dataclass
class TemperatureRecalcView:
    """
    Everything needed for temperature profile recalculation.
    """
    deposit: np.ndarray
    temp: np.ndarray
    temp_surface: np.ndarray
    D_temp: Union[np.ndarray, float]
    slice_no_sub: slice


class DataViewManager:
    """
    Manages the creation and caching of optimized numpy views and indices
    for the main simulation data arrays held in the Structure object.

    This class centralizes all performance-critical slicing and indexing logic, keeping
    the main simulation pipeline clean and focused on physics rather than
    on data indexing.

    The core premise of using views and indices is the locality of the modelled process
    and the sparsity of the data. In other words, for 3D structures represented by 3D arrays,
    data arrays are sparsely filled only to ~15%.
    """

    def __init__(self, state: SimulationState, acceleration_enabled: bool = True):
        """
        Initializes the DataViewManager.
        
        :param state: The simulation state containing structure and all data arrays.
        :type state: SimulationState
        :param acceleration_enabled: Master switch for acceleration grid optimization. If True, generates and caches fancy indices and flattened arrays. If False, uses full-array operations with np.s_[:] indexing.
        :type acceleration_enabled: bool
        """
        self.state = state
        self.acceleration_enabled: bool = acceleration_enabled

        # --- Private state for cached views and indices ---

        # Cached flattened arrays (only populated when acceleration_enabled=True)
        self.beam_matrix_surface_flat = None
        self.beam_matrix_deposition_flat = None

        # Cached slice objects (always computed)
        self._slice_irradiated_2d = None
        self._slice_irradiated_3d = None
        self._slice_irradiated_2d_no_sub = None

        # Cached index tuples (only populated when acceleration_enabled=True)
        self._index_deposition_2d: tuple = None
        self._index_deposition_3d: tuple = None
        self._index_surface_2d: tuple = None
        self._index_semi_surface_2d: tuple = None
        self._index_surface_all_2d: tuple = None
        self._index_surface_all_2d_prev: tuple = None

        # Metrics
        self.n_surface_cells: int = 0
        self.n_semi_surface_cells: int = 0
        self.n_beam_flux_points: int = 0

        # Initialize slices (always needed, regardless of acceleration mode)
        self._slice_irradiated_2d = self._define_irradiated_slice_2d()
        self._slice_irradiated_2d_no_sub = self._define_irradiated_slice_2d_no_sub()

        # Generate surface indices (ALWAYS needed for diffusion algorithm)
        surface_2d_view = self.structure.surface_bool[self._slice_irradiated_2d]
        semi_surface_2d_view = self.structure.semi_surface_bool[self._slice_irradiated_2d]

        self._index_surface_2d = self.get_index(surface_2d_view)
        self.n_surface_cells = self._index_surface_2d[0].size
        self._index_semi_surface_2d = self.get_index(semi_surface_2d_view)
        self.n_semi_surface_cells = self._index_semi_surface_2d[0].size
        self._index_surface_all_2d = concat_index(self._index_surface_2d, self._index_semi_surface_2d)
        self._surface_all[self._slice_irradiated_2d][self._index_surface_all_2d] = True
        self._index_surface_all_2d_prev = self._index_surface_all_2d

    def update_roi(self, beam_matrix: np.ndarray = None):
        """
        Compatibility wrapper for legacy callers.

        New code should call:
        1) ``update_after_cell_filling()``
        2) ``update_after_beam_matrix()``
        
        :param beam_matrix: The new secondary electron flux matrix. If None, uses beam_matrix from state.
        :type beam_matrix: np.ndarray, optional
        """
        if beam_matrix is not None:
            if beam_matrix.shape != self.structure.shape:
                raise ValueError("Beam matrix shape must match the structure shape.")
            self.beam_matrix[:] = beam_matrix

        self._recalculate_views_and_indices()

    def _recalculate_views_and_indices(self):
        """
        Private method to re-compute all slices and indices.
        This centralizes the update logic that was scattered across Process.py.
        Only generates fancy indices and flattened arrays when acceleration_enabled=True.
        """
        # 1. Calculate the 2D slice that covers the entire structure height (always needed)
        self._slice_irradiated_2d = self._define_irradiated_slice_2d()
        # 1a. Calculate the 2D slice that covers the entire structure height without substrate
        self._slice_irradiated_2d_no_sub = self._define_irradiated_slice_2d_no_sub()

        # 2. Generate surface-related indices (ALWAYS needed for diffusion algorithm)
        surface_2d_view = self.structure.surface_bool[self._slice_irradiated_2d]
        semi_surface_2d_view = self.structure.semi_surface_bool[self._slice_irradiated_2d]

        self._index_surface_2d = self.get_index(surface_2d_view)
        self._index_semi_surface_2d = self.get_index(semi_surface_2d_view)
        self._index_surface_all_2d_prev = self._index_surface_all_2d
        self._index_surface_all_2d = concat_index(self._index_surface_2d, self._index_semi_surface_2d)

        # Only generate deposition indices and flattened arrays if acceleration is enabled
        if self.acceleration_enabled:
            # 3. Get the view of the beam matrix within that 2D slice
            beam_matrix_2d_view = self.beam_matrix[self._slice_irradiated_2d]

            # 4. Find the indices of irradiated cells within the 2D view
            self._index_deposition_2d = self.get_index(beam_matrix_2d_view)

            # 5. Based on those indices, calculate the tighter 3D slice
            self._slice_irradiated_3d = self._define_irradiated_slice_3d()

            # 6. Transform the 2D deposition index into the coordinate system of the 3D slice
            self._index_deposition_3d = self._transform_deposition_index_to_3d()

            # 7. Get the flattened effective beam flux for deposition
            self.get_effective_beam_flux_for_deposition()
            # 8. Get the flattened beam flux for surface and semi-surface cells
            self.get_beam_flux_for_surface()
        else:
            # Acceleration disabled: compute simple 3D slice, no fancy indices for deposition
            self._slice_irradiated_3d = np.s_[self.substrate_height:self.max_z, :, :]

    # --- Slice and Index Definition Methods (formerly properties in Process) ---

    def _define_irradiated_slice_2d(self):
        """
        Defines a slice encapsulating the whole surface from just above the
        substrate to the max structure height.
        """
        return np.s_[self.substrate_height - 1:self.max_z, :, :]

    def _define_irradiated_slice_2d_no_sub(self):
        """
        Returns a slice encapsulating the whole surface without the substrate
        """
        return np.s_[self.substrate_height + 1:self.max_z, :, :]

    def _define_irradiated_slice_3d(self) -> slice:
        """
        Defines a tighter 3D slice of the currently irradiated area based on
        the beam matrix.
        """
        if self._index_deposition_2d[0].size == 0:
            # Handle case with no irradiation
            return np.s_[0:0, 0:0, 0:0]

        # Note: self._index_deposition_2d is in the coordinate system of the 2D slice view
        ymin, ymax = self._index_deposition_2d[1].min(), self._index_deposition_2d[1].max()
        xmin, xmax = self._index_deposition_2d[2].min(), self._index_deposition_2d[2].max()

        # The z-dimension comes from the full 2D slice
        z_slice = self._slice_irradiated_2d[0]
        return np.s_[z_slice, ymin:ymax + 1, xmin:xmax + 1]

    def _transform_deposition_index_to_3d(self) -> tuple:
        """
        Transforms the deposition indices from the 2D view's coordinate system
        to the 3D view's coordinate system.
        """
        if self._index_deposition_2d[0].size == 0:
            return np.array([]), np.array([]), np.array([])

        z, y, x = self._index_deposition_2d

        # The 3D slice starts at a y and x offset relative to the full array.
        # We need to subtract these offsets to get coordinates local to the 3D view.
        y_offset = self._slice_irradiated_3d[1].start
        x_offset = self._slice_irradiated_3d[2].start

        return z.copy(), y - y_offset, x - x_offset

    # --- Public API for Retrieving Views and Data ---
    def get_effective_beam_flux_for_deposition(self) -> np.ndarray:
        """
        Returns a flattened array of the beam flux values at the specific
        locations where deposition will occur.
        """
        beam_matrix_3d_view = self.beam_matrix[self._slice_irradiated_3d]
        self.beam_matrix_deposition_flat = beam_matrix_3d_view[self._index_deposition_3d]
        return self.beam_matrix_deposition_flat

    def get_beam_flux_for_surface(self) -> np.ndarray:
        """
        Returns a flattened array of the beam flux values for all surface
        and semi-surface cells.
        """
        # Get aligned views local and new beam matrix arrays
        beam_matrix_2d_view = self.beam_matrix[self._slice_irradiated_2d]
        beam_matrix_surface_2d_view = self.beam_matrix_surface[self._slice_irradiated_2d]
        # Clear previous values
        beam_matrix_surface_2d_view[:] = 0
        index_surface = self._index_surface_2d
        self.beam_matrix_surface_flat = beam_matrix_surface_2d_view[index_surface] = beam_matrix_2d_view[index_surface]
        return self.beam_matrix_surface_flat

    def get_index(self, view) -> tuple:
        """
        Returns the index tuple of nonzero elements in the specified view of a 3D array. C-friendly.
        """
        return cast_index_to_int(index_where(view))

    # --- New View Methods for Uniform Expression Approach ---

    def get_deposition_view(self) -> DepositionView:
        """
        Returns a DepositionView for deposition calculations.
        
        When acceleration_enabled=True:
            - Uses tight 3D bounding box around irradiated area
            - Returns fancy index tuple for sparse access
            - beam_matrix is 1D flattened array
        
        When acceleration_enabled=False:
            - Uses full 2D slice (substrate_height:max_z)
            - Returns np.s_[:] for full-array access
            - beam_matrix is full 3D array
        
        :return: DepositionView with appropriate arrays and indices for current mode.
        """
        if self.acceleration_enabled:
            # Acceleration ON: optimized with fancy indexing
            return DepositionView(
                deposit=self.structure.deposit[self._slice_irradiated_3d],
                precursor=self.structure.precursor[self._slice_irradiated_3d],
                beam_matrix=self.beam_matrix_deposition_flat,  # 1D flattened array
                index=self._index_deposition_3d,  # Fancy index tuple (z, y, x)
                acceleration_enabled=True
            )
        else:
            # Acceleration OFF: full-array operations
            slice_2d = self._slice_irradiated_2d
            return DepositionView(
                deposit=self.structure.deposit[slice_2d],
                precursor=self.structure.precursor[slice_2d],
                beam_matrix=self.beam_matrix[slice_2d],  # Full 3D array
                index=np.s_[:],  # slice(None) - select all
                acceleration_enabled=False
            )

    def get_precursor_density_view(self, temp_manager = None) -> PrecursorDensityView:
        """
        Returns a PrecursorDensityView for precursor density (RDE) calculations.
        
        Note: The `surface` attribute in this view is surface_bool (not surface_all).
        Semi-surface cells are handled separately via semi_surface_index.
        
        When acceleration_enabled=True:
            - Uses tight 2D bounding box around irradiated area
            - beam_matrix is 1D flattened array (for surface cells only)
            - tau is 1D flattened for surface cells ONLY (if temp tracking) or scalar
        
        When acceleration_enabled=False:
            - Uses full 2D slice (substrate_height:max_z)
            - beam_matrix is full 3D array
            - tau is full array (if temp tracking) or scalar
        
        :param temp_manager: If None, tau and D are set to None.
        :type temp_manager: TemperatureManager for temperature-dependent coefficients.
        
        :return: PrecursorDensityView with appropriate arrays for current mode.
        """
        # Get temperature-dependent coefficients from TemperatureManager
        if temp_manager is not None:
            # Get full arrays from temp_manager (raw data, no slicing)
            tau_full = temp_manager.get_tau()  # Scalar or full array
            D_full = temp_manager.get_D()      # Scalar or full array

            # Slice to irradiated region if arrays
            slice_2d = self._slice_irradiated_2d
            if isinstance(tau_full, np.ndarray):
                tau_2d = tau_full[slice_2d]
            else:
                tau_2d = tau_full  # Scalar

            if isinstance(D_full, np.ndarray):
                D_2d = D_full[slice_2d]
            else:
                D_2d = D_full  # Scalar

            # Flatten for acceleration mode if needed
            if self.acceleration_enabled and isinstance(tau_2d, np.ndarray):
                # Flatten tau to 1D for surface cells ONLY (not semi_surface)
                # CRITICAL: Must match precursor[surface] size in physics_engine._rk4_with_ftcs()
                tau = tau_2d[self.structure.surface_bool[slice_2d]]
            else:
                tau = tau_2d

            D = D_2d  # D stays 2D or scalar for RDE view
        else:
            # No temperature manager available.
            tau = None
            D = None

        # Build view
        slice_2d = self._slice_irradiated_2d
        precursor_2d = self.structure.precursor[slice_2d]
        surface = self.structure.surface_bool[slice_2d]
        semi_surface_index = self._index_semi_surface_2d

        return PrecursorDensityView(
            precursor=precursor_2d,
            surface=surface,
            semi_surface_index=semi_surface_index,
            beam_matrix=self.beam_matrix_surface_flat,  # 1D flattened array (surface cells only)
            tau=tau,
            D=D,
            acceleration_enabled=self.acceleration_enabled
        )

    def get_diffusion_view(self, temp_manager = None) -> DiffusionView:
        """
        Returns a DiffusionView for diffusion (FTCS) calculations.
        
        Note: surface_all_index is always a fancy index tuple because diffusion
        is evaluated only on surface+semi-surface cells.
        
        When acceleration_enabled=True:
            - Uses tight 2D bounding box around irradiated area
            - D is 2D view (if temp tracking) or scalar
        
        When acceleration_enabled=False:
            - Uses full 2D slice (substrate_height:max_z)
            - D is full array (if temp tracking) or scalar
        
        :param temp_manager: If None, D is set to None.
        :type temp_manager: TemperatureManager for temperature-dependent coefficients.
        
        :return: DiffusionView with appropriate arrays and surface indices.
        """
        slice_2d = self._slice_irradiated_2d
        surface_all = self._surface_all[slice_2d]

        # Get temperature-dependent diffusion coefficient
        if temp_manager is not None:
            D_full = temp_manager.get_D()  # Scalar or full array

            # Slice to irradiated region if array
            if isinstance(D_full, np.ndarray):
                D = D_full[slice_2d]
            else:
                D = D_full  # Scalar
        else:
            # No temperature manager available.
            D = None

        # ALWAYS use the actual fancy index tuple for surface cells (diffusion needs this)
        return DiffusionView(
            surface_all=surface_all,
            surface_all_index=self._index_surface_all_2d,  # Always fancy index tuple
            D=D,
            acceleration_enabled=self.acceleration_enabled
        )

    def get_surface_update_view(self) -> SurfaceUpdateView:
        """
        Returns a SurfaceUpdateView for cell filling updates.
        
        This view always uses basic slicing (not fancy indexing) because cell filling
        operations need full neighborhood access for the cellular automata.
        
        :return: SurfaceUpdateView with 3D views around the irradiated area.
        """
        # Always use the 3D slice (whether tight or full depends on acceleration mode)
        slice_3d = self._slice_irradiated_3d

        # Get temperature array if temperature tracking is enabled
        temp = self.state.structure.temperature
        return SurfaceUpdateView(
            deposit=self.structure.deposit[slice_3d],
            precursor=self.structure.precursor[slice_3d],
            surface=self.structure.surface_bool[slice_3d],
            semi_surface=self.structure.semi_surface_bool[slice_3d],
            ghosts=self.structure.ghosts_bool[slice_3d],
            temp=temp[slice_3d] if temp is not None else None,
            surface_neighbors=self.structure.surface_neighbors_bool[slice_3d],
            irradiated_area_3d=slice_3d  # For converting local to global indices
        )

    def get_temperature_recalc_view(self) -> TemperatureRecalcView:
        """
        Returns a TemperatureRecalcView for temperature profile recalculation.

            This view is used when temperature tracking is enabled and we need to
            recalculate the temperature profile after cell filling.
        """
        slice_2d_no_sub = self._slice_irradiated_2d_no_sub
        return TemperatureRecalcView(
            deposit=self.structure.deposit[slice_2d_no_sub],
            temp=self.structure.temperature[slice_2d_no_sub],
            temp_surface=self.state.temp_surface[slice_2d_no_sub],
            D_temp=self.state.D_temp[slice_2d_no_sub],  # D_temp is typically a scalar, but could be an array
            slice_no_sub=slice_2d_no_sub
        )

    def update_after_cell_filling(self, structure_extended=False):
        """
        Phase 1 update: Updates surface-related indices after cell filling.

        This method updates only the elements that depend on structure topology
        (max_z, surface_bool, semi_surface_bool). Does NOT touch beam-dependent
        indices or flattened arrays.

        Call this after:
        - cell_filled_routine() completes
        - extend_structure() completes

        Phase 1 updates:
        - Recalculate 2D slices (depend on max_z)
        - Regenerate surface indices (depend on surface_bool, semi_surface_bool)
        """

        # 1. Calculate the 2D slice that covers the entire structure height (always needed)
        self._slice_irradiated_2d = self._define_irradiated_slice_2d()
        # 1a. Calculate the 2D slice that covers the entire structure height without substrate
        self._slice_irradiated_2d_no_sub = self._define_irradiated_slice_2d_no_sub()

        # 2. Generate surface-related indices (ALWAYS needed for diffusion algorithm)
        surface_2d_view = self.structure.surface_bool[self._slice_irradiated_2d]
        semi_surface_2d_view = self.structure.semi_surface_bool[self._slice_irradiated_2d]

        # Sequence is important: regenerate surface and semi-surface first.
        self._index_surface_2d = self.get_index(surface_2d_view)
        self.n_surface_cells = self._index_surface_2d[0].size
        self._index_semi_surface_2d = self.get_index(semi_surface_2d_view)
        self.n_semi_surface_cells = self._index_semi_surface_2d[0].size
        # Then combine indexes for diffusion access.
        self._index_surface_all_2d = concat_index(self._index_surface_2d, self._index_semi_surface_2d)

    def update_after_beam_matrix(self):
        """
        Phase 2 update: Updates beam-dependent indices after MC simulation.

        This method updates only the elements that depend on beam_matrix pattern.
        Surface indices from Phase 1 are assumed to be already up-to-date.

        Call this after:
        - set_beam_matrix() completes (MC simulation finished)

        Phase 2 updates:
        - Regenerate deposition indices (depend on beam_matrix pattern)
        - Recalculate 3D slice (depends on deposition indices)
        - Regenerate flattened beam arrays
        """
        # 3. Get the view of the beam matrix within that 2D slice
        beam_matrix_2d_view = self.beam_matrix[self._slice_irradiated_2d]
        # 8. Get the flattened beam flux for surface and semi-surface cells
        self.get_beam_flux_for_surface()
        # Only generate deposition indices and flattened arrays if acceleration is enabled
        if self.acceleration_enabled:


            # 4. Find the indices of irradiated cells within the 2D view
            self._index_deposition_2d = self.get_index(beam_matrix_2d_view)
            self.n_beam_flux_points = self._index_deposition_2d[0].size

            # 5. Based on those indices, calculate the tighter 3D slice
            self._slice_irradiated_3d = self._define_irradiated_slice_3d()

            # 6. Transform the 2D deposition index into the coordinate system of the 3D slice
            self._index_deposition_3d = self._transform_deposition_index_to_3d()

            # 7. Get the flattened effective beam flux for deposition
            self.get_effective_beam_flux_for_deposition()

        else:
            # Acceleration disabled: compute simple 3D slice, no fancy indices for deposition
            self._slice_irradiated_3d = self._define_irradiated_slice_2d()
            self.n_beam_flux_points = np.count_nonzero(beam_matrix_2d_view)

    def get_surface_index(self):
        """
        Returns the current surface index tuple (fancy index for surface cells).
        """
        if self._index_surface_2d is None:
            raise ValueError("Surface index has not been generated yet. Call update_after_cell_filling() first.")
        return self._index_surface_2d

    def get_semi_surface_index(self):
        """
        Returns the current semi-surface index tuple (fancy index for semi-surface cells).
        """
        if self._index_semi_surface_2d is None:
            raise ValueError("Semi-surface index has not been generated yet. Call update_after_cell_filling() first.")
        return self._index_semi_surface_2d

    def get_surface_all_index(self):
        """
        Returns the current combined surface+semi-surface index tuple (fancy index for all surface cells).
        """
        if self._index_surface_all_2d is None:
            raise ValueError("Combined surface index has not been generated yet. Call update_after_cell_filling() first.")
        return self._index_surface_all_2d

    def get_full_active_volume_slice(self):
        """
        Returns the full active volume slice (from substrate_height to max_z) for 3D arrays.
        This is used when access to all surface is required.
        """
        return self._slice_irradiated_2d

    # --- Convenience properties for cleaner internal access ---
    @property
    def structure(self):
        """Convenience property to access structure from state."""
        return self.state.structure

    @property
    def beam_matrix(self):
        """Convenience property to access beam_matrix from state."""
        return self.state.beam_matrix

    @property
    def beam_matrix_surface(self):
        """Convenience property to access beam_matrix_surface from state."""
        return self.state.beam_matrix_surface

    @property
    def max_z(self):
        """Convenience property to access max_z from state."""
        return self.state.max_z

    @property
    def substrate_height(self):
        """Convenience property to access substrate_height from state."""
        return self.state.substrate_height

    @property
    def _surface_all(self):
        """Convenience property to access surface_all from state."""
        return self.state.surface_all
