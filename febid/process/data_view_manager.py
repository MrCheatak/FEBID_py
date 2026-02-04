import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple
from febid.slice_trics import index_where, get_index_in_parent, concat_index, cast_index_to_int
from febid.process.simulation_state import SimulationState


@dataclass
class DepositionView:
    """Everything needed for deposition calculation.

    Attributes:
        deposit: 3D array view (restricted if acceleration on, full if off)
        precursor: 3D array view (restricted if acceleration on, full if off)
        beam_matrix: 1D flattened array (acceleration on) or full 3D array (acceleration off)
        index: Fancy index tuple (acceleration on) or np.s_[:] (acceleration off)
        acceleration_enabled: Flag indicating which mode is active
    """
    deposit: np.ndarray
    precursor: np.ndarray
    beam_matrix: np.ndarray
    index: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], slice]
    acceleration_enabled: bool


@dataclass
class PrecursorDensityView:
    """Everything needed for precursor density calculation.

    Attributes:
        precursor: 2D array view (restricted if acceleration on, full if off)
        surface_all: Combined surface + semi_surface boolean array
        beam_matrix: 1D flattened array (acceleration on) or full 3D array (acceleration off)
        tau: 1D flattened (acceleration on) or full array or scalar (acceleration off)
        D: 2D view (restricted if acceleration on) or full array or scalar (acceleration off)
        acceleration_enabled: Flag indicating which mode is active
    """
    precursor: np.ndarray
    surface_all: np.ndarray
    beam_matrix: np.ndarray
    tau: Union[np.ndarray, float]
    D: Union[np.ndarray, float]
    acceleration_enabled: bool


@dataclass
class DiffusionView:
    """Everything needed for diffusion calculation.

    Attributes:
        precursor: 2D array view (restricted if acceleration on, full if off)
        surface_all: Combined surface + semi_surface boolean array
        surface_all_index: Fancy index tuple (acceleration on) or np.s_[:] (acceleration off)
        D: 2D view (restricted if acceleration on) or full array or scalar (acceleration off)
        acceleration_enabled: Flag indicating which mode is active
    """
    precursor: np.ndarray
    surface_all: np.ndarray
    surface_all_index: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], slice]
    D: Union[np.ndarray, float]
    acceleration_enabled: bool


@dataclass
class SurfaceUpdateView:
    """Everything needed for cell filling updates.

    This view always uses basic slicing (not fancy indexing) because cell filling
    operations need full neighborhood access.

    Attributes:
        deposit: 3D view around irradiated area
        precursor: 3D view around irradiated area
        surface: 3D view around irradiated area
        semi_surface: 3D view around irradiated area
        ghosts: 3D view around irradiated area
        temp: 3D view around irradiated area (if temperature tracking enabled)
        surface_neighbors: 3D view around irradiated area
        irradiated_area_3d: The slice object for converting local to global indices
    """
    deposit: np.ndarray
    precursor: np.ndarray
    surface: np.ndarray
    semi_surface: np.ndarray
    ghosts: np.ndarray
    temp: Union[np.ndarray, None]
    surface_neighbors: np.ndarray
    irradiated_area_3d: slice


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

        Args:
            state (SimulationState): The simulation state containing structure and all data arrays.
            acceleration_enabled (bool): Master switch for acceleration grid optimization.
                If True, generates and caches fancy indices and flattened arrays.
                If False, uses full-array operations with np.s_[:] indexing.
        """
        self.state = state
        self.acceleration_enabled: bool = acceleration_enabled

        # --- Private state for cached views and indices ---
        # These will be updated by the `update_roi` method.

        # Cached flattened arrays (only populated when acceleration_enabled=True)
        self.beam_matrix_surface = None
        self.beam_matrix_effective = None

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

        # Initialize slices (always needed, regardless of acceleration mode)
        self._slice_irradiated_2d = self._define_irradiated_slice_2d()
        self._slice_irradiated_2d_no_sub = self._define_irradiated_slice_2d_no_sub()

        # Generate surface indices (ALWAYS needed for diffusion algorithm)
        surface_2d_view = self.structure.surface_bool[self._slice_irradiated_2d]
        semi_surface_2d_view = self.structure.semi_surface_bool[self._slice_irradiated_2d]

        self._index_surface_2d = self.get_index(surface_2d_view)
        self._index_semi_surface_2d = self.get_index(semi_surface_2d_view)
        self._index_surface_all_2d = concat_index(self._index_surface_2d, self._index_semi_surface_2d)
        self._index_surface_all_2d_prev = self._index_surface_all_2d

    def update_roi(self, beam_matrix: np.ndarray = None):
        """
        Updates the internal state based on a new beam matrix. This is the
        primary trigger for re-calculating all dependent views and indices.

        Args:
            beam_matrix (np.ndarray, optional): The new secondary electron flux matrix.
                If None, uses beam_matrix from state.
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

        (Logic from Process._irradiated_area_2d)
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

        (Logic from Process.__irradiated_area_3d)
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

        (Logic from Process.__generate_deposition_index)
        """
        if self._index_deposition_2d[0].size == 0:
            return (np.array([]), np.array([]), np.array([]))

        z, y, x = self._index_deposition_2d

        # The 3D slice starts at a y and x offset relative to the full array.
        # We need to subtract these offsets to get coordinates local to the 3D view.
        y_offset = self._slice_irradiated_3d[1].start
        x_offset = self._slice_irradiated_3d[2].start

        return (z.copy(), y - y_offset, x - x_offset)

    # --- Public API for Retrieving Views and Data ---

    def get_view(self, array_name: str, view_type: str = '3d'):
        """
        Returns a sliced view of a specified data array from the Structure.

        Args:
            array_name (str): The name of the array in the Structure object
                              (e.g., 'deposit', 'precursor').
            view_type (str): The type of view, '2d' or '3d'.

        Returns:
            A numpy array view.
        """
        if not hasattr(self.structure, array_name):
            raise AttributeError(f"Structure has no attribute named '{array_name}'")

        source_array = getattr(self.structure, array_name)

        if view_type == '3d':
            return source_array[self._slice_irradiated_3d]
        elif view_type == '2d':
            return source_array[self._slice_irradiated_2d]
        else:
            raise ValueError(f"Unknown view_type: '{view_type}'. Must be '2d' or '3d'.")

    def get_effective_beam_flux_for_deposition(self) -> np.ndarray:
        """
        Returns a flattened array of the beam flux values at the specific
        locations where deposition will occur.

        (Logic from Process.__flatten_beam_matrix_effective)
        """
        beam_matrix_3d_view = self.beam_matrix[self._slice_irradiated_3d]
        self.beam_matrix_effective = beam_matrix_3d_view[self._index_deposition_3d]
        return self.beam_matrix_effective

    def get_beam_flux_for_surface(self) -> np.ndarray:
        """
        Returns a flattened array of the beam flux values for all surface
        and semi-surface cells.

        (Logic from Process.__flatten_beam_matrix_surface)
        """
        beam_matrix_2d_view = self.beam_matrix[self._slice_irradiated_2d]
        self.beam_matrix_surface = beam_matrix_2d_view[self._index_surface_all_2d]
        return self.beam_matrix_surface

    def get_surface_all_indices_2d(self) -> tuple:
        """Returns the combined indices for surface and semi-surface cells."""
        return self._index_surface_all_2d

    def get_deposition_indices_3d(self) -> tuple:
        """Returns the indices where deposition occurs, relative to the 3D view."""
        return self._index_deposition_3d

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

        Returns:
            DepositionView with appropriate arrays and indices for current mode.
        """
        if self.acceleration_enabled:
            # Acceleration ON: optimized with fancy indexing
            return DepositionView(
                deposit=self.structure.deposit[self._slice_irradiated_3d],
                precursor=self.structure.precursor[self._slice_irradiated_3d],
                beam_matrix=self.beam_matrix_effective,  # 1D flattened array
                index=self._index_deposition_3d,  # Fancy index tuple (z, y, x)
                acceleration_enabled=True
            )
        else:
            # Acceleration OFF: full-array operations
            slice_2d = np.s_[self.substrate_height:self.max_z, :, :]
            return DepositionView(
                deposit=self.structure.deposit[slice_2d],
                precursor=self.structure.precursor[slice_2d],
                beam_matrix=self.beam_matrix[slice_2d],  # Full 3D array
                index=np.s_[:],  # slice(None) - select all
                acceleration_enabled=False
            )

    def get_precursor_density_view(self) -> PrecursorDensityView:
        """
        Returns a PrecursorDensityView for precursor density (RDE) calculations.

        Note: This view uses surface_all boolean array, NOT surface_all_index.
        The index tuple is only used in diffusion calculations.

        When acceleration_enabled=True:
            - Uses tight 2D bounding box around irradiated area
            - beam_matrix is 1D flattened array (for surface cells only)
            - tau is 1D flattened (if temp tracking) or scalar

        When acceleration_enabled=False:
            - Uses full 2D slice (substrate_height:max_z)
            - beam_matrix is full 3D array
            - tau is full array (if temp tracking) or scalar

        Returns:
            PrecursorDensityView with appropriate arrays for current mode.
        """
        if self.acceleration_enabled:
            # Acceleration ON: optimized with tight slicing
            precursor_2d = self.structure.precursor[self._slice_irradiated_2d]
            surface_2d = self.structure.surface_bool[self._slice_irradiated_2d]
            semi_surface_2d = self.structure.semi_surface_bool[self._slice_irradiated_2d]
            surface_all = surface_2d | semi_surface_2d

            # Get tau: check if temperature tracking is enabled
            if self.state.tau_temp.any():
                # Temperature-dependent: use flattened tau array (for surface cells only)
                tau = self.state.tau_temp[self._slice_irradiated_2d][self._index_surface_all_2d]
            else:
                # Constant tau from precursor parameters
                tau = self.state.precursor.tau

            # Get D: check if temperature tracking is enabled
            if self.state.D_temp.any():
                # Temperature-dependent: use 2D view
                D = self.state.D_temp[self._slice_irradiated_2d]
            else:
                # Constant D from precursor parameters
                D = self.state.precursor.D

            return PrecursorDensityView(
                precursor=precursor_2d,
                surface_all=surface_all,
                beam_matrix=self.beam_matrix_surface,  # 1D flattened array (surface cells only)
                tau=tau,
                D=D,
                acceleration_enabled=True
            )
        else:
            # Acceleration OFF: full-array operations
            slice_2d = np.s_[self.substrate_height:self.max_z, :, :]
            precursor_2d = self.structure.precursor[slice_2d]
            surface_2d = self.structure.surface_bool[slice_2d]
            semi_surface_2d = self.structure.semi_surface_bool[slice_2d]
            surface_all = surface_2d | semi_surface_2d

            # Get tau: check if temperature tracking is enabled
            if self.state.tau_temp.any():
                tau = self.state.tau_temp[slice_2d]
            else:
                tau = self.state.precursor.tau

            # Get D: check if temperature tracking is enabled
            if self.state.D_temp.any():
                D = self.state.D_temp[slice_2d]
            else:
                D = self.state.precursor.D

            return PrecursorDensityView(
                precursor=precursor_2d,
                surface_all=surface_all,
                beam_matrix=self.beam_matrix[slice_2d],  # Full 3D array
                tau=tau,
                D=D,
                acceleration_enabled=False
            )

    def get_diffusion_view(self) -> DiffusionView:
        """
        Returns a DiffusionView for diffusion (FTCS) calculations.

        Note: surface_all_index is ALWAYS a fancy index tuple (never np.s_[:])
        because the diffusion algorithm needs to identify surface cells
        regardless of acceleration mode.

        When acceleration_enabled=True:
            - Uses tight 2D bounding box around irradiated area
            - D is 2D view (if temp tracking) or scalar

        When acceleration_enabled=False:
            - Uses full 2D slice (substrate_height:max_z)
            - D is full array (if temp tracking) or scalar

        Returns:
            DiffusionView with appropriate arrays and surface indices.
        """
        slice_2d = self._slice_irradiated_2d if self.acceleration_enabled else np.s_[self.substrate_height:self.max_z, :, :]

        precursor_2d = self.structure.precursor[slice_2d]
        surface_2d = self.structure.surface_bool[slice_2d]
        semi_surface_2d = self.structure.semi_surface_bool[slice_2d]
        surface_all = surface_2d | semi_surface_2d

        # Get D: check if temperature tracking is enabled
        if self.state.D_temp.any():
            D = self.state.D_temp[slice_2d]
        else:
            D = self.state.precursor.D

        # ALWAYS use the actual fancy index tuple for surface cells (diffusion needs this)
        return DiffusionView(
            precursor=precursor_2d,
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

        Returns:
            SurfaceUpdateView with 3D views around the irradiated area.
        """
        # Always use the 3D slice (whether tight or full depends on acceleration mode)
        slice_3d = self._slice_irradiated_3d

        # Get temperature array if temperature tracking is enabled
        temp = self.state.surface_temp if hasattr(self.state, 'surface_temp') else None

        return SurfaceUpdateView(
            deposit=self.structure.deposit[slice_3d],
            precursor=self.structure.precursor[slice_3d],
            surface=self.structure.surface_bool[slice_3d],
            semi_surface=self.structure.semi_surface_bool[slice_3d],
            ghosts=self.structure.ghost_cells[slice_3d],
            temp=temp[slice_3d] if temp is not None else None,
            surface_neighbors=self.structure.surface_neighbors[slice_3d],
            irradiated_area_3d=slice_3d  # For converting local to global indices
        )

    def invalidate_and_rebuild(self):
        """
        Invalidates and rebuilds all cached indices and flattened arrays.

        This should be called when the structure changes:
        - After cell filling (structure topology changed)
        - After beam matrix update (irradiated regions changed)
        - After structure extension (dimensions changed)

        If acceleration_enabled=False, this is essentially a no-op
        (only updates slices, skips expensive index generation).
        """
        self._recalculate_views_and_indices()

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
    def max_z(self):
        """Convenience property to access max_z from state."""
        return self.state.max_z

    @property
    def substrate_height(self):
        """Convenience property to access substrate_height from state."""
        return self.state.substrate_height