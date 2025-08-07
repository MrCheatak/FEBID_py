import numpy as np
from febid.Structure import Structure
from febid.slice_trics import index_where, get_index_in_parent, concat_index, cast_index_to_int


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

    def __init__(self, structure: Structure, max_z: int):
        """
        Initializes the DataViewManager.

        Args:
            structure (Structure): The main data container for the simulation.
        """
        self.structure: Structure = structure

        # --- Private state for cached views and indices ---
        # These will be updated by the `update_for_beam_matrix` method.
        self._beam_matrix: np.ndarray = np.zeros_like(self.structure.deposit, dtype=np.int32)
        self.beam_matrix_surface = None
        self.beam_matrix_effective = None

        # Cached slice objects
        self._slice_irradiated_2d = None
        self._slice_irradiated_3d = None

        # Cached index tuples
        self._index_deposition_2d: tuple = None
        self._index_deposition_3d: tuple = None
        self._index_surface_2d: tuple = None
        self._index_semi_surface_2d: tuple = None
        self._index_surface_all_2d: tuple = None
        self._index_surface_all_2d_prev: tuple = None

        self._max_z = max_z

        # 1. Calculate the 2D slice that covers the entire structure height
        self._slice_irradiated_2d = self._define_irradiated_slice_2d()
        self._slice_irradiated_2d_no_sub = self._define_irradiated_slice_2d_no_sub()
        # 6. Generate surface-related indices based on the 2D slice
        surface_2d_view = self.structure.surface_bool[self._slice_irradiated_2d]
        semi_surface_2d_view = self.structure.semi_surface_bool[self._slice_irradiated_2d]

        self._index_surface_2d = self.get_index(surface_2d_view)
        self._index_semi_surface_2d = self.get_index(semi_surface_2d_view)
        self._index_surface_all_2d = concat_index(self._index_surface_2d, self._index_semi_surface_2d)
        self._index_surface_all_2d_prev = self._index_surface_all_2d

    def update_roi(self, beam_matrix: np.ndarray, max_z: int = None):
        """
        Updates the internal state based on a new beam matrix. This is the
        primary trigger for re-calculating all dependent views and indices.

        Args:
            beam_matrix (np.ndarray): The new secondary electron flux matrix.
        """
        if beam_matrix.shape != self.structure.shape:
            raise ValueError("Beam matrix shape must match the structure shape.")

        if max_z:
            self._max_z = max_z
        else:
            # If max_z is not provided, calculate it from the structure's deposit
            non_zero_indices = self.structure.deposit.nonzero()[0]
            self._max_z = non_zero_indices.max() + 3 if non_zero_indices.size > 0 else self.structure.substrate_height + 3
        self._beam_matrix = beam_matrix
        self._recalculate_views_and_indices()

    def _recalculate_views_and_indices(self):
        """
        Private method to re-compute all slices and indices.
        This centralizes the update logic that was scattered across Process.py.
        """
        # 1. Calculate the 2D slice that covers the entire structure height
        self._slice_irradiated_2d = self._define_irradiated_slice_2d()
        # 1a. Calculate the 2D slice that covers the entire structure height without substrate
        self._slice_irradiated_2d_no_sub = self._define_irradiated_slice_2d_no_sub()

        # 2. Get the view of the beam matrix within that 2D slice
        beam_matrix_2d_view = self._beam_matrix[self._slice_irradiated_2d]

        # 3. Find the indices of irradiated cells within the 2D view
        self._index_deposition_2d = self.get_index(beam_matrix_2d_view)

        # 4. Based on those indices, calculate the tighter 3D slice
        self._slice_irradiated_3d = self._define_irradiated_slice_3d()

        # 5. Transform the 2D deposition index into the coordinate system of the 3D slice
        self._index_deposition_3d = self._transform_deposition_index_to_3d()

        # 6. Generate surface-related indices based on the 2D slice
        surface_2d_view = self.structure.surface_bool[self._slice_irradiated_2d]
        semi_surface_2d_view = self.structure.semi_surface_bool[self._slice_irradiated_2d]

        self._index_surface_2d = self.get_index(surface_2d_view)
        self._index_semi_surface_2d = self.get_index(semi_surface_2d_view)
        self._index_surface_all_2d_prev = self._index_surface_all_2d
        self._index_surface_all_2d = concat_index(self._index_surface_2d, self._index_semi_surface_2d)

        # 7. Get the flattened effective beam flux for deposition
        self.get_effective_beam_flux_for_deposition()
        # 8. Get the flattened beam flux for surface and semi-surface cells
        self.get_beam_flux_for_surface()

    # --- Slice and Index Definition Methods (formerly properties in Process) ---

    def _define_irradiated_slice_2d(self):
        """
        Defines a slice encapsulating the whole surface from just above the
        substrate to the max structure height.

        (Logic from Process._irradiated_area_2d)
        """
        return np.s_[self.structure.substrate_height - 1:self._max_z, :, :]

    def _define_irradiated_slice_2d_no_sub(self):
        """
        Returns a slice encapsulating the whole surface without the substrate
        """
        return np.s_[self.structure.substrate_height + 1:self._max_z, :, :]

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
        beam_matrix_3d_view = self._beam_matrix[self._slice_irradiated_3d]
        self.beam_matrix_effective = beam_matrix_3d_view[self._index_deposition_3d]
        return self.beam_matrix_effective

    def get_beam_flux_for_surface(self) -> np.ndarray:
        """
        Returns a flattened array of the beam flux values for all surface
        and semi-surface cells.

        (Logic from Process.__flatten_beam_matrix_surface)
        """
        beam_matrix_2d_view = self._beam_matrix[self._slice_irradiated_2d]
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