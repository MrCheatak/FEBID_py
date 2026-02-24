"""
GPUFacade - GPU-accelerated physics calculations

This module provides GPU-based implementations of physics calculations
that mirror the PhysicsEngine interface for seamless CPU/GPU switching.

Stage 4 of Process refactoring.
"""

import numpy as np
from febid.process.simulation_state import SimulationState
from febid.process.data_view_manager import DataViewManager
from febid.thermal.temperature_manager import TemperatureManager
from febid.kernel_modules import GPU
from febid.logging_config import setup_logger

logger = setup_logger(__name__)


class GPUFacade:
    """
    GPU-accelerated physics engine for FEBID simulation.

    Mirrors PhysicsEngine interface while delegating to OpenCL kernels.

    Design principles:
    - Same interface as PhysicsEngine (compute_deposition, compute_precursor_density, check_cells_filled)
    - Manages GPU memory and data transfers internally
    - Acceleration-agnostic: receives same view structures as CPU path
    - Temperature-integrated: gets tau(T), D(T) from TemperatureManager
    """

    def __init__(self, state: SimulationState, view_manager: DataViewManager,
                 temp_manager: TemperatureManager, device=None):
        """
        Initialize GPU facade with OpenCL context.

        Parameters
        ----------
        state : SimulationState
            Read-only access to simulation data
        view_manager : DataViewManager
            Provides optimized views and indices (not directly used by GPU)
        temp_manager : TemperatureManager
            Provides temperature-dependent coefficients
        device : optional
            GPU device specification (passed to GPU class)
        """
        self.state = state
        self.view_manager = view_manager
        self.temp_manager = temp_manager

        # Create GPU context and compile kernels
        self.knl = GPU(device)

        # Initialize GPU buffers and upload structure data
        self.initialize_kernels()

        # Track cell filling flag
        self._filled_flag = False

    # PUBLIC INTERFACE (mirrors PhysicsEngine)
    def compute_deposition(self, dt: float) -> None:
        """
        GPU-accelerated deposition calculation.

        Mirrors PhysicsEngine.compute_deposition() interface.

        This operation is BLOCKING because we need the filled cell flag immediately after to update the surface.

        Parameters
        ----------
        dt : float
            Time step in seconds
        """
        # Calculate constant (same as CPU version)
        const = (self.state.precursor.sigma * self.state.precursor.V * dt * 1e6 *
                 self.state.deposition_scaling / self.state.cell_V *
                 self.state.cell_size ** 2)

        # Execute GPU kernel
        flag = self.knl.deposit_gpu(const, blocking=True)

        # Note: Results stay on GPU until explicit retrieval
        # Filled cell flag is returned for immediate checking
        self._filled_flag = bool(flag)

    def compute_precursor_density(self, dt: float) -> None:
        """
        GPU-accelerated RDE solver.

        Mirrors PhysicsEngine.compute_precursor_density() interface.

        This operation is non-BLOCKING. Other routines are safe to run while the GPU computes the new precursor density.

        Parameters
        ----------
        dt : float
            Time step in seconds
        """
        # Get temperature-dependent coefficients
        D = self.temp_manager.get_D()
        tau = self.temp_manager.get_tau()

        # Execute GPU RK4 kernel sequence (includes FTCS diffusion at each RK stage)
        self.knl.precur_den_gpu(
            dt=dt,
            D=D,
            F=self.state.precursor.F,
            n0=self.state.precursor.n0,
            tau=tau,
            sigma=self.state.precursor.sigma,
            cell_size=self.state.cell_size,
            blocking=True
        )

    def check_cells_filled(self) -> bool:
        """
        Check if any cells filled on GPU.

        Mirrors PhysicsEngine.check_cells_filled() interface.

        Returns
        -------
        bool
            True if at least one cell is filled (deposit >= 1.0)
        """
        # GPU kernel already set flag during deposition
        return self._filled_flag

    def equilibrate(self, dt: float, max_it: int = 10000, eps: float = 1e-8) -> int:
        """
        Bring precursor to steady state using GPU.

        Parameters
        ----------
        dt : float
            Time step for equilibration
        max_it : int, optional
            Maximum iterations (default: 10000)
        eps : float, optional
            Convergence threshold (default: 1e-8)

        Returns
        -------
        int
            Number of iterations performed
        """
        logger.info("Equilibrating precursor density on GPU...")

        # Get initial precursor state
        precursor_old = self.retrieve_array('precursor', blocking=True).copy()

        for i in range(max_it):
            # Update precursor density
            self.compute_precursor_density(dt)

            # Check convergence every 100 iterations
            if i % 100 == 0:
                precursor_new = self.retrieve_array('precursor', blocking=True)
                delta = np.abs(precursor_new - precursor_old).max()
                precursor_old = precursor_new.copy()

                if delta < eps:
                    logger.info(f"Equilibration converged after {i} iterations (delta={delta:.2e})")
                    return i

        logger.warning(f"Equilibration did not converge after {max_it} iterations")
        return max_it

    # GPU-SPECIFIC OPERATIONS (internal)
    def initialize_kernels(self) -> None:
        """
        Load structure data to GPU buffers.

        This method extracts necessary data from the state and uploads it to the GPU.
        """
        # Get irradiated area indices (needed by GPU)
        irr_ind_2d = self._get_irradiated_indices()

        # Get surface_all array (combines surface and semi_surface)
        surface_all = np.logical_or(self.state.structure.surface_bool,
                                    self.state.structure.semi_surface_bool)

        # Load flattened structure arrays to GPU
        self.knl.load_structure(
            self.state.structure.precursor,
            self.state.structure.deposit,
            surface_all,
            self.state.structure.surface_bool,
            self.state.structure.semi_surface_bool,
            self.state.structure.ghosts_bool,
            irr_ind_2d
        )

    def upload_structure(self, blocking: bool = True) -> None:
        """
        Upload full structure to GPU.

        Parameters
        ----------
        blocking : bool, optional
            Wait for operation to complete (default: True)
        """
        # Get irradiated area indices
        irr_ind_2d = self._get_irradiated_indices()

        # Get surface_all array
        surface_all = np.logical_or(self.state.structure.surface_bool,
                                    self.state.structure.semi_surface_bool)

        # Update all structure arrays on GPU
        self.knl.update_structure(
            self.state.structure.precursor,
            self.state.structure.deposit,
            surface_all,
            self.state.structure.surface_bool,
            self.state.structure.semi_surface_bool,
            self.state.structure.ghosts_bool,
            irr_ind_2d,
            cells=None,
            blocking=blocking
        )

    def update_structure_partial(self, cells=None, blocking: bool = True) -> None:
        """
        Upload only changed cells to GPU.

        Parameters
        ----------
        cells : list of tuples, optional
            List of (z, y, x) cell indices to update. If None, updates all.
        blocking : bool, optional
            Wait for operation to complete (default: True)
        """
        # Get irradiated area indices
        irr_ind_2d = self._get_irradiated_indices()

        # Get surface_all array
        surface_all = np.logical_or(self.state.structure.surface_bool,
                                    self.state.structure.semi_surface_bool)

        # Update structure arrays on GPU (partial if cells provided)
        self.knl.update_structure(
            self.state.structure.precursor,
            self.state.structure.deposit,
            surface_all,
            self.state.structure.surface_bool,
            self.state.structure.semi_surface_bool,
            self.state.structure.ghosts_bool,
            irr_ind_2d,
            cells=cells,
            blocking=blocking
        )

    def retrieve_structure(self, blocking: bool = True) -> dict:
        """
        Retrieve all arrays from GPU.

        Parameters
        ----------
        blocking : bool, optional
            Wait for operation to complete (default: True)

        Returns
        -------
        dict
            Dictionary of array names to numpy arrays
        """
        retrieved = self.knl.get_updated_structure(blocking)

        # Update structure arrays in-place
        data_dict = self.state.structure.data_dict
        names_retrieved = set(retrieved.keys())
        names_local = set(data_dict.keys())
        names = set.intersection(names_retrieved, names_local)

        if len(names) == 0:
            raise ValueError('Got no common arrays to retrieve from GPU!')

        for name in names:
            try:
                data_dict[name][...] = retrieved[name]
            except KeyError as e:
                logger.error(f'Got an unknown array name in Structure from GPU kernel: {name}')
                raise e

        return retrieved

    def retrieve_array(self, name: str, blocking: bool = True) -> np.ndarray:
        """
        Retrieve single array from GPU.

        Parameters
        ----------
        name : str
            Name of the array to retrieve ('deposit', 'precursor', 'surface_bool', etc.)
        blocking : bool, optional
            Wait for operation to complete (default: True)

        Returns
        -------
        np.ndarray
            Retrieved array with restored 3D shape
        """
        data_dict = self.state.structure.data_dict

        if name not in data_dict:
            raise ValueError(f'Array name "{name}" is not present in Structure.')

        # Retrieve from GPU
        array = self.knl.get_structure_partial(name, blocking).reshape(self.state.structure.shape)

        # Update structure array in-place
        data_dict[name][...] = array

        return array

    def retrieve_for_visualization(self, stats_gathering: bool = False, displayed_data: str = None) -> None:
        """
        Selective retrieval for stats and visualization.

        Only retrieves arrays that are actually needed, minimizing data transfer overhead.

        Parameters
        ----------
        stats_gathering : bool, optional
            Whether statistics gathering is active (default: False)
        displayed_data : str, optional
            Name of data array being displayed (default: None)
        """
        necessary_data = []

        # Determine what to retrieve based on active features
        if stats_gathering:
            necessary_data += ['precursor', 'deposit']

        if displayed_data is not None:
            necessary_data += [displayed_data]

        # Remove duplicates
        necessary_data = set(necessary_data)

        # Retrieve each array
        for data_name in necessary_data:
            self.retrieve_array(data_name)

    def set_beam_matrix(self, beam_matrix: np.ndarray, blocking: bool = True) -> None:
        """
        Update beam matrix on GPU.

        Parameters
        ----------
        beam_matrix : np.ndarray
            Secondary electron flux matrix
        blocking : bool, optional
            Wait for operation to complete (default: True)
        """
        # Check if beam matrix buffer exists - if not, load for first time
        if not hasattr(self.knl, 'beam_matrix_buf'):
            self.knl.load_beam_matrix(beam_matrix, blocking=blocking)
        else:
            self.knl.update_beam_matrix(beam_matrix, blocking=blocking)

    def reload_beam_matrix(self, beam_matrix: np.ndarray, blocking: bool = True) -> None:
        """
        Reload beam matrix with new size (after structure resize).

        Parameters
        ----------
        beam_matrix : np.ndarray
            Secondary electron flux matrix with new shape
        blocking : bool, optional
            Wait for operation to complete (default: True)
        """
        self.knl.reload_beam_matrix(beam_matrix, blocking=blocking)

    def release_buffers(self) -> None:
        """
        Release all GPU buffers.

        Used before structure resize to free GPU memory.
        """
        self.knl._GPU__release_all_buffers()

    def reinitialize_after_resize(self) -> None:
        """
        Reinitialize GPU after structure resize.

        This releases old buffers and creates new ones with the new structure size.
        """
        # Release old buffers
        self.release_buffers()

        # Reinitialize with new size
        self.initialize_kernels()

        # Reload beam matrix with new size
        self.set_beam_matrix(self.state.beam_matrix)

    def synchronize(self) -> None:
        """
        Wait for all GPU operations to finish.

        Useful for synchronization before retrieving data or updating the surface.
        """
        self.knl.queue.finish()

    def get_dimensions(self) -> tuple:
        """
        Get GPU grid dimensions.

        Returns
        -------
        tuple
            (zdim, ydim, xdim) dimensions of the GPU grid
        """
        return (self.knl.zdim, self.knl.ydim, self.knl.xdim)

    def set_zdim_max(self, zdim_max: int) -> None:
        """
        Set maximum z dimension and update dependent GPU parameters.

        Parameters
        ----------
        zdim_max : int
            New maximum z dimension
        """
        self.knl.zdim_max = zdim_max
        self.knl.len_lap = (zdim_max - self.knl.zdim_min) * self.knl.xdim * self.knl.ydim

    # INTERNAL HELPER METHODS
    def _get_irradiated_indices(self) -> np.ndarray:
        """
        Get 2D indices for irradiated area along z-axis.

        Returns
        -------
        np.ndarray
            2D index array for GPU [start_z, end_z]
        """
        # Extract z-range from irradiated area slice
        # irradiated_area_2D is typically np.s_[z_start:z_end, :, :]
        z_slice = self.state.substrate_height - 1
        z_max = self.state.max_z

        irr_ind_2d = np.array([z_slice, z_max], dtype=np.int32)
        return irr_ind_2d

    def return_beam_matrix(self, blocking: bool = True) -> np.ndarray:
        """
        Get beam matrix from GPU.

        This is used during cell filling to detect filled cells.

        Parameters
        ----------
        blocking : bool, optional
            Wait for operation to complete (default: True)

        Returns
        -------
        np.ndarray
            Beam matrix array (filled cells marked with -1)
        """
        return self.knl.return_beam_matrix(blocking=blocking)

    def return_slice(self, index: np.ndarray, index_shape: int):
        """
        Return slice of arrays from GPU for surface neighbor updates.

        Parameters
        ----------
        index : np.ndarray
            Flattened 3D indices
        index_shape : int
            Number of elements in the slice

        Returns
        -------
        tuple
            (deposit_slice, surface_slice) arrays
        """
        return self.knl.return_slice(index, index_shape)

    def update_surface(self, full_cells: np.ndarray) -> None:
        """
        Update surface on GPU after cells filled.

        Parameters
        ----------
        full_cells : np.ndarray
            Array of filled cell indices
        """
        self.knl.update_surface(full_cells)

    def index_1d_to_3d(self, index_1d: int) -> tuple:
        """
        Convert 1D flattened index to 3D (z, y, x) coordinates.

        Parameters
        ----------
        index_1d : int
            Flattened index

        Returns
        -------
        tuple
            (z, y, x) coordinates
        """
        return self.knl.index_1d_to_3d(index_1d)
