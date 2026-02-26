# Default packages

# Core packages
import warnings
from threading import Lock
import numpy as np

# Local packages
from febid.Structure import Structure
from febid.continuum_model_base import ContinuumModel
import febid.diffusion as diffusion
from febid.mlcca import MultiLayerdCellCellularAutomata as MLCCA
from .slice_trics import get_3d_slice, get_index_in_parent, index_where
from .process.simulation_state import SimulationState
from .process.data_view_manager import DataViewManager, DepositionView, SurfaceUpdateView
from .process.physics_engine import PhysicsEngine
from .process.simulation_stats import SimulationStats
from febid.thermal.temperature_manager import TemperatureManager
from febid.process.gpu_facade import GPUFacade
from febid.logging_config import setup_logger
# Setup logger
logger = setup_logger(__name__)

# Diffusion is computed with stencil updates on precursor fields.
# Ghost-shell arrays are still maintained for topology bookkeeping and visualization.
class Process:
    """
    Class representing the core deposition process.
    It contains all necessary arrays, variables, parameters and methods to construct a continuous deposition process.
    """

    # A note_ to value correspondence:
    # The main reaction equation operates in absolute values of precursor density per nanometer.
    # The precursor array stores values as such.
    # The deposited volume is though calculated and stored as a fraction of the cell's volume.
    # Thus, the precursor density has to be multiplied by the cell's base area divided by the cell volume.

    def __init__(self, structure: Structure, equation_values, deposition_scaling=1, temp_tracking=True,
                 acceleration_enabled=True, device=None, n_init="full", name=None):
        """Initialize process state, managers, and optional GPU acceleration.

        :param structure: Simulation structure with geometry and field arrays.
        :type structure: Structure
        :param equation_values: Precomputed model constants and process settings.
        :type equation_values: dict
        :param deposition_scaling: Scaling factor between simulated and physical deposition time.
        :type deposition_scaling: float
        :param temp_tracking: Enable temperature-dependent physics updates.
        :type temp_tracking: bool
        :param acceleration_enabled: Enable accelerated array indexing paths.
        :type acceleration_enabled: bool
        :param device: OpenCL device identifier for GPU execution.
        :type device: object
        :param n_init: Initial precursor coverage mode or scalar value.
        :type n_init: str
        :param name: Optional process identifier.
        :type name: str
        :return: None
        """
        super().__init__()
        if not name:
            self.name = str(np.random.randint(000000, 999999, 1)[0])
        else:
            self.name = name

        # Create simulation state (will be fully populated in __set_structure and __set_constants)
        self.state = SimulationState(
            structure=structure,
            model=ContinuumModel(),
            heat_cond=equation_values.get('heat_cond', 0),
            room_temp=294
        )
        self.state.deposition_scaling = deposition_scaling
        self.state.temperature_tracking = temp_tracking  # Set temperature tracking flag in state

        # Main arrays
        # Semi-surface cells are virtual cells that can hold precursor density value, but cannot create deposit.
        # Their role is to serve as pipes on steps, where regular surface cells are not in contact.
        # Therefore, these cells take part in diffusion process, but are not taken into account when calculating
        # other terms in the FEBID equation or deposit increment.

        # Cellular automata engine
        self._local_mcca: MLCCA = MLCCA()

        # Timings
        self.t = 0
        self._dt = None
        self.__forced_dt = False

        # Accuracy
        self.solution_accuracy = 1e-8

        # Utility variables
        self.redraw = True  # flag for external functions saying that surface has been updated
        self.request_temp_recalc = False
        self._temp_step = 10000  # amount of volume to be deposited before next temperature calculation
        self._temp_step_cells = 0  # number of cells to be filled before next temperature calculation
        self._temp_calc_count = 0  # counting number of times temperature has been calculated

        # Statistics
        self.filled_cells = 0  # current number of filled cells
        self.max_T = 0  # cached maximum temperature for reporting
        self._stats_frequency = 1e-3  # s, default calculation of stats and offloading from GPU for visualisation
        self.x0 = 0
        self.y0 = 0
        self.full_cells = None  # indices of the filled cells, used for beam matrix update
        self.last_full_cells = None # indices of the last filled cells
        self.n_surface_cells = 0  # current number of surface cells
        self.n_semi_surface_cells = 0  # current number of semi-surface cells
        self.n_beam_matrix_points = 0  # current number of cells with non-zero electron beam flux

        self.lock = Lock()
        self.device = device

        self.displayed_data = None
        self.stats_gathering = None

        # Initialization sequence
        self.__set_structure(structure)
        self.__set_constants(equation_values)

        # Initialize view manager
        self.view_manager = DataViewManager(self.state, acceleration_enabled=acceleration_enabled)

        # Initialize temperature manager
        self.temp_manager = TemperatureManager(self.state, self.view_manager)

        # Initialize simulation statistics
        stats_enabled = self.stats_gathering if self.stats_gathering is not None else True
        stats_freq = getattr(self, '_stats_frequency', 1e-3)
        self.stats = SimulationStats(
            state=self.state,
            temp_manager=self.temp_manager,
            gathering_enabled=stats_enabled,
            stats_frequency=stats_freq
        )

        # Initialize CPU physics engine
        self.physics_engine = PhysicsEngine(self.state, self.view_manager, self.temp_manager)

        # Initialize GPU facade when device is available
        if device:
            self.gpu_facade = GPUFacade(self.state, self.view_manager, self.temp_manager, device)
        else:
            self.gpu_facade = None

        self._temp_step_cells = self._temp_step / self.state.cell_V
        if n_init == "full":
            self.state.structure.precursor[self.state.structure.surface_bool] = self.state.model.nr
            self.state.structure.precursor[self.state.structure.semi_surface_bool] = self.state.model.nr
        elif n_init == "empty":
            self.state.structure.precursor[self.state.structure.surface_bool] = 1e-6
            self.state.structure.precursor[self.state.structure.semi_surface_bool] = 1e-6
        elif n_init is float:
            self.state.structure.precursor[self.state.structure.surface_bool] = n_init
            self.state.structure.precursor[self.state.structure.semi_surface_bool] = n_init

        # Initialize temperature if tracking enabled
        if self.state.temperature_tracking:
            self.temp_manager.update_full()

        # GPU buffers were initialized before n_init assignment above.
        # Refresh GPU-side arrays so deposition starts from the same initial precursor field as CPU.
        if self.device and self.gpu_facade is not None:
            self.gpu_facade.upload_structure(blocking=True)

    # Initialization methods
    def __set_structure(self, structure: Structure):
        """Initialize structure-dependent bounds used during simulation.

        :param structure: Structure object that defines deposit geometry.
        :type structure: Structure
        :return: None
        """
        self.state.max_z = self.structure.deposit.nonzero()[0].max() + 3

    def __set_constants(self, params):
        """Load precursor, thermal, and scaling constants into process state.

        :param params: Dictionary with precursor and process constants.
        :type params: dict
        :return: None
        """
        # Set precursor parameters (these are stored in self.state.precursor via self.model)
        self.state.precursor.F = params['F']
        self.state.precursor.n0 = params['n0']
        self.state.precursor.V = params['V']
        self.state.precursor.sigma = params['sigma']
        self.state.precursor.tau = params['tau']
        self.state.precursor.k0 = params.get('k0', 0)
        self.state.precursor.Ea = params.get('Ea', 0)
        self.state.precursor.D = params['D']
        self.state.precursor.D0 = params.get('D0', 0)
        self.state.precursor.Ed = params.get('Ed', 0)

        # Update state constants
        self.state.heat_cond = params['heat_cond']
        self.state.deposition_scaling = params['deposition_scaling']

        # Keep convenience aliases used by external callers.
        self.kb = self.state.kb
        self.heat_cond = self.state.heat_cond
        self.deposition_scaling = self.state.deposition_scaling
        if self.state.temperature_tracking:
            if not all([self.state.precursor.k0, self.state.precursor.Ea, self.state.precursor.D0, self.state.precursor.Ed]):
                warnings.warn('Some of the temperature dependent parameters were not found! \n '
                              'Switch to static temperature mode? y/n')
                self.state.temperature_tracking = False

    def print_dt(self, units='µs'):
        """Print the active stable time-step estimate in the selected unit.

        :param units: Time unit label (`s`, `ms`, `µs`, or `ns`).
        :type units: str
        :return: None
        """
        m = 1E6
        if units not in ['s', 'ms', 'µs', 'ns']:
            print('Unacceptable input for time units, use one of the following: s, ms, µs, ns.')
        if units == 's':
            m = 1
        if units == 'ms':
            m = 1E3
        if units == 'µs':
            m = 1E6
        if units == 'ns':
            m = 1E9
        print(f'Current time step is {self.dt * m} {units}. \n'
              f'Time step is evaluated as the shortest stability time \n'
              f'of the following process divided by 5: \n'
              f'  Diffusion: \t Dissociation: \t Desorption: \n'
              f'  {self.dt_diff * m} {units} \t {self.dt_diss * m} {units} \t {self.dt_des * m} {units}')

    # Computational methods
    def check_cells_filled(self):
        """
        Check if any deposit cells are fully filled

        :return: bool
        """
        if self.device:
            return self.gpu_facade.check_cells_filled()
        else:
            return self.physics_engine.check_cells_filled()

    def cell_filled_routine(self):
        """
        Updates all data arrays after a cell is filled.

        :return: flag if the structure was resized
        """
        # Use data views for localized array access.

        # What here actually done is marking the filled cell as a solid and a ghost cell and then updating surface,
        # semi-surface, ghosts and precursor to describe the surface geometry around the newly filled cell.
        # The approach is cell-centric, which means all the surroundings are processed
        structure_extended = False

        # Get deposition view to find filled cells (deposit >= 1)
        dep_view: DepositionView = self.view_manager.get_deposition_view()

        # Find cells that are filled using the view's deposit array and index
        if dep_view.acceleration_enabled:
            # Acceleration ON: use sparse index
            nd = index_where(dep_view.deposit[dep_view.index], '>=', 1)[0]
            nd = dep_view.index[0][nd], dep_view.index[1][nd], dep_view.index[2][nd]
        else:
            # Acceleration OFF: use boolean masking on full view
            nd = index_where(dep_view.deposit, '>=', 1)


        new_deposits = [(nd[0][i], nd[1][i], nd[2][i]) for i in range(nd[0].shape[0])]

        # Get surface update view for cell configuration updates
        surf_view: SurfaceUpdateView = self.view_manager.get_surface_update_view()
        cells_abs = [get_index_in_parent(cell, surf_view.irradiated_area_3d) for cell in new_deposits]
        self.last_full_cells = cells_abs
        self.full_cells = (self.full_cells or []) + self.last_full_cells
        self.filled_cells += len(new_deposits)
        for cell in new_deposits:
            self._update_cell_config(cell, surf_view)
            # Updating nearest neighbors profile
            self.update_nearest_neighbors(cell, surf_view)
        # Post cell update routines
        cell_abs_arr = np.array(cells_abs)
        if np.any(cell_abs_arr[:, 0] + 4 > self.state.max_z):
            self.state.max_z += 1

        if self.state.max_z + 5 > self.structure.shape[0]:
            # Here the Structure is extended in height
            structure_extended = self.extend_structure()

        # Phase 1 update - surface topology changed
        self.view_manager.update_after_cell_filling(structure_extended)
        self.n_surface_cells = self.view_manager.n_surface_cells
        self.n_semi_surface_cells = self.view_manager.n_semi_surface_cells

        if self.state.temperature_tracking:
            for cell in cells_abs:
                # Initialize temperature in newly solid cells.
                self.temp_manager.initialize_cell_temperature(cell)
                self.temp_manager.update_local(cell)
            # Update recalculation request for the next thermal solve.
            self.temp_manager.check_and_request_recalculation(self.filled_cells)
            self.request_temp_recalc = self.temp_manager.requires_recalculation
            if not self.request_temp_recalc:  # skip coefficient update if full recalculation is pending
                self.temp_manager.update_full()

        return structure_extended

    def _update_cell_config(self, cell, surf_view):
        """
        Updates all data arrays after a cell is filled.

        :param cell: filled cell indices (local coordinates in view)
        :param surf_view: SurfaceUpdateView with 3D array views
        :return:
        """
        # What here actually done is marking the filled cell as a solid and a ghost cell and then updating surface,
        # semi-surface, ghosts and precursor to describe the surface geometry around the newly filled cell.
        # The approach is cell-centric, which means all the surroundings are processed

        surplus_deposit = surf_view.deposit[cell] - 1  # saving deposit overfill to distribute among the neighbors later
        precursor_cov = surf_view.precursor[cell]
        surf_view.deposit[cell] = -1  # a fully deposited cell is always a minus unity
        if surf_view.temp is not None:
            surf_view.temp[cell] = self.state.room_temp
        surf_view.precursor[cell] = 0
        surf_view.ghosts[cell] = True  # deposited cell belongs to ghost shell
        surf_view.surface[cell] = False  # deposited cell is not a surface cell
        surf_view.semi_surface[cell] = False  # deposited cell is not a semi-surface cell
        cell_abs = get_index_in_parent(cell, surf_view.irradiated_area_3d)  # cell's absolute position in array

        # Getting new converged configuration
        updated_slice, surface_bool, semi_s_bool, ghosts_bool = self._local_mcca.get_converged_configuration(
            cell_abs, self.structure.deposit < 0,
            self.structure.surface_bool,
            self.structure.semi_surface_bool,
            self.structure.ghosts_bool,)
        surf_bool_prev = self.structure.surface_bool[updated_slice].copy()
        semi_s_bool_prev = self.structure.semi_surface_bool[updated_slice].copy()
        # Updating data arrays
        deposit_kern = self.structure.deposit[updated_slice]
        precursor_kern = self.structure.precursor[updated_slice]
        surf_kern = self.structure.surface_bool[updated_slice]
        surf_semi_kern = self.structure.semi_surface_bool[updated_slice]
        ghosts_kern = self.structure.ghosts_bool[updated_slice]
        surf_kern[:] = surface_bool
        surf_semi_kern[:] = semi_s_bool
        ghosts_kern[:] = ghosts_bool
        surf_diff = surf_bool_prev ^ surf_kern  # difference in surface
        semi_s_diff = semi_s_bool_prev ^ surf_semi_kern  # difference in semi-surface
        surf_resid_sum = np.count_nonzero(surf_diff)

        # Update surface all array
        surf_all_kern = self.state.surface_all[updated_slice]
        surf_all_kern[:] = surf_kern | surf_semi_kern
        # This condition covers one specific case
        # Typically, a newly filled cells spawns at least one surface cell, but it is not always so.
        # If the new cell is located in the corner or at the corner edge, it will not spawn any new surface cells.
        # This means that there will be no new surface cells to distribute the surplus deposit to.
        if surf_resid_sum == 0:
            deposit_kern[surf_diff] += surplus_deposit  # redistribute excess deposit
        else:
            deposit_kern[surf_diff] += surplus_deposit / surf_resid_sum  # redistribute excess deposit
        condition = (semi_s_diff | surf_diff) & (precursor_kern < 1e-6)
        precursor_kern[condition] = precursor_cov  # assign average precursor coverage to new surface cells

    def extend_structure(self):
        """
        Increase structure height and resize the structure object.


        :return: True if the structure was resized
        """
        with Lock():  # blocks run with Lock should exclude calls of decorated functions, otherwise the thread will hang
            shape_old = self.structure.shape
            beam_matrix_old = self.state.beam_matrix.copy()  # Save old beam matrix
            beam_matrix_surface_old = self.state.beam_matrix_surface.copy()  # Save old beam matrix surface
            self.structure.resize_structure(200)
            self._local_mcca.compute_surface_neighbors(
                self.structure.deposit,
                self.structure.surface_bool,
                n=self.state.max_neib,
                out=self.structure.surface_neighbors_bool
            )

            # Recreate state arrays with new size
            self.state.beam_matrix = np.zeros_like(self.structure.deposit, dtype=np.int32)
            self.state.surface_temp = np.zeros_like(self.structure.temperature)
            self.state.D_temp = np.zeros_like(self.structure.precursor)
            self.state.tau_temp = np.zeros_like(self.structure.precursor)
            self.state.surface_all = np.logical_or(self.structure.surface_bool, self.structure.semi_surface_bool)

        self.__set_structure(self.structure)
        # Restore old beam matrix values
        self.state.beam_matrix[:shape_old[0], :shape_old[1], :shape_old[2]] = beam_matrix_old
        self.state.beam_matrix_surface[:shape_old[0], :shape_old[1], :shape_old[2]] = beam_matrix_surface_old

        # Reinitialize GPU buffers after resize.
        if self.device:
            self.gpu_facade.reinitialize_after_resize()

        self.redraw = True

        # Basically, none of the slices have to be updated, because they use indexes, not references.
        return True

    def update_nearest_neighbors(self, cell, data_view):
        """
        Update surface nearest neighbors surrounding the cell.

        This updates the Hausdroff distances used for electron escape depth estimation.

        :param cell: cell indices
        :param data_view: SurfaceUpdateView with local active array views
        :return:
        """
        n_3d, _ = get_3d_slice(cell, data_view.deposit.shape, self.state.max_neib)
        neighbors_neighbs = data_view.surface_neighbors[n_3d]
        deposit_neighbs = data_view.deposit[n_3d]
        surface_neighbs = data_view.surface[n_3d]
        self._local_mcca.compute_surface_neighbors(
            deposit_neighbs,
            surface_neighbs,
            n=self.state.max_neib,
            out=neighbors_neighbs
        )

    def deposition(self):
        """
        Calculate an increment of a deposited volume for all irradiated cells over a time step

        :return:
        """
        dt = self.dt
        if self.device:
            self.gpu_facade.compute_deposition(dt)
        else:
            self.physics_engine.compute_deposition(dt)

    def precursor_density(self):
        """
        Calculate an increment of the precursor density for every surface cell

        :return:
        """
        dt = self.dt
        if self.device:
            self.gpu_facade.compute_precursor_density(dt)
        else:
            self.physics_engine.compute_precursor_density(dt)

    def heat_transfer(self, heating):
        """
        Define heating effect on the process

        Delegates to TemperatureManager for all temperature calculations.

        :param heating: volumetric heat sources distribution
        :return:
        """
        self.temp_manager.update_temperature_field(heating)

        # Keep cached maximum temperature for UI/statistics consumers.
        self.max_T = self.temp_manager.max_temperature

    def set_beam_matrix(self, beam_matrix):
        """
        Set secondary electron flux matrix and perform related auxiliary tasks.

        :param beam_matrix: flux matrix array
        :return:
        """
        self.state.beam_matrix[:, :, :] = beam_matrix
        if type(beam_matrix) is not np.ndarray:
            beam_matrix = np.array(beam_matrix)
        else:
            self.state.beam.f0 = beam_matrix.max()
        # Refresh cached views and indices that depend on beam flux matrix
        self.view_manager.update_after_beam_matrix()
        self.n_beam_matrix_points = self.view_manager.n_beam_flux_points

        # Mirror beam matrix to GPU buffers when GPU mode is active.
        if self.device:
            self.gpu_facade.set_beam_matrix(beam_matrix)

    # Properties
    @property
    def max_temperature(self):
        """
        Get the highest current temperature of the structure.

        :return:
        """
        view = self.view_manager.get_temperature_recalc_view()
        temp_2d = view.temp
        return temp_2d.max()

    @property
    def _deposited_vol(self):
        """
        Get total deposited volume.

        NOTE: This property is used by diagnostic scripts.

        :return: Deposited volume (nm³)
        """
        view1 = self.view_manager.get_temperature_recalc_view()
        view2 = self.view_manager.get_precursor_density_view()
        deposit = view1.deposit
        surface = view2.surface
        return (self.filled_cells + deposit[surface].sum()) * self.state.cell_V

    @property
    def precursor_min(self):
        """
        Get the lowest precursor density at the surface.

        :return:
        """
        view = self.view_manager.get_precursor_density_view()
        precursor = view.precursor
        surface = view.surface
        return precursor[surface].min()

    @property
    def dt_diff(self):
        """
        Returns a time step for diffusion process, s
        """
        D = self.temp_manager.get_D()
        if type(D) is np.ndarray:
            D = D.max()
        if D > 0:
            return diffusion.get_diffusion_stability_time(D, self.state.cell_size)
        else:
            return 1

    @property
    def dt_des(self):
        """
        Returns a time step for desorption process, s
        """
        tau = self.temp_manager.get_tau()
        if type(tau) is np.ndarray:
            tau = tau.max()
        return tau

    @property
    def dt_diss(self):
        """
        Return dissociation time step, s
        """
        return self.state.model.dt_diss

    @property
    def dt(self):
        """
        Returns a time step, s
        """
        if self.__forced_dt:
            return self._dt
        dt = min(self.dt_diff, self.dt_des, self.dt_diss) * 0.9
        self._dt = dt
        return dt

    @dt.setter
    def dt(self, val):
        """
        Set a time step. Set value is locked and will not be recalculated until reset_dt() is called.
        """
        dt = self.dt
        if val > dt:
            print(f'Not allowed to increase time step. \nTime step larger than {dt} s will crash the solution.')
        self._dt = val
        self.__forced_dt = True

    def reset_dt(self):
        """
        Reset time step
        """
        self.__forced_dt = False

    # Statistics properties

    @property
    def stats_frequency(self):
        """Time interval between statistics gathering (seconds)."""
        return self.stats.stats_frequency

    @stats_frequency.setter
    def stats_frequency(self, val):
        """Set time interval between statistics gathering."""
        self.stats.stats_frequency = val

    def gather_stats(self):
        """
        Collect statistics of the process.

        Delegates to SimulationStats for calculation and caching.
        """
        self.stats.gather(t=self.t, filled_cells=self.filled_cells)

    @property
    def _irradiated_area_2d(self):
        """
        Returns a slice encapsulating the whole surface
        """
        return np.s_[self.state.substrate_height - 1:self.state.max_z, :, :]  # a volume encapsulating the whole surface

    @property
    def max_neib(self):
        """
        Get maximum number of neighbors for surface cells.
        The number of surface nearest neighbors that could be escaped by a SE

        :return:
        """
        return self.state.max_neib

    @max_neib.setter
    def max_neib(self, val):
        """
        Set maximum number of neighbors for surface cells.

        :param val: new maximum number of neighbors
        :return:
        """
        self.state.max_neib = val
        self._local_mcca.compute_surface_neighbors(
            self.structure.deposit,
            self.structure.surface_bool,
            n=val,
            out=self.structure.surface_neighbors_bool
        )

    @property
    def structure(self):
        """
        Return the active structure instance.
        """
        return self.state.structure

    @property
    def cell_size(self):
        """Return simulation cell edge size.

        :return: Cell size in nanometers.
        """
        return self.state.cell_size

    @property
    def max_z(self):
        """Return current upper z-bound used for active simulation volume.

        :return: Active maximum z-index in structure coordinates.
        """
        return self.state.max_z

    @property
    def substrate_height(self):
        """Return substrate height index in structure coordinates.

        :return: Substrate height index.
        """
        return self.state.substrate_height

    @property
    def beam_matrix(self):
        """Return current secondary-electron flux matrix.

        :return: Beam flux matrix aligned with structure shape.
        """
        return self.state.beam_matrix


if __name__ == '__main__':
    print("Current script does not have an entry point.....")
    input('Press Enter to exit.')
