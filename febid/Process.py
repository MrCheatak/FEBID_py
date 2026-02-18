# Default packages

# Core packages
import warnings
from threading import Lock
import numpy as np
import numexpr_mod as ne

# Local packages
from febid.Structure import Structure
from febid.continuum_model_base import ContinuumModel
import febid.diffusion as diffusion
from febid.libraries.rolling.roll import beam_matrix_semi_surface_av
from febid.mlcca import MultiLayerdCellCellularAutomata as MLCCA
from .slice_trics import get_3d_slice, get_index_in_parent, index_where, any_where
from .expressions import cache_numexpr_expressions
from .kernel_modules import GPU
from .process.simulation_state import SimulationState
from .process.data_view_manager import DataViewManager, DiffusionView, PrecursorDensityView, DepositionView, SurfaceUpdateView
from febid.thermal.temperature_manager import TemperatureManager
from febid.logging_config import setup_logger
# Setup logger
logger = setup_logger(__name__)

from timeit import default_timer as df


# TODO: look into k-d trees


# Deprecation note:
# At some point, due to efficiency advantages, the diffusion calculation approach switched from 'rolling' to 'stencil'.
# The rolling approach explicitly requires the array of ghost cells, while stencil does not, although still relying
# on this approach. Instead of the ghost cell array, it checks the same 'precursor' array, that it gets as a base argument,
# for zero cells.
# The ghost array is still kept and maintained throughout the simulation for conceptual clearness and visualisation

# TODO: Extract temperature tracking into a separate class
# TODO: It may be possible to extract acceleration framework into separate class
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
                 acceleration_enabled=True, device=None, name=None):
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
        self._t_prev = 0
        self._vol_prev = 0
        self.growth_rate = 0
        self.request_temp_recalc = False
        self._temp_step = 10000  # amount of volume to be deposited before next temperature calculation
        self._temp_step_cells = 0  # number of cells to be filled before next temperature calculation
        self._temp_calc_count = 0  # counting number of times temperature has been calculated

        # Statistics
        self.filled_cells = 0  # current number of filled cells
        self.growth_rate = 0  # average growth rate
        self.dep_vol = 0  # deposited volume
        self.max_T = 0
        self._stats_frequency = 1e-3  # s, default calculation of stats and offloading from GPU for visualisation
        self.min_precursor_coverage = 0
        self.x0 = 0
        self.y0 = 0
        self.full_cells = None  # indices of the filled cells, used for beam matrix update
        self.last_full_cells = None # indices of the last filled cells
        self.n_surface_cells = 0  # current number of surface cells
        self.n_semi_surface_cells = 0  # current number of semi-surface cells
        self.n_beam_matrix_points = 0  # current number of cells with non-zero electron beam flux

        self.lock = Lock()
        self.device = device
        if device:
            self.knl = GPU(device)

        self.displayed_data = None
        self.stats_gathering = None

        # Initialization sequence
        self.__set_structure(structure)
        self.__set_constants(equation_values)

        # Initialize DataViewManager (Stage 2 refactoring)
        self.view_manager = DataViewManager(self.state, acceleration_enabled=acceleration_enabled)

        # Initialize TemperatureManager (Stage 5 refactoring)
        self.temp_manager = TemperatureManager(self.state, self.view_manager)

        self._temp_step_cells = self._temp_step / self.state.cell_V
        self.state.structure.precursor[self.state.structure.surface_bool] = self.state.model.nr

        # Initialize temperature if tracking enabled
        if self.state.temperature_tracking:
            self.temp_manager.update_full()

        self.__expressions()

    # Initialization methods
    def __set_structure(self, structure: Structure):
        self.state.max_z = self.structure.deposit.nonzero()[0].max() + 3
        if self.device:
            self.load_kernel()

    def __set_constants(self, params):
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

        # Backward compatibility: Keep references
        self.kb = self.state.kb
        self.heat_cond = self.state.heat_cond
        self.deposition_scaling = self.state.deposition_scaling
        if self.state.temperature_tracking:
            if not all([self.state.precursor.k0, self.state.precursor.Ea, self.state.precursor.D0, self.state.precursor.Ed]):
                warnings.warn('Some of the temperature dependent parameters were not found! \n '
                              'Switch to static temperature mode? y/n')
                self.state.temperature_tracking = False

    def print_dt(self, units='µs'):
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
        # Searching in reverse is faster because growth is typically from the bottom to the top
        view = self.view_manager.get_deposition_view()
        surface_cells = view.deposit[view.index]
        cells_filled = any_where(surface_cells, '>=', 1, reverse=True)
        return cells_filled

    def cell_filled_routine(self):
        """
        Updates all data arrays after a cell is filled.

        :return: flag if the structure was resized
        """
        # Stage 2: Uses DataViewManager for optimized array access

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
                # Updating temperature in the new cell (Stage 5: use TemperatureManager)
                self.temp_manager.initialize_cell_temperature(cell)
                self.temp_manager.update_local(cell)
            # Stage 5: Check if temperature recalculation needed (sets flag)
            self.temp_manager.check_and_request_recalculation(self.filled_cells)
            # Update local tracking for backward compatibility
            self.request_temp_recalc = self.temp_manager.requires_recalculation
            # Stage 5: Phase 1 temperature update - coefficients for NEW topology with CURRENT temps
            if not self.request_temp_recalc:  # skipping coeffs update if temperature recalculation is pending
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

        # Stage 2: Uses SurfaceUpdateView for array access
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
            self.structure.resize_structure(200)
            self.structure.define_surface_neighbors(self.state.max_neib)

            # Recreate state arrays with new size
            self.state.beam_matrix = np.zeros_like(self.structure.deposit, dtype=np.int32)
            self.state.surface_temp = np.zeros_like(self.structure.temperature)
            self.state.D_temp = np.zeros_like(self.structure.precursor)
            self.state.tau_temp = np.zeros_like(self.structure.precursor)
            self.state.surface_all = np.logical_or(self.structure.surface_bool, self.structure.semi_surface_bool)

        self.__set_structure(self.structure)
        # Restore old beam matrix values
        self.state.beam_matrix[:shape_old[0], :shape_old[1], :shape_old[2]] = beam_matrix_old
        self.redraw = True

        # Basically, none of the slices have to be updated, because they use indexes, not references.
        return True

    def update_nearest_neighbors(self, cell, data_view):
        """
        Update surface nearest neighbors surrounding the cell.

        This updates the Hausdroff distances used for electron escape depth estimation.

        :param cell: cell indices
        :return:
        """
        n_3d, _ = get_3d_slice(cell, data_view.deposit.shape, self.state.max_neib)
        neighbors_neighbs = data_view.surface_neighbors[n_3d]
        deposit_neighbs = data_view.deposit[n_3d]
        surface_neighbs = data_view.surface[n_3d]
        self.structure.define_surface_neighbors(self.state.max_neib,
                                                       deposit_neighbs,
                                                       surface_neighbs,
                                                       neighbors_neighbs)

    def deposition(self):
        """
        Calculate an increment of a deposited volume for all irradiated cells over a time step

        :return:
        """
        # Stage 2: Use DataViewManager for uniform expression approach
        view: DepositionView = self.view_manager.get_deposition_view()

        # Calculate constant (multiplying by 1e6 to preserve accuracy, np.float32 — ~1E-7, produced value — ~1E-10)
        const = (self.state.precursor.sigma * self.state.precursor.V * self.dt * 1e6 *
                 self.state.deposition_scaling / self.state.cell_V * self.state.cell_size ** 2)

        # UNIFORM EXPRESSION: Works for both acceleration ON and OFF
        # When acceleration ON: view.index is fancy tuple (z,y,x), beam_matrix is 1D
        # When acceleration OFF: view.index is np.s_[:], beam_matrix is 3D
        view.deposit[view.index] += view.precursor[view.index] * view.beam_matrix * const / 1e6

    def precursor_density(self):
        """
        Calculate an increment of the precursor density for every surface cell

        :return:
        """
        # Stage 2: Use DataViewManager for precursor density view
        # Stage 5: Pass TemperatureManager to populate tau and D coefficients
        view: PrecursorDensityView = self.view_manager.get_precursor_density_view(self.temp_manager)

        # Here, surface_all represents surface+semi_surface cells.
        # Boolean indexing: precursor[surface_all] extracts values at surface cells (1D flat array)
        view.precursor[view.surface_all] += self.__rk4_with_ftcs(view)

    def equilibrate(self, max_it=10000, eps=1e-8):
        """
        Bring precursor coverage to a steady state with a given accuracy

        It is advised to run this method after updating the surface in order to determine a more accurate precursor
        density value for newly acquired cells

        :param max_it: number of iterations
        :param eps: desired accuracy
        """
        # Stage 5: Pass TemperatureManager to view
        view = self.view_manager.get_precursor_density_view(self.temp_manager)
        start = df()
        for i in range(max_it):
            p_prev = view.precursor.copy()
            self.precursor_density()
            norm = np.linalg.norm(view.precursor - p_prev)/ np.linalg.norm(view.precursor)
            if norm < eps:
                print(f'Took {i+1} iteration(s) to equilibrate, took {df() - start}')
                return 1
        else:
            acc = str(norm)[:int(3-np.log10(eps))]
            warnings.warn(f'Failed to reach {eps} accuracy in {max_it} iterations in Process.equilibrate. Achieved accuracy: {acc} \n'
                          f'Terminating loop.', RuntimeWarning)
            print(f'Took {i + 1} iteration(s) to equilibrate, took {df() - start}')

    # GPU methods
    def load_kernel(self):
        """
        values are transfered to compute device
        """
        self.knl.load_structure(self.structure.precursor, self.structure.deposit, self._surface_all,
                                self.structure.surface_bool, self.structure.semi_surface_bool,
                                self.structure.ghosts_bool, self._irr_ind_2D)

    def precursor_density_gpu(self, blocking=True):
        """
        precursor density, deposition and check cells filled are calculated on compute device at once
        """
        dt = self.dt
        D = self.temp_manager.get_D()
        cell_size = self.state.cell_size
        F = self.state.precursor.F
        n0 = self.state.precursor.n0
        tau = self.temp_manager.get_tau()
        sigma = self.state.precursor.sigma
        out = self.knl.precur_den_gpu(0, dt * D / (cell_size ** 2), F * dt, (F * dt * tau + n0 * dt) / (tau * n0),
                                      sigma * dt, blocking)

    def deposition_gpu(self, blocking=True):
        """
        precursor density, deposition and check cells filled are calculated on compute device at once
        """
        dt = self.dt
        cell_size = self.state.cell_size
        sigma = self.state.precursor.sigma
        V = self.state.precursor.V
        const = (sigma * V * dt * 1e6 * self.state.deposition_scaling / self.state.cell_V * cell_size ** 2)
        out = self.knl.deposit_gpu(const, blocking)
        return out

    def update_surface_GPU(self):
        """
        Updates all data arrays on compute device

        :return:
        """
        # Here we use the beam matrix as an indicator for filled cells to avoid unnecessary data transfer.
        # Conventionally, the deposit array is used for this purpose, but it is float32 that takes longer to copy back, while the beam matrix is int32.
        # The cells are filled in the kernel, so the filled cells are marked with -1 in the beam matrix there.
        beam_matrix = self.knl.return_beam_matrix()
        full_cells = np.argwhere(beam_matrix < 0)
        self.filled_cells += full_cells.size
        self.last_full_cells = np.argwhere(beam_matrix.reshape(self.structure.shape) < 0)
        self.knl.update_surface(full_cells)
        # self.redraw = True

        for cell in full_cells[0]:
            # z_coord = cell // (self.knl.ydim * self.knl.xdim)
            # y_coord = (cell - z_coord * self.knl.ydim * self.knl.xdim) // self.knl.xdim
            # x_coord = cell - (z_coord * self.knl.ydim * self.knl.xdim) - (y_coord * self.knl.xdim)
            z_coord, y_coord, x_coord = self.knl.index_1d_to_3d(cell)

            if z_coord + 4 > self.state.max_z:
                self.state.max_z = z_coord + 4
                self.knl.zdim_max = z_coord + 4
                self.knl.len_lap = (z_coord + 4 - self.knl.zdim_min) * self.knl.xdim * self.knl.ydim

            self.irradiated_area_2D = np.s_[self.state.substrate_height - 1:self.state.max_z, :, :]

            if z_coord - 3 < 0:
                z_min = 0
            else:
                z_min = z_coord - 3
            if z_coord + 4 > self.knl.zdim:
                z_max = self.knl.zdim
            else:
                z_max = z_coord + 4
            if y_coord - 3 < 0:
                y_min = 0
            else:
                y_min = y_coord - 3
            if y_coord + 4 > self.knl.ydim:
                y_max = self.knl.ydim
            else:
                y_max = y_coord + 4
            if x_coord - 3 < 0:
                x_min = 0
            else:
                x_min = x_coord - 3
            if x_coord + 4 > self.knl.xdim:
                x_max = self.knl.xdim
            else:
                x_max = x_coord + 4
            n_3d = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]
            ind_arr = [[], [], []]
            for z in range(z_min, z_max, 1):
                for y in range(y_min, y_max, 1):
                    for x in range(x_min, x_max, 1):
                        ind_arr[0].append(z)
                        ind_arr[1].append(y)
                        ind_arr[2].append(x)
            arr_size = len(ind_arr[0])
            ind_arr = np.array(ind_arr).reshape(-1).astype(np.int32)
            # start = timeit.default_timer()
            deposit, surface = self.knl.return_slice(ind_arr, arr_size)
            # out = timeit.default_timer() - start
            deposit = deposit.reshape(z_max - z_min, y_max - y_min, x_max - x_min)
            surface = surface.reshape(z_max - z_min, y_max - y_min, x_max - x_min)
            self.structure.define_surface_neighbors(self.state.max_neib,
                                                          deposit,
                                                          surface,
                                                          self.structure.surface_neighbors_bool[n_3d])

        if self.state.max_z + 5 > self.structure.shape[0]:
            # Here the Structure is extended in height
            # and all the references to the data arrays are renewed
            self.offload_structure_from_gpu_all()
            flag = self.extend_structure()
            # Basically, none of the slices have to be updated, because they use indexes, not references.
            return flag

        return False

    def offload_structure_from_gpu_all(self, blocking=True):
        """
        Offloads all data from compute device

        :param blocking: wait until the operation is finished
        """
        data_dict = self.structure.data_dict
        retrieved = self.knl.get_updated_structure(blocking)
        names_retrieved  = set(retrieved.keys())
        names_local= set(data_dict.keys())
        names = set.intersection(names_retrieved, names_local)
        if len(names) == 0:
            raise ValueError('Got no common arrays to offload!')
        for name in names:
            try:
                data_dict[name][...] = retrieved[name]
            except KeyError as e:
                print('Got an unknown array name in Structure from GPU kernel.')
                raise e


    def offload_from_gpu_partial(self, data_name, blocking=True):
        """
        Offloads data from compute device

        :param data_name: name of the data to be offloaded
        :param blocking: wait until the operation is finished
        """
        data_dict = self.structure.data_dict
        if data_name in data_dict:
            array = self.knl.get_structure_partial(data_name, blocking).reshape(self.structure.shape)
            data_dict[data_name][...] = array
        else:
            raise ValueError('Got an array name that is not present in Structure.')

    def onload_structure_to_gpu(self, blocking=True):
        """
        Loads structure data to the GPU

        :param blocking: wait until the operation is finished

        """
        self.knl.update_structure(self.structure.precursor, self.structure.deposit, self._surface_all,
                                  self.structure.surface_bool, self.structure.semi_surface_bool,
                                  self.structure.ghosts_bool, self._irr_ind_2D, blocking=blocking)

    def update_structure_to_gpu(self, blocking=True):
        """
        Updates structure data on the GPU

        :param blocking: wait until the operation is finished
        """
        self.knl.update_structure(self.structure.precursor, self.structure.deposit, self._surface_all,
                                  self.structure.surface_bool, self.structure.semi_surface_bool,
                                  self.structure.ghosts_bool, self._irr_ind_2D, cells=self.last_full_cells, blocking=blocking)

    def get_data(self):
        """
        Offload data necessary for visualization and statistics from compute device.
        """
        necessary_data = []
        if self.stats_gathering:
            necessary_data += ['precursor', 'deposit']
        if self.displayed_data is not None:
            necessary_data += [self.displayed_data]
        necessary_data = set(necessary_data)  # removing duplicates
        if necessary_data:
            [self.offload_from_gpu_partial(data) for data in necessary_data]

    ###

    def __rk4(self, precursor, beam_matrix):
        """
        Calculates increment of precursor density by Runge-Kutta method

        :param precursor: flat precursor array
        :param beam_matrix: flat surface electron flux array
        :return:
        """
        k1 = self.__precursor_density_increment(precursor, beam_matrix,
                                                self.dt)  # this is actually an array of k1 coefficients
        k2 = self.__precursor_density_increment(precursor, beam_matrix, self.dt / 2, k1 / 2)
        k3 = self.__precursor_density_increment(precursor, beam_matrix, self.dt / 2, k2 / 2)
        k4 = self.__precursor_density_increment(precursor, beam_matrix, self.dt, k3)
        return ne.re_evaluate("rk4", casting='same_kind', local_dict={'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4})

    def __rk4_with_ftcs(self, view):
        beam_matrix = view.beam_matrix
        surface_all = view.surface_all
        precursor = view.precursor
        prec_flat = precursor[surface_all]
        if np.any(view.D) == 0:
            return self.__rk4(prec_flat, beam_matrix)
        diff_flat = self._diffusion(precursor, surface_all, flat=True) # 3D
        k1 = self.__precursor_density_increment(prec_flat, beam_matrix, self.dt, diff_flat)
        k1_div = k1 / 2
        diff_flat = self._diffusion(precursor, surface_all, add=k1_div, flat=True)
        k2 = self.__precursor_density_increment(prec_flat, beam_matrix, self.dt / 2, diff_flat, k1_div)
        k2_div = k2 / 2
        diff_flat = self._diffusion(precursor, surface_all, add=k2_div, flat=True)
        k3 = self.__precursor_density_increment(prec_flat, beam_matrix, self.dt / 2, diff_flat, k2_div)
        diff_flat = self._diffusion(precursor, surface_all, add=k3, flat=True)
        k4 = self.__precursor_density_increment(prec_flat, beam_matrix, self.dt, diff_flat, k3)
        return ne.re_evaluate("rk4", casting='same_kind', local_dict={'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4})

    def __precursor_density_increment(self, precursor, beam_matrix, dt, diffusion_matrix=0, addon=0.0):
        """
        Calculates increment of the precursor density without a diffusion term

        :param precursor: flat precursor array
        :param beam_matrix: flat surface electron flux array
        :param dt: time step
        :param addon: Runge Kutta term
        :return:
        """
        view: PrecursorDensityView = self.view_manager.get_precursor_density_view(self.temp_manager)
        tau = view.tau
        n_d = diffusion_matrix
        try:
            return ne.re_evaluate('rde_temp',
                              local_dict={'F': self.state.precursor.F, 'dt': dt, 'n0': self.state.precursor.n0,
                                          'sigma': self.state.precursor.sigma, 'n': precursor + addon, 'tau': tau,
                                          'se_flux': beam_matrix, 'n_d' : n_d}, casting='same_kind')
        except ValueError as e:
            logger.error(f"Failed numexpr.re_evaluate() in Process.__precursor_density_increment due to array size mismatch. \n"
                         f"Filled cells count: {self.filled_cells} \n"
                         f"Surface cells: {self.n_surface_cells} \n"
                         f"Semi-surface cells: {self.n_semi_surface_cells} \n"
                         f"Beam flux points: {self.n_beam_matrix_points} \n"
                         f"Precursor array size: {precursor.size} \n"
                         f"Beam matrix size: {beam_matrix.size} \n"
                         f"Diffusion matrix size: {n_d.size} \n"
                         )
            raise e

    def _diffusion(self, grid, surface, dt=0.0, add=0, flat=False):
        """
        Calculates diffusion term of the reaction-diffusion equation for all surface cells.

        :param grid: precursor coverage array
        :param surface: boolean surface array
        :param add: Runge-Kutta intermediate member
        :return: flat ndarray
        """
        if not dt:
            dt = self.dt

        # Stage 2: Use DataViewManager for surface_all_index (always needed for diffusion)
        # Stage 5: Use TemperatureManager for D coefficient
        view: DiffusionView = self.view_manager.get_diffusion_view(self.temp_manager)
        D = view.D  # Get D from view (already sliced and shaped appropriately)

        return diffusion.diffusion_ftcs(grid, surface, D, dt, self.state.cell_size, view.surface_all_index, flat=flat,
                                        add=add)

    def heat_transfer(self, heating):
        """
        Define heating effect on the process

        Delegates to TemperatureManager for all temperature calculations.

        :param heating: volumetric heat sources distribution
        :return:
        """
        # Stage 5: Delegate to TemperatureManager
        self.temp_manager.update_temperature_field(heating)

        # Update local tracking (backward compatibility)
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
        # Stage 2: Phase 2 update - beam pattern changed
        self.view_manager.update_after_beam_matrix()
        self.n_beam_matrix_points = self.view_manager.n_beam_flux_points
        beam_matrix_2d = self.state.beam_matrix[self.view_manager._slice_irradiated_2d]
        index = self.view_manager._index_semi_surface_2d
        beam_matrix_semi_surface_av(beam_matrix_2d, beam_matrix_2d, *index)

    # Miscellaneous
    def __expressions(self):
        """
        Prepare math expressions for faster calculations. Expression are stored in the package.
        This method should be called only once.

        :return:
        """
        cache_numexpr_expressions()

    # Properties
    @property
    def max_temperature(self):
        """
        Get the highest current temperature of the structure.

        :return:
        """
        temp_2d = self.state.surface_temp[self.view_manager._slice_irradiated_2d]
        return temp_2d.max()

    @property
    def _deposited_vol(self):
        """
        Get total deposited volume.

        :return:
        """
        s = self.view_manager._slice_irradiated_2d
        deposit = self.structure.deposit[s]
        surface = self.structure.surface_bool[s]
        return (self.filled_cells + deposit[surface].sum()) * self.state.cell_V

    @property
    def precursor_min(self):
        """
        Get the lowest precursor density at the surface.

        :return:
        """
        s = self.view_manager._slice_irradiated_3d
        precursor = self.structure.precursor[s]
        surface = self.structure.surface_bool[s]
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

    #Generate a setter and getter for self.stats_frequency
    @property
    def stats_frequency(self):
        return self._stats_frequency

    @stats_frequency.setter
    def stats_frequency(self, val):
        self._stats_frequency = val

    def _gather_stats(self):
        """
        Collect statistics of the process

        :return:
        """
        self.growth_rate = (self.filled_cells - self._vol_prev) / (self.t - self._t_prev)
        self.dep_vol = self._deposited_vol
        self.min_precursor_coverage = self.precursor_min

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
        self.structure.define_surface_neighbors(val)

    @property
    def structure(self):
        """
        Returns the structure instance.
        For backward compatability.
        """
        return self.state.structure

    @property
    def cell_size(self):
        return self.state.cell_size

    @property
    def max_z(self):
        return self.state.max_z

    @property
    def substrate_height(self):
        return self.state.substrate_height


if __name__ == '__main__':
    print("Current script does not have an entry point.....")
    input('Press Enter to exit.')
