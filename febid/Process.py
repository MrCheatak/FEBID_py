# Default packages

# Core packages
import warnings
from threading import Lock
import numpy as np
import numexpr_mod as ne

# Local packages
from febid.Structure import Structure
from febid.continuum_model_base import BeamSettings, PrecursorParams, ContinuumModel
import febid.diffusion as diffusion
import febid.heat_transfer as heat_transfer
from febid.libraries.rolling.roll import surface_temp_av
from febid.mcca import MixedCellCellularAutomata as MCCA
from .slice_trics import get_3d_slice
from .expressions import cache_numexpr_expressions

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
                 name=None):
        super().__init__()
        if not name:
            self.name = str(np.random.randint(000000, 999999, 1)[0])
        else:
            self.name = name
        # Declaring necessary properties
        self.beam: BeamSettings = None
        self.precursor: PrecursorParams = None
        self.model: ContinuumModel = None
        self.set_model(ContinuumModel())
        self.structure: Structure = structure
        self.cell_size = None
        self.cell_V = None
        self.heat_cond = 0
        # Main arrays
        # Semi-surface cells are virtual cells that can hold precursor density value, but cannot create deposit.
        # Their role is to serve as pipes on steps, where regular surface cells are not in contact.
        # Therefore, these cells take part in diffusion process, but are not taken into account when calculating
        # other terms in the FEBID equation or deposit increment.
        self._beam_matrix = None  # contains values of the SE surface flux
        self.surface_temp = None

        # Cellular automata engine
        self._local_mcca = MCCA()

        # Working arrays
        self.__deposit_reduced_3d = None
        self.__precursor_reduced_3d = None
        self.__surface_reduced_3d = None
        self.__semi_surface_reduced_3d = None
        self.__ghosts_reduced_3d = None
        self.__temp_reduced_3d = None
        self.__beam_matrix_reduced_2d = None
        self._D_temp = None
        self._tau_temp = None

        # Helpers
        self._surface_all = None
        self.__beam_matrix_surface = None
        self.__beam_matrix_effective = None
        self.__deposition_index = None
        self.__surface_index = None
        self.__semi_surface_index = None
        self._solid_index = None
        self.__surface_all_index = None
        self.__tau_flat = None

        # Monte Carlo simulation instance
        self.sim = None

        # Physical variables
        self.room_temp = 294  # room temperature

        # Timings
        self.t = 0
        self._dt = None
        self.__forced_dt = False

        # Accuracy
        self.solution_accuracy = 1e-8

        # Utility variables
        self.deposition_scaling = deposition_scaling  # multiplier of the deposit increment; used to speed up the process
        self.redraw = True  # flag for external functions saying that surface has been updated
        self.t_prev = 0
        self.vol_prev = 0
        self.growth_rate = 0
        self.temperature_tracking = temp_tracking
        self.request_temp_recalc = temp_tracking
        self._temp_step = 10000  # amount of volume to be deposited before next temperature calculation
        self._temp_step_cells = 0  # number of cells to be filled before next temperature calculation
        self._temp_calc_count = 0  # counting number of times temperature has been calculated

        # Statistics
        self.substrate_height = 0  # Thickness of the substrate
        self.n_substrate_cells = 0  # the number of the cells in the substrate
        self.max_neib = 0  # the number of surface nearest neighbors that could be escaped by a SE
        self.max_z = 0  # maximum height of the deposited structure, cells
        self.max_z_prev = 0
        self.filled_cells = 0  # current number of filled cells
        self.n_filled_cells = []
        self.growth_rate = []
        self.dep_vol = 0  # deposited volume
        self.max_T = 0
        self._stats_frequency = 1e-2  # s
        self.min_precursor_coverage = 0
        self.x0 = 0
        self.y0 = 0

        self.lock = Lock()

        # Initialization sequence
        self.__set_structure(structure)
        self.__set_constants(equation_values)
        self.__update_views_2d()
        self.__generate_surface_index()
        self._get_solid_index()
        self._temp_step_cells = self._temp_step / self.cell_V
        self.structure.precursor[self.structure.surface_bool] = self.model.nr
        if self.temperature_tracking:
            self.__get_surface_temp()
            self._residence_time_profile()
            self._diffusion_coefficient_profile()
        self.__expressions()

    # Initialization methods
    def __set_structure(self, structure: Structure):
        self.structure = structure
        self._surface_all = np.logical_or(self.structure.surface_bool, self.structure.semi_surface_bool)
        self._beam_matrix = np.zeros_like(structure.deposit, dtype=np.int32)
        self.surface_temp = np.zeros_like(self.structure.temperature)
        self._D_temp = np.zeros_like(self.structure.precursor)
        self._tau_temp = np.zeros_like(self.structure.precursor)
        self.cell_size = self.structure.cell_size
        self.cell_V = self.cell_size ** 3
        self.__set_max_z()
        self.substrate_height = structure.substrate_height
        self.n_substrate_cells = self.structure.deposit[:structure.substrate_height].size

    def __set_constants(self, params):
        self.kb = 0.00008617
        self.precursor.F = params['F']
        self.precursor.n0 = params['n0']
        self.precursor.V = params['V']
        self.precursor.sigma = params['sigma']
        self.precursor.tau = params['tau']
        self.precursor.k0 = params.get('k0', 0)
        self.precursor.Ea = params.get('Ea', 0)
        self.precursor.D = params['D']
        self.precursor.D0 = params.get('D0', 0)
        self.precursor.Ed = params.get('Ed', 0)
        self.heat_cond = params['heat_cond']
        self.deposition_scaling = params['deposition_scaling']
        if self.temperature_tracking:
            if not all([self.precursor.k0, self.precursor.Ea, self.precursor.D0, self.precursor.Ed]):
                warnings.warn('Some of the temperature dependent parameters were not found! \n '
                              'Switch to static temperature mode? y/n')
                self.temperature_tracking = False

    def set_model(self, model: ContinuumModel):
        self.model = model
        self.precursor = model.precursor
        self.beam = model.beam

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
        if self.__deposit_reduced_3d.max() >= 1:
            return True
        return False

    def cell_filled_routine(self):
        """
        Updates all data arrays after a cell is filled.

        :return: flag if the structure was resized
        """

        # What here actually done is marking the filled cell as a solid and a ghost cell and then updating surface,
        # semi-surface, ghosts and precursor to describe the surface geometry around the newly filled cell.
        # The approach is cell-centric, which means all the surroundings are processed

        nd = (self.__deposit_reduced_3d >= 1).nonzero()
        new_deposits = [(nd[0][i], nd[1][i], nd[2][i]) for i in range(nd[0].shape[0])]
        self.filled_cells += len(new_deposits)
        for cell in new_deposits:
            self._update_cell_config(cell)
            # Updating temperature in the new cell
            self.update_cell_temperature(cell)
            # Updating nearest neighbors profile
            self.update_nearest_neighbors(cell)
        if len(new_deposits) > 0:
            # Post cell update routines
            self.__set_max_z()
            self.__update_views_2d()
            if self.temperature_tracking and self.max_z - self.substrate_height - 3 > 2:
                self.request_temp_recalc = self.filled_cells > self._temp_calc_count * self._temp_step_cells
                if self.request_temp_recalc:
                    self._get_solid_index()

        if self.max_z + 5 > self.structure.shape[0]:
            # Here the Structure is extended in height
            return self.extend_structure()
        return False

    def _update_cell_config(self, cell):
        """
        Updates all data arrays after a cell is filled.

        :param cell: filled cell indices
        :return:
        """
        # What here actually done is marking the filled cell as a solid and a ghost cell and then updating surface,
        # semi-surface, ghosts and precursor to describe the surface geometry around the newly filled cell.
        # The approach is cell-centric, which means all the surroundings are processed
        surplus_deposit = self.__deposit_reduced_3d[
                              cell] - 1  # saving deposit overfill to distribute among the neighbors later
        precursor_cov = self.__precursor_reduced_3d[cell]
        self.__deposit_reduced_3d[cell] = -1  # a fully deposited cell is always a minus unity
        self.__temp_reduced_3d[cell] = self.room_temp
        self.__precursor_reduced_3d[cell] = 0
        self.__ghosts_reduced_3d[cell] = True  # deposited cell belongs to ghost shell
        self.__surface_reduced_3d[cell] = False  # deposited cell is not a surface cell
        self.__semi_surface_reduced_3d[cell] = False  # deposited cell is not a semi-surface cell
        # Getting new converged configuration
        updated_slice, surface_bool, semi_s_bool, ghosts_bool = self._local_mcca.get_converged_configuration(
            cell, self.__deposit_reduced_3d.astype(bool),
            self.__surface_reduced_3d,
            self.__semi_surface_reduced_3d,
            self.__ghosts_reduced_3d)
        surf_bool_prev = self.__surface_reduced_3d[updated_slice].copy()
        semi_s_bool_prev = self.__semi_surface_reduced_3d[updated_slice].copy()
        # Updating data arrays
        deposit_kern = self.__deposit_reduced_3d[updated_slice]
        precursor_kern = self.__precursor_reduced_3d[updated_slice]
        surf_kern = self.__surface_reduced_3d[updated_slice]
        surf_semi_kern = self.__semi_surface_reduced_3d[updated_slice]
        ghosts_kern = self.__ghosts_reduced_3d[updated_slice]
        surf_kern[:] = surface_bool
        surf_semi_kern[:] = semi_s_bool
        ghosts_kern[:] = ghosts_bool
        surf_diff = surf_bool_prev ^ surf_kern  # difference in surface
        semi_s_diff = semi_s_bool_prev ^ surf_semi_kern  # difference in semi-surface
        deposit_kern[surf_diff] += surplus_deposit / np.count_nonzero(surf_diff)  # redistribute excess deposit
        condition = (semi_s_diff | surf_diff) & (precursor_kern < 1e-6)
        precursor_kern[condition] = precursor_cov  # assign average precursor coverage to new surface cells

    def extend_structure(self):
        """
        Increase structure height and resize the structure object.


        :return: True if the structure was resized
        """
        with self.lock:  # blocks run with Lock should exclude calls of decorated functions, otherwise the thread will hang
            shape_old = self.structure.shape
            self.structure.resize_structure(200)
            self.structure.define_surface_neighbors(self.max_neib)
            beam_matrix = self._beam_matrix  # taking care of the beam_matrix, because __set_structure creates it empty
        self.__set_structure(self.structure)
        self._beam_matrix[:shape_old[0], :shape_old[1], :shape_old[2]] = beam_matrix
        self.redraw = True
        # Basically, none of the slices have to be updated, because they use indexes, not references.
        return True

    def update_cell_temperature(self, cell):
        """
        Update temperature of the cell by assigning it an average of the surrounding cells.

        :param cell: cell indices
        :return:
        """
        temp_slice, _ = get_3d_slice(cell, self.__temp_reduced_3d.shape, 2)
        temp_kern = self.__temp_reduced_3d[temp_slice]
        condition = (temp_kern > self.room_temp)
        if np.any(condition):
            self.__temp_reduced_3d[cell] = temp_kern[condition].sum() / np.count_nonzero(condition)

    def update_nearest_neighbors(self, cell):
        """
        Update surface nearest neighbors surrounding the cell.

        This updates the Hausdroff distances used for electron escape depth estimation.

        :param cell: cell indices
        :return:
        """
        n_3d, _ = get_3d_slice(cell, self.__deposit_reduced_3d.shape, self.max_neib)
        neighbors_neighbs = self.__surface_neighbors_reduced_3d[n_3d]
        deposit_neighbs = self.__deposit_reduced_3d[n_3d]
        surface_neighbs = self.__surface_reduced_3d[n_3d]
        self.structure.define_surface_neighbors(self.max_neib,
                                                deposit_neighbs,
                                                surface_neighbs,
                                                neighbors_neighbs)

    def deposition(self):
        """
        Calculate an increment of a deposited volume for all irradiated cells over a time step

        :return:
        """
        # Instead of processing cell by cell and on the whole surface, it is implemented to process only (effectively)
        # irradiated area and array-wise(thanks to Numpy)
        # np.float32 — ~1E-7, produced value — ~1E-10
        const = (self.precursor.sigma * self.precursor.V * self.dt * 1e6 *
                 self.deposition_scaling / self.cell_V * self.cell_size ** 2)  # multiplying by 1e6 to preserve accuracy
        self.__deposit_reduced_3d[self.__deposition_index] += (
                self.__precursor_reduced_3d[self.__deposition_index] * self.__beam_matrix_effective * const / 1e6)

    def precursor_density(self):
        """
        Calculate an increment of the precursor density for every surface cell

        :return:
        """
        # Here, surface_all represents surface+semi_surface cells.
        # It is only used in diffusion calculation, because semi_surface cells cannot take part in deposition process
        precursor = self.__precursor_reduced_2d
        surface_all = self.__surface_all_reduced_2d
        surface = self.__surface_reduced_2d
        diffusion_matrix = self.__rk4_diffusion(precursor,
                                                surface_all)  # Diffusion term is calculated separately and added in the end
        precursor[surface] += self.__rk4(precursor[surface],
                                         self.__beam_matrix_surface)  # An increment is calculated through Runge-Kutta method without the diffusion term
        precursor[surface_all] += diffusion_matrix[surface_all]  # finally adding diffusion term

    def equilibrate(self, max_it=10000):
        """
        Bring precursor coverage to a steady state with a given accuracy

        It is advised to run this method after updating the surface in order to determine a more accurate precursor
        density value for newly acquired cells

        :param max_it: number of iterations
        """
        start = df()
        for i in range(max_it):
            # p_prev = self.__precursor_reduced_2d.copy()
            self.precursor_density()
            # norm = np.linalg.norm(self.__precursor_reduced_2d - p_prev)/ np.linalg.norm(self.__precursor_reduced_2d)
            # if norm < eps:
            #     print(f'Took {i+1} iteration(s) to equilibrate, took {df() - start}')
            #     return 1
        else:
            # acc = str(norm)[:int(3-math.log10(eps))]
            # warnings.warn(f'Failed to reach {eps} accuracy in {max_it} iterations in Process.equilibrate. Achieved accuracy: {acc} \n'
            #               f'Terminating loop.', RuntimeWarning)
            print(f'Took {i + 1} iteration(s) to equilibrate, took {df() - start}')

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
        return ne.re_evaluate("rk4", casting='same_kind')

    def __rk4_diffusion(self, grid, surface):
        """
        Apply Runge-Kutta 4 method to the calculation of the diffusion term.

        :param grid: precursor coverage array
        :param surface: surface boolean array
        :return:
        """
        dt = self.dt
        k1 = self._diffusion(grid, surface, dt, flat=False)
        k1[surface] /= 2
        k2 = self._diffusion(grid, surface, dt / 2, add=k1, flat=False)
        k2[surface] /= 2
        k3 = self._diffusion(grid, surface, dt / 2, add=k2, flat=False)
        k4 = self._diffusion(grid, surface, dt, add=k3, flat=False)
        return ne.re_evaluate("rk4", casting='same_kind')

    def __precursor_density_increment(self, precursor, beam_matrix, dt, addon=0.0):
        """
        Calculates increment of the precursor density without a diffusion term

        :param precursor: flat precursor array
        :param beam_matrix: flat surface electron flux array
        :param dt: time step
        :param addon: Runge Kutta term
        :return:
        """
        tau = self._get_tau()
        return ne.re_evaluate('precursor_temp',
                              local_dict={'F': self.precursor.F, 'dt': dt, 'n0': self.precursor.n0,
                                          'sigma': self.precursor.sigma, 'n': precursor + addon, 'tau': tau,
                                          'se_flux': beam_matrix}, casting='same_kind')

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
        D = self._get_D()
        return diffusion.diffusion_ftcs(grid, surface, D, dt, self.cell_size, self.__surface_all_index, flat=flat,
                                        add=add)

    def heat_transfer(self, heating):
        """
        Define heating effect on the process

        :param heating: volumetric heat sources distribution
        :return:
        """
        # Calculating temperature profile
        if self.max_z - self.substrate_height - 3 > 2:
            if self.request_temp_recalc:
                self._temp_calc_count += 1
                slice_2d = self.__irradiated_area_2D_no_sub  # using only top layer of the substrate
                if type(heating) is np.ndarray:
                    heat = heating[slice_2d]
                else:
                    heat = heating
                # Running solution of the heat equation
                start = df()
                print(f'Current max. temperature: {self.max_T} K')
                print(f'Total heating power: {heat.sum() / 1e6:.3f} W/nm/K/1e6')
                heat_transfer.heat_transfer_steady_sor(self.structure.temperature[slice_2d], self.heat_cond,
                                                       self.cell_size, heat, self.solution_accuracy)
                print(f'New max. temperature {self.structure.temperature.max():.3f} K')
                print(f'Temperature recalculation took {df() - start:.4f} s')
        self.structure.temperature[self.substrate_height] = self.room_temp
        self.__get_surface_temp()  # estimating surface temperature
        self._diffusion_coefficient_profile()  # calculating surface diffusion coefficients
        self._residence_time_profile()  # calculating residence times

    def _diffusion_coefficient_profile(self):
        """
        Calculate surface diffusion coefficient for every surface cell.

        :return:
        """
        self._D_temp[self._surface_all] = self.precursor.diffusion_coefficient_at_T(self.surface_temp[self._surface_all])

    def _residence_time_profile(self):
        """
        Calculate residence time for every surface cell.

        :return:
        """
        self.__tau_flat = self.precursor.residence_time_at_T(self.__surface_temp_reduced_2d[self.__surface_reduced_2d])
        self.__tau_temp_reduced_2d[self.__surface_reduced_2d] = self.__tau_flat

    def set_beam_matrix(self, beam_matrix):
        """
        Set secondary electron flux matrix

        :param beam_matrix: flux matrix array
        :return:
        """
        self._beam_matrix[:, :, :] = beam_matrix
        if type(beam_matrix) is not np.ndarray:
            beam_matrix = np.array(beam_matrix)
        else:
            self.beam.f0 = beam_matrix.max()
        self._update_helper_arrays()

    # Data maintenance methods
    # These methods support an optimization path that provides up to 100x speed up
    # 1. By selecting and processing chunks of arrays (views) that are effectively changing
    # 2. By preparing indexes for arrays
    def _update_helper_arrays(self):
        """
        Define new views to data arrays, create axillary indexes and flatten beam_matrix array

        :return:
        """
        # Basically, procedure provided by this method is optional and serves as an optimisation.
        self.__update_views_3d()
        self.__generate_deposition_index()
        self.__generate_surface_index()
        self.__flatten_beam_matrix_effective()
        self.__flatten_beam_matrix_surface()

    def __update_views_3d(self):
        """
        Update view-arrays in accordance with the currently irradiated area

        :return:
        """
        # This is a part of a helper routine destined to reduce the number of cells processed on every iteration
        # All the methods operating on main arrays (deposit, precursor, etc.) use a corresponding view
        #  on the necessary array.
        # '3D view' mentioned here can be referred to as a volume that encapsulates all cells that have been irradiated
        slice3d = self._irradiated_area_3D
        self.__deposit_reduced_3d = self.structure.deposit[slice3d]
        self.__precursor_reduced_3d = self.structure.precursor[slice3d]
        self.__surface_reduced_3d = self.structure.surface_bool[slice3d]
        self.__semi_surface_reduced_3d = self.structure.semi_surface_bool[slice3d]
        self.__ghosts_reduced_3d = self.structure.ghosts_bool[slice3d]
        self.__temp_reduced_3d = self.structure.temperature[slice3d]
        self.__beam_matrix_reduced_3d = self._beam_matrix[slice3d]
        self.__surface_neighbors_reduced_3d = self.structure.surface_neighbors_bool[slice3d]

    def __update_views_2d(self):
        """
        Update view-arrays in accordance with the current max height of the structure

        :return:
        """
        # This is a part of a helper routine destined to reduce the number of cells processed on every iteration
        # All the methods operating on main arrays (deposit, precursor, etc.) use a corresponding view
        #  on the necessary array.
        # '2D view' taken here can be referred to as a volume that encapsulates
        #  the whole surface of the deposited structure. This means it takes a view only along the z-axis.
        slice2d = self._irradiated_area_2D
        self.__deposit_reduced_2d = self.structure.deposit[slice2d]
        self.__precursor_reduced_2d = self.structure.precursor[slice2d]
        self.__surface_reduced_2d = self.structure.surface_bool[slice2d]
        self.__semi_surface_reduced_2d = self.structure.semi_surface_bool[slice2d]
        self.__surface_all_reduced_2d = self._surface_all[slice2d]
        self.__beam_matrix_reduced_2d = self._beam_matrix[slice2d]
        self.__temp_reduced_2d = self.structure.temperature[slice2d]
        self.__surface_temp_reduced_2d = self.surface_temp[slice2d]
        self.__D_temp_reduced_2d = self._D_temp[slice2d]
        self.__tau_temp_reduced_2d = self._tau_temp[slice2d]

    def __generate_deposition_index(self):
        """
        Generate a tuple of indices of the cells that are irradiated for faster indexing in 'deposition' method

        :return:
        """
        self.__deposition_index = self.__beam_matrix_reduced_3d.nonzero()

    def __generate_surface_index(self):
        """
        Generate a tuple of indices for faster indexing in 'laplace_term' method

        :return:
        """
        self.__surface_all_reduced_2d[:, :, :] = np.logical_or(self.__surface_reduced_2d,
                                                               self.__semi_surface_reduced_2d)
        index = self.__surface_all_reduced_2d.nonzero()
        self.__surface_all_index = (np.intc(index[0]), np.intc(index[1]), np.intc(index[2]))
        index = self.__surface_reduced_2d.nonzero()
        self.__surface_index = (np.intc(index[0]), np.intc(index[1]), np.intc(index[2]))
        index = self.__semi_surface_reduced_2d.nonzero()
        self.__semi_surface_index = (np.intc(index[0]), np.intc(index[1]), np.intc(index[2]))

    def __flatten_beam_matrix_effective(self):
        """
        Extract a flattened array of non-zero elements from beam_matrix array

        :return:
        """
        self.__beam_matrix_effective = self.__beam_matrix_reduced_3d[self.__deposition_index]

    def __flatten_beam_matrix_surface(self):
        """
        Extract a flattened array of beam_matrix array

        :return:
        """
        self.__beam_matrix_surface = self.__beam_matrix_reduced_2d[self.__surface_reduced_2d]

    def _get_solid_index(self):
        index = self.structure.deposit[self.__irradiated_area_2D_no_sub]
        index = (index < 0).nonzero()
        self._solid_index = (np.intc(index[0]), np.intc(index[1]), np.intc(index[2]))
        return self._solid_index

    def __set_max_z(self):
        """
        Set z position of the highest not empty cell in the structure

        :return:
        """
        self.max_z = self.structure.deposit.nonzero()[0].max() + 3

    def __get_surface_temp(self):
        self.__surface_temp_reduced_2d[...] = 0
        surface_temp_av(self.__surface_temp_reduced_2d, self.__temp_reduced_2d, *self.__surface_index)
        surface_temp_av(self.__surface_temp_reduced_2d, self.__surface_temp_reduced_2d, *self.__semi_surface_index)
        self.max_T = self.max_temperature

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
        return self.__surface_temp_reduced_2d.max()

    @property
    def deposited_vol(self):
        """
        Get total deposited volume.

        :return:
        """
        return (self.filled_cells + self.__deposit_reduced_2d[self.__surface_reduced_2d].sum()) * self.cell_V

    @property
    def precursor_min(self):
        """
        Get the lowest precursor density at the surface.

        :return:
        """
        return self.__precursor_reduced_2d[self.__surface_reduced_2d].min()

    def _get_tau(self):
        """
        Returns single value of the residence time if temperature tracking is off
        or returns an array of values otherwise
        """
        if self.temperature_tracking:
            tau = self.__tau_flat
        else:
            tau = self.precursor.tau
        return tau

    def _get_D(self):
        """
        Returns single value of the diffusion coefficient if temperature tracking is off
        or returns an array of values otherwise
        """
        if self.temperature_tracking:
            D = self.__D_temp_reduced_2d
        else:
            D = self.precursor.D
        return D

    @property
    def dt_diff(self):
        """
        Returns a time step for diffusion process, s
        """
        D = self._get_D()
        if type(D) is np.ndarray:
            D = D.max()
        if D > 0:
            return diffusion.get_diffusion_stability_time(D, self.cell_size)
        else:
            return 1

    @property
    def dt_des(self):
        """
        Returns a time step for desorption process, s
        """
        tau = self._get_tau()
        if type(tau) is np.ndarray:
            tau = tau.max()
        return tau

    @property
    def dt_diss(self):
        """
        Return dissociation time step, s
        """
        return self.model.dt_diss

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

    @property
    def _irradiated_area_2D(self):
        """
        Returns a slice encapsulating the whole surface
        """
        return np.s_[self.structure.substrate_height - 1:self.max_z, :, :]  # a volume encapsulating the whole surface

    @property
    def _irradiated_area_3D(self):
        """
        Returns a slice of the currently irradiated area
        """
        indices = np.nonzero(self.__beam_matrix_reduced_2d)
        y_start, y_end, x_start, x_end = indices[1].min(), indices[1].max() + 1, indices[2].min(), indices[2].max() + 1
        irradiated_area_3D = np.s_[self.structure.substrate_height-1:self.max_z, y_start:y_end,
                             x_start:x_end]  # a slice of the currently irradiated area
        return irradiated_area_3D  # a slice of the currently irradiated area

    @property
    def __irradiated_area_2D_no_sub(self):
        """
        Returns a slice encapsulating the whole surface without the substrate
        """
        return np.s_[self.substrate_height + 1:self.max_z, :, :]


if __name__ == '__main__':
    print("Current script does not have an entry point.....")
    input('Press Enter to exit.')
