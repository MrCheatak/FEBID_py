# Default packages
import math
import os, sys

# Core packages
import warnings
from threading import Lock
import numpy as np
import numexpr_mod as ne

# Local packages
from febid.Structure import Structure
import febid.diffusion as diffusion
import febid.heat_transfer as heat_transfer
from febid.libraries.rolling.roll import surface_temp_av

from timeit import default_timer as df


# TODO: look into k-d trees

def restrict(func):
    """
    Prevent simultaneous call of the decorated methods
    """

    def inner(self):
        flag = self.lock.acquire()
        while not flag:
            flag = self.lock.acquire()
        return_vals = func(self)
        flag_e = self.lock.release()
        return return_vals

    return inner


# Deprication note:
# At some point, due to efficiency advantages, the diffusion calculation approach switched from 'rolling' to 'stencil'.
# The rolling approach explicitly requires the array of ghost cells, while stencil does not, although still relying
# on this approach. Instead of the ghost cell array, it checks the same 'precursor' array, that it gets as a base argument,
# for zero cells.
# The ghost array is still kept and maintained throughout the simulation for conceptual clearness and visualisation
class Process():
    """
    Class representing the core deposition process.
    It contains all necessary arrays, variables, parameters and methods to construct a continuous deposition process.
    """

    ### A note_ to value correspondance:
    # The main reaction equation operates in absolute values of precursor density per nanometer.
    # The precursor array stores values as such.
    # The deposited volume is though calculated and stored as a fraction of the cell's volume.
    # Thus, the precursor density has to be multiplied by the cell's base area divided by the cell volume.

    def __init__(self, structure:Structure, equation_values, timings, deposition_scaling=1, temp_tracking=True, name=None):
        if not name:
            self.name = str(np.random.randint(000000, 999999, 1)[0])
        else:
            self.name = name
        # Declaring necessary  properties
        self.structure = None
        self.cell_dimension = None
        self.cell_V = None
        # Main arrays
        # Semi-surface cells are virtual cells that can hold precursor density value, but cannot create deposit.
        # Their role is to serve as pipes on steps, where regular surface cells are not in contact.
        # Therefore, these cells take part in diffusion process, but are not taken into account when calculating
        # other terms in the FEBID equation or deposit increment.
        self.deposit = None  # contains values from 0 to 1 describing the filling state of a cell
        self.precursor = None  # contains values from 0 to 1 describing normalized precursor density in a cell
        self.surface = None  # a boolean array, surface cells are True
        self.semi_surface = None  # a boolean array, semi-surface cells are True
        self.surface_n_neighbors = None  # a boolean array, surface n-nearest neighbors are True
        self.ghosts = None  # a boolean array, ghost cells are True
        self.beam_matrix = None  # contains values of the SE surface flux
        self.temp = None  # contains temperatures of each cell
        self.surface_temp = None

        # Working arrays
        self.__deposit_reduced_3d = None
        self.__precursor_reduced_3d = None
        self.__surface_reduced_3d = None
        self.__semi_surface_reduced_3d = None
        self.__surface_all_reduced_3d = None
        self.__ghosts_reduced_3d = None
        self.__temp_reduced_3d = None
        self.__ghosts_reduced_2d = None
        self.__beam_matrix_reduced_2d = None
        self.D_temp = None
        self.tau_temp = None

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
        self.F = 0  # precursor surface flux
        self.n0 = 0  # maximum absolute precursor density
        self.sigma = 0  # integral cross-section
        self.tau = 0  # residence time
        self.V = 0  # deposit volume of a dissociated precursor molecule
        self.D = 0  # surface diffusion coefficient
        self.cp = 0  # heat capacity
        self.heat_cond = 0  # heat conductance coefficient
        self.rho = 0  # density
        self.room_temp = 294  # room temperature

        # Timings
        self.t_diffusion = 0
        self.t_dissociation = 0
        self.t_desorption = 0
        self.dt = 0
        self.t = 0

        # Utility variables
        self.__neibs_sides = None  # a stencil array used when surface is updated
        self.__neibs_edges = None  # a similar stencil
        self.irradiated_area_3D = np.s_[:, :, :]
        self.irradiated_area_2D = np.s_[:, :, :]
        self.__area_2D_no_sub = np.s_[:, :, :]
        self.deposition_scaling = deposition_scaling  # multiplier of the deposit increment; used to speed up the process
        self.redraw = True  # flag for external functions saying that surface has been updated
        self.t_prev = 0
        self.vol_prev = 0
        self.growth_rate = 0
        self.temperature_tracking = temp_tracking
        self.request_temp_recalc = temp_tracking
        self.temp_step = 10000  # amount of volume to be deposited before next temperature calculation
        self.temp_step_cells = 0  # number of cells to be filled before next temperature calculation
        self.temp_calc_count = 0  # counting number of times temperature has been calculated

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
        self.execution_speed = 0
        self.profiler = None
        self.stats_freq = 1e-2  # s
        self.min_precursor_covearge = 0

        self.lock = Lock()

        # Initialization sequence
        self.__set_structure(structure)
        self.__set_constants(equation_values)
        self.precursor[self.surface] = self.nr
        self.__get_timings(timings)
        self.__expressions()
        self.__get_utils()

    # Initialization methods
    def __set_structure(self, structure: Structure):
        self.structure = structure
        self.deposit = self.structure.deposit
        self.precursor = self.structure.precursor
        self.surface = self.structure.surface_bool
        self.semi_surface = self.structure.semi_surface_bool
        self.surface_n_neighbors = self.structure.surface_neighbors_bool
        self._surface_all = np.logical_or(self.surface, self.semi_surface)
        self.ghosts = self.structure.ghosts_bool
        self.beam_matrix = np.zeros_like(structure.deposit, dtype=np.int32)
        self.temp = self.structure.temperature
        self.surface_temp = np.zeros_like(self.temp)
        self.D_temp = np.zeros_like(self.precursor)
        self.tau_temp = np.zeros_like(self.precursor)
        self.cell_dimension = self.structure.cell_dimension
        self.cell_V = self.cell_dimension ** 3
        self.__get_max_z()
        self.substrate_height = structure.substrate_height
        self.irradiated_area_2D = np.s_[self.structure.substrate_height - 1:self.max_z, :, :]
        self.__area_2D_no_sub = np.s_[self.substrate_height + 1:self.max_z, :, :]
        self.__update_views_2d()
        self.__update_views_3d()
        self.__generate_surface_index()
        self.__generate_deposition_index()
        self._get_solid_index()
        self.n_substrate_cells = self.deposit[:structure.substrate_height].size
        self.temp_step_cells = self.temp_step / self.cell_V

    def __set_constants(self, params):
        self.kb = 0.00008617
        self.F = params['F']
        self.n0 = params['n0']
        self.V = params['V']
        self.sigma = params['sigma']
        self.tau = params['tau']
        self.k0 = params.get('k0', 0)
        self.Ea = params.get('Ea', 0)
        self.D = params['D']
        self.D0 = params.get('D0', 0)
        self.Ed = params.get('Ed', 0)
        self.cp = params['cp']
        self.heat_cond = params['heat_cond']
        self.rho = params['rho']
        self.deposition_scaling = params['deposition_scaling']
        if self.temperature_tracking and not all([self.k0, self.Ea, self.D0, self.Ed]):
            warnings.warn('Some of the temperature dependent parameters were not found! \n '
                          'Switch to static temperature mode? y/n')
            answer: str = input().lower()
            if answer in ['y', 'n']:
                if answer == 'y':
                    self.temperature_tracking = False
                    print('Switching to static temperature mode.')
                    return
                if answer == 'n':
                    print('Terminating.')
                    sys.exit('Exiting due to insufficient parameters for temperature tracking')
        self.__get_surface_temp()
        self.residence_time()
        self.diffusion_coefficient()

    def __get_timings(self, timings):
        # Stability time steps
        self.t_diffusion = diffusion.get_diffusion_stability_time(self.D, self.cell_dimension)
        self.t_dissociation = timings['t_flux']
        self.t_desorption = timings['t_desorption']
        # self.get_dt()
        self.dt = min(self.t_desorption, self.t_diffusion, self.t_dissociation)
        self.dt = self.dt - self.dt / 10
        self.t = self.dt

    def get_dt(self):
        if self.temperature_tracking:
            D_max = self.__D_temp_reduced_2d.max()
            self.t_diffusion = diffusion.get_diffusion_stability_time(D_max, self.cell_dimension)
            self.t_desorption = self.__tau_temp_reduced_2d[self.__surface_reduced_2d].min()
        self.t_dissociation = 1 / self.beam_matrix.max() / self.sigma
        self.dt = min(self.t_desorption, self.t_diffusion, self.t_dissociation)
        self.dt = self.dt - self.dt / 10

    def view_dt(self, units='µs'):
        m = 1E6
        if units not in ['s', 'ms', 'µs', 'ns']:
            print(f'Unacceptable input for time units, use one of the following: s, ms, µs, ns.')
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
              f'  {self.t_diffusion * m} {units} \t {self.t_dissociation * m} {units} \t {self.t_desorption * m} {units}')

    # Computational methods
    def check_cells_filled(self):
        """
        Check if any deposit cells are fully filled

        :return: bool
        """
        if self.__deposit_reduced_3d.max() >= 1:
            return True
        return False

    def update_surface(self):
        """
        Updates all data arrays after a cell is filled.

        :return:
        """
        # What here actually done is marking the filled cell as a solid and a ghost cell and then updating surface,
        # semi-surface, ghosts and precursor to describe the surface geometry around the newly filled cell.
        # The approach is cell-centric, which means all the surroundings are processed
        nd = (self.__deposit_reduced_3d >= 1).nonzero()
        new_deposits = [(nd[0][i], nd[1][i], nd[2][i]) for i in range(nd[0].shape[0])]
        self.filled_cells += len(new_deposits)
        for cell in new_deposits:
            surplus_deposit = self.__deposit_reduced_3d[
                                  cell] - 1  # saving deposit overfill to distribute among the neighbors later
            surplus_precursor = self.__precursor_reduced_3d[cell]
            self.__deposit_reduced_3d[cell] = -1  # a fully deposited cell is always a minus unity
            self.__temp_reduced_3d[cell] = self.room_temp
            self.__precursor_reduced_3d[cell] = 0
            self.__ghosts_reduced_3d[cell] = True  # deposited cell belongs to ghost shell
            self.__surface_reduced_3d[cell] = False  # rising the surface one cell up (new cell)
            # self.redraw = True

            # Instead of using classical conditions, boolean arrays are used to select elements
            # First, a condition array is created, that picks only elements that satisfy conditions
            # Then this array is used as index

            neibs_sides, neibs_edges = self.__neibs_sides, self.__neibs_edges

            # Creating a view with the 1st nearest neighbors to the deposited cell
            z_min, z_max, y_min, y_max, x_min, x_max = 0, 0, 0, 0, 0, 0
            # Taking into account cases when the cell is located at the edge:
            # Small note_: views should be first decreased from the end ([:2])
            # and then from the beginning. Otherwise, decreasing from the end will have no effect.
            if cell[0] + 2 > self.__deposit_reduced_3d.shape[0]:
                z_max = self.__deposit_reduced_3d.shape[0]
                neibs_sides = neibs_sides[:2, :, :]
                neibs_edges = neibs_edges[:2, :, :]
            else:
                z_max = cell[0] + 2
            if cell[0] - 1 < 0:
                z_min = 0
                neibs_sides = neibs_sides[1:, :, :]
                neibs_edges = neibs_edges[1:, :, :]
            else:
                z_min = cell[0] - 1
            if cell[1] + 2 > self.__deposit_reduced_3d.shape[1]:
                y_max = self.__deposit_reduced_3d.shape[1]
                neibs_sides = neibs_sides[:, :2, :]
                neibs_edges = neibs_edges[:, :2, :]
            else:
                y_max = cell[1] + 2
            if cell[1] - 1 < 0:
                y_min = 0
                neibs_sides = neibs_sides[:, 1:, :]
                neibs_edges = neibs_edges[:, 1:, :]
            else:
                y_min = cell[1] - 1
            if cell[2] + 2 > self.__deposit_reduced_3d.shape[2]:
                x_max = self.__deposit_reduced_3d.shape[2]
                neibs_sides = neibs_sides[:, :, :2]
                neibs_edges = neibs_edges[:, :, :2]
            else:
                x_max = cell[2] + 2
            if cell[2] - 1 < 0:
                x_min = 0
                neibs_sides = neibs_sides[:, :, 1:]
                neibs_edges = neibs_edges[:, :, 1:]
            else:
                x_min = cell[2] - 1
            neighbors_1st = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]
            # Creating a view with the 2nd nearest neighbors to the deposited cell
            if cell[0] - 2 < 0:
                z_min = 0
            else:
                z_min = cell[0] - 2
            if cell[0] + 3 > self.__deposit_reduced_3d.shape[0]:
                z_max = self.__deposit_reduced_3d.shape[0]
            else:
                z_max = cell[0] + 3
            if cell[1] - 2 < 0:
                y_min = 0
            else:
                y_min = cell[1] - 2
            if cell[1] + 3 > self.__deposit_reduced_3d.shape[1]:
                y_max = self.__deposit_reduced_3d.shape[1]
            else:
                y_max = cell[1] + 3
            if cell[2] - 2 < 0:
                x_min = 0
            else:
                x_min = cell[2] - 2
            if cell[2] + 3 > self.__deposit_reduced_3d.shape[2]:
                x_max = self.__deposit_reduced_3d.shape[2]
            else:
                x_max = cell[2] + 3
            neighbors_2nd = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]

            # Processing cell configuration according to the cell evolution rules
            deposit_kern = self.__deposit_reduced_3d[neighbors_1st]
            semi_s_kern = self.__semi_surface_reduced_3d[neighbors_1st]
            surf_kern = self.__surface_reduced_3d[neighbors_1st]
            temp_kern = self.__temp_reduced_3d[neighbors_1st]
            # Creating condition array
            condition = np.logical_and(deposit_kern == 0,
                                       neibs_sides)  # True for elements that are not deposited and are side neighbors
            # Updating main arrays
            semi_s_kern[condition] = False
            surf_kern[condition] = True
            condition = np.logical_and(np.logical_and(deposit_kern == 0, surf_kern == 0),
                                       neibs_edges)  # True for elements that are not deposited, not surface cells and are edge neighbors
            semi_s_kern[condition] = True
            ghosts_kern = self.__ghosts_reduced_3d[neighbors_1st]
            ghosts_kern[...] = False
            deposit_kern[surf_kern] += surplus_deposit / np.count_nonzero(surf_kern)  # distributing among the neighbors
            condition = (temp_kern > self.room_temp)
            if np.any(condition):
                self.__temp_reduced_3d[cell] = temp_kern[condition].sum() / np.count_nonzero(condition)

            surf_kern = self.__surface_reduced_3d[neighbors_2nd]
            semi_s_kern = self.__semi_surface_reduced_3d[neighbors_2nd]
            ghosts_kern = self.__ghosts_reduced_3d[neighbors_2nd]
            condition = np.logical_and(surf_kern == 0,
                                       semi_s_kern == 0)  # True for elements that are neither surface nor semi-surface cells
            ghosts_kern[condition] = True

            self.__deposit_reduced_3d[cell] = -1  # a fully deposited cell is always a minus unity
            self.__precursor_reduced_3d[cell] = 0 # precursor density in the deposited cell is always 0
            self.__ghosts_reduced_3d[cell] = True  # deposited cell belongs to ghost shell
            self.__surface_reduced_3d[cell] = False  # deposited cell is no longer a surface cell
            precursor_kern = self.__precursor_reduced_3d[neighbors_2nd]
            condition = (semi_s_kern | surf_kern) & (precursor_kern < 1e-6)
            precursor_kern[condition] = surplus_precursor
            # precursor_kern[condition] += 0.000001  # only for plotting purpose (to pass vtk threshold filter)
            # precursor_kern[condition] += 0.000001

        else:
            self.__get_max_z()
            if self.temperature_tracking and self.max_z - self.substrate_height - 3 > 2:
                self.request_temp_recalc = self.filled_cells > self.temp_calc_count * self.temp_step_cells
                if self.request_temp_recalc:
                    self._get_solid_index()

            self.__area_2D_no_sub = np.s_[self.substrate_height + 1:self.max_z, :, :]
            self.irradiated_area_2D = np.s_[self.structure.substrate_height - 1:self.max_z, :,
                                      :]  # a volume encapsulating the whole surface
            self.__update_views_2d()
            # Updating local surface nearest neighbors
            vert_slice = (self.irradiated_area_2D[1], self.irradiated_area_3D[1], self.irradiated_area_3D[2])
            deposit_red_2d = self.deposit[vert_slice]
            surface_red_2d = self.surface[vert_slice]
            neighbors_2d = self.surface_n_neighbors[vert_slice]
            cell = (cell[0] + self.irradiated_area_3D[0].start, cell[1], cell[2])
            if cell[0] - 3 < 0:
                z_min = 0
            else:
                z_min = cell[0] - 3
            if cell[0] + 4 > deposit_red_2d.shape[0]:
                z_max = deposit_red_2d.shape[0]
            else:
                z_max = cell[0] + 4
            if cell[1] - 3 < 0:
                y_min = 0
            else:
                y_min = cell[1] - 3
            if cell[1] + 4 > deposit_red_2d.shape[1]:
                y_max = deposit_red_2d.shape[1]
            else:
                y_max = cell[1] + 4
            if cell[2] - 3 < 0:
                x_min = 0
            else:
                x_min = cell[2] - 3
            if cell[2] + 4 > deposit_red_2d.shape[2]:
                x_max = deposit_red_2d.shape[2]
            else:
                x_max = cell[2] + 4
            n_3d = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]
            self.structure.define_surface_neighbors(self.max_neib,
                                                    deposit_red_2d[n_3d],
                                                    surface_red_2d[n_3d],
                                                    neighbors_2d[n_3d])

        if self.max_z + 5 > self.structure.shape[0]:
            # Here the Structure is extended in height
            # and all the references to the data arrays are renewed
            with self.lock:  # blocks run with Lock should exclude calls of decorated functions, otherwise the thread will hang
                shape_old = self.structure.shape
                self.structure.resize_structure(200)
                self.structure.define_surface_neighbors(self.max_neib)
                beam_matrix = self.beam_matrix  # taking care of the beam_matrix, because __set_structure creates it empty
            self.__set_structure(self.structure)
            self.beam_matrix[:shape_old[0], :shape_old[1], :shape_old[2]] = beam_matrix
            self.redraw = True
            # Basically, none of the slices have to be updated, because they use indexes, not references.
            return True
        return False

    def deposition(self):
        """
        Calculate an increment of a deposited volume for all irradiated cells over a time step

        :return:
        """
        # Instead of processing cell by cell and on the whole surface, it is implemented to process only (effectively)
        # irradiated area and array-wise(thanks to Numpy)
        # np.float32 — ~1E-7, produced value — ~1E-10
        const = self.sigma * self.V * self.dt * 1e6 * self.deposition_scaling / self.cell_V * self.cell_dimension ** 2  # multiplying by 1e6 to preserve accuracy
        self.__deposit_reduced_3d[self.__deposition_index] += self.__precursor_reduced_3d[
                                                                  self.__deposition_index] * self.__beam_matrix_effective * const / 1e6

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
        # diffusion_matrix = self._laplace_term_stencil(precursor, surface_all)  # Diffusion term is calculated separately and added in the end
        diffusion_matrix = self.__rk4_diffusion(precursor, surface_all)
        precursor[surface] += self.__rk4(precursor[surface],
                                         self.__beam_matrix_surface)  # An increment is calculated through Runge-Kutta method without the diffusion term
        precursor[surface_all] += diffusion_matrix[surface_all]  # finally adding diffusion term

    def equilibrate(self, eps=1e-4, max_it=10000):
        """
        Bring precursor coverage to a steady state with a given accuracy

        It is advised to run this method after updating the surface in order to determine a more accurate precursor
        density value for newly acquired cells

        :param eps: desired accuracy
        """
        start = df()
        for i in range(20):
            # p_prev = self.__precursor_reduced_2d.copy()
            self.precursor_density()
            # norm = np.linalg.norm(self.__precursor_reduced_2d - p_prev)/ np.linalg.norm(self.__precursor_reduced_2d)
            # if norm < eps:
            #     print(f'Took {i+1} iteration(s) to equilibrate, took {df() - start}')
            #     return 1
        else:
            # acc = str(norm)[:int(3-math.log10(eps))]
            # warnings.warn(f'Failed to reach {eps} accuracy in {max_it} iterations in Process.equilibrate. Acheived accuracy: {acc} \n'
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
        if self.temperature_tracking:
            tau = self.__tau_flat
        else:
            tau = self.tau
        return ne.re_evaluate('precursor_temp',
                              local_dict={'F': self.F, 'dt': dt, 'n0': self.n0,
                                          'sigma': self.sigma, 'n': precursor + addon, 'tau': tau,
                                          'se_flux': beam_matrix}, casting='same_kind')

    def _diffusion(self, grid, surface, dt=0, add=0, flat=False):
        """
        Calculates diffusion term of the reaction-diffusion equation for all surface cells.

        :param grid: precursor coverage array
        :param surface: boolean surface array
        :param add: Runge-Kutta intermediate member
        :return: flat ndarray
        """
        if not dt:
            dt = self.dt
        if self.temperature_tracking:
            D_param = self.__D_temp_reduced_2d
        else:
            D_param = self.D
        return diffusion.diffusion_ftcs(grid, surface, D_param, dt, self.cell_dimension, self.__surface_all_index,
                                        flat=flat, add=add)

    def heat_transfer(self, heating):
        """
        Define heating effect on the process

        :param heating: volumetric heat sources distribution
        :return:
        """
        # Calculating temperature profile
        if self.max_z - self.substrate_height - 3 > 2:
            if self.request_temp_recalc:
                self.temp_calc_count += 1
                slice = self.__area_2D_no_sub  # using only top layer of the substrate
                if type(heating) is np.ndarray:
                    heat = heating[slice]
                else:
                    heat = heating
                # Running solution of the heat equation
                start = df()
                print(f'Current max. temperature: {self.max_T} K')
                print(f'Total heating power: {heat.sum() / 1e6:.3f} W/nm/K/1e6')
                heat_transfer.heat_transfer_steady_sor(self.temp[slice], self.heat_cond, self.cell_dimension, heat,
                                                       1e-8)
                print(f'New max. temperature {self.temp.max():.3f} K')
                print(f'Temperature recalculation took {df() - start:.4f} s')
        self.temp[self.substrate_height] = self.room_temp
        self.__get_surface_temp() # estimating surface temperature
        self.diffusion_coefficient() # calculating surface diffusion coefficients
        self.residence_time() # calculating residence times

    def diffusion_coefficient(self):
        """
        Calculate surface diffusion coefficient for every surface cell.

        :return:
        """
        self.D_temp[self._surface_all] = self.diffusion_coefficient_expression(self.surface_temp[self._surface_all])

    def diffusion_coefficient_expression(self, temp=294):
        """
        Calculate surface diffusion coefficient at a specified temperature.

        :param temp: temperature, K
        :return:
        """
        return self.D0 * np.exp(-self.Ed / self.kb / temp)

    def residence_time(self):
        """
        Calculate residence time for every surface cell.

        :return:
        """
        self.__tau_flat = self.residence_time_expression(self.__surface_temp_reduced_2d[self.__surface_reduced_2d])
        self.__tau_temp_reduced_2d[self.__surface_reduced_2d] = self.__tau_flat

    def residence_time_expression(self, temp=294):
        """
        Calculate residence time at the given temperature
        :param temp: temperature, K
        :return:
        """
        return 1 / self.k0 * np.exp(self.Ea / self.kb / temp)

    # Data maintenance methods
    # These methods support an optimization path that provides up to 100x speed up
    # 1. By selecting and processing chunks of arrays (views) that are effectively changing
    # 2. By preparing indexes for arrays
    def update_helper_arrays(self):
        """
        Define new views to data arrays, create axillary indexes and flatten beam_matrix array

        :return:
        """
        # Basically, procedure provided by this method is optional and serves as an optimisation.
        self.__get_irradiated_area()
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
        # All the methods operating on main arrays (deposit, precursor, etc) use a corresponding view
        #  on the necessary array.
        # '3D view' mentioned here can be referred to as a volume that encapsulates all cells that have been irradiated
        self.__deposit_reduced_3d = self.deposit[self.irradiated_area_3D]
        self.__precursor_reduced_3d = self.precursor[self.irradiated_area_3D]
        self.__surface_reduced_3d = self.surface[self.irradiated_area_3D]
        self.__semi_surface_reduced_3d = self.semi_surface[self.irradiated_area_3D]
        self.__surface_all_reduced_3d = self._surface_all[self.irradiated_area_3D]
        self.__ghosts_reduced_3d = self.ghosts[self.irradiated_area_3D]
        self.__temp_reduced_3d = self.temp[self.irradiated_area_3D]
        self.__beam_matrix_reduced_3d = self.beam_matrix[self.irradiated_area_3D]

    def __update_views_2d(self):
        """
        Update view-arrays in accordance with the current max height of the structure

        :return:
        """
        # This is a part of a helper routine destined to reduce the number of cells processed on every iteration
        # All the methods operating on main arrays (deposit, precursor, etc) use a corresponding view
        #  on the necessary array.
        # '2D view' taken here can be referred to as a volume that encapsulates
        #  the whole surface of the deposited structure. This means it takes a view only along the z-axis.
        self.__deposit_reduced_2d = self.deposit[self.irradiated_area_2D]
        self.__precursor_reduced_2d = self.precursor[self.irradiated_area_2D]
        self.__surface_reduced_2d = self.surface[self.irradiated_area_2D]
        self.__semi_surface_reduced_2d = self.semi_surface[self.irradiated_area_2D]
        self.__surface_all_reduced_2d = self._surface_all[self.irradiated_area_2D]
        self.__ghosts_reduced_2d = self.ghosts[self.irradiated_area_2D]
        self.__beam_matrix_reduced_2d = self.beam_matrix[self.irradiated_area_2D]
        self.__temp_reduced_2d = self.temp[self.irradiated_area_2D]
        self.__surface_temp_reduced_2d = self.surface_temp[self.irradiated_area_2D]
        self.__D_temp_reduced_2d = self.D_temp[self.irradiated_area_2D]
        self.__tau_temp_reduced_2d = self.tau_temp[self.irradiated_area_2D]

    def __get_irradiated_area(self):
        """
        Get boundaries of the irradiated area in XY plane

        :return:
        """
        indices = np.nonzero(self.__beam_matrix_reduced_2d)
        y_start, y_end, x_start, x_end = indices[1].min(), indices[1].max() + 1, indices[2].min(), indices[2].max() + 1
        self.irradiated_area_3D = np.s_[self.structure.substrate_height:self.max_z, y_start:y_end,
                                  x_start:x_end]  # a slice of the currently irradiated area
        self.__update_views_3d()

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
        index = self.deposit[self.__area_2D_no_sub]
        index = (index < 0).nonzero()
        self._solid_index = (np.intc(index[0]), np.intc(index[1]), np.intc(index[2]))
        return self._solid_index

    @restrict
    def __get_max_z(self):
        """
        Get z position of the highest not empty cell in the structure

        :return:
        """
        self.max_z = self.deposit.nonzero()[0].max() + 3

    def __get_surface_temp(self):
        self.__surface_temp_reduced_2d[...] = 0
        surface_temp_av(self.__surface_temp_reduced_2d, self.__temp_reduced_2d, *self.__surface_index)
        surface_temp_av(self.__surface_temp_reduced_2d, self.__surface_temp_reduced_2d, *self.__semi_surface_index)
        self.max_T = self.max_temperature

    # Misc
    def __get_utils(self):
        # Kernels for choosing cells
        self.__neibs_sides = np.array([[[0, 0, 0],  # chooses side neighbors
                                        [0, 1, 0],
                                        [0, 0, 0]],
                                       [[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]],
                                       [[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]]])
        self.__neibs_edges = np.array([[[0, 1, 0],  # chooses edge neighbors
                                        [1, 0, 1],
                                        [0, 1, 0]],
                                       [[1, 0, 1],
                                        [0, 0, 0],
                                        [1, 0, 1]],
                                       [[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]]])

    def __expressions(self):
        """
        Prepare math expressions for faster calculations

        :return:
        """
        # Precompiled expressions for numexpr_mod.reevaluate function
        # Creating dummy variables of necessary types
        k1, k2, k3, k4, F, n, n0, tau, sigma, dt = np.arange(10, dtype=np.float64)
        se_flux = np.arange(1, dtype=np.int64)
        ne.cache_expression("(k1+k4)/6 +(k2+k3)/3", 'rk4')
        ne.cache_expression("(F * (1 - n / n0) - n / tau - n * sigma * se_flux) * dt", 'precursor')
        ne.cache_expression("F * dt * (1 - n / n0) - n * dt / tau - n * sigma * se_flux * dt", 'precursor_temp')

    # Properties
    @property
    def kd(self):
        if self.temperature_tracking:
            tau = self.residence_time_expression(self.room_temp)
        else:
            tau = self.tau
        return self.F / self.n0 + 1 / tau + self.sigma * self.beam_matrix.max()

    @property
    def kr(self):
        if self.temperature_tracking:
            tau = self.residence_time_expression(self.room_temp)
        else:
            tau = self.tau
        return self.F / self.n0 + 1 / tau

    @property
    def nr(self):
        """
        Calculate replenished precursor coverage

        :return:
        """
        return self.F / self.kr

    @property
    def nd(self):
        """
        Calculate depleted precursor coverage

        :return:
        """
        return self.F / self.kd

    @property
    @restrict
    def max_temperature(self):
        """
        Get the highest current temperature of the structure.

        :return:
        """
        return self.__surface_temp_reduced_2d.max()

    @property
    @restrict
    def deposited_vol(self):
        """
        Get total deposited volume.

        :return:
        """
        return (self.filled_cells + self.__deposit_reduced_2d[self.__surface_reduced_2d].sum()) * self.cell_V

    @property
    @restrict
    def precursor_min(self):
        """
        Get the lowest precursor density at the surface.

        :return:
        """
        return self.__precursor_reduced_2d[self.__surface_reduced_2d].min()


if __name__ == '__main__':
    print("Current script does not have an entry point.....")
    input('Press Enter to exit.')
