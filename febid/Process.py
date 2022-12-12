# Default packages
import os, sys

# Core packages
import numpy as np
from numexpr_mod import evaluate_cached, cache_expression

# Local packages
from febid import Structure
import febid.diffusion as diffusion
import febid.heat_transfer as heat_transfer
from febid.libraries.rolling.roll import surface_temp_av

from timeit import default_timer as df

# TODO: look into k-d trees
# TODO: add a benchmark to determine optimal threads number for current machine


class Process():
    """
    Class representing the core deposition process.
    It contains all necessary arrays, variables, parameters and methods to support a continuous deposition process.
    """

    ### A note_ to value correspondance:
    # The main reaction equation operates in absolute values of precursor density per nanometer.
    # The precursor array stores values as such.
    # The deposited volume is though calculated and stored as a fraction of the cell's volume.
    # Thus, the precursor density has to be multiplied by the cell's base area divided by the cell volume.

    def __init__(self, structure, equation_values, timings, deposition_scaling=1, temp_tracking=True, name=None):
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
        self.deposit = None # contains values from 0 to 1 describing the filling state of a cell
        self.precursor = None # contains values from 0 to 1 describing normalized precursor density in a cell
        self.surface = None # a boolean array, surface cells are True
        self.semi_surface = None # a boolean array, semi-surface cells are True
        self.surface_n_neighbors = None # a boolean array, surface n-nearest neighbors are True
        self.ghosts = None # a boolean array, ghost cells are True
        self.beam_matrix = None # contains values of the SE surface flux
        self.temp = None # contains temperatures of each cell
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
        self.F = 0 # precursor surface flux
        self.n0 = 0 # maximum absolute precursor density
        self.sigma = 0 # integral cross-section
        self.tau = 0 # residence time
        self.V = 0 # deposit volume of a dissociated precursor molecule
        self.D = 0 # surface diffusion coefficient
        self.cp = 0 # heat capacity
        self.heat_cond = 0 # heat conductance coefficient
        self.rho = 0 # density
        self.room_temp = 294 # room temperature

        # Timings
        self.t_diffusion = 0
        self.t_dissociation = 0
        self.t_desorption = 0
        self.dt = 0
        self.t = 0
        
        # Utility variables
        self.__neibs_sides = None # a stencil array used when surface is updated
        self.__neibs_edges = None # a similar stencil
        self.irradiated_area_3D = np.s_[:,:,:]
        self.irradiated_area_2D = np.s_[:,:,:]
        self.__area_2D_no_sub = np.s_[:, :, :]
        self.deposition_scaling = deposition_scaling # multiplier of the deposit increment; used to speed up the process
        self.redraw = True # flag for external functions saying that surface has been updated
        self.t_prev = 0
        self.vol_prev = 0
        self.temperature_tracking = True
        self.temp_step = 10000 # amount of volume to be deposited before next temperature calculation
        self.temp_step_cells = 0 # number of cells to be filled before next temperature calculation
        self.temp_calc_count = 0 # counting number of times temperature has been calculated
        
        # Statistics
        self.substrate_height = 0 # Thickness of the substrate
        self.n_substrate_cells = 0 # the number of the cells in the substrate
        self.max_neib = 0 # the number of surface nearest neighbors that could be escaped by a SE
        self.max_z = 0 # maximum height of the deposited structure, cells
        self.filled_cells = 0 # current number of filled cells
        self.n_filled_cells = []
        self.growth_rate = []
        self.dep_vol = 0 # deposited volume
        self.max_T = 0
        self.execution_speed = 0
        self.profiler= None

        # Initialization sequence
        self.__set_structure(structure)
        self.__set_constants(equation_values)
        self.__get_timings(timings)
        self.__expressions()
        self.__get_utils()

    # Initialization methods
    def __set_structure(self, structure:Structure):
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
        self.cell_V = self.cell_dimension**3
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
        self.temp_step_cells = self.temp_step/self.cell_V

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
            answer:str = input().lower()
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


    def __setup_MC_module(self, params):
        pass

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
        self.t_dissociation = 1/self.beam_matrix.max()/self.sigma
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
        print(f'Current time step is {self.dt*m} {units}. \n'
              f'Time step is evaluated as the shortest stability time \n'
              f'of the following process divided by 5: \n'
              f'  Diffusion: \t Dissociation: \t Desorption: \n'
              f'  {self.t_diffusion*m} {units} \t {self.t_dissociation*m} {units} \t {self.t_desorption*m} {units}')

    # Computational methods
    def check_cells_filled(self):
        """
        Check if any deposit cells are fully filled

        :return: bool
        """
        if self.__deposit_reduced_3d.max()>=1:
            return True
        return False

    def update_surface(self):
        """
        Updates all data arrays

        :return:
        """
        # What here actually done is marking the filled cell as a solid and a ghost cell and then updating surface,
        # semi-surface, ghosts and precursor to describe the surface geometry around the newly filled cell.
        # The approach is cell-centric, which means all the surroundings are processed
        nd = (self.__deposit_reduced_3d >= 1).nonzero()
        new_deposits = [(nd[0][i], nd[1][i], nd[2][i]) for i in range(nd[0].shape[0])]
        self.filled_cells += len(new_deposits)
        for cell in new_deposits:
            # deposit[cell[0]+1, cell[1], cell[2]] += deposit[cell[0], cell[1], cell[2]] - 1  # if the cell was filled above unity, transferring that surplus to the cell above
            surplus = self.__deposit_reduced_3d[cell] - 1 # saving deposit overfill to distribute among the neighbors later
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
            # Taking into account cases when the cell is at the edge:
            # Small note_: views should be first decreased from the end ([:2])
            # and then from the begining. Otherwise desreasing from the end will have no effect.
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
            # neighbors_1st = s_[cell[0]-1:cell[0]+2, cell[1]-1:cell[1]+2, cell[2]-1:cell[2]+2]
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
            # neighbors_2nd = s_[cell[0]-2:cell[0]+3, cell[1]-2:cell[1]+3, cell[2]-2:cell[2]+3]
            neighbors_2nd = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]

            deposit_kern = self.__deposit_reduced_3d[neighbors_1st]
            semi_s_kern = self.__semi_surface_reduced_3d[neighbors_1st]
            surf_kern = self.__surface_reduced_3d[neighbors_1st]
            temp_kern = self.__temp_reduced_3d[neighbors_1st]
            # Creating condition array
            condition = np.logical_and(deposit_kern == 0, neibs_sides)  # True for elements that are not deposited and are side neighbors
            # Updating main arrays
            semi_s_kern[condition] = False
            surf_kern[condition] = True
            condition = np.logical_and(np.logical_and(deposit_kern == 0, surf_kern == 0), neibs_edges)  # True for elements that are not deposited, not surface cells and are edge neighbors
            semi_s_kern[condition] = True
            ghosts_kern = self.__ghosts_reduced_3d[neighbors_1st]
            ghosts_kern[...] = False
            deposit_kern[surf_kern] += surplus/np.count_nonzero(surf_kern) # distributing among the neighbors
            condition = (temp_kern > 0)
            self.__temp_reduced_3d[cell] = temp_kern[condition].sum()/np.count_nonzero(condition)

            surf_kern = self.__surface_reduced_3d[neighbors_2nd]
            semi_s_kern = self.__semi_surface_reduced_3d[neighbors_2nd]
            ghosts_kern = self.__ghosts_reduced_3d[neighbors_2nd]
            condition = np.logical_and(surf_kern == 0, semi_s_kern == 0)  # True for elements that are neither surface nor semi-surface cells
            ghosts_kern[condition] = True

            self.__deposit_reduced_3d[cell] = -1  # a fully deposited cell is always a minus unity
            self.__precursor_reduced_3d[cell] = 0
            self.__ghosts_reduced_3d[cell] = True  # deposited cell belongs to ghost shell
            self.__surface_reduced_3d[cell] = False  # rising the surface one cell up (new cell)
            precursor_kern = self.__precursor_reduced_3d[neighbors_2nd]
            precursor_kern[semi_s_kern] += 0.000001  # for stensil and plotting purpose (to pass vtk threshold filter)
            precursor_kern[surf_kern] += 0.000001

            self.__get_max_z()
            if self.temperature_tracking:
                self.__temp_reduced_3d[cell] = 0
                temp_kern = self.__temp_reduced_3d[neighbors_1st]
                condition = (temp_kern > 0)
                self.__temp_reduced_3d[cell] = temp_kern[condition].sum() / np.count_nonzero(condition)
                self.request_temp_recalc = self.filled_cells > self.temp_calc_count * self.temp_step_cells
                if self.request_temp_recalc:
                    self._get_solid_index()

            self.__area_2D_no_sub = np.s_[self.substrate_height + 1:self.max_z, :, :]
            self.irradiated_area_2D = np.s_[self.structure.substrate_height-1:self.max_z, :, :] # a volume encapsulating the whole surface
            self.__update_views_2d()
            # Updating surface nearest neighbors array
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
            shape_old = self.structure.shape
            self.structure.resize_structure(200)
            self.structure.define_surface_neighbors(self.max_neib)
            beam_matrix = self.beam_matrix # taking care of the beam_matrix, because __set_structure creates it empty
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
        # Instead of processing cell by cell and on the whole surface, it is implemented to process only (effectively) irradiated area and array-wise(thanks to Numpy)
        # np.float32 — ~1E-7, produced value — ~1E-10
        const = self.sigma*self.V*self.dt * 1e6 * self.deposition_scaling / self.cell_V * self.cell_dimension**2 # multiplying by 1e6 to preserve accuracy
        self.__deposit_reduced_3d[self.__deposition_index] += self.__precursor_reduced_3d[self.__deposition_index] * self.__beam_matrix_effective * const / 1e6

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
        diffusion_matrix = self._laplace_term_stencil(precursor, surface_all)  # Diffusion term is calculated separately and added in the end
        precursor[surface] += self.__rk4(precursor[surface], self.__beam_matrix_surface)  # An increment is calculated through Runge-Kutta method without the diffusion term
        precursor[surface_all] += diffusion_matrix  # finally adding diffusion term

    def __rk4(self, precursor, beam_matrix):
        """
        Calculates increment of precursor density by Runge-Kutta method

        :param precursor: flat precursor array
        :param beam_matrix: flat surface electron flux array
        :return:
        """
        k1 = self.__precursor_density_increment(precursor, beam_matrix, self.dt)  # this is actually an array of k1 coefficients
        k2 = self.__precursor_density_increment(precursor, beam_matrix, self.dt / 2, k1 / 2)
        k3 = self.__precursor_density_increment(precursor, beam_matrix, self.dt / 2, k2 / 2)
        k4 = self.__precursor_density_increment(precursor, beam_matrix, self.dt, k3)
        return evaluate_cached(self.expressions["rk4"], casting='same_kind')

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
            return evaluate_cached(self.expressions['precursor_density_temp'],
                                    local_dict={'F': self.F, 'dt': dt, 'n0':self.n0,
                                    'sigma':self.sigma, 'sub': precursor+addon, 'tau': tau,
                                    'flux_matrix': beam_matrix}, casting='same_kind')
        else:
            return evaluate_cached(self.expressions["precursor_density"],
                               local_dict={'F_dt': self.F * dt,
                                           'F_dt_n0_1_tau_dt': (self.F * dt * self.tau + self.n0 * dt) / (
                                                       self.tau * self.n0),
                                           'addon': addon, 'flux_matrix': beam_matrix, 'sigma_dt': self.sigma * dt,
                                           'sub': precursor}, casting='same_kind')

    def _laplace_term_stencil(self, grid, surface, add=0, div=0):
        """
        Calculates diffusion term for all surface cells using stencil operator

        :param grid: precursor array
        :param surface: boolean surface array
        :param add: Runge-Kutta intermediate member
        :param div:
        :return: flat ndarray
        """
        if self.temperature_tracking:
            D_param = self.__D_temp_reduced_2d
        else:
            D_param = self.D
        return diffusion.diffusion_stencil(grid, surface, D_param, self.dt, self.cell_dimension, self.__surface_all_index, add=add, div=div)
        # return evaluate_cached(expressions["laplace1"], local_dict={'dt_D': dt*D, 'grid_out':grid_out[surface]}, casting='same_kind')

    def _laplace_term_rolling(self, grid, surface, ghosts=None, add=0, div: int = 0):
        """
        Calculates diffusion term for all surface cells using rolling


        :param grid: 3D precursor density array
        :param surface: boolean surface array
        :param ghosts: array representing ghost cells
        :param add: Runge-Kutta intermediate member
        :param div:
        :return: to grid array
        """

        return diffusion.diffusion_rolling(grid, surface, ghosts, self.D, self.dt, self.cell_dimension, flat=True, add=add, div=div)

    def heat_transfer(self, heating):
        if self.max_z - self.substrate_height - 3 > 2:
            if self.request_temp_recalc:
                self.temp_calc_count += 1
                slice = self.__area_2D_no_sub # using only top layer of the substrate
                if type(heating) is np.ndarray:
                    heat = heating[slice]
                else:
                    heat = heating
                # heat_transfer.heat_transfer_BE(self.temp[slice], 'heatsink', self.heat_cond, self.cp,
                #                                    self.rho, self.dt, self.cell_dimension, heat,)
                # self.temp[slice] += heat_transfer.temperature_stencil(self.temp[slice], self.heat_cond, self.cp,
                #                                    self.rho, self.dt, self.cell_dimension, heat,)
                start = df()
                heat_transfer.heat_transfer_steady_sor(self.temp[slice], self.heat_cond, self.cell_dimension, heat, 1e-7, self._solid_index)
                print(f'Temperature recalculation took {df() - start:.4f} s')
        self.temp[self.substrate_height] = self.room_temp
        self.__get_surface_temp()
        self.diffusion_coefficient()
        self.residence_time()

    def diffusion_coefficient(self):
        self.D_temp[self._surface_all] = self.D0 * np.exp(-self.Ed / self.kb / self.surface_temp[self._surface_all])

    def residence_time(self):
        self.__tau_flat = 1 / self.k0 * np.exp(self.Ea / self.kb / self.__surface_temp_reduced_2d[self.__surface_reduced_2d])
        self.__tau_temp_reduced_2d[self.__surface_reduced_2d] = self.__tau_flat

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
        self.__fix_bad_cells()
        indices = np.nonzero(self.__beam_matrix_reduced_2d)
        y_start, y_end, x_start, x_end = indices[1].min(), indices[1].max()+1, indices[2].min(), indices[2].max()+1
        self.irradiated_area_3D = np.s_[self.structure.substrate_height:self.max_z, y_start:y_end, x_start:x_end]  # a slice of the currently irradiated area
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
        self.__surface_all_reduced_2d[:,:,:] = np.logical_or(self.__surface_reduced_2d, self.__semi_surface_reduced_2d)
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
        index = (index<0).nonzero()
        self._solid_index = (np.intc(index[0]), np.intc(index[1]), np.intc(index[2]))
        return self._solid_index

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
    def __fix_bad_cells(self):
        # Temporary fix
        # Very rarely Monte Carlo primary electron simulation produces duplicate coordinates,
        # that lead to division by zero(distance=0) and then to type size overflow(infinity),
        # which does not raise exceptions in Cython.
        # This results in random cells having huge negative numbers
        if self.beam_matrix.min() < 0:
            self.__beam_matrix_reduced_2d[self.__beam_matrix_reduced_2d < 0] = 0

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
        # Precompiled expressions for numexpr_mod.evaluate_cached function
        self.expressions = dict(rk4=cache_expression("(k1+k4)/6 +(k2+k3)/3",
                                                     signature=[('k1', np.float64), ('k2', np.float64),
                                                                ('k3', np.float64), ('k4', np.float64)]),
                                # precursor_density_=cache_expression("(F * (1 - (sub + addon) / n0) - (sub + addon) / tau - (sub + addon) * sigma * flux_matrix)*dt", [('F', np.int64), ('addon', np.float64), ('dt', np.float64), ('flux_matrix', np.int64), ('n0', np.float64), ('sigma',np.float64), ('sub', np.float64), ('tau', np.float64)]),
                                precursor_density=cache_expression(
                                    "F_dt - (sub + addon) * (F_dt_n0_1_tau_dt + sigma_dt * flux_matrix)",
                                    signature=[('F_dt', np.float64), ('F_dt_n0_1_tau_dt', np.float64),
                                               ('addon', np.float64), ('flux_matrix', np.int64),
                                               ('sigma_dt', np.float64), ('sub', np.float64)]),
                                precursor_density_temp = cache_expression("F*dt*(1-sub/n0) - sub*dt/tau - sub*sigma*flux_matrix*dt",
                                            signature=[('F', np.float64), ('dt', np.float64), ('flux_matrix', np.int32), ('n0', np.float64),
                                                        ('sigma', np.float64), ('sub', np.float64), ('tau', np.float64), ]),
                                laplace1=cache_expression("grid_out*dt_D",
                                                          signature=[('dt_D', np.float64), ('grid_out', np.float64)]),
                                laplace2=cache_expression("grid_out*dt_D_div", signature=[('dt_D_div', np.float64),
                                                                                          ('grid_out', np.float64)]))

    def printing(self, x, y, dwell_time):
        pass
    @property
    def kd(self):
        return self.F/self.n0 + 1/self.tau + self.sigma * self.beam_matrix.max()
    @property
    def kr(self):
        return self.F/self.n0 + 1/self.tau
    @property
    def nr(self):
        return self.F/self.kr
    @property
    def nd(self):
        return self.F/self.kd
    @property
    def max_temperature(self):
        return self.__surface_temp_reduced_2d.max()
    @property
    def deposited_vol(self):
        return (self.filled_cells + self.deposit[self.surface].sum()) * self.cell_V

if __name__ == '__main__':
    print("Current script does not have an entry point.....")
    input('Press Enter to exit.')
