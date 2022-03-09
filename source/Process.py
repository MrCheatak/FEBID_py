# Default packages
import os, sys, datetime
import copy, math
from contextlib import suppress

# Core packages
import numpy as np
from numexpr_mod import evaluate_cached, cache_expression

# Axillary packeges
import scipy.constants as scpc
from tqdm import tqdm
from tkinter import filedialog as fd
import pyvista as pv
import pickle
import yaml
import timeit
import line_profiler

# Local packages
from Structure import Structure
import VTK_Rendering as vr
import etraj3d
from libraries.rolling import roll

# TODO: look into k-d trees
# TODO: add a benchmark to determine optimal threads number for current machine


# <editor-fold desc="Parameters">
# td = 1E-6  # dwell time of a beam, s
# Ie = 1E-10  # beam current, A
# beam_d = 10  # electron beam diameter, nm
# effective_diameter = beam_d * 3.3 # radius of an area which gets 99% of the electron beam
# f = Ie / scpc.elementary_charge / (math.pi * beam_d * beam_d / 4)  # electron flux at the surface, 1/(nm^2*s)
# F = 3000  # precursor flux at the surface, 1/(nm^2*s)   here assumed a constant, but may be dependent on time and position
# tau = 500E-6  # average residence time, s; may be dependent on temperature
#
# # Precursor properties
# sigma = 2.2E-2  # dissociation cross section, nm^2; is averaged from cross sections of all electron types (PE,BSE, SE1, SE2)
# n0 = 1.9  # inversed molecule size, Me3PtCpMe, 1/nm^2
# M_Me3PtCpMe = 305  # molar mass of the precursor Me3Pt(IV)CpMe, g/mole
# p_Me3PtCpMe = 1.5E-20  # density of the precursor Me3Pt(IV)CpMe, g/nm^3
# V = 4 / 3 * math.pi * math.pow(0.139, 3)  # atomic volume of the deposited atom (Pt), nm^3
# D = np.float32(1E5)  # diffusion coefficient, nm^2/s
#
#
# kd = F / n0 + 1 / tau + sigma * f  # depletion rate
# kr = F / n0 + 1 / tau  # replenishment rate
# nr = F / kr  # absolute density after long time
# nd = F / kd  # depleted absolute density
# t_out = 1 / (1 / tau + F / n0)  # effective residence time
# p_out = 2 * math.sqrt(D * t_out) / beam_d
# cell_dimension = 5  # side length of a square cell, nm
#
# effective_radius_relative = math.floor(effective_diameter / cell_dimension / 2)
#
# nn=1 # default number of threads for numexpr
# </editor-fold>

# <editor-fold desc="Timings">
# dt = np.float32(1E-6)  # time step, s
# t_flux = 1/(sigma*f)  # dissociation event time
# diffusion_dt = math.pow(cell_dimension * cell_dimension, 2) / (2 * D * (cell_dimension * cell_dimension + cell_dimension * cell_dimension))   # maximum stability lime of the diffusion solution
# tau = 500E-6  # average residence time, s; may be dependent on temperature
# </editor-fold>

# <editor-fold desc="Framework" >
# Main cell matrices
# system_size = 50
# height_multiplyer = 2
# substrate = zeros((system_size*height_multiplyer, system_size, system_size), dtype=np.float32) # substrate[z,x,y] holds precursor density
# deposit = zeros((system_size*height_multiplyer, system_size, system_size), dtype=np.float32) # deposit[z,y,x] holds deposit density
# substrate[0, :, :] = nr  # filling substrate surface with initial precursor density
# # deposit[0, 20:40, 20:40] = 0.95
# zmax, ymax, xmax = substrate.shape # dimensions of the grid

# It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus able produce deposit under irradiation.
# The idea is to avoid iterating through the whole 3D matrix and address only surface cells
# Thus the number of surface cells is fixed.

# Semi-surface cells are cells that have precursor density but do not have deposit right under them
# Thus they cannot produce deposit and their precursor density is calculated without disosisiation term.
# They are introduced to allow diffusion on the walls of the deposit.
# Basically these are all other surface cells
# </editor-fold>

# <editor-fold desc="Helpers">
# center = effective_radius_relative * cell_dimension  # beam center in array-coordinates
# index_y, index_x = mgrid[0:(effective_radius_relative*2+1), 0:(effective_radius_relative*2+1)] # for indexing purposes of flux matrix
# index_yy, index_xx = index_y*cell_dimension-center, index_x*cell_dimension-center

# A dictionary of expressions for numexpr.evaluate_cached
# Debug note: before introducing a new cached expression, that expression should be run with the default 'evaluate' function for fetching the signature list.
# This is required, because variables in it must be in the same order as Numexpr fetches them, otherwise Numexpr compiler will throw an error
# TODO: this has to go the Structure class
expressions = dict(pe_flux=cache_expression("f*exp(-r*r/(2*beam_d*beam_d))", signature=[('beam_d', np.int32), ('f', np.float64), ('r', np.float64)]),
                   rk4=cache_expression("(k1+k4)/6 +(k2+k3)/3", signature=[('k1', np.float64), ('k2', np.float64), ('k3', np.float64), ('k4', np.float64)]),
                   #precursor_density_=cache_expression("(F * (1 - (sub + addon) / n0) - (sub + addon) / tau - (sub + addon) * sigma * flux_matrix)*dt", [('F', np.int64), ('addon', np.float64), ('dt', np.float64), ('flux_matrix', np.int64), ('n0', np.float64), ('sigma',np.float64), ('sub', np.float64), ('tau', np.float64)]),
                   precursor_density=cache_expression("F_dt - (sub + addon) * (F_dt_n0_1_tau_dt + sigma_dt * flux_matrix)", signature=[('F_dt', np.float64), ('F_dt_n0_1_tau_dt', np.float64), ('addon', np.float64), ('flux_matrix', np.int64), ('sigma_dt',np.float64), ('sub', np.float64)]),
                   laplace1=cache_expression("grid_out*dt_D", signature=[('dt_D', np.float64), ('grid_out', np.float64)]),
                   laplace2=cache_expression("grid_out*dt_D_div", signature=[('dt_D_div', np.float64), ('grid_out', np.float64)]),
                   flux_matrix=cache_expression("((index_xx-center)*(index_xx-center)+(index_yy-center)*(index_yy-center))**0.5", signature=[('center', np.int32), ('index_xx', np.int32), ('index_yy', np.int32)]))
# </editor-fold>

class Process():
    """
    Class representing the core deposition process.
    It contains all necessary arrays, variables, parameters and methods to support a continuous deposition process.
    """
    def __init__(self, structure, mc_config, equation_values, timings, time_spent=datetime.datetime.now(), deposition_scaling=1, name=None):
        if not name:
            self.name = str(np.random.randint(000000, 999999, 1)[0])
        else:
            self.name = name
        # Declaring necessary  properties
        self.structure = None
        self.cell_dimension = None
        # Main arrays
        # Semi-surface cells are virtual cells that can hold precursor density value, but cannot create deposit.
        # Their role is to serve as pipes on steps, where regular surface cells are not in contact.
        # Therefore, these cells take part in diffusion process, but are not taken into account when calculating
        # other terms in the FEBID equation or deposit increment.
        self.deposit = None # contains values from 0 to 1 describing the filling state of a cell
        self.precursor = None # contains values from 0 to 1 describing normalized precursor density in a cell
        self.surface = None # a boolean array, surface cells are True
        self.semi_surface = None # a boolean array, semi-surface cells are True
        self.ghosts = None # a boolean array, ghost cells are True
        self.beam_matrix = None # contains values of the SE surface flux
        
        # Working arrays
        self.__deposit_reduced_3d = None
        self.__precursor_reduced_3d = None
        self.__surface_reduced_3d = None
        self.__semi_surface_reduced_3d = None
        self.__surface_all_reduced_3d = None
        self.__ghosts_reduced_3d = None
        self.__ghosts_reduced_2d = None
        self.__beam_matrix_reduced_2d = None

        # Helpers
        self._surface_all = None
        self.__beam_matrix_surface = None
        self.__beam_matrix_effective = None
        self.__deposition_index = None
        self.__surface_index = None

        # Monte Carlo simulation instance
        self.sim = None
        
        # Physical variables
        self.F = 0 # precursor surface flux
        self.n0 = 0 # maximum absolute precursor density
        self.sigma = 0 # integral cross-section
        self.tau = 0 # residence time
        self.V = 0 # deposit volume of a dissociated precursor molecule
        self.D = 0 # surface diffusion coefficient
        
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
        self.deposition_scaling = deposition_scaling # multiplier of the deposit increment; used to speed up the process
        self.redraw = True # flag for external functions saying that surface has been updated
        
        # Statistics
        self.substrate_height = 0 # Thickness of the substrate
        self.n_substrate_cells = 0 # the number of the cells in the substrate
        self.max_z = 0 # maximum height of the deposited structure, cells
        self.n_filled_cells = []
        self.growth_rate = []
        self.time_spent = time_spent
        self.start_time = datetime.datetime.now()
        self.execution_speed = 0
        self.profiler= None

        # Initialization sequence
        self.__set_structure(structure)
        self.__set_constants(equation_values)
        self.__setup_MC_module(mc_config)
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
        self._surface_all = np.logical_or(self.surface, self.semi_surface)
        self.ghosts = self.structure.ghosts_bool
        self.beam_matrix = np.zeros_like(structure.deposit, dtype=np.int32)
        self.cell_dimension = self.structure.cell_dimension
        self.__get_max_z()
        self.substrate_height = structure.substrate_height
        self.irradiated_area_2D = np.s_[self.structure.substrate_height - 1:self.max_z, :, :]
        self.__update_views_2d()
        self.__update_views_3d()
        self.__generate_surface_index()
        self.__generate_deposition_index()
        self.n_substrate_cells = self.deposit[:structure.substrate_height].size

    def __set_constants(self, params):
        self.F = params['F']
        self.n0 = params['n0']
        self.V = params['V']
        self.sigma = params['sigma']
        self.tau = params['tau']
        self.D = params['D']
        self.deposition_scaling = params['deposition_scaling']

    def __setup_MC_module(self, params):
        mc_sim = etraj3d.cache_params(params, self.deposit, self.surface)

    def __get_timings(self, timings):
        # Stability time steps
        self.t_diffusion = timings['t_diff']
        self.t_dissociation = timings['t_flux']
        self.t_desorption = timings['t_desorption']
        self.get_dt()
        self.t = 1E-10

    def get_dt(self):
        self.dt = min(self.t_desorption, self.t_diffusion, self.t_dissociation)/5

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
        # What here actually done is marking the filled cell as solid and ghost cell and then updating surface,
        # semi-surface, ghosts and precursor to describe the surface geometry around the newly filled cell.
        # The approach is cell-centric, which means all the surroundings are processed
        nd = (self.__deposit_reduced_3d >= 1).nonzero()
        new_deposits = [(nd[0][i], nd[1][i], nd[2][i]) for i in range(nd[0].shape[0])]
        for cell in new_deposits:
            # deposit[cell[0]+1, cell[1], cell[2]] += deposit[cell[0], cell[1], cell[2]] - 1  # if the cell was filled above unity, transferring that surplus to the cell above
            surplus = self.__deposit_reduced_3d[cell] - 1 # saving deposit overfill to distribute among the neighbors later
            self.__deposit_reduced_3d[cell] = -1  # a fully deposited cell is always a minus unity
            self.__precursor_reduced_3d[cell] = 0
            self.__ghosts_reduced_3d[cell] = True  # deposited cell belongs to ghost shell
            self.__surface_reduced_3d[cell] = False  # rising the surface one cell up (new cell)
            self.redraw = True

            # Instead of using classical conditions, boolean arrays are used to select elements
            # First, a condition array is created, that picks only elements that satisfy conditions
            # Then this array is used as index

            neibs_sides, neibs_edges = self.__neibs_sides, self.__neibs_edges

            # Creating a view with the 1st nearest neighbors to the deposited cell
            z_min, z_max, y_min, y_max, x_min, x_max = 0, 0, 0, 0, 0, 0
            # Taking into account cases when the cell is at the edge:
            if cell[0] - 1 < 0:
                z_min = 0
                neibs_sides = neibs_sides[1:, :, :]
                neibs_edges = neibs_edges[1:, :, :]
            else:
                z_min = cell[0] - 1
            if cell[0] + 2 > self.__deposit_reduced_3d.shape[0]:
                z_max = self.__deposit_reduced_3d.shape[0]
                neibs_sides = neibs_sides[:1, :, :]
                neibs_edges = neibs_edges[:1, :, :]
            else:
                z_max = cell[0] + 2
            if cell[1] - 1 < 0:
                y_min = 0
                neibs_sides = neibs_sides[:, 1:, :]
                neibs_edges = neibs_edges[:, 1:, :]
            else:
                y_min = cell[1] - 1
            if cell[1] + 2 > self.__deposit_reduced_3d.shape[1]:
                y_max = self.__deposit_reduced_3d.shape[1]
                neibs_sides = neibs_sides[:, :1, :]
                neibs_edges = neibs_edges[:, :1, :]
            else:
                y_max = cell[1] + 2
            if cell[2] - 1 < 0:
                x_min = 0
                neibs_sides = neibs_sides[:, :, 1:]
                neibs_edges = neibs_edges[:, :, 1:]
            else:
                x_min = cell[2] - 1
            if cell[2] + 2 > self.__deposit_reduced_3d.shape[2]:
                x_max = self.__deposit_reduced_3d.shape[2]
                neibs_sides = neibs_sides[:, :, :1]
                neibs_edges = neibs_edges[:, :, :1]
            else:
                x_max = cell[2] + 2
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
            precursor_kern[semi_s_kern] += 0.000001  # only for plotting purpose (to pass vtk threshold filter)
            precursor_kern[surf_kern] += 0.000001

            self.__get_max_z()
            self.irradiated_area_2D = np.s_[self.structure.substrate_height-1:self.max_z, :, :] # a volume encapsulating the whole surface
            self.__update_views_2d()

        if self.max_z + 5 > self.structure.shape[0]:
            # Here the Structure is extended in height
            # and all the references to the data arrays are renewed
            shape_old = self.structure.shape
            self.structure.resize_structure(200)
            beam_matrix = self.beam_matrix # taking care of the beam_matrix, because __set_structure creates it empty
            self.__set_structure(self.structure)
            self.beam_matrix[:shape_old[0], :shape_old[1], :shape_old[2]] = beam_matrix
            # Basically, none of the slices have to be updated, because they use indexes, not references.
        return False

    def deposition(self):
        """
        Calculate an increment of a deposited volume for all irradiated cells over a time step

        :return:
        """
        # Instead of processing cell by cell and on the whole surface, it is implemented to process only (effectively) irradiated area and array-wise(thanks to Numpy)
        # Math here cannot be efficiently simplified, because multiplication of constant variables here produces a value below np.float32 accuracy
        # np.float32 — ~1E-7, produced value — ~1E-10
        const = self.sigma*self.V*self.dt * self.deposition_scaling
        self.__deposit_reduced_3d[self.__deposition_index] += self.__precursor_reduced_3d[self.__deposition_index] * self.__beam_matrix_effective * const

    def precursor_density(self):
        """
        Calculate an increment of the precursor density for every surface cell surface

        :return:
        """
        precursor = self.__precursor_reduced_2d
        surface_all = self.__surface_all_reduced_2d
        surface = self.__surface_reduced_2d
        diffusion_matrix = self._laplace_term_stencil(precursor, surface_all)  # Diffusion term is calculated separately and added in the end
        precursor[surface] += self.__rk4(precursor[surface], self.__beam_matrix_surface )  # An increment is calculated through Runge-Kutta method without the diffusion term
        # precursor[semi_surf_bool] += rk4(dt, precursor[semi_surf_bool])  # same process for semi-cells, but without dissociation term
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
        grid += add
        grid_out = -6 * grid
        roll.stencil(grid_out, grid, *self.__surface_index)
        grid -= add
        return grid_out[surface] * self.dt * self.D
        # return evaluate_cached(expressions["laplace1"], local_dict={'dt_D': dt*D, 'grid_out':grid_out[surface]}, casting='same_kind')

    def _laplace_term_rolling(self, grid, ghosts=None, add=0, div: int = 0):
        """
        Calculates diffusion term for all surface cells using rolling


        :param grid: 3D precursor density array
        :param ghosts: array representing ghost cells
        :param add: Runge-Kutta intermediate member
        :param div:
        :return: to grid array
        """

        # Debugging note_: it would be more elegant to just use numpy.roll() on the ghosts_bool to assign neighboring values
        # to ghost cells. But Numpy doesn't retain array structure when utilizing boolean index streaming. It rather extracts all the cells
        # (that correspond to True in our case) and processes them as a flat array. It caused the shifted values for ghost cells to
        # be assigned to the previous(first) layer, which was not processed by numpy.roll() when it rolled backwards.
        # Thus, borders(planes) that are not taking part in rolling(shifting) are cut off by using views to an array
        if ghosts is None:
            ghosts = self.ghosts
        grid = grid + add
        grid_out = grid * (-6)

        # X axis:
        # No need to have a separate array of values, when whe can conveniently call them from the original data
        shore = grid[:, :, 1:]
        wave = grid[:, :, :-1]
        shore[ghosts[:, :, 1:]] = wave[
            ghosts[:, :, 1:]]  # assigning values to ghost cells forward along X-axis
        # grid_out[:,:, :-1]+=grid[:,:, 1:] #rolling forward (actually backwards)
        roll.rolling_3d(grid_out[:, :, :-1], grid[:, :, 1:])
        index = ghosts.reshape(-1).nonzero()
        # grid_out[:,:,-1] += grid[:,:,-1] #taking care of edge values
        roll.rolling_2d(grid_out[:, :, -1], grid[:, :, -1])
        # grid[ghosts_bool] = 0 # flushing ghost cells
        grid.reshape(-1)[index] = 0
        # Doing the same, but in reverse
        shore = grid[:, :, :-1]
        wave = grid[:, :, 1:]
        shore[ghosts[:, :, :-1]] = wave[ghosts[:, :, :-1]]
        # grid_out[:,:,1:] += grid[:,:,:-1] #rolling backwards
        roll.rolling_3d(grid_out[:, :, 1:], grid[:, :, :-1])
        # grid_out[:, :, 0] += grid[:, :, 0]
        roll.rolling_2d(grid_out[:, :, 0], grid[:, :, 0])
        # grid[ghosts_bool] = 0
        grid.reshape(-1)[index] = 0

        # Y axis:
        shore = grid[:, 1:, :]
        wave = grid[:, :-1, :]
        shore[ghosts[:, 1:, :]] = wave[ghosts[:, 1:, :]]
        # grid_out[:, :-1, :] += grid[:, 1:, :]
        roll.rolling_3d(grid_out[:, :-1, :], grid[:, 1:, :])
        # grid_out[:, -1, :] += grid[:, -1, :]
        roll.rolling_2d(grid_out[:, -1, :], grid[:, -1, :])
        # grid[ghosts_bool] = 0
        grid.reshape(-1)[index] = 0
        shore = grid[:, :-1, :]
        wave = grid[:, 1:, :]
        shore[ghosts[:, :-1, :]] = wave[ghosts[:, :-1, :]]
        # grid_out[:, 1:, :] += grid[:, :-1, :]
        roll.rolling_3d(grid_out[:, 1:, :], grid[:, :-1, :])
        # grid_out[:, 0, :] += grid[:, 0, :]
        roll.rolling_2d(grid_out[:, 0, :], grid[:, 0, :])
        # grid[ghosts_bool] = 0
        grid.reshape(-1)[index] = 0

        # Z axis:
        shore = grid[1:, :, :]
        wave = grid[:-1, :, :]
        shore[ghosts[1:, :, :]] = wave[ghosts[1:, :, :]]
        # c
        roll.rolling_3d(grid_out[:-1, :, :], grid[1:, :, :])
        # grid_out[-1, :, :] += grid[-1, :, :]
        roll.rolling_2d(grid_out[-1, :, :], grid[-1, :, :])
        # grid[ghosts_bool] = 0
        grid.reshape(-1)[index] = 0
        shore = grid[:-1, :, :]
        wave = grid[1:, :, :]
        shore[ghosts[:-1, :, :]] = wave[ghosts[:-1, :, :]]
        # grid_out[1:, :, :] += grid[:-1, :, :]
        roll.rolling_3d(grid_out[1:, :, :], grid[:-1, :, :])
        # grid_out[0, :, :] += grid[0, :, :]
        roll.rolling_2d(grid_out[0, :, :], grid[0, :, :])
        # grid[ghosts_bool] = 0
        grid.reshape(-1)[index] = 0
        # grid_out[ghosts_bool]=0
        grid_out.reshape(-1)[index] = 0  # result also has to be cleaned as it contains redundant values in ghost cells
        # numexpr: 1 core performs better
        # numexpr.set_num_threads(nn)
        return evaluate_cached(self.expressions["laplace1"],
                               local_dict={'dt_D': self.dt * self.D, 'grid_out': grid_out},
                               casting='same_kind')
        # else:
        #     return evaluate_cached(expressions["laplace2"], local_dict={'dt_D_div': dt*D/div, 'grid_out':grid_out}, casting='same_kind')

    # Data maintenance methods
    # These methods represent an optimization path that provides up to 100x speed up
    # 1. By selecting and processing chunks of arrays (views) that are effectively changing
    # 2. By preparing indexes for arrays
    def update_helper_arrays(self):
        """
        Define new views to data arrays, create axillary indexes and flatten beam_matrix array
        :return:
        """
        # Basically, procedure provided by this method is optional and serves as an optimisation.
        # Although,
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
        Generate a tuple of indices for faster indexing in 'deposition' method

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
        self.__surface_index = (np.intc(index[0]), np.intc(index[1]), np.intc(index[2]))

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

    def __get_max_z(self):
        """
        Get z position of the highest not empty cell in the structure
        :return:
        """
        self.max_z = self.deposit.nonzero()[0].max() + 3

    # Misc
    def __fix_bad_cells(self):
        # Temporary fix
        # Very rarely Monte Carlo primary electron simulation produces duplicate coordinates,
        # that lead to division by zero(distance=0) and then to type size overflow(infinity),
        # which does not raise exceptions in Cython.
        # This results in random cells having huge negative numbers
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
                                laplace1=cache_expression("grid_out*dt_D",
                                                          signature=[('dt_D', np.float64), ('grid_out', np.float64)]),
                                laplace2=cache_expression("grid_out*dt_D_div", signature=[('dt_D_div', np.float64),
                                                                                          ('grid_out', np.float64)]))

    def printing(self, x, y, dwell_time):
        pass



# @jit(nopython=True, parallel=True)
def deposition(deposit, precursor, flux_flat, index, sigma_V_dt, gr = 1.0):

    """
    Calculates deposition on the surface for a given time step dt (outer loop)

    :param deposit: 3D deposit array
    :param precursor: 3D precursor density array
    :param flux_matrix: matrix of electron flux distribution
    :param dt: time step
    :return: writes back to deposit array
    """
    # Instead of processing cell by cell and on the whole surface, it is implemented to process only (effectively) irradiated area and array-wise(thanks to Numpy)
    # Math here cannot be efficiently simplified, because multiplication of constant variables here produces a value below np.float32 accuracy
    # np.float32 — ~1E-7, produced value — ~1E-10
    # deposit[surface_bool] += precursor[surface_bool] * flux_matrix[surface_bool] * sigma_V_dt * gr
    const = sigma_V_dt * gr
    deposit[index] += precursor[index] * flux_flat * const



# @jit(nopython=True, parallel=True)
def update_surface(deposit, precursor, surface_bool, semi_surf_bool, ghosts_bool):
    """
    Evolves surface upon a full deposition of a cell. This method holds has the vast majority of logic

    :param deposit: 3D deposit array
    :param precursor: 3D precursor density array
    :param surface_bool: array representing surface cells
    :param semi_surf_bool: array representing semi-surface cells
    :param ghosts_bool: array representing ghost cells
    :return: changes surface, semi-surface and ghosts arrays
    """
    # new_deposits = np.argwhere(deposit >= 1)  # looking for new deposits
    if deposit.max()>=1:
        nd = (deposit>=1).nonzero()
        new_deposits = [(nd[0][i], nd[1][i], nd[2][i]) for i in range(nd[0].shape[0])]
        for cell in new_deposits:
            # deposit[cell[0]+1, cell[1], cell[2]] += deposit[cell] - 1  # if the cell was filled above unity, transferring that surplus to the cell above
            deposit[cell] = -1  # a fully deposited cell is always a minus unity
            precursor[cell] = 0
            ghosts_bool[cell] = True  # deposited cell belongs to ghost shell
            surface_bool[cell] = False  # rising the surface one cell up (new cell)

            # Instead of using classical conditions, boolean arrays are used to select elements
            # First, a condition array is created, that picks only elements that satisfy conditions
            # Then this array is used as index

            # Kernels for choosing cells
            neibs_sides = np.array([[[0, 0, 0],  # chooses side neighbors
                                     [0, 1, 0],
                                     [0, 0, 0]],
                                    [[0, 1, 0],
                                     [1, 0, 1],
                                     [0, 1, 0]],
                                    [[0, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]]])
            neibs_edges = np.array([[[0, 1, 0],  # chooses edge neighbors
                                     [1, 0, 1],
                                     [0, 1, 0]],
                                    [[1, 0, 1],
                                     [0, 0, 0],
                                     [1, 0, 1]],
                                    [[0, 1, 0],
                                     [1, 0, 1],
                                     [0, 1, 0]]])

            # Creating a view with the 1st nearest neighbors to the deposited cell
            z_min, z_max, y_min,y_max, x_min, x_max = 0,0,0,0,0,0
            if cell[0]-1 < 0:
                z_min = 0
                neibs_sides = neibs_sides[1:,:,:]
                neibs_edges = neibs_edges[1:,:,:]
            else:
                z_min = cell[0] - 1
            if cell[0]+2 > deposit.shape[0]:
                z_max = deposit.shape[0]
                neibs_sides = neibs_sides[:1,:,:]
                neibs_edges = neibs_edges[:1,:,:]
            else:
                z_max = cell[0] + 2
            if cell[1]-1<0:
                y_min = 0
                neibs_sides = neibs_sides[:,1:,:]
                neibs_edges = neibs_edges[:,1:,:]
            else:
                y_min = cell[1] - 1
            if cell[1]+2 > deposit.shape[1]:
                y_max = deposit.shape[1]
                neibs_sides = neibs_sides[:,:1,:]
                neibs_edges = neibs_edges[:,:1,:]
            else:
                y_max = cell[1] + 2
            if cell[2]-1 < 0:
                x_min = 0
                neibs_sides = neibs_sides[:,:,1:]
                neibs_edges = neibs_edges[:,:,1:]
            else:
                x_min = cell[2] - 1
            if cell[2]+2>deposit.shape[2]:
                x_max = deposit.shape[2]
                neibs_sides = neibs_sides[:, :, :1]
                neibs_edges = neibs_edges[:, :, :1]
            else:
                x_max = cell[2] + 2
            # neighbors_1st = s_[cell[0]-1:cell[0]+2, cell[1]-1:cell[1]+2, cell[2]-1:cell[2]+2]
            neighbors_1st = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]
            # Creating a view with the 2nd nearest neighbors to the deposited cell
            if cell[0]-2 < 0:
                z_min = 0
            else:
                z_min = cell[0] - 2
            if cell[0]+3 > deposit.shape[0]:
                z_max = deposit.shape[0]
            else:
                z_max = cell[0] + 3
            if cell[1]-2<0:
                y_min = 0
            else:
                y_min = cell[1] - 2
            if cell[1]+3 > deposit.shape[1]:
                y_max = deposit.shape[1]
            else:
                y_max = cell[1] + 3
            if cell[2]-2 < 0:
                x_min = 0
            else:
                x_min = cell[2] - 2
            if cell[2]+3>deposit.shape[2]:
                x_max = deposit.shape[2]
            else:
                x_max = cell[2] + 3
            # neighbors_2nd = s_[cell[0]-2:cell[0]+3, cell[1]-2:cell[1]+3, cell[2]-2:cell[2]+3]
            neighbors_2nd = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]


            deposit_kern = deposit[neighbors_1st]
            semi_s_kern = semi_surf_bool[neighbors_1st]
            surf_kern = surface_bool[neighbors_1st]
            # Creating condition array
            condition = np.logical_and(deposit_kern==0, neibs_sides) # True for elements that are not deposited and are side neighbors
            semi_s_kern[condition] = False
            surf_kern[condition] = True
            condition = np.logical_and(np.logical_and(deposit_kern==0, surf_kern==0), neibs_edges) # True for elements that are not deposited, not surface cells and are edge neighbors
            semi_s_kern[condition] = True
            ghosts_kern = ghosts_bool[neighbors_1st]
            ghosts_kern[...] = False

            surf_kern = surface_bool[neighbors_2nd]
            semi_s_kern = semi_surf_bool[neighbors_2nd]
            ghosts_kern = ghosts_bool[neighbors_2nd]
            condition = np.logical_and(surf_kern==0, semi_s_kern==0) # True for elements that are neither surface nor semi-surface cells
            ghosts_kern[condition] = True

            deposit[cell] = -1  # a fully deposited cell is always a minus unity
            precursor[cell] = 0
            ghosts_bool[cell] = True # deposited cell belongs to ghost shell
            surface_bool[cell] = False  # rising the surface one cell up (new cell)
            precursor_kern = precursor[neighbors_2nd]
            precursor_kern[semi_s_kern] += 0.0001 # only for plotting purpose (to pass vtk threshold filter)
            precursor_kern[surf_kern] +=0.0001
            # refresh(precursor, surface_bool, semi_surf_bool, ghosts_bool, cell[0] + 1, cell[1], cell[2])
        else: # when loop finishes
        #     if cell[0] > h_max-3:
        #         h_max += 1
            return True

    return False


# @jit(nopython=True) # parallel=True)
def refresh(precursor, surface_bool, semi_s_bool, ghosts_bool, z,y,x):
    """
    Updates surface, semi-surface and ghost cells collections according to the provided coordinate of a newly deposited cell

    :param precursor: 3D precursor density array
    :param semi_s_bool: array representing semi-surface cells
    :param ghosts_bool: array representing ghost cells
    :param z: z-coordinate of the cell above the new deposit
    :param y: y-coordinate of the deposit
    :param x: x-coordinate of the deposit
    :return: changes surface array, semi-surface and ghosts collections
    """
    # this is needed, due to the precursor view being 2 cells wider in case of semi-surface or ghost cell falling out of the bounds of the view
    semi_s_bool[z, y, x] = False # removing the new cell from the semi_surface collection, because it belongs to surface now
    semi_s_bool[z-1, y-1, x] = False
    surface_bool[z-1, y-1, x] = True
    semi_s_bool[z-1, y+1, x] = False
    surface_bool[z-1, y+1, x] = True
    semi_s_bool[z-1, y, x-1] = False
    surface_bool[z-1, y, x-1] = True
    semi_s_bool[z-1, y, x+1] = False
    surface_bool[z-1, y, x+1] = True
    ghosts_bool[z, y, x] = False # removing the new cell from the ghost shell collection
    precursor[z, y, x] += precursor[z - 1, y, x] # if the deposited cell had precursor in it, transfer that surplus to the cell above
    # this may lead to an overfilling of a cell above unity, but it is not causing any anomalies due to diffusion process
    precursor[z - 1, y, x] = -1  # precursor density is NaN in the fully deposited cells (it was previously set to zero, but later some of the zero cells were added back to semi-surface)
    if precursor[z+1, y, x] == 0: # if the cell that is above the new cell is empty, then add it to the ghost shell collection
        ghosts_bool[z+1, y, x] = True
    # Adding neighbors(in x-y plane) of the new cell to the semi_surface collection
    # and updating ghost shell for every neighbor:
    with suppress(IndexError): # It basically skips operations that occur out of the array
        if precursor[z, y - 1, x] == 0:
            semi_s_bool[z, y - 1, x] = True
            precursor[z, y - 1, x] += 1E-7 # this "marks" cell as a surface one, because some of the checks refer to if the cell is empty. This assignment is essential. It corresponds to the smallest value that float32 can hold and should be changed corrspondingly to the variable type.
            refresh_ghosts(precursor, ghosts_bool, x, y-1, z) # update ghost shell around
        if precursor[z, y + 1, x] == 0:
            semi_s_bool[z, y + 1, x] = True
            precursor[z, y + 1, x] += 1E-7
            refresh_ghosts(precursor, ghosts_bool,  x, y+1, z)
        if precursor[z, y, x - 1] == 0:
            semi_s_bool[z, y, x - 1] = True
            precursor[z, y, x - 1] += 1E-7
            refresh_ghosts(precursor, ghosts_bool, x-1, y, z)
        if precursor[z, y, x + 1] == 0:
            semi_s_bool[z, y, x + 1] = True
            precursor[z, y, x + 1] += 1E-7
            refresh_ghosts(precursor, ghosts_bool, x+1, y, z)

# @jit(nopython=True) # parallel=True)
def refresh_ghosts(precursor, ghosts_bool, x, y, z):
    """
    Updates ghost cells collection around the specified cell

    :param precursor: 3D precursor density array
    :param ghosts_bool: array representing ghost cells
    :param x: x-coordinates of the cell
    :param y: y-coordinates of the cell
    :param z: z-coordinates of the cell
    :return: changes ghosts array
    """
    # It must be kept in mind, that z-coordinate here is absolute, but x and y are relative to the view
    # Firstly, deleting current cell from ghost shell and then adding all neighboring cells(along all axis) if they are zero
    ghosts_bool[z, y, x] = False
    if precursor[z - 1, y, x] == 0:
        ghosts_bool[z - 1, y, x] = True
    if precursor[z + 1, y, x] == 0:
        ghosts_bool[z + 1, y, x] = True
    if precursor[z, y - 1, x] == 0:
        ghosts_bool[z, y - 1, x] = True
    if precursor[z, y + 1, x] == 0:
        ghosts_bool[z, y + 1, x] = True
    if precursor[z, y, x - 1] == 0:
        ghosts_bool[z, y, x - 1] = True
    if precursor[z, y, x + 1] == 0:
        ghosts_bool[z, y, x + 1] = True


def precursor_density(flux_matrix_flat, precursor, surface_bool, surface_full_bool, surface_index, ghosts, F, n0, tau, sigma, D, dt):
    """
    Recalculates precursor density on the whole surface

    :param flux_matrix: matrix of electron flux distribution
    :param precursor: 3D precursor density array
    :param surface_bool: array representing surface cells
    :param semi_surf_bool: array representing semi-surface cells
    :param ghosts_bool: array representing ghost cells
    :param dt: time step
    :return: changes precursor array
    """
    # diffusion_matrix = laplace_term_rolling(precursor, surface_full_bool, ghosts, D, dt)  # Diffusion term is calculated separately and added in the end
    diffusion_matrix = laplace_term_stencil(precursor, surface_full_bool, surface_index, D, dt) # Diffusion term is calculated separately and added in the end
    precursor[surface_bool] += rk4(dt, precursor[surface_bool], F, n0, tau, sigma, flux_matrix_flat) # An increment is calculated through Runge-Kutta method without the diffusion term
    # precursor[semi_surf_bool] += rk4(dt, precursor[semi_surf_bool])  # same process for semi-cells, but without dissociation term
    precursor[surface_full_bool]+=diffusion_matrix # finally adding diffusion term
    # if precursor[precursor > 1] != 0:
    #     raise Exception("Normalized precursor density is more than 1")


# @jit(nopython=False, parallel=True)
def rk4(dt, sub, F, n0, tau, sigma, flux_matrix=0):
    """
    Calculates increment of precursor density by Runge-Kutta method

    :param dt: time step
    :param sub: array of surface cells
    :param flux_matrix: matrix of electron flux distribution
    :return:
    """
    k1 = precursor_density_increment(dt, sub, flux_matrix, F, n0, tau, sigma,) # this is actually an array of k1 coefficients
    k2 = precursor_density_increment(dt/2, sub, flux_matrix, F, n0, tau, sigma, k1 / 2)
    k3 = precursor_density_increment(dt/2, sub, flux_matrix, F, n0, tau, sigma, k2 / 2)
    k4 = precursor_density_increment(dt, sub, flux_matrix, F, n0, tau, sigma, k3)
    # numexpr: 1 core performs better
    # numexpr.set_num_threads(nn)
    return evaluate_cached(expressions["rk4"], casting='same_kind')


# @jit(nopython=False, parallel=True)
# noinspection PyUnusedLocal
def precursor_density_increment(dt, sub, flux_matrix, F, n0, tau, sigma, addon=0):
    """
    Calculates increment of the precursor density without a diffusion term

    :param dt: time step
    :param sub: array of surface cells (2D for surface cells, 1D dor semi-surface cells)
    :param flux_matrix: matrix of electron flux distribution
    :param addon: Runge Kutta term
    :return: to sub array
    """
    # numexpr: 1 core performs better
    # numexpr.set_num_threads(nn)
    return evaluate_cached(expressions["precursor_density"], local_dict={'F_dt':F*dt, 'F_dt_n0_1_tau_dt': (F*dt*tau+n0*dt)/(tau*n0), 'addon':addon, 'flux_matrix':flux_matrix, 'sigma_dt':sigma*dt, 'sub':sub}, casting='same_kind')


def rk4_diffusion(grid, ghosts_bool, D, dt):
    '''
    Calculates increment of the diffusion term

    :param grid: 3D array
    :param ghosts_bool: array representing ghost cells
    :param D: diffusion coefficient
    :param dt: time step
    :return: 3D array with the increments
    '''
    k1=laplace_term_rolling(grid, ghosts_bool, D, dt)
    k2=laplace_term_rolling(grid, ghosts_bool, D, dt, add=k1/2)
    k3=laplace_term_rolling(grid, ghosts_bool, D, dt, add=k2/2)
    k4=laplace_term_rolling(grid, ghosts_bool, D, dt, add=k3)
    # numexpr.set_num_threads(nn)
    return evaluate_cached(expressions["rk4"], casting='same_kind')

def laplace_term_stencil(grid, surface, index, D, dt, add=0, div=0):
    """
    Calculates diffusion term for all surface cells using stencil operator

    :param grid: 3D precursor density array
    :param surface: 3D boolean surface array
    :param index: a tuple of 3 indices (z,y,x) of surface cells
    :param D: diffusion coefficient
    :param dt: time step
    :param add: Runge-Kutta intermediate member
    :param div:
    :return: to grid array
    """
    grid += add
    # grid = grid + add
    # grid_out = copy.copy(grid)
    # grid_out *= -6
    grid_out = -6*grid
    # Creating index
    # index = surface.nonzero()
    # z = np.intc(index[0])
    # y = np.intc(index[1])
    # x = np.intc(index[2])
    roll.stencil(grid_out, grid, *index )
    grid -= add
    return grid_out[surface]*dt*D
    # return evaluate_cached(expressions["laplace1"], local_dict={'dt_D': dt*D, 'grid_out':grid_out[surface]}, casting='same_kind')


# @jit(nopython=True, parallel=True, forceobj=False)
def laplace_term_rolling(grid, surface, ghosts_bool, D, dt, add = 0, div: int = 0):
    """
    Calculates diffusion term for all surface cells using rolling


    :param grid: 3D precursor density array
    :param surface: 3D boolean surface array
    :param ghosts_bool: array representing ghost cells
    :param D: diffusion coefficient
    :param dt: time step
    :param add: Runge-Kutta intermediate member
    :param div:
    :return: to grid array
    """

    # Debugging note_: it would be more elegant to just use numpy.roll() on the ghosts_bool to assign neighboring values
    # to ghost cells. But Numpy doesn't retain array structure when utilizing boolean index streaming. It rather extracts all the cells
    # (that correspond to True in our case) and processes them as a flat array. It caused the shifted values for ghost cells to
    # be assigned to the previous(first) layer, which was not processed by numpy.roll() when it rolled backwards.
    # Thus, borders(planes) that are not taking part in rolling(shifting) are cut off by using views to an array
    grid = grid + add
    grid_out = copy.copy(grid)
    grid_out *= -6

    # X axis:
    # No need to have a separate array of values, when whe can conveniently call them from the original data
    shore = grid[:, :, 1:]
    wave = grid[:, :, :-1]
    shore[ghosts_bool[:, :, 1:]] = wave[ghosts_bool[:, :, 1:]] # assigning values to ghost cells forward along X-axis
    # outcome = grid_out[:,:, :-1]
    # income = grid[:,:, 1:]
    # gs_out = gs[:,:,:-1].reshape(-1).nonzero()
    # gs_in = gs[:,:,1:].reshape(-1).nonzero()
    # outcome.reshape(-1)[gs_out] += income.reshape(-1)[gs_in]
    # outcome.reshape(-1)[...] += income.reshape(-1)[...]
    # roll.rolling_1d(outcome.reshape(-1), income.reshape(-1))
    # grid_out[:,:, :-1]+=grid[:,:, 1:] #rolling forward (actually backwards)
    roll.rolling_3d(grid_out[:,:,:-1], grid[:,:,1:])
    index = ghosts_bool.reshape(-1).nonzero()
    # grid_out[:,:,-1] += grid[:,:,-1] #taking care of edge values
    roll.rolling_2d(grid_out[:,:,-1], grid[:,:,-1])
    # grid[ghosts_bool] = 0 # flushing ghost cells
    grid.reshape(-1)[index] = 0
    # Doing the same, but in reverse
    shore = grid[:, :, :-1]
    wave = grid[:, :, 1:]
    shore[ghosts_bool[:, :, :-1]] = wave[ghosts_bool[:, :, :-1]]
    # grid_out[:,:,1:] += grid[:,:,:-1] #rolling backwards
    roll.rolling_3d(grid_out[:,:,1:], grid[:,:,:-1])
    # grid_out[:, :, 0] += grid[:, :, 0]
    roll.rolling_2d(grid_out[:, :, 0], grid[:, :, 0])
    # grid[ghosts_bool] = 0
    grid.reshape(-1)[index] = 0

    # Y axis:
    shore = grid[:, 1:, :]
    wave = grid[:, :-1, :]
    shore[ghosts_bool[:, 1:, :]] = wave[ghosts_bool[:, 1:, :]]
    # grid_out[:, :-1, :] += grid[:, 1:, :]
    roll.rolling_3d(grid_out[:, :-1, :], grid[:, 1:, :])
    # grid_out[:, -1, :] += grid[:, -1, :]
    roll.rolling_2d(grid_out[:, -1, :], grid[:, -1, :])
    # grid[ghosts_bool] = 0
    grid.reshape(-1)[index] = 0
    shore = grid[:, :-1, :]
    wave = grid[:, 1:, :]
    shore[ghosts_bool[:, :-1, :]] = wave[ghosts_bool[:, :-1, :]]
    # grid_out[:, 1:, :] += grid[:, :-1, :]
    roll.rolling_3d(grid_out[:, 1:, :], grid[:, :-1, :])
    # grid_out[:, 0, :] += grid[:, 0, :]
    roll.rolling_2d(grid_out[:, 0, :], grid[:, 0, :])
    # grid[ghosts_bool] = 0
    grid.reshape(-1)[index] = 0

    # Z axis:
    shore = grid[1:, :, :]
    wave = grid[:-1, :, :]
    shore[ghosts_bool[1:, :, :]] = wave[ghosts_bool[1:, :, :]]
    # c
    roll.rolling_3d(grid_out[:-1, :, :], grid[1:, :, :])
    # grid_out[-1, :, :] += grid[-1, :, :]
    roll.rolling_2d(grid_out[-1, :, :], grid[-1, :, :])
    # grid[ghosts_bool] = 0
    grid.reshape(-1)[index] = 0
    shore = grid[:-1, :, :]
    wave = grid[1:, :, :]
    shore[ghosts_bool[:-1, :, :]] = wave[ghosts_bool[:-1, :, :]]
    # grid_out[1:, :, :] += grid[:-1, :, :]
    roll.rolling_3d(grid_out[1:, :, :], grid[:-1, :, :])
    # grid_out[0, :, :] += grid[0, :, :]
    roll.rolling_2d(grid_out[0, :, :], grid[0, :, :])
    # grid[ghosts_bool] = 0
    grid.reshape(-1)[index] = 0
    # grid_out[ghosts_bool]=0
    grid_out.reshape(-1)[index] = 0 # result also has to be cleaned as it contains redundant values in ghost cells
    # numexpr: 1 core performs better
    # numexpr.set_num_threads(nn)
    return grid_out[surface]*dt*D
    # return evaluate_cached(expressions["laplace1"], local_dict={'dt_D': dt*D, 'grid_out':grid_out}, casting='same_kind')
    # else:
    #     return evaluate_cached(expressions["laplace2"], local_dict={'dt_D_div': dt*D/div, 'grid_out':grid_out}, casting='same_kind')


def define_irr_area(beam_matrix):
    indices = np.nonzero(beam_matrix)
    return indices[1].min(), indices[1].max(), indices[2].min(), indices[2].max()

# def path_generator(type, height=1, width=5, length=5)

# @jit(nopython=True, parallel=True, cache=True)
def show_yield(deposit, summ, summ1, res):
    summ1 = np.sum(deposit)
    res = summ1-summ
    return summ, summ1, res


def initialize_framework(from_file=False):
    """
    Prepare simulation framework and parameters from input configuration files
    :param from_file: load structure from vtk file
    :return:
    """
    precursor = yaml.load(open(f'{sys.path[0]}{os.sep}Me3PtCpMe.yml', 'r'),
                          Loader=yaml.Loader)  # Precursor and substrate properties(substrate here is the top layer)
    settings = yaml.load(open(f'{sys.path[0]}{os.sep}Parameters.yml', 'r'),
                         Loader=yaml.Loader)  # Parameters of the beam, dwell time and precursor flux
    sim_params = yaml.load(open(f'{sys.path[0]}{os.sep}Simulation.yml', 'r'),
                           Loader=yaml.Loader)  # Size of the chamber, cell size and time step
    mc_config, equation_values, timings, nr = buffer_constants(precursor, settings, sim_params)
    structure = Structure()
    if from_file:
        try:
            vtk_file = '/Users/sandrik1742/Documents/PycharmProjects/FEBID/source/Profiling/Pillar.vtk'#
            vtk_obj = pv.read(vtk_file)
            structure.load_from_vtk(vtk_obj)
            structure.precursor[structure.surface_bool] = nr
            structure.precursor[structure.semi_surface_bool] = nr
        except FileNotFoundError:
            print("File not found.")
            structure.create_from_parameters(sim_params['cell_dimension'], sim_params['width'], sim_params['length'],
                                  sim_params['height'], sim_params['substrate_height'], nr)
    else:
        structure.create_from_parameters(sim_params['cell_dimension'], sim_params['width'], sim_params['length'],
                              sim_params['height'], sim_params['substrate_height'], nr)
    return structure, mc_config, equation_values, timings

def buffer_constants(precursor: dict, settings: dict, sim_params: dict):
    """
    Calculate necessary constants and prepare parameters for modules

    :param precursor: precursor properties
    :param settings: simulation conditions
    :param sim_params: parameters of the simulation
    :return:
    """
    td = settings["dwell_time"]  # dwell time of a beam, s
    Ie = settings["beam_current"]  # beam current, A
    beam_FWHM = 2.36*settings["gauss_dev"]  # electron beam diameter, nm
    F = settings["precursor_flux"]  # precursor flux at the surface, 1/(nm^2*s)   here assumed a constant, but may be dependent on time and position
    effective_diameter = beam_FWHM * 3.3  # radius of an area which gets 99% of the electron beam
    f = Ie / scpc.elementary_charge / (math.pi * beam_FWHM * beam_FWHM / 4)  # electron flux at the surface, 1/(nm^2*s)
    e = precursor["SE_emission_activation_energy"]
    l = precursor["SE_mean_free_path"]

    # Precursor properties
    sigma = precursor["cross_section"]  # dissociation cross section, nm^2; is averaged from cross sections of all electron types (PE,BSE, SE1, SE2)
    n0 = precursor["max_density"]  # inversed molecule size, Me3PtCpMe, 1/nm^2
    molar = precursor["molar_mass_precursor"]  # molar mass of the precursor Me3Pt(IV)CpMe, g/mole
    # density = 1.5E-20  # density of the precursor Me3Pt(IV)CpMe, g/nm^3
    V = precursor["dissociated_volume"]  # atomic volume of the deposited atom (Pt), nm^3
    D = precursor["diffusion_coefficient"]  # diffusion coefficient, nm^2/s
    tau = precursor["residence_time"] * 1E-6  # average residence time, s; may be dependent on temperature

    kd = F / n0 + 1 / tau + sigma * f  # depletion rate
    kr = F / n0 + 1 / tau  # replenishment rate
    nr = F / kr  # absolute density after long time
    nd = F / kd  # depleted absolute density
    t_out = 1 / (1 / tau + F / n0)  # effective residence time
    p_out = 2 * math.sqrt(D * t_out) / beam_FWHM

    # Initializing framework
    # dt = sim_params["time_step"]
    cell_dimension = sim_params["cell_dimension"]  # side length of a square cell, nm

    t_flux = 1 / (sigma + f)  # dissociation event time
    diffusion_dt = math.pow(cell_dimension * cell_dimension, 2) / (2 * D * (
            cell_dimension * cell_dimension + cell_dimension * cell_dimension))  # maximum stability
    dt = np.min([t_flux, diffusion_dt, tau])

    # Parameters for Monte-Carlo simulation
    mc_config = {'name': precursor["deposit"], 'E0': settings["beam_energy"], 'Emin': settings["minimum_energy"],
              'Z': precursor["average_element_number"],
              'A': precursor["average_element_mol_mass"], 'rho': precursor["average_density"],
              'I0': settings["beam_current"], 'sigma': settings["gauss_dev"],
              'N': Ie * dt / scpc.elementary_charge, 'sub': settings["substrate_element"],
              'cell_dim': sim_params["cell_dimension"],
              'e': precursor["SE_emission_activation_energy"], 'l': precursor["SE_mean_free_path"]}
    # Parameters for reaction-equation solver
    equation_values = {'F': settings["precursor_flux"], 'n0': precursor["max_density"],
                       'sigma': precursor["cross_section"], 'tau': precursor["residence_time"] * 1E-6,
                       'V': precursor["dissociated_volume"], 'D': precursor["diffusion_coefficient"],
                       'dt': dt}
    # Stability time steps
    timings = {'t_diff': diffusion_dt, 't_flux': t_flux, 't_desorption': tau, 'dt': dt}

    # effective_radius_relative = math.floor(effective_diameter / cell_dimension / 2)
    return mc_config, equation_values, timings, nr
    
# /The printing loop.
# @jit(nopython=True)
def printing(loops=1, dwell_time=1):
    """
    Performs FEBID printing process in a zig-zag manner for given number of times

    :param loops: number of repetitions of the route
    :return: changes deposit and precursor arrays
    """



    structure, mc_config, equation_values, timings = initialize_framework(False)

    F = equation_values['F']
    n0 = equation_values['n0']
    sigma = equation_values['sigma']
    tau = equation_values['tau']
    V = equation_values['V']
    D = equation_values['D']
    dt = equation_values['dt']
    sub_h = structure.substrate_height

    t = 0 # absolute time, s
    t+=dt
    refresh_dt = dt*2 # dt for precursor density recalculation

    beam_matrix = np.zeros(structure.shape, dtype=int)  # matrix for electron beam flux distribution

    surface_bool = structure.surface_bool
    semi_surface_bool = structure.semi_surface_bool
    ghosts_bool = structure.ghosts_bool
    deposit = structure.deposit
    precursor = structure.precursor
    precursor[precursor == -1] = 0
    surface_full_bool = np.logical_or(surface_bool, semi_surface_bool) # this is an axillary array used to apply diffusion term for both surface and semi-surface cells

    max_z=int(sub_h + np.nonzero(surface_bool)[0].max()+3) # used to track the highest point

    # Surface index
    index = surface_full_bool[sub_h:max_z,:,:].nonzero()
    z = np.intc(index[0])
    y = np.intc(index[1])
    x = np.intc(index[2])
    surface_index = (z,y,x)

    # Deposition index
    deposition_index = None
    beam_matrix_flat = None
    beam_matrix_surface = beam_matrix[sub_h:max_z,:,:][surface_bool[sub_h:max_z,:,:]]

    sim = etraj3d.cache_params(mc_config, deposit, surface_bool)

    y0, x0 = structure.ydim / 2 * structure.cell_dimension, structure.xdim / 2 * structure.cell_dimension


    y_start, y_end, x_start, x_end = 0, 0, 0, 0
    irradiated_area_3D = None

    # Deposition index
    deposition_index = None
    beam_matrix_flat = None

    flag = True
    # precursor[sub_h, 30, 30] = 0.01
    # render = vr.Render(structure.cell_dimension)
    # render._add_3Darray(precursor, 0.0000001, 1, opacity=1, nan_opacity=1, scalar_name='Precursor',
    #                    button_name='precursor', cmap='plasma')
    # render.show(interactive_update=True, cam_pos=[(206.34055818793468, 197.6510638707941, 100.47106597548205),
    #                                               (0.0, 0.0, 0.0),
    #                                               (-0.23307751464125356, -0.236197909312718, 0.9433373838690787)])
    frame_rate = 0.5
    frame = 0
    time_step = 60 # s
    time_stamp = 0# 2*24*60*60 + 16*60*60 + 0*60 # s
    start_time = 0
    init_cells = np.count_nonzero(deposit[deposit == -2]) # substrate layer
    total_dep_cells = [np.count_nonzero(deposit[deposit < 0]) - init_cells]  # total number of fully deposited cells
    growth_rate = []  # growth rate on each step
    i = 0
    redraw_flag = True
    a = 1
    start_time = timeit.default_timer()
    time_stamp = timeit.default_timer()
    # def show_threaded(frame, i, redraw_flag):
    #     frame += frame_rate
    #     i += 1
    #     if redraw_flag:
    #         render.p.clear()
    #         total_dep_cells.append(np.count_nonzero(deposit[deposit < 0]) - init_cells)
    #         growth_rate.append((total_dep_cells[i] - total_dep_cells[i - 1]) / (time_stamp - time_step + 0.001) * 60 * 60)
    #         render._add_3Darray(precursor, 0.0000001, 1, opacity=0.5, show_edges=True, exclude_zeros=False,
    #                             scalar_name='Precursor',
    #                             button_name='precursor', cmap='plasma')
    #         render.meshes_count += 1
    #         # render.add_3Dar1ray(deposit, structure.cell_dimension, -2, -0.5, 0.7, scalar_name='Deposit',
    #         #            button_name='Deposit', color='white', show_scalar_bar=False)
    #         render.p.add_text(f'Time: {str(datetime.timedelta(seconds=int(time_stamp)))} \n'
    #                           f'Sim. time: {(t):.8f} s \n'
    #                           f'Speed: {(t / time_stamp):.8f} \n'  # showing time passed
    #                           f'Relative growth rate: {int(total_dep_cells[i] / time_stamp * 60 * 60)} cell/h \n'  # showing average growth rate
    #                           f'Real growth rate: {int(total_dep_cells[i] / t * 60)} cell/min \n',
    #                           position='upper_left',
    #                           font_size=12)  # showing average growth rate
    #         render.p.add_text(f'Cells: {total_dep_cells[i]} \n'  # showing total number of deposited cells
    #                           f'Height: {max_z * structure.cell_dimension} nm \n', position='upper_right',
    #                           font_size=12)  # showing current height of the structure
    #         redraw_flag = False
    #     else:
    #         # render.p.mesh['precursor'] = precursor[precursor!=0]
    #         render.p.update_scalars(precursor[precursor > 0])
    #         render.p.update_scalar_bar_range(clim=[precursor[precursor > 0].min(), precursor.max()])
    #     render.p.update()
    #     return frame, redraw_flag

    for l in tqdm(range(0, loops)):  # loop repeats
        # if timeit.default_timer() >= start_time:
        #
        #     # structure.deposit = deposit
        #     # structure.precursor = precursor
        #     # structure.ghosts_bool = ghosts_bool
        #     # structure.surface_bool = surface_bool
        #     # vr.save_deposited_structure(structure, f'{l/1000}k_of_{loops/1000}_loops_k_gr4 ')
        #     time_stamp += time_step
        #     start_time += time_step
        #     print(f'Time passed: {time_stamp}, Av.speed: {l/time_stamp}')

        # if timeit.default_timer() > frame:
        #     frame, a = show_threaded(frame, i, a)
        if flag:
            start = timeit.default_timer()
            beam_matrix = etraj3d.rerun_simulation(y0, x0, deposit, surface_bool, sim, dt)
            print(f'Took total {timeit.default_timer() - start}')
            beam_matrix[beam_matrix<0] = 0
            y_start, y_end, x_start, x_end = define_irr_area(beam_matrix[sub_h:max_z])
            max_z = int(sub_h + np.nonzero(surface_bool)[0].max() + 2)
            flag = False
            irradiated_area_3D = np.s_[sub_h:max_z, y_start:y_end,x_start:x_end]  # a slice of the currently irradiated area
            deposition_index = beam_matrix[irradiated_area_3D].nonzero()
            beam_matrix_flat = beam_matrix[irradiated_area_3D][deposition_index]
        deposition(deposit[irradiated_area_3D],
                   precursor[irradiated_area_3D],
                   beam_matrix_flat,
                   deposition_index, sigma*V*dt, 4)  # depositing on a selected area
        redraw_flag = flag = update_surface(deposit[irradiated_area_3D],
                               precursor[irradiated_area_3D],
                               surface_bool[irradiated_area_3D],
                               semi_surface_bool[irradiated_area_3D],
                               ghosts_bool[irradiated_area_3D],)  # updating surface on a selected area
        if flag:
            surface_full_bool = np.logical_or(surface_bool, semi_surface_bool)
            index = surface_full_bool[sub_h:max_z,:,:].nonzero()
            z = np.intc(index[0])
            y = np.intc(index[1])
            x = np.intc(index[2])
            surface_index = (z,y,x)
            beam_matrix_surface = beam_matrix[sub_h:max_z, :, :][surface_bool[sub_h:max_z, :, :]]
            a = 1
        # if t % refresh_dt < 1E-6:
        # TODO: look into DASK for processing arrays by chunks in parallel
        precursor_density(beam_matrix_surface,
                          precursor[sub_h:max_z, :, :],
                          surface_bool[sub_h:max_z, :, :],
                          surface_full_bool[sub_h:max_z,:,:],
                          surface_index,
                          ghosts_bool[sub_h:max_z,:,:], F, n0, tau, sigma, D, dt)
        t += dt
    a=0
    b=0




def open_params():
    input("Press Enter and specify your Precursor&Substrate properties file:")
    precursor_cfg = fd.askopenfilename()
    input("Press Enter and specify your Technical parameters file:")
    tech_cfg = fd.askopenfilename()
    input("Press Enter and specify your Simulation parameters file:")
    sim_cfg = fd.askopenfilename()
    return precursor_cfg, tech_cfg, sim_cfg

profiler = line_profiler.LineProfiler()
# profiled_func = profiler(laplace_term_rolling)
# profiled_func = profiler(laplace_term_stencil)
# profiled_func = profiler(printing)
# profiled_func = profiler(deposition)
# profiled_func = profiler(update_surface)
# profiled_func = profiler(precursor_density)

if __name__ == '__main__':
    # precursor_cfg, tech_cfg, sim_cfg = open_params()
    # profiled_func(1000)
    printing(10000000000)
    # profiler.print_stats()
    q=0
