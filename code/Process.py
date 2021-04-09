###################################################################
#
#  FEBID Simulation
#
#  Version 0.9
#
####################################################################

import numpy as np
from numpy import asarray, zeros, copy, where, mgrid, s_
import scipy.constants as scpc
import math
import matplotlib.pyplot as plt
from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# import ipyvolume as ipv
from numexpr import evaluate, evaluate_from_cache, cache_expression
import numexpr
from modified_libaries import numexpr as ne
import cProfile
# from timebudget import timebudget
from numba import jit, typeof, generated_jit
from numba.experimental import jitclass
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import sys
import line_profiler
# TODO: look into k-d trees
# TODO: add a benchmark to determine optimal threads number for current machine
interval_def = sys.getswitchinterval()
sys.setswitchinterval(1000)

class Laplace_Convolution:
    # @jit(nopython=False, parallel=True)
    def laplace_term(grid, surfa, semi_surf, D, dt):
        """
        Calculates diffusion term for all surface cells using convolution

        :param grid: 3D precursor density array
        :param surfa: surface cells collection
        :param semi_surf: semi-surface cells collection
        :param D: diffusion coefficient
        :param dt: time step
        :return: to grid array
        """
        p_grid = np.pad(grid, 1, mode='constant', constant_values=0)
        sub_grid = copy(grid)
        Laplace_Convolution.convolute(sub_grid, p_grid, surfa, semi_surf)
        return evaluate("sub_grid*D*dt")

    # @jit(nopython=False, parallel=True)
    def convolute(grid_out, grid, coords, coords1):
        """
        Selectively applies convolution operator to the cells with provided coordinates

        :param grid_out: 3D array with convolution results
        :param grid: original 3D array
        :param coords: coordinates of surface cells
        :param coords1: coordinates of semi-surface cells
        :return: to grid_out array
        """
        grid_out *= -6
        with np.nditer(coords, flags=['multi_index']) as it:
            for z in it:
                kernel = grid[z:z + 3, it.multi_index[0]:it.multi_index[0] + 3,
                         it.multi_index[1]:it.multi_index[1] + 3]  # taking a 3x3x3 view around current cell
                # kernel = grid[z - 1:z + 2, it.multi_index[0] - 1:it.multi_index[0] + 2, it.multi_index[1] - 1:it.multi_index[1]+2]
                grid_out[z, it.multi_index[0], it.multi_index[1]] += Laplace_Convolution.kernel_convolution(kernel)
        if any(coords1):
            for pos in coords1:
                kernel = grid[pos[0]:pos[0] + 3, pos[1]:pos[1] + 3, pos[2]:pos[2] + 3]
                grid_out[pos] += Laplace_Convolution.kernel_convolution(kernel)

    @jit(nopython=False)
    def kernel_convolution(grid):
        """
        Applies convolution to the central cell in the given 3D grid.

        If one of the neighbors is Zero, it is replaced by the central value.

        :param grid: array to convolute
        :return: changes grid array
        """
        # mesh = [(0, 1, 1), (2, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1)]
        summ = 0
        if 1 > grid[0, 1, 1] > 0:
            summ += grid[0, 1, 1]
        else:
            summ += grid[1, 1, 1]
        if 1 > grid[2, 1, 1] > 0:
            summ += grid[2, 1, 1]
        else:
            summ += grid[1, 1, 1]
        if 1 > grid[1, 1, 0] > 0:
            summ += grid[1, 1, 0]
        else:
            summ += grid[1, 1, 1]
        if 1 > grid[1, 1, 2] > 0:
            summ += grid[1, 1, 2]
        else:
            summ += grid[1, 1, 1]
        if 1 > grid[1, 0, 1] > 0:
            summ += grid[1, 0, 1]
        else:
            summ += grid[1, 1, 1]
        if 1 > grid[1, 2, 1] > 0:
            summ += grid[1, 2, 1]
        else:
            summ += grid[1, 1, 1]

        return summ

# TODO: implement import of parameters from file
# <editor-fold desc="Parameters">
td = 1E-6  # dwell time of a beam, s
Ie = 1E-10  # beam current, A
beam_d = 10  # electron beam diameter, nm
effective_diameter = beam_d * 3.3 # radius of an area which gets 99% of the electron beam
f = Ie / scpc.elementary_charge / (math.pi * beam_d * beam_d / 4)  # electron flux at the surface, 1/(nm^2*s)
F = 3000  # precursor flux at the surface, 1/(nm^2*s)   here assumed a constant, but may be dependent on time and position
tau = 500E-6  # average residence time, s; may be dependent on temperature

# Precursor properties
sigma = 2.2E-2  # dissociation cross section, nm^2; is averaged from cross sections of all electron types (PE,BSE, SE1, SE2)
n0 = 1.9  # inversed molecule size, Me3PtCpMe, 1/nm^2
M_Me3PtCpMe = 305  # molar mass of the precursor Me3Pt(IV)CpMe, g/mole
p_Me3PtCpMe = 1.5E-20  # density of the precursor Me3Pt(IV)CpMe, g/nm^3
V = 4 / 3 * math.pi * math.pow(0.139, 3)  # atomic volume of the deposited atom (Pt), nm^3
D = np.float32(1E5)  # diffusion coefficient, nm^2/s


kd = F / n0 + 1 / tau + sigma * f  # depletion rate
kr = F / n0 + 1 / tau  # replenishment rate
nr = F / kr  # absolute density after long time
nd = F / kd  # depleted absolute density
t_out = 1 / (1 / tau + F / n0)  # effective residence time
p_out = 2 * math.sqrt(D * t_out) / beam_d
cell_dimension = 5  # side length of a square cell, nm

effective_radius_relative = math.floor(effective_diameter / cell_dimension / 2)

nn=1 # default number of threads for numexpr
# </editor-fold>

# <editor-fold desc="Timings">
dt = np.float32(1E-6)  # time step, s
t_flux = 1/(sigma+f)  # dissociation event time
diffusion_dt = math.pow(cell_dimension * cell_dimension, 2) / (2 * D * (cell_dimension * cell_dimension + cell_dimension * cell_dimension))   # maximum stability lime of the diffusion solution
tau = 500E-6  # average residence time, s; may be dependent on temperature
# </editor-fold>

# <editor-fold desc="Framework" >
# Main cell matrices
system_size = 50
height_multiplyer = 2
substrate = zeros((system_size*height_multiplyer, system_size, system_size), dtype=np.float32) # substrate[z,x,y] holds precursor density
deposit = zeros((system_size*height_multiplyer, system_size, system_size), dtype=np.float32) # deposit[z,y,x] holds deposit density
substrate[0, :, :] = nr  # filling substrate surface with initial precursor density
# deposit[0, 20:40, 20:40] = 0.95
zmax, ymax, xmax = substrate.shape # dimensions of the grid

# It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus able produce deposit under irradiation.
# The idea is to avoid iterating through the whole 3D matrix and address only surface cells
# Thus the number of surface cells is fixed.

# Semi-surface cells are cells that have precursor density but do not have deposit right under them
# Thus they cannot produce deposit and their precursor density is calculated without disosisiation term.
# They are introduced to allow diffusion on the walls of the deposit.
# Basically these are all other surface cells
# </editor-fold>

# <editor-fold desc="Helpers">
center = effective_radius_relative * cell_dimension  # beam center in array-coordinates
index_y, index_x = mgrid[0:(effective_radius_relative*2+1), 0:(effective_radius_relative*2+1)] # for indexing purposes of flux matrix
index_yy, index_xx = index_y*cell_dimension-center, index_x*cell_dimension-center

# A dictionary of expressions for numexpr.evaluate_from_cache
# Debug note: before introducing a new cached expression, that expression should be run with the default 'evaluate' function for fetching the signature list.
# This is required, because variables in it must be in the same order as Numexpr fetches them, otherwise Numexpr compiler will throw an error
expressions = dict(pe_flux=cache_expression("f*exp(-r*r/(2*beam_d*beam_d))", [('beam_d', np.int32), ('f', np.float64), ('r', np.float64)]),
                   rk4=cache_expression("(k1+k4)/6 +(k2+k3)/3", [('k1', np.float64), ('k2', np.float64), ('k3', np.float64), ('k4', np.float64)]),
                   #precursor_density_=cache_expression("(F * (1 - (sub + addon) / n0) - (sub + addon) / tau - (sub + addon) * sigma * flux_matrix)*dt", [('F', np.int64), ('addon', np.float64), ('dt', np.float64), ('flux_matrix', np.int64), ('n0', np.float64), ('sigma',np.float64), ('sub', np.float64), ('tau', np.float64)]),
                   precursor_density=cache_expression("F_dt - (sub + addon) * (F_dt_n0_1_tau_dt + sigma_dt * flux_matrix)", [('F_dt', np.float64), ('F_dt_n0_1_tau_dt', np.float64), ('addon', np.float64), ('flux_matrix', np.int64), ('sigma_dt',np.float64), ('sub', np.float64)]),
                   laplace1=cache_expression("grid_out*dt_D", [('dt_D', np.float64), ('grid_out', np.float64)]),
                   laplace2=cache_expression("grid_out*dt_D_div", [('dt_D_div', np.float64), ('grid_out', np.float64)]),
                   flux_matrix=cache_expression("((index_xx-center)*(index_xx-center)+(index_yy-center)*(index_yy-center))**0.5", [('center', np.int32), ('index_xx', np.int32), ('index_yy', np.int32)]))
# </editor-fold>

# @jit(nopython=True)
def pe_flux(r):
    """Calculates PE flux at the given radius according to Gaussian distribution.

    :param r: radius from the center of the beam.
    """
    # numexpr: no impact from number of cores or vml
    return evaluate_from_cache(expressions["pe_flux"])


# @jit(nopython=False, forceobj=True)
def make_tuple(arr):
    """
    DEPRICATED
    Converts a matrix array of z-indices(y=i, x=j) to a sequence of indices

    :param arr: array to convert
    :return: tuple(list[z], list[y], list[x]), indexing-ready
    """
    yy, xx = mgrid[0:arr.shape[0], 0:arr.shape[1]]
    return (arr.flatten(),yy.flatten(), xx.flatten())


@jit(nopython=True)
def flush_structure(substrate: np.ndarray, deposit: np.ndarray, init_density: np.float32 = nr, init_deposit = .0, volume_prefill = .0):
    """
    Resets and prepares initial state of the printing framework

    :param substrate: 3D precursor density array
    :param deposit: 3D deposit array
    :param init_density: initial precursor density on the surface
    :param init_deposit: initial deposit on the surface, can be a 2D array with the same size as deposit array along 0 and 1 dimensions
    :param volume_prefill: initial deposit in the volume, can be a predefined structure in an 3D array same size as deposit array (constant value is virtual and used for code development)
    :return:
    """
    substrate[...] = 0
    substrate[0, :, :] = init_density  # filling substrate surface with initial precursor density
    if volume_prefill == 0:
        deposit[...] = 0
    else:
        deposit[...] = volume_prefill # partially filling cells with deposit
        if init_deposit != 0:
            deposit[0, :, :] = init_deposit # partially fills surface cells with deposit


# @jit(nopython=True, parallel=True)
def deposition(deposit, substrate, flux_matrix, surface_bool, dt):

    """
    Calculates deposition on the surface for a given time step dt (outer loop)

    :param deposit: 3D deposit array
    :param substrate: 3D precursor density array
    :param flux_matrix: matrix of electron flux distribution
    :param dt: time step
    :return: writes back to deposit array
    """
    # Instead of processing cell by cell and on the whole surface, it is implemented to process only (effectively) irradiated area and array-wise(thanks to Numpy)
    # Math here cannot be efficiently simplified, because multiplication of constant variables here produces a value below np.float32 accuracy
    # np.float32 — ~1E-7, produced value — ~1E-10
    deposit[surface_bool] += substrate[surface_bool] * sigma * flux_matrix[surface_bool] * V * dt


# @jit(nopython=False)
def update_surface(deposit, substrate, surface_bool, semi_surf_bool, ghosts_bool, flux_matrix, z_max):
    """
    Evolves surface upon a full deposition of a cell. This method holds has the vast majority of logic

    :param deposit: 3D deposit array
    :param substrate: 3D precursor density array
    :param surface_bool: array representing surface cells
    :param semi_surf_bool: array representing semi-surface cells
    :param ghosts_bool: array representing ghost cells
    :param flux_matrix: matrix of electron flux distribution
    :param z_max: highest point along Z axis
    :return: changes surface, semi-surface and ghosts arrays
    """
    # Because all arrays are sent to the function as views of the currently irradiated area (relative coordinate system), offsets are required to update semi-surface and ghost cells collection, because they are stored in absolute coordinates
    new_deposits = np.argwhere(deposit>=1) # looking for new deposits
    if len(new_deposits) != 0:
        for cell in new_deposits:
            ghosts_bool[cell[0], cell[1]+2, cell[2]+2] = True # deposited cell belongs to ghost shell
            surface_bool[cell[0], cell[1], cell[2]] = False  # rising the surface one cell up (new cell)
            surface_bool[cell[0] + 1, cell[1], cell[2]] = True
            flux_matrix[cell[0]+1, cell[1], cell[2]] = flux_matrix[cell[0], cell[1], cell[2]] # flux_matrix shape must follow surface shape
            flux_matrix[cell[0], cell[1], cell[2]] = 0
            deposit[cell[0]+1, cell[1], cell[2]] += deposit[cell[0], cell[1], cell[2]] - 1  # if the cell was filled above unity, transferring that surplus to the cell above
            deposit[cell[0], cell[1], cell[2]] = -1  # a fully deposited cell is always a minus unity
            refresh(substrate, semi_surf_bool, ghosts_bool, cell[0] + 1, cell[1] + 2, cell[2] + 2)
        else:
            if cell[0] > z_max-4:
                z_max += 1
    return z_max


# @jit(nopython=True)
def refresh(substrate, semi_s_bool, ghosts_bool, z,y,x):
    """
    Updates surface, semi-surface and ghost cells collections according to the provided coordinate of a newly deposited cell

    :param substrate: 3D precursor density array
    :param semi_s_bool: array representing semi-surface cells
    :param ghosts_bool: array representing ghost cells
    :param z: z-coordinate of the cell above the new deposit
    :param y: y-coordinate of the deposit
    :param x: x-coordinate of the deposit
    :return: changes surface array, semi-surface and ghosts collections
    """
    # this is needed, due to the substrate view being 2 cells wider in case of semi-surface or ghost cell falling out of the bounds of the view
    semi_s_bool[z, y, x] = False # removing the new cell from the semi_surface collection
    ghosts_bool[z, y, x] = False # removing the new cell from the ghost shell collection
    substrate[z, y, x] += substrate[z - 1, y, x] # if the deposited cell had precursor in it, transfer that surplus to the cell above
    # this may lead to an overfilling of a cell above unity, but it is not causing any anomalies due to diffusion process
    substrate[z - 1, y, x] = np.nan  # precursor density is NaN in the fully deposited cells (it was previously set to zero, but later some of the zero cells were added back to semi-surface)
    if substrate[z+1, y, x] == 0: # if the cell that is above the new cell is empty, then add it to the ghost shell collection
        ghosts_bool[z+1, y, x] = True
    # Adding neighbors(in x-y plane) of the new cell to the semi_surface collection
    # and updating ghost shell for every neighbor:
    if substrate[z, y - 1, x] == 0:
        semi_s_bool[z, y - 1, x] = True
        substrate[z, y - 1, x] += 1E-7 # this "marks" cell as a surface one, because some of the checks refer to if the cell is empty. This assignment is essential. It corresponds to the smallest value that float32 can hold and should be changed corrspondingly to the variable type.
        refresh_ghosts(substrate, ghosts_bool, x, y-1, z) # update ghost shell around
    if substrate[z, y + 1, x] == 0:
        semi_s_bool[z, y + 1, x] = True
        substrate[z, y + 1, x] += 1E-7
        refresh_ghosts(substrate, ghosts_bool,  x, y+1, z)
    if substrate[z, y, x - 1] == 0:
        semi_s_bool[z, y, x - 1] = True
        substrate[z, y, x - 1] += 1E-7
        refresh_ghosts(substrate, ghosts_bool, x-1, y, z)
    if substrate[z, y, x + 1] == 0:
        semi_s_bool[z, y, x + 1] = True
        substrate[z, y, x + 1] += 1E-7
        refresh_ghosts(substrate, ghosts_bool, x+1, y, z)

# @jit(nopython=True)
def refresh_ghosts(substrate, ghosts_bool, x, y, z):
    """
    Updates ghost cells collection around the specified cell

    :param substrate: 3D precursor density array
    :param ghosts_bool: array representing ghost cells
    :param x: x-coordinates of the cell
    :param y: y-coordinates of the cell
    :param z: z-coordinates of the cell
    :return: changes ghosts array
    """
    # It must be kept in mind, that z-coordinate here is absolute, but x and y are relative to the view
    # Firstly, deleting current cell from ghost shell and then adding all neighboring cells(along all axis) if they are zero
    ghosts_bool[z, y, x] = False
    if substrate[z - 1, y, x] == 0:
        ghosts_bool[z - 1, y, x] = True
    if substrate[z + 1, y, x] == 0:
        ghosts_bool[z + 1, y, x] = True
    if substrate[z, y - 1, x] == 0:
        ghosts_bool[z, y - 1, x] = True
    if substrate[z, y + 1, x] == 0:
        ghosts_bool[z, y + 1, x] = True
    if substrate[z, y, x - 1] == 0:
        ghosts_bool[z, y, x - 1] = True
    if substrate[z, y, x + 1] == 0:
        ghosts_bool[z, y, x + 1] = True


# @profile
def precursor_density(flux_matrix, substrate, surface_bool, semi_surf_bool, ghosts_bool, dt):
    """
    Recalculates precursor density on the whole surface

    :param flux_matrix: matrix of electron flux distribution
    :param substrate: 3D precursor density array
    :param surface_bool: array representing surface cells
    :param semi_surf_bool: array representing semi-surface cells
    :param ghosts_bool: array representing ghost cells
    :param dt: time step
    :return: changes substrate array
    """
    diffusion_matrix = laplace_term_rolling(substrate, ghosts_bool, D, dt)  # Diffusion term is calculated separately and added in the end
    substrate[surface_bool] += rk4(dt, substrate[surface_bool], flux_matrix[surface_bool]) # An increment is calculated through Runge-Kutta method without the diffusion term
    substrate[semi_surf_bool] += rk4(dt, substrate[semi_surf_bool])  # same process for semi-cells, but without dissociation term
    substrate+=diffusion_matrix # finally adding diffusion term
    # if substrate[substrate > 1] != 0:
    #     raise Exception("Normalized precursor density is more than 1")


# @jit(nopython=False, parallel=True)
# noinspection PyUnusedLocal
def rk4(dt, sub, flux_matrix=0):
    """
    Calculates increment of precursor density by Runge-Kutta method

    :param dt: time step
    :param sub: array of surface cells
    :param flux_matrix: matrix of electron flux distribution
    :return:
    """
    k1 = precursor_density_increment(dt, sub, flux_matrix) # this is actually an array of k1 coefficients
    k2 = precursor_density_increment(dt/2, sub, flux_matrix, k1 / 2)
    k3 = precursor_density_increment(dt/2, sub, flux_matrix, k2 / 2)
    k4 = precursor_density_increment(dt, sub, flux_matrix, k3)
    # numexpr: 1 core performs better
    numexpr.set_num_threads(nn)
    return evaluate_from_cache(expressions["rk4"], casting='same_kind')


# @jit(nopython=False, parallel=True)
# noinspection PyUnusedLocal
def precursor_density_increment(dt, sub, flux_matrix, addon=0):
    """
    Calculates increment of the precursor density without a diffusion term

    :param dt: time step
    :param sub: array of surface cells (2D for surface cells, 1D dor semi-surface cells)
    :param flux_matrix: matrix of electron flux distribution
    :param addon: Runge Kutta term
    :return: to sub array
    """
    # numexpr: 1 core performs better
    numexpr.set_num_threads(nn)
    return evaluate_from_cache(expressions["precursor_density"], local_dict={'F_dt':F*dt, 'F_dt_n0_1_tau_dt': (F*dt*tau+n0*dt)/(tau*n0), 'addon':addon, 'flux_matrix':flux_matrix, 'sigma_dt':sigma*dt, 'sub':sub}, casting='same_kind')


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
    numexpr.set_num_threads(nn)
    return evaluate_from_cache(expressions["rk4"], casting='same_kind')


# @jit(nopython=True, parallel=True, forceobj=False)
def laplace_term_rolling(grid, ghosts_bool, D, dt, add = 0, div: int = 0):
    """
    Calculates diffusion term for all surface cells using rolling


    :param grid: 3D precursor density array
    :param ghosts_bool: array representing ghost cells
    :param D: diffusion coefficient
    :param dt: time step
    :param add: Runge-Kutta intermediate member
    :param div:
    :return: to grid array
    """

    # Debugging note: it would be more elegant to just use numpy.roll() on the ghosts_bool to assign neighboring values
    # to ghost cells. But Numpy doesn't retain array structure when utilizing boolean index streaming. It rather extracts all the cells
    # (that correspond to True in our case) and processes them as a flat array. It caused the shifted values for ghost cells to
    # be assigned to the previous(first) layer, which was not processed by numpy.roll() when it rolled backwards.
    # Thus, borders(planes) that are not taking part in rolling(shifting) are cut off by using views to an array
    grid = grid + add
    grid_out = copy(grid)
    grid_out *= -6

    # X axis:
    # No need to have a separate array of values, when whe can conveniently call them from the original data
    shore = grid[:, :, 1:]
    wave = grid[:, :, :-1]
    shore[ghosts_bool[:, :, 1:]] = wave[ghosts_bool[:, :, 1:]] # assigning values to ghost cells forward along X-axis
    grid_out[:,:, :-1]+=grid[:,:, 1:] #rolling forward (actually backwards)
    grid_out[:,:,-1] += grid[:,:,-1] #taking care of edge values
    grid[ghosts_bool] = 0 # flushing ghost cells
    # Doing the same, but in reverse
    shore = grid[:, :, :-1]
    wave = grid[:, :, 1:]
    shore[ghosts_bool[:, :, :-1]] = wave[ghosts_bool[:, :, :-1]]
    grid_out[:,:,1:] += grid[:,:,:-1] #rolling backwards
    grid_out[:, :, 0] += grid[:, :, 0]
    grid[ghosts_bool] = 0

    # Y axis:
    shore = grid[:, 1:, :]
    wave = grid[:, :-1, :]
    shore[ghosts_bool[:, 1:, :]] = wave[ghosts_bool[:, 1:, :]]
    grid_out[:, :-1, :] += grid[:, 1:, :]
    grid_out[:, -1, :] += grid[:, -1, :]
    grid[ghosts_bool] = 0
    shore = grid[:, :-1, :]
    wave = grid[:, 1:, :]
    shore[ghosts_bool[:, :-1, :]] = wave[ghosts_bool[:, :-1, :]]
    grid_out[:, 1:, :] += grid[:, :-1, :]
    grid_out[:, 0, :] += grid[:, 0, :]
    grid[ghosts_bool] = 0

    # Z axis:
    shore = grid[1:, :, :]
    wave = grid[:-1, :, :]
    shore[ghosts_bool[1:, :, :]] = wave[ghosts_bool[1:, :, :]]
    grid_out[:-1, :, :] += grid[1:, :, :]
    grid_out[-1, :, :] += grid[-1, :, :]
    grid[ghosts_bool] = 0
    shore = grid[:-1, :, :]
    wave = grid[1:, :, :]
    shore[ghosts_bool[:-1, :, :]] = wave[ghosts_bool[:-1, :, :]]
    grid_out[1:, :, :] += grid[:-1, :, :]
    grid_out[0, :, :] += grid[0, :, :]
    grid[ghosts_bool] = 0
    grid_out[ghosts_bool]=0 # result also has to be cleaned as it contains redundant values in ghost cells
    # numexpr: 1 core performs better
    numexpr.set_num_threads(nn)
    if div == 0:
        return evaluate_from_cache(expressions["laplace1"], local_dict={'dt_D': dt*D, 'grid_out':grid_out}, casting='same_kind')
    # else:
    #     return evaluate_from_cache(expressions["laplace2"], local_dict={'dt_D_div': dt*D/div, 'grid_out':grid_out}, casting='same_kind')


def define_ghosts(substrate, surface):
    """
    Defines ghost cells for every axis and direction separately.

    :param substrate: 3D precursor density array
    :param surface: surface cells matrix
    :return: six lists with coordinates of ghost cells and six lists of corresponding values
    """
    xxf, xyf, xzf, yxf, yyf, yzf, zxf,zyf,zzf =[],[],[],[],[],[],[],[],[]
    xxb, xyb, xzb, yxb, yyb, yzb, zxb, zyb, zzb = [], [], [], [], [], [], [], [], []
    gxf, gxb, gyf, gyb, gzf, gzb = [],[],[],[],[],[]
    for y in range(surface.shape[0]):
        for x in range(surface.shape[1]):
            try:
                if substrate[surface[y,x]-1,y,x] == 0:
                    zxb.append(x)
                    zyb.append(y)
                    zzb.append(surface[y,x]-1)
                    gzb.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass
            try:
                if substrate[surface[y,x]+1,y,x] == 0:
                    zxf.append(x)
                    zyf.append(y)
                    zzf.append(surface[y,x]+1)
                    gzf.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass
            try:
                if substrate[surface[y,x],y-1,x] == 0:
                    yxb.append(x)
                    yyb.append(y-1)
                    yzb.append(surface[y,x])
                    gyb.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass
            try:
                if substrate[surface[y,x],y+1,x] == 0:
                    yxf.append(x)
                    yyf.append(y+1)
                    yzf.append(surface[y,x])
                    gyf.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass
            try:
                if substrate[surface[y,x],y,x-1] == 0:
                    xxb.append(x-1)
                    xyb.append(y)
                    xzb.append(surface[y,x])
                    gxb.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass
            try:
                if substrate[surface[y,x],y,x+1] == 0:
                    xxf.append(x+1)
                    xyf.append(y)
                    xzf.append(surface[y,x])
                    gxf.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass

    return [zzf, zyf, zxf] # , [zzb, zyb, zxb], [yzf, yyf, yxf], [yzb, yyb, yxb], [xzf, xyf, xxf], [xzb, xyb, xxb], gzf, gzb, gyf, gyb, gxf, gxb


# @jit(nopython=False, parallel=True)
# noinspection PyIncorrectDocstring
def flux_matrix(matrix, surface_irradiated):
    """
    Calculates a matrix with electron flux distribution

    :param matrix: output matrix
    :param surface_irradiated: irradiated area
    :return: to matrix array
    """
    # TODO: with the same beam size, electron flux distribution is actually always the same, it just travels on the surface, thus no need to calculate it every time
    # TODO: distances from the center will always be the same, so the matrix of distances can be pre-calculated
    matrix[surface_irradiated] = pe_flux(np.hypot(index_xx, index_yy).reshape(-1))


# @jit(nopython=True)
def define_irradiated_area(y, x, effective_radius_relative:int):
    """
    Defines boundaries of the effectively irradiated area

    :param y: y-coordinate of the beam
    :param x: x-coordinate of the beam
    :param effective_radius_relative: a distance at which intensity is lower than 99% of the distribution
    :return: four values defining an area in x-y plane
    """
    norm_y_start = 0
    norm_y_end = y + effective_radius_relative
    norm_x_start = 0
    norm_x_end = x + effective_radius_relative
    temp = y - effective_radius_relative
    if temp > 0:
        norm_y_start = temp
    if norm_y_end > ymax:
        norm_y_end = ymax
    temp = x - effective_radius_relative
    if temp > 0:
        norm_x_start = temp
    if norm_x_end > xmax:
        norm_x_end = xmax
    return  norm_y_start, norm_y_end+1, norm_x_start, norm_x_end+1


@jit(nopython=True, parallel=True, cache=True)
def show_yield(deposit, summ, summ1, res):
    summ1 = np.sum(deposit)
    res = summ1-summ
    return summ, summ1, res


# /The printing loop.
# @jit(nopython=True)
# @profile
def printing(loops=1):
    """
    Performs FEBID printing process in a zig-zag manner for given number of times

    :param loops: number of repetitions of the route
    :return: changes deposit and substrate arrays
    """
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

    flush_structure(substrate, deposit, init_deposit = 0.97, volume_prefill=0.7)

    ghosts = set(zip(*define_ghosts(substrate, zeros((system_size, system_size), dtype=int))))
    temp = tuple(zip(*ghosts))  # casting a set of coordinates to a list of index sequences for every dimension
    ghosts_index = (asarray(temp[0]), asarray(temp[1]), asarray(temp[2]))  # constructing a tuple of ndarray sequences
    ghosts_bool = np.full(substrate.shape, False, dtype=bool)
    ghosts_bool[ghosts_index]=True

    surface_bool = np.full((zmax, ymax, xmax), False, dtype=bool)
    surface_bool[0, :, :] = True # assumed that we have a flat substrate with no deposit

    semi_surface_bool = np.full((zmax, ymax, xmax), False, dtype=bool)

    t = 2E-6 # absolute time, s
    refresh_dt = dt*2 # dt for precursor density recalculation

    dwell_step = 2 # int(beam_d / 2/cell_dimension)
    x_offset = 15  # offset on the X-axis on both sides
    y_offset = 15  # offset on the Y-axis on both sides
    x_limit = xmax - x_offset
    y_limit = ymax - y_offset

    beam_matrix = zeros((zmax,ymax,xmax), dtype=int)  # matrix for electron beam flux distribution
    beam_exposure = zeros((zmax,ymax,xmax), dtype=int)  # see usage in the loop

    summ1,summ, result=0,0,0

    max_z=4 # used to track the highest point
    ext = 2 # a constant for inner update_surface logic

    for l in range(loops):  # loop repeats, currently beam travels in a zig-zack manner (array indexing)
        # summ1, summ, result = show_yield(deposit, summ, summ1,result)
        # print(f'Deposit yield:{result}  Loop:{l}')
        print(f'Loop:{l}')
        for y in range(y_offset, y_limit, dwell_step):  # beam travel along Y-axis
            for x in range(x_offset, x_limit, dwell_step):  # beam travel alon X-axis
                y_start, y_end, x_start, x_end = define_irradiated_area(y, x, effective_radius_relative) # Determining the area around the beam that is effectively irradiated
                # WARNING: there is no protection from falling out of an array!
                # There always must be at least 3 cells margin from irradiated area to array edges
                irradiated_area_3D=s_[:max_z, y_start:y_end, x_start:x_end] # a slice of the currently irradiated area
                irradiated_area_3D_ext = s_[:max_z, y_start-ext:y_end+ext, x_start-ext:x_end+ext]
                flux_matrix(beam_matrix[irradiated_area_3D], surface_bool[irradiated_area_3D]) # getting electron beam flux distribution matrix
                beam_exposure[irradiated_area_3D] += beam_matrix[irradiated_area_3D] # accumulates beam exposure for precursor density if it is called with an interval bigger that dt
                while True:
                    deposition(deposit[irradiated_area_3D],
                               substrate[irradiated_area_3D],
                               beam_matrix[irradiated_area_3D],
                               surface_bool[irradiated_area_3D], dt)  # depositing on a selected area
                    max_z = update_surface(deposit[irradiated_area_3D],
                                           substrate[irradiated_area_3D_ext],
                                           surface_bool[irradiated_area_3D],
                                           semi_surface_bool[irradiated_area_3D_ext],
                                           ghosts_bool[irradiated_area_3D_ext],
                                           beam_matrix[irradiated_area_3D],
                                           max_z)  # updating surface on a selected area
                    if t % refresh_dt < 1E-6:
                        precursor_density(beam_matrix[:max_z, :, :],
                                          substrate[:max_z, :, :],
                                          surface_bool[:max_z, :, :],
                                          semi_surface_bool[:max_z, :, :],
                                          ghosts_bool[:max_z, :, :], refresh_dt)
                        # if l==3:
                        #     profiler = line_profiler.LineProfiler()
                        #     profiled_func = profiler(precursor_density)
                        #     try:
                        #         profiled_func(beam_matrix, substrate[:surface.max()+3,:,:], surface, ghosts_index, refresh_dt)
                        #     finally:
                        #         profiler.print_stats()
                        beam_exposure[:max_z, y_offset-effective_radius_relative:y_limit+effective_radius_relative,x_offset-effective_radius_relative:x_limit+effective_radius_relative] = 0 # flushing accumulated radiation
                    t += dt

                    if not t % td > 1E-6:
                        break
                beam_matrix[irradiated_area_3D] = 0


if __name__ == '__main__':
    repetitions = int(input("Enter your value: "))
    # test_grid = np.full((2,50,50), 0.83, dtype=np.float32)
    printing(3)
    cProfile.runctx('printing(repetitions)',globals(),locals())
    # <editor-fold desc="Plot">
    mat = substrate > 0
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(mat, **{'cmap': cm.plasma})
    plt.show()
    # </editor-fold>
    section = zeros((substrate.shape[0], ymax), dtype=float)
    # fig, (ax0) = plt.subplots(1)
    pos = np.int16(xmax/2)
    for i in range(0, substrate.shape[0]):
        for j in range(0, substrate.shape[1]):
            section[i,j] = substrate[i, j,pos]
    plt.show()
    q=0


