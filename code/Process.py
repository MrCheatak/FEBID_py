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

# class Positionb:
#     def __init__(self, z:int, y:int, x:int):
#         self.x = np.int16(x)
#         self.y = np.int16(y)
#         self.z = np.int16(z)
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
# substrate[z,x,y] holds precursor density,
# deposit[z,y,x] holds deposit density.
system_size = 50
substrate = zeros((system_size*2, system_size, system_size), dtype=np.float32)
deposit = np.full((system_size*2, system_size, system_size),0 , dtype=np.float32)
substrate[0, :, :] = nr  # filling substrate surface with initial precursor density
deposit[0, :, :] = 0.97
# deposit[0, 20:40, 20:40] = 0.95
zmax, ymax, xmax = substrate.shape # dimensions of the grid

# Holds z coordinates of the surface cells that are allowed to produce deposit.
# Indices of every cell correspond to their x&y position
# The idea is to avoid iterating through the whole 3D matrix and address only surface cells
# It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus able produce deposit under irradiation.
# Thus the number of surface cells is fixed.
# TODO: is it possible to create a non-contiguous and evolvable view to the 3D array? Later on all the surface should be able to produce deposit
surface = zeros((system_size, system_size), dtype=np.int64)
surf =[] # a stub for holding a sequence of indices of surface cells for quick processing

# ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb = set(), set(), set(), set(), set(), set()
# Sets where chosen for maintaining both semi-surface cells and ghost cells as they:
# 1. Doesn't require search to get an element — O(1)
# 2. Are faster than lists at adding/removing/getting values
# 3. Doesn't allow duplicates: if set already contains an item being added, effectively nothing happens
# 4. Have a convenient set().discard function, that quietly passes if an item is not in the set
# ghosts = set() # all the ghost cells, that represent a closed shell of the surface to prevent diffusion into the void or back to deposit
# ghosts_index = ()

# Semi-surface cells are cells that have precursor density but do not have deposit right under them
# Thus they cannot produce deposit and their precursor density is calculated without disosisiation term.
# They are introduced to allow diffusion on the walls of the deposit.
# Basically these are all other surface cells
semi_surface = set()
# TODO: maybe combine surface and semi-surface by utilizing subsets?
# </editor-fold>

# <editor-fold desc="Helpers">
# tempy = mgrid[0:ymax, 0:xmax]
stub = np.mgrid[0:ymax, 0:xmax] # for indexing purposes of surface array
# tempy = mgrid[0:effective_radius_relative*2+1, 0:effective_radius_relative*2+1]
index_y, index_x = mgrid[0:(effective_radius_relative*2+1), 0:(effective_radius_relative*2+1)] # for indexing purposes of flux matrix
index_yy, index_xx = index_y*cell_dimension, index_x*cell_dimension

# A dictionary of expressions for numexpr.evaluate_from_cache
expressions = dict(pe_flux=cache_expression("f*exp(-r*r/(2*beam_d*beam_d))", [('beam_d', np.int64), ('f', np.float64), ('r', np.float64)]),
                   rk4=cache_expression("(k1+k4)/6 +(k2+k3)/3", [('k1', np.float64), ('k2', np.float64), ('k3', np.float64), ('k4', np.float64)]),
                   precursor_density=cache_expression("(F * (1 - (sub + addon) / n0) - (sub + addon) / tau - (sub + addon) * sigma * flux_matrix)*dt", [('F', np.int64), ('addon', np.float64), ('dt', np.float64), ('flux_matrix', np.int64), ('n0', np.float64), ('sigma',np.float64), ('sub', np.float64), ('tau', np.float64)]),
                   laplace1=cache_expression("grid_out*dt*D", [('D', float), ('dt', np.float64), ('grid_out', float)]),
                   laplace2=cache_expression("grid_out*dt*D/div", [('D', float), ('dt', np.float64), ('grid_out', float), ('div', np.int64)]),
                   flux_matrix=cache_expression("((index_xx-center)*(index_xx-center)+(index_yy-center)*(index_yy-center))**0.5", [('center', np.int64), ('index_xx', np.int64), ('index_yy', np.int64)]))
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
    Converts a matrix array of z-indices(y=i, x=j) to a sequence of indices

    :param arr: array to convert
    :return: tuple(list[z], list[y], list[x]), indexing-ready
    """
    yy, xx = mgrid[0:arr.shape[0], 0:arr.shape[1]]
    return (arr.flatten(),yy.flatten(), xx.flatten())


@jit(nopython=True)
def flush_structure(substrate, deposit, init_density = nr, init_deposit = .0, volume_prefill = .0):
    """
    Resets printing framework

    :param substrate:
    :param deposit:
    :param init_density:
    :param init_deposit:
    :param volume_prefill:
    :return:
    """
    substrate[...] = 0
    substrate[0, :, :] = init_density  # filling substrate surface with initial precursor density
    if volume_prefill == 0:
        deposit[...] = 0
    else:
        deposit[...] = volume_prefill # partially filling cells with deposit
    deposit[0, :, :] = init_deposit # partially fills surface cells with deposit


# @jit(nopython=True, parallel=True)
def deposition(deposit, substrate, flux_matrix, surf, dt):
    """
    Calculates deposition on the surface for a given time step dt (outer loop)

    :param deposit: 3D deposit array
    :param substrate: 3D precursor density array
    :param flux_matrix: matrix of electron flux distribution
    :param surf: collection of surface cells indices (synchronized with th view)
    :param dt: time step
    :return: writes back to deposition array
    """
    # Instead of processing cell by cell and on the whole surface, it is implemented to process only (effectively) irradiated area and array-wise(thanks to Numpy)
    deposit[surf] += substrate[surf] * sigma * flux_matrix.view().reshape(flux_matrix.shape[0] ** 2) * V * dt


# @jit(nopython=False)
def update_surface(deposit, substrate, surface, surf, semi_surface, ghosts, init_y=0, init_x=0):
    """
    Evolves surface upon a full deposition of a cell

    :param deposit: 3D deposit array
    :param substrate: 3D precursor density array
    :param surface: array corresponding to surface cells
    :param surf: collection of surface cells indices
    :param semi_surface: collection of semi-surface cells
    :param ghosts: collection of ghost cells
    :param init_y: offset for y-axis
    :param init_x: offset for x-axis
    :return: changes surface array, semi-surface and ghosts collections
    """
    # Because all arrays are sent to the function as views of the currently irradiated area (relative coordinate system), offsets are required to update semi-surface and ghost cells collection, because they are stored in absolute coordinates
    new_deposits = np.argwhere(deposit>=1) # looking for new deposits
    if len(new_deposits) != 0:
        for cell in new_deposits:
            ghosts.add((cell[0], cell[1] + init_y, cell[2] + init_x))  # add fully deposited cell to the ghost shell
            surface[cell[1], cell[2]] +=1  # rising the surface one cell up (new cell)
            refresh(deposit, substrate, semi_surface, ghosts, cell[0]+1, cell[1], cell[2], init_y, init_x)
        else:
            temp = tuple(zip(*ghosts))  # casting a set of coordinates to a list of index sequences for every dimension
            # ghosts_ind = (asarray(temp[0]), asarray(temp[1]), asarray(temp[2]))  # constructing a tuple of ndarray sequences
            surf = make_tuple(surface)
    return surf, True # boolean True serves as a flag, to later regenerate a semi-surface index sequence


# @jit(nopython=True)
def refresh(deposit, substrate, semi_surface, ghosts, z,y,x, init_y=0, init_x=0):
    """
    Updates surface, semi-surface and ghost cells collections according to the provided coordinate of a newly deposited cell

    :param deposit: 3D deposit array
    :param substrate: 3D precursor density array
    :param semi_surface: collection of semi-surface cells
    :param z: z-coordinate of the cell above the new deposit
    :param y: y-coordinate of the deposit
    :param x: x-coordinate of the deposit
    :param init_y: offset for y-axis
    :param init_x: offset for x-axis
    :return: changes surface array, semi-surface and ghosts collections
    """
    semi_surface.discard((0,0,0))
    semi_surface.discard((z, y+init_y, x+init_x))  # removing the new cell from the semi_surface collection
    ghosts.discard((z, y + init_y, x + init_x)) # removing the new cell from the ghost shell collection
    deposit[z, y, x] += deposit[z - 1, y, x] - 1  # if the cell was filled above unity, transferring that surplus to the cell above
    deposit[z - 1, y, x] = -1  # a fully deposited cell is always a unity
    xx = x + 2 # this is needed, due to the substrate view being 2 cells wider in case of semi-surface or ghost cell falling out of the bounds of the view
    yy = y + 2
    substrate[z, yy, xx] += substrate[z - 1, yy, xx] # if the deposited cell had precursor in it, transfer that surplus to the cell above
    # this may lead to an overfilling of a cell above unity, but it is not causing any anomalies due to diffusion process
    substrate[z - 1, yy, xx] = np.nan  # precursor density is NaN in the fully deposited cells (it was previously set to zero, but later some of the zero cells were added back to semi-surface)
    if substrate[z+1, yy, xx] == 0: # if the cell that is above the new cell is empty, then add it to the ghost shell collection
        ghosts.add((z+1, y+init_y, x+init_x))
    # Adding neighbors(in x-y plane) of the new cell to the semi_surface collection
    # and updating ghost shell for every neighbor:
    if substrate[z, yy - 1, xx] == 0:
        semi_surface.add((z, y - 1+init_y, x+init_x))  # adding cell to the list
        substrate[z, yy - 1, xx] += 1E-7 # "marks" cell as a surface one, because some of the checks refer to if the cell is empty. This assignment is essential. This corresponds to the smallest value that float32 can hold and should be changed corrspondingly to the type.
        refresh_ghosts(substrate, ghosts, x + init_x, xx, y-1 + init_y, yy-1, z) # update ghost shell around
    if substrate[z, yy + 1, xx] == 0:
        semi_surface.add((z, y + 1+init_y, x+init_x))
        substrate[z, yy + 1, xx] += 1E-7
        refresh_ghosts(substrate, ghosts, x + init_x, xx, y+1 + init_y, yy+1, z)
    if substrate[z, yy, xx - 1] == 0:
        semi_surface.add((z, y+init_y, x - 1+init_x))
        substrate[z, yy, xx - 1] += 1E-7
        refresh_ghosts(substrate, ghosts, x -1 + init_x, xx-1, y + init_y, yy, z)
    if substrate[z, yy, xx + 1] == 0:
        semi_surface.add((z, y+init_y, x + 1+init_x))
        substrate[z, yy, xx + 1] += 1E-7
        refresh_ghosts(substrate, ghosts, x + 1 + init_x, xx+1, y + init_y, yy, z)

# @jit(nopython=True)
def refresh_ghosts(substrate, ghosts, x, xx, y, yy, z):
    """
    Updates ghost cells collection around the specified cell

    :param substrate: 3D precursor density array
    :param ghosts: collection of ghost cells
    :param x: absolute x-coordinates of the cell
    :param xx: substrate view-relative x-coordinates of the cell
    :param y: absolute y-coordinates of the cell
    :param yy: substrate view-relative y-coordinates of the cell
    :param z: absolute z-coordinates of the cell
    :return: changes ghosts collection
    """
    # z-coordinates are same for both coordinates, because the view to a substrate is taken only from the x-y plane
    # Firstly, deleting current cell from ghost shell and then adding all neighboring cells(along all axises) if they are zero
    ghosts.discard((z, y, x))
    if substrate[z - 1, yy, xx] == 0:
        ghosts.add((z - 1, y, x))
    if substrate[z + 1, yy, xx] == 0:
        ghosts.add((z + 1, y, x))
    if substrate[z, yy - 1, xx] == 0:
        ghosts.add((z, y - 1, x))
    if substrate[z, yy + 1, xx] == 0:
        ghosts.add((z, y + 1, x))
    if substrate[z, yy, xx - 1] == 0:
        ghosts.add((z, y, x - 1))
    if substrate[z, yy, xx + 1] == 0:
        ghosts.add((z, y, x + 1))

# @profile
def precursor_density(flux_matrix, substrate, surface, semi_surface_index, ghosts_index, dt):
    """
    Recalculates precursor density on the whole surface

    :param flux_matrix: matrix of electron flux distribution
    :param substrate: 3D precursor density array
    :param dt: time step
    :return: changes substrate array
    """
    sub = zeros([ymax, xmax])  # surface cells array that will be processed to calculate a precursor density increment
    # semi_sub = zeros((len(semi_surface))) # same for semi-surface cells
    sub[stub[0], stub[1]] = substrate[surface, stub[0], stub[1]] # not using np.nditer speeded up the program by 10 times
    diffusion_matrix = laplace_term_rolling(substrate, ghosts_index, D, dt)  # Diffusion term is calculated seperately and added in the end
    rk4(dt, sub, flux_matrix) # An increment is calculated through Runge-Kutta method without the diffusion term
    substrate[surface, stub[0], stub[1]] += sub[stub[0], stub[1]]
    if any(semi_surface_index[0]): # same process for semi-cells
        semi_sub=substrate[semi_surface_index]
        rk4(dt,semi_sub)
        substrate[semi_surface_index]+=semi_sub
    substrate+=diffusion_matrix # finally adding diffusion term
    # if substrate[substrate > 1] != 0:
    #     raise Exception("Normalized precursor density is more than 1")


# @jit(nopython=False, parallel=True)
# noinspection PyUnusedLocal
def rk4(dt, sub, flux_matrix=0):
    """
    Calculates increment of precursor density by Runge-Kutta method

    :param dt: time step
    :param sub: array of surface cells (2D for surface cells, 1D for semi-surface cells)
    :param flux_matrix: matrix of electron flux distribution
    :return: to sub array
    """
    k1 = precursor_density_increment(dt, sub, flux_matrix) # this is actually an array of k1 coefficients
    k2 = precursor_density_increment(dt/2, sub, flux_matrix, k1 / 2)
    k3 = precursor_density_increment(dt/2, sub, flux_matrix, k2 / 2)
    k4 = precursor_density_increment(dt, sub, flux_matrix, k3)
    # numexpr: 1 core performs better
    numexpr.set_num_threads(nn)
    evaluate_from_cache(expressions["rk4"], out=sub, casting='same_kind')


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
    return evaluate_from_cache(expressions["precursor_density"])


def rk4_diffusion(grid, ghosts_index, D, dt):
    k1=laplace_term_rolling(grid, ghosts_index, D, dt)
    k2=laplace_term_rolling(grid, ghosts_index, D, dt, k1,2)
    k3=laplace_term_rolling(grid, ghosts_index, D, dt, k2,2)
    k4=laplace_term_rolling(grid, ghosts_index, D, dt, k3)
    numexpr.set_num_threads(nn)
    return evaluate("(k1+k4)/6 +(k2+k3)/3", casting='same_kind')


# @jit(nopython=True, parallel=True, forceobj=False)
def laplace_term_rolling(grid, ghosts_index, D, dt , add = 0, div = 0):
    """
    Calculates diffusion term for all surface cells using rolling

    :param grid: 3D precursor density array
    :param D: diffusion coefficient
    :param dt: time step
    :return: to grid array
    """
    grid = grid + add
    grid_out = copy(grid)
    grid_out *= -6
    # X axis:
    # No need to have a separate array of values, when whe can conveniently call them from the origin:
    grid[ghosts_index] = grid[ghosts_index[0], ghosts_index[1], ghosts_index[2]-1] # assinging precursor density values to ghost cells along the rolling axis and direction
    grid_out[:,:, :-1]+=grid[:,:, 1:] #rolling forward
    grid_out[:,:,-1] += grid[:,:,-1] #taking care of edge values
    grid[ghosts_index] = 0 # flushing ghost cells
    # While Numpy allows negative indicies, indicies that are greater than the given dimention cause IndexiError and thus has to be taken care of
    temp = where(ghosts_index[2] > system_size - 2, ghosts_index[2] - 1, ghosts_index[2]) # decreasing all the edge indices by one to exclude falling out of the array
    grid[ghosts_index] = grid[ghosts_index[0], ghosts_index[1], temp+1]
    grid_out[:,:,1:] += grid[:,:,:-1] #rolling backwards
    grid_out[:, :, 0] += grid[:, :, 0]
    grid[ghosts_index] = 0
    # Y axis:
    grid[ghosts_index] = grid[ghosts_index[0], ghosts_index[1]-1, ghosts_index[2]]
    grid_out[:, :-1, :] += grid[:, 1:, :]
    grid_out[:, -1, :] += grid[:, -1, :]
    grid[ghosts_index] = 0
    temp = where(ghosts_index[1] > system_size - 2, ghosts_index[1] - 1, ghosts_index[1])
    grid[ghosts_index] = grid[ghosts_index[0], temp+1, ghosts_index[2]]
    grid_out[:, 1:, :] += grid[:, :-1, :]
    grid_out[:, 0, :] += grid[:, 0, :]
    grid[ghosts_index] = 0
    # Z-axis:
    grid[ghosts_index] = grid[ghosts_index[0]-1, ghosts_index[1], ghosts_index[2]]
    grid_out[:-1, :, :] += grid[1:, :, :]
    grid_out[-1, :, :] += grid[-1, :, :]
    grid[ghosts_index] = 0
    temp = where(ghosts_index[0] > system_size - 2, ghosts_index[0] - 1, ghosts_index[0])
    grid[ghosts_index] = grid[temp+1, ghosts_index[1], ghosts_index[2]]
    grid_out[1:, :, :] += grid[:-1, :, :]
    grid_out[0, :, :] += grid[0, :, :]
    grid[ghosts_index] = 0
    grid_out[ghosts_index]=0 # result has also to be cleaned as it has redundant values in ghost cells
    # numexpr: 1 core performs better
    numexpr.set_num_threads(nn)
    if div == 0:
        return evaluate_from_cache(expressions["laplace1"], casting='same_kind')
    else:
        return evaluate_from_cache(expressions["laplace2"], casting='same_kind')


def define_ghosts(substrate, surface, semi_surface =[] ):
    """
    Defines ghost cells for every axis and direction separately.

    :param substrate: 3D precursor density array
    :param surface: surface cells matrix
    :param semi_surface: semi-surface cells list(set)
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

    for cell in semi_surface:
        try:
            if substrate[cell[0]-1, cell[1], cell[2]] == 0:
                zzb.append(cell[0]-1)
                zyb.append(cell[1])
                zxb.append(cell[2])
                gzb.append(substrate[cell])
        except IndexError:
            pass
        try:
            if substrate[cell[0]+1, cell[1], cell[2]] == 0:
                zzf.append(cell[0]+1)
                zyf.append(cell[1])
                zxf.append(cell[2])
                gzf.append(substrate[cell])
        except IndexError:
            pass
        try:
            if substrate[cell[0], cell[1]-1, cell[2]] == 0:
                yzb.append(cell[0])
                yyb.append(cell[1]-1)
                yxb.append(cell[2])
                gyb.append(substrate[cell])
        except IndexError:
            pass
        try:
            if substrate[cell[0], cell[1]+1, cell[2]] == 0:
                yzf.append(cell[0])
                yyf.append(cell[1]+1)
                yxf.append(cell[2])
                gyf.append(substrate[cell])
        except IndexError:
            pass
        try:
            if substrate[cell[0], cell[1], cell[2]-1] == 0:
                xzb.append(cell[0])
                xyb.append(cell[1])
                xxb.append(cell[2]+1)
                gxb.append(substrate[cell])
        except IndexError:
            pass
        try:
            if substrate[cell[0], cell[1], cell[2]+1] == 0:
                xzf.append(cell[0])
                xyf.append(cell[1])
                xxf.append(cell[2]+1)
                gxf.append(substrate[cell])
        except IndexError:
            pass

    return [zzf, zyf, zxf] # , [zzb, zyb, zxb], [yzf, yyf, yxf], [yzb, yyb, yxb], [xzf, xyf, xxf], [xzb, xyb, xxb], gzf, gzb, gyf, gyb, gxf, gxb


# @jit(nopython=False, parallel=True)
# noinspection PyIncorrectDocstring
def flux_matrix(matrix, y1:int, y2:int, x1:int, x2:int):
    """
    Calculates a matrix with electron flux distribution

    :param matrix: output matrix
    :param y1: y2, x1 – x2 – boundaries of the effectively irradiated area
    :return: to matrix array
    """
    matrix[:, :] = 0  # flushing previous values
    irradiated_area = matrix[y1:y2, x1:x2]
    center = effective_radius_relative*cell_dimension # beam center in array-coordinates
    # numexpr: expression is not affected by the number of cores
    dist = evaluate_from_cache(expressions["flux_matrix"])
    irradiated_area[index_y, index_x] = pe_flux(dist)


# @jit(nopython=True)
def define_irradiated_area(y, x, effective_radius_relative:int):
    """
    Defines boundaries of the effectively irradiated area

    :param x: x-coordinate of the beam
    :param y: y-coordinate of the beam
    :param effective_radius_relative: a distance at which intensity is lower than 99% of the distribution
    :return: four values limiting an area in x-y plane
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
def printing(loops=1):
    """
    Performs FEBID printing process in a zig-zag manner for given number of times

    :param loops: number of repetitions of the route
    :return: changes deposit and substrate arrays
    """
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

    flush_structure(substrate, deposit, init_deposit = 0.97, volume_prefill=0.7)
    surface[...]=0
    semi_surface.clear()
    semi_surface_index=([], [], [])
    ghosts = set(zip(*define_ghosts(substrate, surface)))
    temp = tuple(zip(*ghosts))  # casting a set of coordinates to a list of index sequences for every dimension
    ghosts_index = (asarray(temp[0]), asarray(temp[1]), asarray(temp[2]))  # constructing a tuple of ndarray sequences

    t = 2E-6 # absolute time, s
    refresh_dt = dt*2 # dt for precursor density recalculation

    dwell_step = 2 # int(beam_d / 2/cell_dimension)
    x_offset = 20  # offset on the X-axis on both sides
    y_offset = 20  # offset on the Y-axis on both sides
    x_limit = xmax - x_offset
    y_limit = ymax - y_offset

    # beam = Positionb(0, y_offset, x_offset)  # coordinates of the beam
    beam_matrix = zeros((system_size,system_size), dtype=int) # matrix for electron beam flux distribution
    beam_exposure = zeros((system_size,system_size), dtype=int) # see usage in the loop

    summ1,summ, result=0,0,0

    for l in range(loops):  # loop repeats, currently beam travels in a zig-zack manner (array indexing)
        # summ1, summ, result = show_yield(deposit, summ, summ1,result)
        # print(f'Deposit yield:{result}  Loop:{l}')
        print(f'Loop:{l}')
        for y in range(y_offset, y_limit, dwell_step):  # beam travel along Y-axis
            for x in range(x_offset, x_limit, dwell_step):  # beam travel alon X-axis
                y_start, y_end, x_start, x_end = define_irradiated_area(y, x, effective_radius_relative) # Determining the area around the beam that is effectively irradiated
                irradiated_area_2D=s_[y_start:y_end, x_start:x_end] # a slice of the currently irradiated area
                irradiated_area_3D=s_[:surface.max()+3, y_start:y_end, x_start:x_end]
                flux_matrix(beam_matrix, y_start, y_end, x_start, x_end)
                beam_exposure += beam_matrix # accumulates beam exposure for precursor density if it is called with an interval bigger that dt
                surf = make_tuple(surface[irradiated_area_2D]) # getting an indexing-ready surface cells coordinates
                semi_surface.add((0, 0, 0))
                while True:
                    deposition(deposit[irradiated_area_3D], substrate[irradiated_area_3D], beam_matrix[irradiated_area_2D], surf, dt) # depositing on a selected area
                    surf, refresh_flag = update_surface(deposit[irradiated_area_3D], substrate[:surface.max()+3, y_start-2:y_end+2, x_start-2:x_end+2], surface[irradiated_area_2D], surf, semi_surface, ghosts,y_start, x_start) # updating surface on a selected area
                    if t % refresh_dt < 1E-6:
                        if refresh_flag == True: # regenerating indexing arrays
                            temp = list(zip(*semi_surface))  # casting a set of coordinates to a list of index sequences
                            semi_surface_index = (asarray(temp[0]), asarray(temp[1]), asarray(temp[2])) # constructing a tuple of ndarray sequences
                            temp = tuple(zip(*ghosts))
                            ghosts_index = (asarray(temp[0]), asarray(temp[1]), asarray(temp[2]))
                            refresh_flag = False
                        precursor_density(beam_matrix, substrate[:surface.max()+3,:,:], surface, semi_surface_index, ghosts_index, refresh_dt)
                        # if l==3:
                        #     profiler = line_profiler.LineProfiler()
                        #     profiled_func = profiler(precursor_density)
                        #     try:
                        #         profiled_func(beam_matrix, substrate[:surface.max()+3,:,:], surface, ghosts_index, refresh_dt)
                        #     finally:
                        #         profiler.print_stats()
                        beam_exposure = 0 # flushing accumulated radiation
                    t += dt

                    if not t % td > 1E-6:
                        break


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
    # if substrate[i,j,pos,0] ==0:
    #     ax0.pcolor(substrate[i,:,pos,0], color="white")
    # else:
    #     if substrate[i,j,pos,1] == 1:
    #         ax0.pcolor([i,j], substrate[i,j,pos,1], cmap='RdBu')
    #     else:
    #         ax0.pcolor([i,j], substrate[i,j,pos,0], color="orange")
    plt.show()
    q=0


