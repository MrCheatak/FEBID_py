"""
Heat equation solution
"""

import math
import warnings

import numpy as np
from febid.libraries.rolling import roll
from febid.libraries.pde import tridiag
from numexpr_mod import cache_expression, evaluate_cached, evaluate
from diffusion import laplace_term_rolling, laplace_term_stencil, prepare_ghosts_index, prepare_surface_index

# Heat transfer is solved according to finite-difference explicit
# FTCS (Forward in Time Central in Space) method.
# Stability condition in 3D space: ∆t<=∆x^2/6k

# Algorithm works with 3-dimensional arrays, which represent a discretized space with a cubic cell.
# A value held in a cell corresponds to the temperature in that point.
# In order to manage and index cells that hold a temperature value, a boolean array with the same shape is used.
# Where True marks cells with a value.

heat_equation = cache_expression('a*temp', signature=[('a', np.float64), ('temp', np.float64)])
eV_J = 1.60217733E-19

def get_heat_transfer_stability_time(k, rho, cp, dx):
    heat_dt = dx**2*cp*rho*1e-24 / (6 * k) # maximum stability
    return heat_dt

def temperature_stencil(grid, k, cp, rho, dt, dl, heat_source=0, solid_index=None, substrate_T=294, flat=False, add=0, div=0):
    """
    Calculates diffusion term for the surface cells using stencil operator

        Nevertheless 'surface_index' is an optional argument,
    it is highly recommended to handle index from the caller function

    :param grid: 3D precursor density array, normalized
    :param surface: 3D boolean surface array
    :param D: diffusion coefficient, nm^2/s
    :param dt: time interval over which diffusion term is calculated, s
    :param dl: grid space step, nm
    :param surface_index: a tuple of indices of surface cells for the 3 dimensions
    :param substrate_T: constant temperature of the substrate
    :param flat: if True, returns a flat array of surface cells. Otherwise returns a 3d array with the same shape as grid.
    :param add: Runge-Kutta intermediate member
    :param div:
    :return: 3d or 1d ndarray
    """
    if solid_index is None:
        solid_index = prepare_surface_index(grid)
    y, x = (grid[0]).nonzero()
    # Getting equation constanats
    a = k / cp / rho * 1e24 # thermal diffusivity
    A = dt * a / dl ** 2 # base matrix coefficient
    S = heat_source * (dt / cp / rho * 1e24 * 1.60217733E-19)# heating source
    grid += add
    grid_out = laplace_term_stencil(grid, solid_index)
    grid -= add
    grid_out += S
    grid[0, y, x] = substrate_T
    grid_out[0, y, x] = 0
    g = grid[:, :, 10]
    if flat:
        return evaluate_cached(heat_equation, local_dict={'a': A, 'temp': grid_out[solid_index]}, casting='same_kind')
    else:
        return evaluate_cached(heat_equation, local_dict={'a': A, 'temp': grid_out}, casting='same_kind')


def temperature_rolling(grid, solid, heat, ghosts_bool, k, cp, rho, dt, cell_dim, ghost_index=None, flat=False, add=0, div=0):
    """
        Calculates diffusion term for all surface cells using rolling

            Nevertheless 'ghosts_index' is an optional argument,
        it is highly recommended to handle index from the caller function

        :param grid: 3D precursor density array, normalized
        :param surface: 3D boolean surface array
        :param ghosts_bool: array representing ghost cells
        :param D: diffusion coefficient, nm^2/s
        :param dt: time interval over which diffusion term is calculated, s
        :param cell_dim: grid space step, nm
        :param ghosts_index: 7 index arrays
        :param flat: if True, returns a flat array of surface cells. Otherwise returns a 3d array with the same shape as grid.
        :param add: Runge-Kutta intermediate member
        :param div:
        :return: 3d or 1d ndarray
        """
    if not ghosts_index:
        ghosts_index = prepare_ghosts_index(ghosts_bool)
    grid += add
    grid_out = laplace_term_rolling(grid, ghosts_bool)
    grid -= add
    a = dt / cp / rho / 1000 * 1e27  # [K*nm^3/W]
    if flat:
        return evaluate_cached(heat_equation, local_dict={'temp':grid_out[solid_index], 'q': heat[solid_index],
                                                          'k': k/cell_dim**2, 'a': a}, casting='same_kind')
    else:
        return evaluate_cached(heat_equation, local_dict={'temp':grid_out, 'q': heat, 'k': k, 'a': a},
                                                           casting='same_kind')


def prepare_solid_index(grid):
    index = grid.nonzero()
    return np.intc(index[0]), np.intc(index[1]), np.intc(index[2])


def heat_transfer_BE(grid, conditions, k, cp, rho, dt, dl, heat_source = 0, substrate_T = 294):
    """
        Calculate temperature distribution after the specified time step by solving the parabollic heat equation.

        The heat equation with the heat source term is solved by backward Euler scheme.

        Fractional step method is used to numerically solve the PDE in 3D space.

            There are two options for boundary conditions:
                'isolated': the structure is isolated from both void and substrate

                "heatsink': the structure disipates heat through the substarete that has a constant temperature.



        :param grid: spatial temperature distribution array
        :param conditions: structure boundary conditions
        :param k: thermal conductivity, [W/K/m]
        :param cp: heat capacity, [J/kg/K]
        :param rho: density, [g/cm^3]
        :param dt: time interval over which diffusion term is calculated, [s]
        :param dl: grid step size, [nm]
        :param heat_source: heating source, [W/nm^3]
        :return: 3d or 1d ndarray
        """
    if conditions not in ['isolated', 'heatsink']:
        warnings.warn('The condition specified is not reconized. Use \'isolated\' or \'heatsink\'. Failed to resolve temperature distribution.')
        return

    # Defining conditions
    if conditions == 'isolated':
        pass
    if conditions == 'heatsink':
        y, x = grid[0].nonzero()
        grid[0, y, x] = substrate_T

    # Getting equation constanats
    a = k / cp / rho * 1e24  # thermal diffusivity
    A = dt * a / dl ** 2  # base matrix coefficient
    S = heat_source * (dt / cp / rho * 1e24 * 1.60217733E-19)  # heating source

    # Subdividing to 1D arrays in each direction,
    # check if there are voids(zeros) in each 1D array and splits it if any
    # This operation accounts for two things:
    #   It divides voxel domain into columns and subdivides voxel columns if they intersect void
    #   And it makes sure there are no zeros in the columns, as they will break tridiagonl matrix solution
    i_x, i_y, i_z = fragmentise(grid)
    # Solving in-place
    # Boundary conditions are set to 2 – no flux
    tridiag.adi_3d_indexing(grid, grid, i_x, i_y, i_z, A, 2)
    grid += S
    return True


def fragmentise(grid):
    """
    Collect columns along each axis that do not contain zero cells

    :param grid: 3d array
    :return: array of index tripples
    """
    index_x = []
    index_y = []
    index_z = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j, :].max() > 0:
                ii = subdivide_list(grid[i, j, :], i, j, axis=2)
                if ii is not None:
                    index_x += ii
    for i in range(grid.shape[0]):
        for j in range(grid.shape[2]):
            if grid[i, :, j].max() > 0:
                ii = subdivide_list(grid[i, :, j], i, j, axis=1)
                if ii is not None:
                    index_y += ii
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            if grid[:, i, j].max() > 0:
                ii = subdivide_list(grid[:, i, j], i, j, axis=0)
                if ii is not None:
                    index_z += ii
    index_x = np.array(index_x)
    index_y = np.array(index_y)
    index_z = np.array(index_z)
    return index_x, index_y, index_z


def subdivide_list(grid, i=0, j=0, axis=2):
    """
    Extract start and end indexes of the non-zero sections in the array.

    This function virtually prevents zeros from appering in a solution matrix by extracting the 'solid' cells along
    the slice.

    :param grid: 1D array
    :param i: first index of the current slice
    :param j: second index of the current slice
    :param axis: axis along which the slice was taken
    :return:
    """
    def get_index(x):
        tripple[axis] = x
        i, j, h = tripple
        return i, j, h
    index_l = []
    if axis == 2:
        tripple = np.array([i, j, 0])
    if axis == 1:
        tripple = np.array([i, 0, j])
    if axis == 0:
        tripple = np.array([0, i, j])
    if grid.min() == 0:
        index = (grid == 0).nonzero()[0]
        if index.shape[0] >= grid.shape[0]-1:
            return None
        index_d = index[1:] - index[:-1]
        index_n = (index_d > 2).nonzero()[0]
        if index.min() == 0:
            first = None
        else:
            if index[0] > 1:
                first = get_index(0)
                second = get_index(index[0])
                index_l.append(first)
                index_l.append(second)
            else:
                first = None
        for k in index_n:
            first = get_index(index[k] + 1)
            second = get_index(index[k+1])
            index_l.append(first)
            index_l.append(second)
        if index[-1] < grid.shape[0]-1:
            first = get_index(index[-1] + 1)
            second = get_index(grid.shape[0])
            index_l.append(first)
            index_l.append(second)
    else:
        first = get_index(0)
        second = get_index(grid.shape[0])
        index_l.append(first)
        index_l.append(second)
    return index_l
