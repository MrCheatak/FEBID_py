"""
Diffusion module
Solution for diffusion equation via FTCS method
"""
import math

import numpy as np
from febid.libraries.rolling import roll


# Diffusion is solved according to finite-difference explicit
# FTCS (Forward in Time Central in Space) method.
# Stability condition in 3D space: ∆t<=∆x^2/6D

# Algorithm works with 3-dimensional arrays, which represent a discretized space with a cubic cell.
# A value held in a cell corresponds to the concentration in that point.

def get_diffusion_stability_time(D, dx):
    """
    Get max stable time step for FTCS solution
    :param D: diffusion coefficient, nm/nm^2
    :param dx: grid spacing, nm
    :return: time step, s
    """
    diffusion_dt = math.pow(dx, 2) / (6 * D)  # maximum stability
    return diffusion_dt


def diffusion_ftcs(grid, surface, D, dt, cell_size, surface_index=None, flat=True, add=0):
    """
    Calculate diffusion term for the surface cells using stencil approach

        Nevertheless the 'surface_index' is an optional argument,
    it is highly recommended to handle index from the caller function

    :param grid: 3D precursor density array, normalized
    :param surface: 3D boolean surface array
    :param D: diffusion coefficient, nm^2/s
    :param dt: time interval over which diffusion term is calculated, s
    :param cell_size: grid space step, nm
    :param surface_index: a tuple of indices of surface cells for the 3 dimensions
    :param flat: if True, returns a flat array of surface cells. Otherwise, returns a 3d array with the same shape as grid.
    :param add: Runge-Kutta intermediate member
    :return: 3d or 1d ndarray
    """
    if surface_index is None:
        surface_index = prepare_surface_index(surface)
    grid += add
    grid_out = laplace_term_stencil(grid, surface_index)
    # stencil_debug(grid_out, grid, *surface_index)
    grid -= add
    if type(D) in [int, float]:
        a = dt * D / (cell_size * cell_size)
    else:
        a = dt * D[surface] / (cell_size * cell_size)
    if flat:
        return grid_out[surface] * a
    else:
        grid_out[surface] *= a
        return grid_out


def laplace_term_stencil(grid, surface_index):
    """
    Apply stencil operator to the selected cells in the grid.

    :param grid: operated grid
    :param surface_index: selected cell index [z, y, x]
    :return:
    """
    grid_out = -6 * grid
    roll.stencil(grid_out, grid, *surface_index)
    return grid_out


def prepare_surface_index(surface: np.ndarray):
    """
    Get a multiindex from the surface array

    :param surface: boolean array defining surface cells position in space
    :return: tuple of 1d ndarrays
    """
    index = surface.nonzero()
    return np.intc(index[0]), np.intc(index[1]), np.intc(index[2])


def stencil_debug(grid_out, grid, z_index, y_index, x_index):
    xdim, ydim, zdim = grid.shape
    shape = (zdim, ydim, xdim)
    l = z_index.size
    cond = 0
    zero_count = 0

    def axis(ind):
        if grid[ind] != 0:
            grid_out[z, y, x] = grid_out[z, y, x] + grid[ind]
        else:
            return 1

    for i in range(l):
        z = z_index[i]
        y = y_index[i]
        x = x_index[i]
        ind_f = (z, y, x)
        ind_b = (z, y, x)
        zero_count = 0
        if zdim - 1 > z > 0:
            cond += 1
            if ydim - 1 > y > 0:
                cond += 1
                if xdim - 1 > x > 0:
                    cond += 1
        if cond == 3:
            for j in range(3):
                ind_f[j] += 1
                ind_b[j] -= 1
                zero_count += axis(ind_f)
                zero_count += axis(ind_b)
                ind_f[j] -= 1
                ind_b[j] += 1
        else:
            for j in range(3):
                c = ind_f[j]
                boundary = shape[j]
                ind_f[j] += 1
                ind_b[j] -= 1
                if c > boundary - 1:
                    zero_count += 1
                else:
                    zero_count += axis(ind_f)
                if c < 1:
                    zero_count += 1
                else:
                    zero_count += axis(ind_b)
                ind_f[j] -= 1
                ind_b[j] += 1
        grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y, x] * zero_count
        zero_count = 0
        cond = 0
