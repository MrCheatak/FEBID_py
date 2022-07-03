import math

import numpy as np
from source.libraries.rolling import roll

# Diffusion is solved according to finite-difference explicit
# FTCS (Forward in Time Central in Space) method.
# Stability condition in 3D space: ∆t<=∆x^2/6D

# Algorithm works with 3-dimensional arrays, which represent a discretized space with a cubic cell.
# A value held in a cell corresponds to the concentration in that point.
# In order to manage and index cells that hold a concentration value, a boolean array with the same shape is used.
# Where True marks cells with a value.

def get_diffusion_stability_time(D, dx):
    diffusion_dt = math.pow(dx, 2) / (6 * D)  # maximum stability
    return diffusion_dt

def laplace_term_stencil(grid, surface, D, dt, cell_dim, surface_index=None, flat=True, add=0, div=0):
    """
    Calculates diffusion term for the surface cells using stencil operator

        Nevertheless 'surface_index' is an optional argument,
    it is highly recommended to handle index from the caller function

    :param grid: 3D precursor density array, normalized
    :param surface: 3D boolean surface array
    :param D: diffusion coefficient, nm^2/s
    :param dt: time interval over which diffusion term is calculated, s
    :param cell_dim: grid space step, nm
    :param surface_index: a tuple of indices of surface cells for the 3 dimensions
    :param flat: if True, returns a flat array of surface cells. Otherwise returns a 3d array with the same shape as grid.
    :param add: Runge-Kutta intermediate member
    :param div:
    :return: 3d or 1d ndarray
    """
    if surface_index is None:
        surface_index = prepare_surface_index(surface)
    grid += add
    grid_out = -6 * grid
    roll.stencil(grid_out, grid, *surface_index)
    # stencil_debug(grid_out, grid, *surface_index)
    grid -= add
    a = dt * D/(cell_dim * cell_dim)
    if flat:
        return grid_out[surface] * a
    else:
        grid_out[surface] *= a
        return grid_out


def laplace_term_rolling(grid, surface, ghosts_bool, D, dt, cell_dim, ghosts_index=None, flat=True, add = 0, div: int = 0):
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

    # Debugging note_: it would be more elegant to just use numpy.roll() on the ghosts_bool to assign neighboring values
    # to ghost cells. But Numpy doesn't retain array structure when utilizing boolean index streaming. It rather extracts all the cells
    # (that correspond to True in our case) and processes them as a flat array. It caused the shifted values for ghost cells to
    # be assigned to the previous(first) layer, which was not processed by numpy.roll() when it rolled backwards.
    # Thus, borders(planes) that are not taking part in rolling(shifting) are cut off by using views to an array
    if not ghosts_index:
        ghosts_index = prepare_ghosts_index(ghosts_bool)

    grid += add
    grid_out = -6 * grid

    # X axis:
    # No need to have a separate array of values, when whe can conveniently call them from the original data
    shore = grid[:, :, 1:]
    wave = grid[:, :, :-1]
    shore[ghosts_bool[:, :, 1:]] = wave[ghosts_bool[:, :, 1:]] # assigning values to ghost cells forward along X-axis
    # grid_out[:,:, :-1]+=grid[:,:, 1:] #rolling forward (actually backwards)
    roll.rolling_3d(grid_out[:,:,:-1], grid[:,:,1:])
    index = ghosts_bool.reshape(-1).nonzero()
    # grid_out[:,:,-1] += grid[:,:,-1] #taking care of edge values
    roll.rolling_2d(grid_out[:,:,-1], grid[:,:,-1])
    grid.reshape(-1)[index] = 0
    # Doing the same, but in reverse
    shore = grid[:, :, :-1]
    wave = grid[:, :, 1:]
    shore[ghosts_bool[:, :, :-1]] = wave[ghosts_bool[:, :, :-1]]
    # grid_out[:,:,1:] += grid[:,:,:-1] #rolling backwards
    roll.rolling_3d(grid_out[:,:,1:], grid[:,:,:-1])
    # grid_out[:, :, 0] += grid[:, :, 0]
    roll.rolling_2d(grid_out[:, :, 0], grid[:, :, 0])
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
    grid.reshape(-1)[index] = 0
    # grid_out[ghosts_bool]=0
    grid_out.reshape(-1)[index] = 0 # result also has to be cleaned as it contains redundant values in ghost cells
    a = dt * D/(cell_dim * cell_dim)
    if flat:
        return grid_out[surface] * a
    else:
        grid_out[surface] *= a
        return grid_out


def prepare_ghosts_index(ghosts_bool:np.ndarray):
    """
    Prepares a collection of indexes for the rolling function

    A tuple of 7 1d indexes is returned in the following order:
        non-shifted

        shifted backwards along 3rd axis (x)

        shifted forward along 3rd axis (x)

        shifted backwards along 2nd axis (y)

        shifed forward along 2nd axis (y)

        shifted backwards along 1st axis (z)

        shifted forward along 1st axis (z)

    :param ghosts_bool: boolean array defining ghost cells position in space
    :return: tuple of 1d ndarrays

    """

    #     Indexing a flattened numpy array is up to 3 times faster than using multidimensional index
    # Method used here is basically about composing flattened index manually.
    # It is assumed that all arrays are in C-order and resulting index will be used on arrays with C-order.
    # A flattened index of an array cell at arr[3,4,5], where arr has dimensions (30,40,50), can be calculated as follows:
    # index = 3*40*50 + 4*50 + 5
    # Function follows the following algorithm:
    # First, multidimensional index is extracted, then shifting along the axes is done and finally 1d indexes
    # are calculated based on the shifted multidimensional index using the example above.
    # Function np.where is used to do bounds check and prevent falling out of the array

    # Getting flattened index from boolean array
    ghosts = ghosts_bool.ravel().nonzero()
    # Getting multidimensional index
    z, y, x = ghosts_bool.nonzero()

    # Preparing shifted indexes along each axis and direction with bounds check
    xb = np.where((x+1)<ghosts_bool.shape[2], x+1, x)
    xf = np.where((x-1)>=0, x-1, x)
    yb = np.where((y+1)<ghosts_bool.shape[1], y+1, y)
    yf = np.where((y-1)>=0, y-1, y)
    zb = np.where((z+1)<ghosts_bool.shape[0], z+1, z)
    zf = np.where((z-1)>=0, z-1, z)

    # Getting final flattened index
    dim0 = ghosts_bool.shape[2] * ghosts_bool.shape[1]
    dim1 = ghosts_bool.shape[2]
    ghosts_xf = (z*dim0 + y*dim1 + xf)
    ghosts_xb = (z*dim0 + y*dim1 + xb)
    ghosts_yf = (z*dim0 + yf*dim1 + x)
    ghosts_yb = (z*dim0 + yb*dim1 + x)
    ghosts_zf = (zf*dim0 + y*dim1 + x)
    ghosts_zb = (zb*dim1 + y*dim1 + x)

    return ghosts, ghosts_xf, ghosts_xb, ghosts_yf, ghosts_yb, ghosts_zf, ghosts_zb


def prepare_surface_index(surface:np.ndarray):
    """
    Get a multiindex from the surface array

    :param surface: boolean array defining surface cells position in space
    :return: tuple of 1d ndarrays
    """
    index = surface.nonzero()
    return np.intc(index[0]), np.intc(index[1]), np.intc(index[2])


def stencil_debug(grid_out, grid, z_index, y_index, x_index):
    xdim, ydim, zdim = grid.shape
    l = z_index.size
    cond = 0
    zero_count = 0
    for i in range(l):
        z = z_index[i]
        y = y_index[i]
        x = x_index[i]
        zero_count = 0
        if z<zdim-1 and z>0:
            cond += 1
            if y<ydim-1 and y>0:
                cond += 1
                if x<xdim-1 and x>0:
                    cond += 1
        if cond == 3:
            # Z - axis
            if grid[z+1, y, x] != 0:
                grid_out[z, y, x] = grid_out[z, y, x] + grid[z+1, y, x]
            else:
                zero_count += 1
            if grid[z-1, y, x] != 0:
                grid_out[z, y, x] = grid_out[z, y, x] + grid[z-1, y, x]
            else:
                zero_count += 1

            # Y - axis
            if grid[z, y+1, x] != 0:
                grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y+1, x]
            else:
                zero_count += 1
            if grid[z, y-1, x] != 0:
                grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y-1, x]
            else:
                zero_count += 1

            # X - axis
            if grid[z, y, x+1] != 0:
                grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y, x+1]
            else:
                zero_count += 1
            if grid[z, y, x-1] != 0:
                grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y, x-1]
            else:
                zero_count += 1
            grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y, x] * zero_count
            zero_count = 0
            cond = 0
        else:
            # Z - axis
            if z>zdim-1:
                zero_count  += 1
            else:
                if grid[z + 1, y, x] != 0:
                    grid_out[z, y, x] = grid_out[z, y, x] + grid[z + 1, y, x]
                else:
                    zero_count += 1
            if z<1:
                zero_count += 1
            else:
                if grid[z - 1, y, x] != 0:
                    grid_out[z, y, x] = grid_out[z, y, x] + grid[z - 1, y, x]
                else:
                    zero_count += 1
            # Y - axis
            if y>ydim-2:
                zero_count += 1
            else:
                if grid[z, y + 1, x] != 0:
                    grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y + 1, x]
                else:
                    zero_count += 1
            if y<1:
                zero_count += 1
            else:
                if grid[z, y - 1, x] != 0:
                    grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y - 1, x]
                else:
                    zero_count += 1
            # X - axis
            if x>xdim-2:
                zero_count += 1
            else:
                if grid[z, y, x + 1] != 0:
                    grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y, x + 1]
                else:
                    zero_count += 1
            if x<1:
                zero_count += 1
            else:
                if grid[z, y, x - 1] != 0:
                    grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y, x - 1]
                else:
                    zero_count += 1
            grid_out[z, y, x] = grid_out[z, y, x] + grid[z, y, x] * zero_count
            zero_count = 0
        cond = 0