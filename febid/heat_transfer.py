"""
Heat transfer module
"""

import warnings

import numpy as np
from febid.libraries.rolling import roll
from febid.libraries.pde import tridiag
import numexpr_mod as ne
from febid.diffusion import laplace_term_stencil, prepare_surface_index

# Heat transfer is currently solved statically for a steady-state condition according to
# Simultaneous Over-Relaxation (SOR) method.

# FTCS (Forward in Time Central in Space) scheme is implemented as well for a real-time solution.
# Stability condition for FTCS in 3D space: ∆t<=∆x^2/6k
# Implicit Euler ADI method is implemented as well.
# Keep in mind, that while relaxation (SOR) requires only thermal conductivity,
# real-time solvers require density and heat capacity as well.

# Algorithm works with 3-dimensional arrays, which represent a discretized space with a cubic cell.
# A value held in a cell corresponds to the temperature of that volume.
# In order to manage and index cells that hold a temperature value, a boolean array with the same shape is used,
# where True flag marks cells with a value.
# This algorithm defines temperature of the solid.
# Surface temperature is calculated later by averaging neighboring solid cells.

a, temp = 0.1, 0.1
heat_equation = ne.cache_expression('a*temp', 'heat_equation')
eV_J = 1.60217733E-19


def get_heat_transfer_stability_time(k, rho, cp, dx):
    """
    Get the largest stable time step for the FTCS scheme.

    :param k: thermal conductivity, [W/m/K]
    :param rho: density, [g/cm^3]
    :param cp: heat capacity, [J/kg/K]
    :param dx: grid step (cell size), nm
    :return: time step in seconds
    """
    heat_dt = dx**2*cp*rho*1e-24 / (6 * k) # maximum stability
    return heat_dt


def temperature_stencil(grid, k, cp, rho, dt, dl, heat_source=0, solid_index=None, substrate_T=294, flat=False, add=0):
    """
    Calculates diffusion term for the surface cells using stencil operator

        Nevertheless, 'solid_index' is an optional argument,
    it is highly recommended to handle index from the caller function.

    :param grid: 3D temperature array
    :param k: thermal conductivity, [W/K/m]
    :param cp: heat capacity, [J/kg/K]
    :param rho: density, [g/cm^3]
    :param dt: time interval over which diffusion term is calculated, s
    :param dl: grid spacing (cell size), nm
    :param heat_source: 3D volumetric heating source array, [W/nm^3]
    :param solid_index: indexes of solid cells
    :param substrate_T: temperature of the substrate
    :param flat: if True, returns a flat array of surface cells. Otherwise, returns a 3d array with the same shape as grid.
    :param add:
    :return: 3d or 1d ndarray
    """
    if solid_index is None:
        solid_index = prepare_surface_index(grid)
    y, x = (grid[0]).nonzero()
    # Getting equation constants
    a = k / cp / rho * 1e24 # thermal diffusivity
    A = dt * a / dl ** 2 # base matrix coefficient
    S = heat_source * (dt / cp / rho * 1e24 * 1.60217733E-19)# heating source
    grid += add
    grid_out = laplace_term_stencil(grid, solid_index)
    grid -= add
    grid_out += S
    grid[0, y, x] = substrate_T
    grid_out[0, y, x] = 0
    if flat:
        return ne.re_evaluate('heat_equation', local_dict={'a': A, 'temp': grid_out[solid_index]}, casting='same_kind')
    else:
        return ne.re_evaluate('heat_equation', local_dict={'a': A, 'temp': grid_out}, casting='same_kind')


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


def heat_transfer_steady_sor(grid, k, dl, heat_source, eps, solid_index=None):
    """
    Find steady-state solution to the heat equation with the given accuracy

    :param grid: 3D temperature array
    :param k: thermal conductivity, [W/K/m]
    :param dl: grid spacing (cell size), nm
    :param heat_source: 3D volumetric heating source array, [W/nm^3]
    :param eps: desired accuracy
    :param solid_index: indexes of solid cells
    :return: 3D temperature array
    """

    # For any arbitrary big structure the number of iterations required is significant for a small difference (0.5 K)
    # and scales with the number of cells, thus there is no need to check for desired accuracy often.
    # Accuracy check requires a full array copy, while SOR calculation is done in-place.
    # The accuracy evaluation is itself incredibly intensive and takes 80% of the function run time,
    # if called on every iteration.
    # Here, accuracy evaluation is done only a few times.
    # Initially, an accuracy check step is evaluated based on the number of cells.
    # After 3 checks the total required number of iterations is predicted
    # and SOR runs uninterrupted to it. This algorithm is then repeated until the desired accuracy is reached.
    # It takes only up to 18 accuracy evaluations and consequently up to 15 predictions for any relaxation task.
    # The overrun(excessive iterations) is up to 6% which goes down for longer runs.
    #
    # The prediction is based on exponential behaviour of accuracy propagation (vs N of iterations).
    # By consequently fitting calculated norms to the exp. equation and calculating the number of required iterations
    # it is possible to keep the number of required accuracy evaluations almost constant for any case.
    # An important remark: the predicted N of iterations has to always be overcompensated, otherwise the algorithm will
    # check accuracy with an increasing frequency.

    def get_index(grid):
        if solid_index:
            z, y, x = solid_index
        else:
            z, y, x = grid.nonzero()
            z, y, x = z.astype(np.intc), y.astype(np.intc), x.astype(np.intc)
        return z, y, x
    print(f'\nFinding steady state solution using Simultaneous Over-Relaxation:')
    p_j = 1 - np.pi ** 2 / grid.shape[0] ** 2 / 2 # spectral radius of the Jacobi iteration
    S = heat_source * dl ** 2 / k * 1.60217733E-19
    grid_gs = np.zeros_like(grid)
    z, y, x = get_index(grid)
    norm = 1  # achieved accuracy
    base_step = np.ceil(z.shape[0] / 1000) * 10 # accuracy checks step
    skip_step = base_step * 5
    skip = skip_step # next planned accuracy check iteration
    prediction_step = skip_step * 3
    n_predictions = 0
    norm_array = []
    iters = []
    for i in range(1000000):
        if i == 0: # over-relaxation parameter
            w = 1 / (1 - p_j ** 2 / 2)
        else:
            w = 1 / (1 - p_j ** 2 * w / 4)
        if i % skip == 0 and i != 0: # skipping array copy
            grid_gs[...] = grid
        roll.stencil_sor(grid, S, w, z, y, x)
        if i % skip == 0 and i != 0: # skipping achieved accuracy evaluation
            norm = (np.linalg.norm(grid[1:]-grid_gs[1:])/ np.linalg.norm(grid[1:]))
            norm_array.append(norm) # recording achieved accuracy for fitting
            iters.append(i)
            skip += skip_step
        if eps > norm:
            print(f'Reached solution with an error of {norm:.3e}')
            return grid
        if i % prediction_step == 0 and i != 0:
            a, b = fit_exponential(iters, norm_array)
            skip = int((np.log(eps) - a)/b) + skip_step * n_predictions # making a prediction with overcompensation
            prediction_step = skip # next prediction will be after another norm is calculated
            n_predictions += 1
    raise RuntimeError(f'Exceeded a million iterations during solving a steady-state heat transfer problem. \n'
                       f'function: febid.heat_transfer.heat_transfer_steady_sor')


def fit_exponential(x0, y0):
    """
    Fit data to an exponential equation y = a*exp(b*x)

    :param x0: x coordinates
    :param y0: y coordinates
    :return: ln(a), b
    """
    x = np.array(x0)
    y = np.array(y0)
    p = np.polyfit(x, np.log(y), 1)
    a = p[1]
    b = p[0]
    # returning ln(a) to directly solve for desired x
    return a, b

def fragmentise(grid):
    """
    Collect columns along each axis that do not contain zero cells

    :param grid: 3d array
    :return: array of index triples
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

    This function virtually prevents zeros from appearing in a solution matrix by extracting the 'solid' cells along
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
