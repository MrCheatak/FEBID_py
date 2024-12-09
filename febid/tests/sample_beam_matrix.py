import numpy as np
from febid.Structure import Structure

def test_beam_matrix(x, y, sim, pr, resized=False):
    """
    Generate a Gaussian beam matrix. For testing purposes only.

    :param x: x position of the beam
    :param y: y position of the beam
    :param sim: MC simulation object
    :param pr: Process object
    :param resized: flag to indicate if the structure was resized

    :return: 3D array of int32
    """
    if resized:
        beam_matrix_buff = None
        filled_cells_init = None
    else:
        beam_matrix_buff = pr.beam_matrix
        filled_cells_init = pr.full_cells
    beam_matrix = sample_beam_matrix(x, y, sim.pe_sim.sigma, pr.structure,
                                     beam_matrix_buff=beam_matrix_buff,
                                     filled_cells_init=filled_cells_init)
    return beam_matrix


def sample_beam_matrix(x_pos, y_pos, a, structure:Structure, f0=1e6, beam_matrix_buff=None, filled_cells_init=None):
    """
    Sample a gaussian beam matrix for testing purposes. Optimized for quick updates based on filled cells and irradiated area slice.

    :param a: Gaussian standard deviation
    :param structure: structure object of the simulation
    :param f0: Gaussian peak value
    :param beam_matrix_buff: buffer for the beam matrix
    :param filled_cells_init: filled cells, must be an array of tripples
    :param irrad_area_2d: irradiated area slice
    :return: 3D array of int32
    """
    # Generate a 2D Gaussian PDF matrix
    _, ydim, xdim = structure.shape
    _, ydim_abs, xdim_abs = structure.shape_abs
    x = np.linspace(0, xdim_abs, xdim) - x_pos
    y = np.linspace(0, ydim_abs, ydim) - y_pos
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x * x + y * y)
    gaussian_2d = (np.exp(-(d ** 2 / (2 * a ** 2))) * f0).astype(np.int32)
    # Create a new array if the buffer is not provided
    if beam_matrix_buff is not None:
        beam_matrix = beam_matrix_buff
    else:
        beam_matrix = np.zeros(structure.shape, dtype=np.int32)
    # Clear filled cells and project the values from the 2D Gaussian onto the 3D structure using surface_bool
    if filled_cells_init is not None:

        beam_matrix[*filled_cells_init.T] = 0
        for n in range(len(filled_cells_init)):
            filled_cells_T = filled_cells_init.T
            slice_3d = np.s_[filled_cells_T[0,n] - 1:filled_cells_T[0,n] + 2,
                       filled_cells_T[1,n] - 1:filled_cells_T[1,n] + 2,
                       filled_cells_T[2,n] - 1:filled_cells_T[2,n] + 2]
            slice_2d = np.s_[filled_cells_T[1,n] - 1:filled_cells_T[1,n] + 2,
                       filled_cells_T[2,n] - 1:filled_cells_T[2,n] + 2]
            beam_matrix_view = beam_matrix[slice_3d]
            index = structure.surface_bool[slice_3d]
            gaussian_2d_view = gaussian_2d[slice_2d]
            for i in range(index.shape[1]):
                for j in range(index.shape[2]):
                    beam_matrix_view[index] = gaussian_2d_view[i, j]
    for n in range(ydim):
        for j in range(xdim):
            if gaussian_2d[n, j] < 10:
                continue
            index = structure.surface_bool[:, n, j]
            beam_matrix_view = beam_matrix[:, n, j]
            beam_matrix_view[index] = gaussian_2d[n, j]
    return beam_matrix.astype(np.int32)
