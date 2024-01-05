"""
A collection of utility functions for manipulating Python slices
"""

import numpy as np


def get_3d_slice(center, shape, n=1):
    """
    Get a 3D slice from a 3D array with a specified center encapsulating n-direct neighbors from each side
    :param center: slice center
    :param shape: initial array shape
    :param n: int
    :return: resulting slice, original center position in the new slice
    """
    z, y, x = center

    # Ensure center is within array bounds
    if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
        raise ValueError("Invalid center coordinates")

    # Define the range for each dimension with consideration for boundaries
    z_range = slice(max(0, z - n), min(shape[0], z + n + 1))
    y_range = slice(max(0, y - n), min(shape[1], y + n + 1))
    x_range = slice(max(0, x - n), min(shape[2], x + n + 1))

    # Extract the 3D slice
    result_slice = np.s_[z_range, y_range, x_range]

    # Adjust the center position
    center_new = (
        z - max(0, z - n),
        y - max(0, y - n),
        x - max(0, x - n)
    )

    return result_slice, center_new


def get_boundary_indices(cell, shape, size=1):
    """
    Return a slice of the n-direct neighbor cells with the specified center into the array of given shape.

    :param cell: The center cell of the slice
    :param shape: The shape of the sliced array
    :param size: The number of neighbors to select
    :return: A slice of the n-direct neighbor cells
    """

    def determine_slice(axis, coord):
        if coord + size + 1 > shape[axis]:
            stop = None
        else:
            stop = coord + size + 1
        if coord - size < 0:
            start = None
        else:
            start = coord - size
        return slice(start, stop)

    z_slice = determine_slice(0, cell[0])
    y_slice = determine_slice(1, cell[1])
    x_slice = determine_slice(2, cell[2])

    result_slice = np.s_[z_slice, y_slice, x_slice]

    return result_slice


def get_slice_into_parent(slc1, slc2):
    def to_zero(var):
        if var is None:
            return 0
        return var

    def get_start(st1, st2):
        if st1 is None and st2 is None:
            return None
        if st1 is None and st2 is not None:
            return 0
        if st1 is not None and st2 is None:
            return st1
        return st1 + st2

    def get_stop(st1, st2, st3, st4):
        """
        :param st1: stop of slc1
        :param st2: stop of slc2
        :param st3: length of slc1
        :param st4: calculated stop
        """
        if st1 is None and st2 is None:
            return None
        if st1 is None and st2 is not None:
            return st2
        if st1 is not None and st2 is None:
            return st3 + st4
        return st3 + st4

    if len(slc1) != len(slc2):
        raise IndexError("Slice has different number of dimensions")
    result_slice = []
    for i in range(len(slc1)):
        if all([True if s is None else False for s in slc1[i]]):
            result_slice.append(slc2[i])
        elif all([True if s is None else False for s in slc2[i]]):
            result_slice.append(slc1[i])
        slc_start = get_start(slc1[i].start, slc2[i].start)
        slc_stop = get_stop(slc1[i].stop, slc2[i].stop, to_zero(slc1[i].stop) - to_zero(slc1[i].start), slc_start)
        slc = slice[slc_start: slc_stop]
        result_slice.append(slc[0])
    return tuple(result_slice)


def calculate_absolute_slice(slice1, slice2):
    """
    Calculate the absolute slice from two slices
    :param slice1: first slice
    :param slice2: second slice
    :return: absolute slice
    """
    # Create a dummy array to infer the original shape
    dummy_array = np.zeros((1,) * len(slice1))

    # Convert slices to numpy slice objects
    slice1_np = np.s_[slice1]
    slice2_np = np.s_[slice2]

    # Combine slices before applying them
    combined_slices = tuple(slice2_np[i] if s == slice(None) else s for i, s in enumerate(slice1_np))

    # Create the combined absolute slice
    combined_absolute_slice = np.s_[combined_slices]

    return combined_absolute_slice


def get_center_view(array, n=1):
    shape = np.array(array.shape)
    # if np.any(shape < n*2+1):
    #     raise ValueError("The larger array must be larger than the view size")

    center_index = shape.max() // 2

    view_slice = []
    for i in range(shape.size):
        start = center_index - n
        end = center_index + n + 1
        view_slice.append(slice(start if start >= 0 else None,
                                end if end < shape[i] else None))
    view_slice = tuple(view_slice)
    center_view = array[view_slice]
    return center_view
