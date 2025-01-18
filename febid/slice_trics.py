"""
A collection of utility functions for manipulating Python slices
"""

import numpy as np
import operator


operator_dict = {
    '==': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le
}


def get_3d_slice(center, shape, n=1):
    """
    Get a 3D slice from a 3D array with a specified center encapsulating n-direct neighbors from each side.
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


def transformSliceToOriginal(slc1, slc2):
    """
    Transform a slice object to operate on the original array.

    An array arr is sliced with slc1 to create a view arr_view1. Then, the slice object slc2 is applied to arr_view to
    create a view arr_view2. This function transforms slc2 to create the view arr_view2 directly from the array arr.

    :param slc1: The slice object applied to the original array
    :param slc2: The slice object applied to the view array

    :return: The transformed slice object to extract view specified by slc2 from the original array
    """
    transformed_slicer = tuple(
        slice(
            slc1[i].start + slc2[i].start,
            slc1[i].start + slc2[i].stop,
            slc2[i].step,
        )
        for i in range(len(slc2))
    )
    return transformed_slicer


def slice_3d_to_1d_index(slicer, shape):
    """
    Convert a 3D slice into a 1D index for a flattened array.
    The idex is to be used to extract the same cells, but from a flattened array.

    :param shape: The original 3D array shape
    :param slicer: The 3D slice object

    :return: The 1D index corresponding to the 3D slice
    """
    x_indices = np.arange(*slicer[0].indices(shape[0]))
    y_indices = np.arange(*slicer[1].indices(shape[1]))
    z_indices = np.arange(*slicer[2].indices(shape[2]))
    x, y, z = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
    flat_indices = np.ravel_multi_index((x.ravel(), y.ravel(), z.ravel()), shape)
    return flat_indices


def get_index_in_parent(index, slicer):
    """
    Get the index in the parent array from the index in the view array.

    :param index: The index in the view array, must be array of tuples, like the result of np.argwhere()
    :param slicer: The slice object applied to the parent array

    :return: The index in the parent array
    """
    if len(index) != len(slicer):
        raise IndexError("Index has different number of dimensions")
    result_index = []
    for i, item in enumerate(index):
        if slicer[i].start is None:
            result_index.append(item)
        else:
            result_index.append(item + slicer[i].start)
    return tuple(result_index)


def find_bounding_slice(array):
    """
    Find a slice that contains all nonzero elements of a 3D array.

    :param array: The 3D array to find the bounding slice for
    :return: a tuple of slice objects
    """
    shape = array.shape
    min_bounds = [shape[0], shape[1], shape[2]]
    max_bounds = [-1, -1, -1]

    # Scan along each axis independently
    for axis in range(3):
        for start in range(shape[axis]):
            # Create slices to scan along this axis
            selector = [slice(None)] * 3
            selector[axis] = slice(start, start + 1)
            if np.any(array[tuple(selector)] != 0):
                min_bounds[axis] = start
                break

        for end in range(shape[axis] - 1, -1, -1):
            selector = [slice(None)] * 3
            selector[axis] = slice(end, end + 1)
            if np.any(array[tuple(selector)] != 0):
                max_bounds[axis] = end
                break

    # Check if no nonzero elements were found
    if max_bounds[0] == -1:
        return None

    # Construct slices
    slicer = tuple(slice(min_bounds[axis], max_bounds[axis] + 1) for axis in range(3))
    return slicer


def index_where(array, condition='!=', value=0):
    """
    Find the indices of elements in an array that satisfy a condition. Condition is a string that would conventionally be
    used in a comparison operation, i.e. '!=0' or '>'.

    By default, the function finds the indices of nonzero elements in the array and would be equal to np.nonzero(array),
    however up to 4x faster.
    It is recommended to use contiguous arrays as the function uses np.ravel() to flatten the array.

    :param array: The array to search
    :param condition: The condition to satisfy
    :param value: The value to compare against
    :return: A tuple of indices
    """
    array_flat = array.ravel()
    operator_func = operator_dict[condition]
    index_1d = np.where(operator_func(array_flat, value))[0]
    index = np.unravel_index(index_1d, array.shape)
    return index


def any_where(array, condition='!=0', value=0, reverse=False, chunk_size=1024*32):
    """
    Check if any element in the array satisfies a condition. Condition is a string that would conventionally be
    used in a comparison operation.

    By default, the function returns True on the first nonzero element. The search is chunked to improve performance (up to 4x faster).
    It is recommended to use contiguous arrays as the function uses np.ravel() to flatten the array.

    :param array: The array to search
    :param condition: The condition to satisfy
    :param value: The value to compare against
    :param reverse: Whether to search in reverse
    :param chunk_size: The chunk size for the search
    :return: True if any element satisfies the condition, False otherwise
    """
    array_flat = array.ravel()
    operator_func = operator_dict[condition]
    # But going in reverse is even faster
    if reverse:
        for start in range(array_flat.size - 1, -1, -chunk_size):
            chunk = array_flat[max(0, start - chunk_size - 1):start + 1]
            if chunk.max() >= value:
                return True
    else:
        for start in range(0, array_flat.size, chunk_size):
            chunk = array_flat[start:min(array_flat.size, start + chunk_size)]
            if chunk.max() >= value:
                return True
    return False


def concat_index(arr1, arr2):
    """
    Concatenate two sets of indices represented as tuple of arrays(np.nonzero-like).

    :param arr1: The first set of indices
    :param arr2: The second set of indices
    :return: tuple of concatenated indices
    """
    if arr1 is None:
        return arr2
    if arr2 is None:
        return arr1
    return tuple(np.concatenate((arr1[i], arr2[i])) for i in range(len(arr1)))