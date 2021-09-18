#cython: language_level=3
#cython: cdivision=True
#in particular enables special integer division

import cython
import numpy as np
from libc.stdlib cimport malloc, realloc, free
from libc.math cimport sqrt
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.array cimport array, clone




####### Notes ##########
# 1. CPython arrays are much quicker to append to a List than a Memoryview slice, even with copying
# 2. Memoryview creation has bigger overhead than creating C-arrays or using malloc() due to a call to Python function
# 3. Classes are created as Python objects, thus their usage is not possible without GIL
# 4. Cython does not yet support VLAs(Variable Length Arrays), which were introduced in C99. That would be a preferred way instead of malloc()


cpdef double traverse_segment(double[:,:,:] energies, double[:,:,:] grid, int cell_dim, double[:,:] p0, double[:,:] pn, double[:,:] direction, double[:,:] t, double[:,:] step_t, double[:] dEs, int max_count):
    """
    Wrapper for Cython function.
    Deposits energies to the structure based on the energy losses.
    
    :param L: distances between segment points
    :param cell_dim: size of a cell
    :param dEs: energies lost on segments
    :param direction: segment pointing direction
    :param energies: structured array of deposited energies
    :param grid: surface array
    :param p0: starting points of segments
    :param pn: c of segments
    :param step_t: increments of t value
    :param t: arbitrary values to detect crossing
    :param N: number of segments
    :return: total deposited energy
    """

    return traverse_segment_c(energies, grid, cell_dim, p0, pn, direction, t, step_t, dEs, dEs.shape[0], max_count)


# cpdef float[:,:] traverse_cells(double[:] p0, double[:] pn, double[:] direction, double[:] t, double[:] step_t, int N):
#     """
#     Wrapper for Cython function.
#     Get a collection of points where ray crossed the cells
#
#     :param p0: starting point
#     :param pn: end-point
#     :param direction: pointing direction
#     :param t: arbitrary values to detect crossing
#     :param step_t: increment of t value
#     :param N: size of output array
#     :return: array of points
#     """
#     cdef:
#         # float[:,:] crossings = np.empty((N,3))
#         float ** crossings = <float**> malloc(N * sizeof(float *))
#     for i in range(N):
#         crossings[i] = <float *> malloc(3 * sizeof(float))
#
#     traverse_cells_c(p0, pn, direction, t, step_t, crossings, N)
#
#     return crossings

cpdef double generate_flux(double[:,:,:] flux, unsigned char[:,:,:] surface, int cell_dim, double[:,:] p0, double[:,:] pn, double[:,:] direction, double[:,:] t, double[:,:] step_t, double[:] n_se, int max_count):
    """
    Wrapper for Cython function.
    Generate surface SE flux. 
    
    :param flux: array to accumulate SEs
    :param surface: array describing surface
    :param cell_dim: size of a grid cell
    :param p0: starting points
    :param pn: end-points
    :param direction: pointing directions(vectors)
    :param t: arbitrary values to detect crossing
    :param step_t: increments of t value
    :param n_se: number of SEs emitted 
    :param max_count: maximum number of crossing events per emission
    :return: total SE yield
    """

    return generate_flux_c(flux, surface, cell_dim, p0, pn, direction, t, step_t, n_se, n_se.shape[0], max_count)

cpdef double det_1d(double[:] vector):
    """
    Calculate the length of a vector
    :param vector: array with 3 elements
    :return: 
    """
    return det_c_debug(vector)

cpdef void det_2d(double[:,:] arr_of_vectors, double[:] out):
    """
    Calculate the length of vectors in an array 
    
    :param arr_of_vectors: array of vectors listed along 0 axis
    :param out: output array, has to be the same length as input's 0 axis
    :return: 
    """
    for i in range(arr_of_vectors.shape[0]):
        out[i] = det_c_debug(arr_of_vectors[i])

# cpdef double[:] det_2d_full(double[:,:] arr_of_vectors):  # returns a memeoryview, not an ndarray
#     """
#     Calculate the length of vectors in an array
#
#     :param arr_of_vectors: array of vectors listed along 0 axis
#     :param out: output array, has to be the same length as input's 0 axis
#     :return:
#     """
#     cdef double[:] out = np.empty(arr_of_vectors.shape[0])
#     for i in range(arr_of_vectors.shape[0]):
#         out[i] = det_c_debug(arr_of_vectors[i])
#     return out

################ Cython Functions #################
## For internal use only

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double generate_flux_c(double[:,:,:] flux, unsigned char[:,:,:] surface, int cell_dim, double[:,:] p0, double[:,:] pn, double[:,:] direction, double[:,:] t, double[:,:] step_t, double[:] n_se, int N, int max_count) nogil:
    cdef:
        char ind
        int i, q, r, cond = 0, count = 0
        double next_t, total_flux=0
        # Allocating memory, the size corresponds to the length of SE trajectory
        float ** crossings = <float**> malloc(max_count * sizeof(float *))
        int ** coords = <int**> malloc(max_count * sizeof(int *))

    for i in range(max_count):
        crossings[i] = <float *> malloc(3 * sizeof(float))
        coords[i] = <int *> malloc(3 * sizeof(int))

    for q in range(N):
        # Traversing cells
        count = traverse_cells_c(p0[q], pn[q], direction[q], t[q], step_t[q], crossings, max_count)
        if count>max_count:
            with gil:
                print(f'FL Length: {det_c_debug(direction[q])}, \n'
                          f'Direction: {direction[q,0],direction[q,1],direction[q,2]} \n'
                          f'p0: {p0[q,0],p0[q,1],p0[q,2]} \n'
                          f'pn: {pn[q,0],pn[q,1],pn[q,2]} \n'
                          f't: {t[q,0],t[q,1],t[q,2]} \n'
                          f'step_t: {step_t[q,0],step_t[q,1],step_t[q,2]} \n'
                          f'Count, max: {count}, {max_count}')
        # Getting coordinates
        for i in range(count):
            for r in range(3):
                coords[i][r] = (<int> crossings[i][r]) / cell_dim
        # Yielding SEs
        for i in range(count):
            #Bounds check
            cond = cond + (coords[i][0]<145)
            cond = cond + (coords[i][1]<70)
            cond = cond + (coords[i][2]<70)
            cond = cond + (coords[i][0]>=0)
            cond = cond + (coords[i][1]>=0)
            cond = cond + (coords[i][2]>=0)
            if cond == 6:
                # Checking if the cell is a surface cell
                if surface[coords[i][0], coords[i][1], coords[i][2]]:
                    # Yielding SEs
                    flux[coords[i][0], coords[i][1], coords[i][2]] += n_se[q]
                    total_flux += n_se[q]
                    cond = 0
                    break # An SE may cross a wall of a surface cell and end in it.
                    # Both positions are recorded by the algorithm,
                    # which would artificially double the yield. This Break line prevents it.
            cond = 0
        count = 0
    # Freeing memory before exiting function
    # It has to be done in reverse to the allocation process
    for i in range(max_count):
        if crossings[i] != NULL: # Trying to free already empty memory may result in exception
            free(crossings[i])
        else: raise MemoryError("Trying to free an empty memory block").with_traceback()
        crossings[i] = NULL
        if coords[i] != NULL:
            free(coords[i])
        else: raise MemoryError("Trying to free an empty memory block").with_traceback()
        coords[i] = NULL
    free(coords)
    coords = NULL
    free(crossings)
    crossings = NULL

    return total_flux


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double traverse_segment_c(double[:,:,:] energies, double[:,:,:] grid, int cell_dim, double[:,:] p0, double[:,:] pn, double[:,:] direction, double[:,:] t, double[:,:] step_t, double[:] dEs, int N, int max_count) nogil:
    cdef:
        char ind
        int i, q, r, count = 0
        double next_t = 0, total_energy = 0
        # Allocating memory, the size corresponds to the longest segment
        float ** crossings = <float**> malloc(max_count * sizeof(float *)) #crossings[1000][3]
        int ** coords = <int**> malloc(max_count * sizeof(int *))
        float ** deltas = <float**> malloc(max_count * sizeof(float *))

    for i in range(max_count):
        crossings[i] = <float *> malloc(3 * sizeof(float))
        deltas[i] = <float *> malloc(3 * sizeof(float))
        coords[i] = <int *> malloc(3 * sizeof(int))
    for q in range(N):  # go through segments, from here segment -> ray
        # Traversing cells
        count = traverse_cells_c(p0[q], pn[q], direction[q], t[q], step_t[q], crossings, max_count)
        count -= 1 # Because we calculate distances between points
        with gil:
            if count>0:
                if count>max_count:
                    print(f'SEG Length: {det_c_debug(direction[q])}, \n'
                          f'Direction: {direction[q,0],direction[q,1],direction[q,2]} \n'
                          f'p0: {p0[q,0],p0[q,1],p0[q,2]} \n'
                          f'pn: {pn[q,0],pn[q,1],pn[q,2]} \n'
                          f't: {t[q,0],t[q,1],t[q,2]} \n'
                          f'step_t: {step_t[q,0],step_t[q,1],step_t[q,2]} \n'
                          f'E: {dEs[q]}, Count, max: {count}, {max_count}')
        # Getting distances between crossings and crossing coordinates
        for i in range(count):
            for r in range(3):
                deltas[i][r] = crossings[i + 1][r] - crossings[i][r]
                coords[i][r] = <int>(crossings[i + 1][r] / cell_dim)
        # Depositing energy
        for i in range(count):
            if grid[coords[i][0], coords[i][1], coords[i][2]] <= -1:
                next_t = det_c(deltas[i]) * dEs[q]
                energies[coords[i][0], coords[i][1], coords[i][2]] += next_t
                total_energy += next_t
        count = 0
    # Freeing memory before exiting function
    # It has to be done in reverse to the allocation process
    for i in range(max_count):
        free(crossings[i])
        crossings[i]=NULL
    # for i in range(count_prev):
        free(coords[i])
        coords[i] = NULL
        free(deltas[i])
        deltas[i] = NULL
    free(coords)
    coords = NULL
    free(deltas)
    deltas = NULL
    free(crossings)
    crossings = NULL

    return total_energy


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int traverse_cells_c(double[:] p0, double[:] pn, double[:] direction, double[:] t, double[:] step_t, float ** crossings, int max) nogil:
    """
    Get coordinates, where the ray crosses walls of grid cells including end coordinates of the ray.
    AABB Ray-Voxel traversal algorithm taken from https://www.shadertoy.com/view/XddcWn#
    
    :param p0: origin of the ray
    :param pn: end point of the ray
    :param direction: vector of the ray
    :param t: arbitrary value indicating a crossing event
    :param step_t: increment of t value, component-wise
    :param crossings: output array for coordinates; the length of the array has to be bigger than the resulting count
    :param max: length of output array
    :return: count of the crossing coordinates
    
    If array has less capacity than the resulting count, it would result in memory 
    leakage and corruption, which is not detected on spot, but only upon garbage collection. 
    Usually Segmentation violation 11 error is emitted.
    """
    cdef:
        char ind
        int i, count = 0
        double next_t

    for i in range(3):
        crossings[count][i] = <float> p0[i]
    count +=1
    while True:  # iterating until all the cells are traversed by the ray
        next_t, ind = arr_min(t)  # minimal t-value corresponds to the box wall crossed; 2x faster than t.min() !!
        if next_t > 1:  # finish if trajectory ends inside a cell (t>1)
            for i in range(3):
                crossings[count][i] = <float> pn[i]
            count += 1
            ###Enable for debugging ###
            # if count-1>= max:
            #     with gil:
            #         print(f'TC Length: {det_c_debug(direction)}, \n'
            #               f'Direction: {direction[0],direction[1],direction[2]} \n'
            #               f'p0: {p0[0],p0[1],p0[2]} \n'
            #               f'pn: {pn[0],pn[1],pn[2]} \n'
            #               f't: {t[0],t[1],t[2]} \n'
            #               f'step_t: {step_t[0],step_t[1],step_t[2]} \n'
            #               f'Crossing: {crossings[count][0],crossings[count][1],crossings[count][2]} ')
            break
        for i in range(3):
            crossings[count][i] = <float> (p0[i] + next_t * direction[i])
        count += 1
        t[ind] = t[ind] + step_t[ind]  # going to the next wall; 7x faster than with (t==next_t)
    return count


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline (double, char) arr_min(double[:] x) nogil:
    """
    Find the minimum value in the array(vector).
    
    :param x: input array, has to have a size of 3
    :return: (min value, index of min value)
    """
    if x[0] >= x[1]:
        if x[1] >= x[2]:
            return x[2], 2
        else:
            return x[1], 1
    else:
        if x[0] >= x[2]:
            return x[2], 2
        else:
            return x[0], 0

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline double det_c(float* vec) nogil:
    """
    Find the length of a vector.
    
    :param vec: vector array
    :return: length
    """
    cdef double length = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    return length

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline double det_c_debug(double[:] vec) nogil:
    cdef double length = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    return length



