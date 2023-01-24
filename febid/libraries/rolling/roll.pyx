#cython: language_level=3

import traceback
cimport cython
from cython.parallel cimport prange


# Cython performs 2-3 times faster than numpy when adding one
# array slice to another. This applies to medium size arrays from [10:10:10] to [500:500:500].
# With smaller arrays bottleneck is the overhead, with bigger arrays Numpy works faster

cpdef int surface_temp_av(double[:,:,:] surface_temp, double[:,:,:] temp, int[:] z, int[:] y, int[:] x) except -1:
    """
    Define temperature of the surface cells by averaging temperature of the neighboring solid cells 

    :param surface_temp: surface temperature array
    :param temp: solid temperature array
    :param z: first array index
    :param y: second array index
    :param x: third array index 
    :return: 
    """
    try:
        return surface_temp_av_cy(surface_temp, temp, z, y, x)
    except Exception as ex:
        traceback.print_exc()
        raise ex


cpdef int stencil_sor(double[:,:,::1] grid, double[:,:,:] s, double w, int[:] z, int[:] y, int[:] x) except -1:
    """
    Stencil operator. Sums all the neighbors to the current cell. 
    If a neighbor is 0 or out of the bounds, then adds cell's current value to itself.
    Arrays must have the same shape.
    :param grid: operated array
    :param s: power source array
    :param w: over-relaxation parameter
    :param z_index: first array index
    :param y_index: second array index
    :param x_index: third array index
    :return: 
    """
    try:
        stencil_sor_cy(grid, s, w, z, y, x)
    except Exception as ex:
        traceback.print_exc()
        raise ex


cpdef int stencil_gs(double[:,:,::1] grid, double[:,:,:] s, int[:] z, int[:] y, int[:] x) except -1:
    """
    Stencil operator. Sums all the neighbors to the current cell. 
    If a neighbor is 0 or out of the bounds, then adds cell's current value to itself.
    Arrays must have the same shape.
    :param grid: operated array
    :param s: power source array
    :param w: over-relaxation parameter
    :param z_index: first array index
    :param y_index: second array index
    :param x_index: third array index
    :return: 
    """
    try:
        stencil_gs_cy(grid, s, z, y, x)
    except Exception as ex:
        traceback.print_exc()
        raise ex


cpdef int stencil(double[:,:,::1] grid_out, double[:,:,::1] grid, int[:] z, int[:] y, int[:] x) except -1:
    """
    Stencil operator. Sums all the neighbors to the current cell. 
    If a neighbor is 0 or out of the bounds, then adds cell's current value to itself.
    Arrays must have the same shape.
    :param grid_out: operated array
    :param grid: source array
    :param z: first array index
    :param y: second array index
    :param x: third array index 
    :return: 
    """
    try:
        stencil_cy(grid_out, grid, z, y, x)
    except Exception as ex:
        traceback.print_exc()
        raise ex


cpdef void rolling_3d(double[:,:,:] arr, const double[:,:,:] brr):
    """
    Analog of the np.roll for 3d arrays
    :param arr: array to add to
    :param brr: addition
    
    :return: 
    """
    rolling_3d_cy(arr, brr)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void rolling_3d_cy(double[:,:,:] arr, const double[:,:,:] brr) nogil:
    cdef int i, j, k
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                arr[i,j,k] = arr[i,j,k] +  brr[i,j,k]



cpdef void rolling_2d(double[:,:] a, const double[:,:] b):
    """
    Analog of the np.roll for 2d arrays
    :param arr: array to add to
    :param brr: addition

    :return: 
    """
    rolling_2d_cy(a, b)


@cython.initializedcheck(False) # turn off initialization check for memoryviews
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int stencil_cy(double[:,:,::1] grid_out, double[:,:,::1] grid, int[:] z_index, int[:] y_index, int[:] x_index) nogil except -1:
    """
    Apply Laplace operator via stencil. 
    Ghost cell method implied: neighbors, that equal to 0 or that are out of the bonds reflect
    the value in the current cell. 
    
    :param grid_out: operated array
    :param grid: source array
    :param z_index: first array index
    :param y_index: second array index
    :param x_index: third array index
    :return: 
    """
    cdef int i, z, y, x, xdim, ydim, zdim, zero_count = 0
    cdef double cum_sum = 0
    xdim = grid_out.shape[2]
    ydim = grid_out.shape[1]
    zdim = grid_out.shape[0]
    l = z_index.shape[0]

    # Assumptions taken for optimization:
    #   1. Most cells are inside the array, thus bounds check should be quick. Cells on the boundary are processed separately.
    #   2. There are at least 3 non-zero neighbors, but usually 4, therefore the first condition is !=0
    for i in range(l):
        z = z_index[i]
        y = y_index[i]
        x = x_index[i]
        zero_count = 0
        cum_sum = 0
        zero_count = stencil_base(&cum_sum, grid, x, xdim, y, ydim, z, zdim)
        # with gil: # show actual calculation for debugging
        #     print(f'{z, y, x}:  {grid_out[z,y,x]+cum_sum + grid[z, y, x] * zero_count}  =  {grid_out[z,y,x]}  {cum_sum}  +  {zero_count}  *  {grid[z,y,x]}')
        grid_out[z, y, x] += cum_sum + grid[z, y, x] * zero_count


@cython.initializedcheck(False) # turn off initialization check for memoryviews
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int stencil_sor_cy(double[:,:,::1] grid, double[:,:,:] s, double w, int[:] z_index, int[:] y_index, int[:] x_index) nogil except -1:
    """
    Stencil operator tuned for SOR. 
    :param grid: operated array
    :param s: power source array
    :param w: over-relaxation parameter
    :param z_index: first array index
    :param y_index: second array index
    :param x_index: third array index
    :return: 
    """
    cdef int i, z, y, x, xdim, ydim, zdim, zero_count=0
    cdef double cell, residual
    xdim = grid.shape[2]
    ydim = grid.shape[1]
    zdim = grid.shape[0]
    l = z_index.shape[0]

    # Assumptions taken for optimization:
    #   1. Most cells are inside the array, thus bounds check should be quick. Cells on the boundary are processed separately.
    #   2. There are at least 3 non-zero neighbors, but usually 4, therefore the first condition is !=0
    for i in range(l):
        z = z_index[i]
        y = y_index[i]
        x = x_index[i]
        zero_count = 0
        residual = 0
        if z == 0:
            continue
        zero_count = stencil_base(&residual, grid, x, xdim, y, ydim, z, zdim)
        cell = grid[z, y, x]
        # with gil: # explicitly show residual calculation for debugging
        #     print(f'{z, y, x}:  {residual+cell*(zero_count-6)+s[z,y,x]}  = '
        #           f' {residual}  +  {cell}  *  ({zero_count} - 6)  +  {s[z,y,x]}')
        residual += cell * (zero_count - 6) + s[z, y, x]
        # with gil: # explicitly show calculation
        #     print(f'{z, y, x}:  {grid[z,y,x]+w*residual/6}  =  {grid[z, y, x]}  +  {w}  *  {residual}  /6')
        grid[z, y, x] += w * residual/6


@cython.initializedcheck(False) # turn off initialization check for memoryviews
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int stencil_gs_cy(double[:,:,::1] grid, double[:,:,:] s, int[:] z_index, int[:] y_index, int[:] x_index) nogil except -1:
    """
    Stencil operator tuned for SOR. 
    :param grid: operated array
    :param s: power source array
    :param z_index: first array index
    :param y_index: second array index
    :param x_index: third array index
    :return: 
    """
    cdef int i, z, y, x, xdim, ydim, zdim, start, zero_count=0
    cdef double cell, residual
    xdim = grid.shape[2]
    ydim = grid.shape[1]
    zdim = grid.shape[0]
    l = z_index.shape[0]

    # Assumptions taken for optimization:
    #   1. Most cells are inside the array, thus bounds check should be quick. Cells on the boundary are processed separately.
    #   2. There are at least 3 non-zero neighbors, but usually 4, therefore the first condition is !=0
    for i in range(l):
        z = z_index[i]
        y = y_index[i]
        x = x_index[i]
        zero_count = 0
        residual = 0
        if z == 0:
            continue
        zero_count = stencil_base(&residual, grid, x, xdim, y, ydim, z, zdim)
        cell = grid[z, y, x]
        residual += cell * zero_count + s[z, y, x]
        # with gil: # explicitly show calculation
        #     print(f'{z, y, x}:  {grid[z,y,x]+w*residual/6}  =  {grid[z, y, x]}  +  {w}  *  {residual}  /6')
        grid[z, y, x] = residual/6


@cython.initializedcheck(False) # turn off initialization check for memoryviews
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int surface_temp_av_cy(double[:,:,:] grid_out, double[:,:,:] grid, int[:] z_index, int[:] y_index, int[:] x_index) nogil except -1:
    """
    Define temperature of the surface cells by averaging temperature of the neighboring solid cells 

    :param surface_temp: surface temperature array
    :param temp: solid temperature array
    :param z: first array index
    :param y: second array index
    :param x: third array index 
    :return: 
    """
    cdef int i, z, y, x, xdim, ydim, zdim, zero_count=0
    cdef double average = 0
    xdim = grid.shape[2]
    ydim = grid.shape[1]
    zdim = grid.shape[0]
    l = z_index.shape[0]

    # Assumptions taken for optimization:
    #   1. Most cells are inside the array, thus bounds check should be quick. Cells on the boundary are processed separately.
    #   2. There are at least 3 non-zero neighbors, but usually 4, therefore the first condition is !=0
    for i in range(l):
        z = z_index[i]
        y = y_index[i]
        x = x_index[i]
        zero_count = 0
        average = 0
        if z == 0:
            continue
        zero_count = stencil_base(&average, grid, x, xdim, y, ydim, z, zdim)
        grid_out[z, y, x] = average / (6 - zero_count)


@cython.initializedcheck(False) # turn off initialization check for memoryviews
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int stencil_base(double* sum, double[:,:,:] grid, int x, int xdim, int y, int ydim, int z, int zdim) nogil except -1:
    """
    Stencil operator. Sums all the neighbors of the current cell. 
    If a neighbor is 0 or out of the bounds, then increase skipped cell counter.
    
    :param cum_sum: returned sum
    :param grid: source array
    :param x: third dimension index
    :param xdim: third dimension length
    :param y: second dimension index
    :param ydim: second dimension length
    :param z: first dimension index
    :param zdim: first dimension length
    :return: number of skipped cells
    """
    cdef:
        int cond = 0
        int zero_count = 0
        double cum_sum = 0
    if z<zdim-1 and z>0:
        cond += 1
        if y<ydim-1 and y>0:
            cond += 1
            if x<xdim-1 and x>0:
                cond += 1
    if cond == 3:
        # Z - axis
        if grid[z + 1, y, x] != 0:
            cum_sum += grid[z + 1, y, x]
        else:
            zero_count += 1
        if grid[z - 1, y, x] != 0:
            cum_sum += grid[z - 1, y, x]
        else:
            zero_count += 1
        # Y - axis
        if grid[z, y + 1, x] != 0:
            cum_sum += grid[z, y + 1, x]
        else:
            zero_count += 1
        if grid[z, y - 1, x] != 0:
            cum_sum += grid[z, y - 1, x]
        else:
            zero_count += 1
        # X - axis
        if grid[z, y, x + 1] != 0:
            cum_sum += grid[z, y, x + 1]
        else:
            zero_count += 1
        if grid[z, y, x - 1] != 0:
            cum_sum += grid[z, y, x - 1]
        else:
            zero_count += 1
    else:
        # Z - axis
        if z > zdim - 2:
            zero_count += 1
        else:
            if grid[z + 1, y, x] != 0:
                cum_sum += grid[z + 1, y, x]
            else:
                zero_count += 1
        if z < 1:
            zero_count += 1
        else:
            if grid[z - 1, y, x] != 0:
                cum_sum += grid[z - 1, y, x]
            else:
                zero_count += 1
        # Y - axis
        if y > ydim - 2:
            zero_count += 1
        else:
            if grid[z, y + 1, x] != 0:
                cum_sum += grid[z, y + 1, x]
            else:
                zero_count += 1
        if y < 1:
            zero_count += 1
        else:
            if grid[z, y - 1, x] != 0:
                cum_sum += grid[z, y - 1, x]
            else:
                zero_count += 1
        # X - axis
        if x > xdim - 2:
            zero_count += 1
        else:
            if grid[z, y, x + 1] != 0:
                cum_sum += grid[z, y, x + 1]
            else:
                zero_count += 1
        if x < 1:
            zero_count += 1
        else:
            if grid[z, y, x - 1] != 0:
                cum_sum += grid[z, y, x - 1]
            else:
                zero_count += 1
    sum[0] = cum_sum
    return zero_count


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void rolling_2d_cy(double[:,:] a, const double[:,:] b) nogil:
    cdef int i, j
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
                a[i,j] = a[i,j] +  b[i,j]


cpdef void rolling_1d(double[:] a, const double[:] b):
    rolling_1d_cy(a,b)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void rolling_1d_cy(double[:] a, const double[:] b) nogil:
    cdef int i
    for i in prange(a.shape[0]):
        a[i] = a[i] + b[i]
