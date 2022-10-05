#cython: language_level=3

cimport cython
from cython.parallel cimport prange


# Cython performs 2-3 times faster than numpy when adding one
# array slice to another. This applies to medium size arrays from [10:10:10] to [500:500:500].
# With smaller arrays bottleneck is the overhead, with bigger arrays Numpy works faster


cpdef void stencil_sor(double[:,:,::1] grid, double[:,:,:] s, double w, int[:] z, int[:] y, int[:] x):
    """
    Stencil-based calculation of a residual vector for the SOR method on a 3d grid
    
    :param grid: array operated in-place
    :param s: power source
    :param z: first array index
    :param y: second array index
    :param x: third array index 
    :return: 
    """
    stencil_sor_cy(grid, s, w, z, y, x)

cpdef void stencil(double[:,:,::1] grid_out, double[:,:,::1] grid, int[:] z, int[:] y, int[:] x):
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
    stencil_cy(grid_out, grid, z, y, x)

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
cdef void stencil_cy(double[:,:,::1] grid_out, double[:,:,::1] grid, int[:] z_index, int[:] y_index, int[:] x_index) nogil:
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
    cdef int i, z, y, x, xdim, ydim, zdim, zero_count=0, cond=0
    xdim = grid_out.shape[2]
    ydim = grid_out.shape[1]
    zdim = grid_out.shape[0]
    l = z_index.shape[0]

    # Assumptions taken for optimization:
    #   1. Most cells are inside the array, thus bounds check should be quick. Cells on the boundary are processed separately.
    #   2. There are at least 3 non-zero neighbors, but usually 4, therefore the first condition is !=0
    for i in prange(l, schedule='static', chunksize=50, num_threads=4):
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
            if z>zdim-2:
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


@cython.initializedcheck(False) # turn off initialization check for memoryviews
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void stencil_sor_cy(double[:,:,::1] grid, double[:,:,:] s, double w, int[:] z_index, int[:] y_index, int[:] x_index) nogil:
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
    cdef int i, z, y, x, xdim, ydim, zdim, zero_count=0, cond=0
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
        if z == 0:
            continue
        if z<zdim-1 and z>0:
            cond += 1
            if y<ydim-1 and y>0:
                cond += 1
                if x<xdim-1 and x>0:
                    cond += 1
        cell = grid[z, y, x]
        if cond == 3:
            # Z - axis
            if grid[z+1, y, x] != 0:
                residual += grid[z+1, y, x]
            else:
                zero_count += 1
            if grid[z-1, y, x] != 0:
                residual += grid[z-1, y, x]
            else:
                zero_count += 1

            # Y - axis
            if grid[z, y+1, x] != 0:
                residual += grid[z, y+1, x]
            else:
                zero_count += 1
            if grid[z, y-1, x] != 0:
                residual += grid[z, y-1, x]
            else:
                zero_count += 1

            # X - axis
            if grid[z, y, x+1] != 0:
                residual += grid[z, y, x+1]
            else:
                zero_count += 1
            if grid[z, y, x-1] != 0:
                residual += grid[z, y, x-1]
            else:
                zero_count += 1
            residual += cell * (zero_count - 6) + s[z, y, x]
            zero_count = 0
            cond = 0
        else:
            # Z - axis
            if z>zdim-2:
                zero_count  += 1
            else:
                if grid[z + 1, y, x] != 0:
                    residual += grid[z + 1, y, x]
                else:
                    zero_count += 1
            if z<1:
                zero_count += 1
            else:
                if grid[z - 1, y, x] != 0:
                    residual += grid[z - 1, y, x]
                else:
                    zero_count += 1
            # Y - axis
            if y>ydim-2:
                zero_count += 1
            else:
                if grid[z, y + 1, x] != 0:
                    residual += grid[z, y + 1, x]
                else:
                    zero_count += 1
            if y<1:
                zero_count += 1
            else:
                if grid[z, y - 1, x] != 0:
                    residual += grid[z, y - 1, x]
                else:
                    zero_count += 1
            # X - axis
            if x>xdim-2:
                zero_count += 1
            else:
                if grid[z, y, x + 1] != 0:
                    residual += grid[z, y, x + 1]
                else:
                    zero_count += 1
            if x<1:
                zero_count += 1
            else:
                if grid[z, y, x - 1] != 0:
                    residual += grid[z, y, x - 1]
                else:
                    zero_count += 1
            residual += cell * (zero_count - 6) + s[z, y, x]
            zero_count = 0
        grid[z, y, x] += w * residual/6
        residual = 0
        cond = 0



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
