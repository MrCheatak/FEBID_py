#cython: language_level=3

cimport cython
from cython.parallel cimport prange


# Cython performs 2-3 times faster than numpy when adding one
# array slice to another. This applies to medium size arrays from [10:10:10] to [500:500:500].
# With smaller arrays bottleneck is the overhead, with bigger arrays Numpy works faster

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
