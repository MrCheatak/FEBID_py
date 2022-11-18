"""
Tridiagonal parallel matrix solver
"""

import cython
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange


"""
A representation of a solution matrix with the function arguments
 in it, that define boundary conditions:
 
|b0  c0  0   0   0 |
|c   b   c   0   0 |
|0   c   b   c   0 |
|0   0   c   b   c |
|0   0   0   c0  b0|

"""


# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef adi_3d_indexing(double[:,:,:] d, double[:,:,:] x, long[:,:] s1, long[:,:] s2, long[:,:] s3, double a, long boundaries):
    """
        Solve a PDE in 3D domain using ADI method.
        Use provided slices to solve for certain regions.  

        :param d: right hand side vector
        :param x: vector to be solved
        :param s1: index triples for x-axis, that define a 1d slice
        :param s2: index triples for y-axis, that define a 1d slice
        :param s3: index triples for z-axis, that define a 1d slice
        :param a: equation coefficient, proportional to diffusivity
        :param boundaries: type of boundary conditions: 0 for 0 at boundaries, 1 for fixed boundaries, 2 for no flow through boundaries
        """
    cdef:
        int i, j
        int u, v, w, q
        double b0, c0, bn, cn  # boundary conditions
    if boundaries == 0:
        b0 = 1 + 2 * a
        c0 = -a
    if boundaries == 1:
        b0 = 1
        c0 = 0
    if boundaries == 2:
        b0 = 1 + 1 * a
        c0 = -a
    b = 1 + 2 * a
    c = -a
    # for j in prange(0, s1.shape[0], 2, num_threads=4, nogil=True):
    for j in range(0, s1.shape[0], 2):
        u = s1[j, 0]
        v = s1[j, 1]
        w = s1[j, 2]
        q = s1[j+1,2]
        tridiag_1d_c(d[u, v, w:q], d[u, v, w:q], b, c, b0, c0, b0, c0)
    # for j in prange(0, s2.shape[0], 2, num_threads=4, nogil=True):
    for j in range(0, s2.shape[0], 2):
        u = s2[j, 0]
        v = s2[j, 1]
        w = s2[j, 2]
        q = s2[j+1,1]
        tridiag_1d_c(d[u, v:q, w], d[u, v:q, w], b, c, b0, c0, b0, c0)
    # for j in prange(0, s3.shape[0], 2, num_threads=4, nogil=True):
    for j in range(0, s3.shape[0], 2):
        u = s3[j, 0]
        v = s3[j, 1]
        w = s3[j, 2]
        q = s3[j+1,0]
        if u == 0:
            bn = 1
            cn = 0
        else:
            bn = b0
            cn = c0
        tridiag_1d_c(d[u:q, v, w], d[u:q, v, w], b, c, bn, cn, b0, c0)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef adi_3d(double[:, :, :] d, double[:, :, :] x, double a, long boundaries):
    """
    Solve a PDE in 3D uniform domain using ADI method with backward Euler scheme. 
    
    :param d: right hand side vector
    :param x: vector to be solved
    :param a: equation coefficient
    :param boundaries: type of boundary conditions: 0 for 0 at boundaries, 1 for fixed boundaries, 2 for no flow through boundaries
    """
    cdef:
        int i, j
        double b0, c0 # boundary conditions
    if boundaries == 0:
        b0 = 1 + 2 * a
        c0 = -a
    if boundaries == 1:
        b0 = 1
        c0 = 0
    if boundaries == 2:
        b0 = 1 + 1 * a
        c0 = -a
    b = 1 + 2 * a
    c = -a
    for i in range(d.shape[0]):
        for j in prange(d.shape[1], num_threads=4, nogil=True):
        # for j in range(100):
            tridiag_1d_c(d[i, j, :], d[i, j, :], b, c, b0, c0, b0, c0)
    for i in range(d.shape[0]):
        for j in prange(d.shape[2], num_threads=4, nogil=True):
            tridiag_1d_c(d[i, :, j], d[i, :, j], b, c, b0, c0, b0, c0)
    for i in range(d.shape[1]):
        for j in prange(d.shape[2], num_threads=4, nogil=True):
            tridiag_1d_c(d[:, i, j], d[:, i, j], b, c, b0, c0, b0, c0)

cpdef tridiag_1d(double[:] d, double[:] x, double b, double c, double b0, double c0):
    """
    Tridiagonal matrix solver
    
    The solver uses Thomas algorithm.
    
    :param d: right hand side vector
    :param x: output vector
    :param b: main diagonal value
    :param c: upper and lower diagonal value
    :param b0: boundary value for main diagonal
    :param c0: boundary value for upper and lower diagonals 
    :return: 
    """
    tridiag_1d_c(d, x, b, c, b0, c0, b0, c0)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void tridiag_1d_c(double[:] d, double[:] x, double b, double c, double b0, double c0, double bn, double cn) nogil:
    """
    Tridiagonal matrix solver
    
    The solver uses Thomas algorithm.
    
    :param d: right hand side vector
    :param x: output vector
    :param b: main diagonal value
    :param c: upper and lower diagonal value
    :param b0: begin boundary value for main diagonal
    :param c0: begin boundary value for upper and lower diagonals
    :param bn: end boundary value for main diagonal
    :param cn: end boundary value for upper and lower diagonals
    """
    cdef:
        int i, N
        double *gamma = <double*>malloc(d.shape[0] * sizeof(double))
        double *rho = <double*>malloc(d.shape[0] * sizeof(double))
    N = d.shape[0] - 1
    gamma[0] = c0/b0
    rho[0] = d[0]/b0
    for i in range(1, N+1):
        gamma[i] = c / (b - c * gamma[i - 1])
        rho[i] = (d[i] - c * rho[i - 1]) / (b - c * gamma[i - 1])
    rho[N] = (d[N] - cn * rho[N-1]) / (bn - cn * gamma[N-1])
    x[N] = rho[N]
    for i in range(N-1, -1, -1):
        x[i] = rho[i] - x[i + 1] * gamma[i]
        if x[i] > 1e6 or x[i] < 0:
            with gil:
                for N in range(x.shape[0]):
                    print(x[N+1])
                raise RuntimeError("Exceeding 1e6 iterations in tridiagonal matrix solution")
    free(gamma)
    free(rho)

