import numpy as np
import Laplace as lp
import timeit

def laplace_numpy(arr, brr):
    for i in range(1):
        arr[:,:,:-1] += brr[:,:,1:]

def laplace_cy(arr, brr):
    for i in range(1):
        lp.rolling_3d(arr[:,:,:-1],brr[:,:,1:])



if __name__ == '__main__':
    a = np.random.uniform(0, 1000, (600,700,70))
    arr = np.copy(a)
    brr = np.zeros_like(arr)
    brr[...] = arr[...]

    start = timeit.default_timer()
    laplace_numpy(a, brr)
    print(f'Numpy took : {timeit.default_timer() - start}')

    # arr = np.copy(a)
    start = timeit.default_timer()
    laplace_cy(arr, brr)
    print(f'Cython took : {timeit.default_timer() - start}')

    lp.rolling_2d(a[:,:,10], brr[:,:,11])