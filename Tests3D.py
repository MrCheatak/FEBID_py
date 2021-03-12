import numpy as np
from timebudget import timebudget
import itertools
import matplotlib.pyplot as plt
import cProfile
import numexpr
import numpy.ma as ma
import numba as nb
import scipy as sc
from scipy import ndimage
from numba import njit, jit
from collections import namedtuple
from recordclass import recordclass

D = 10
dt = 0.001

q=np.arange(25).reshape(5,5)
z=np.zeros((5,5))
w=[(1,3,2,2),(2,4,3,3)]
z[w]=q[w]
q[w]=99
# for ww in w:
#     ww[1]=0
# it = np.nditer(q, flags=['multi_index'])
# for x in it:
#     g=it.multi_index[0]

# w[0] = 99
# w =np.append(w, q[2:3,2:3])
# w[0]=1
grid = np.zeros((1000,1000,100))
grid[480:520,480:520,48:52] = 5
grid[495:505,490:500,49:50] = 20
sample = np.arange(27).reshape(3,3,3)


c=[]
c.append((1,1,1))
c.append((2,2,2))
c.append((3,3,3))
c.append((3,2,1))
c.append((2,1,4))
p = np.nonzero(c)
f = list(zip(*c))
k=(np.asarray(f[0]), np.asarray(f[1]),np.asarray(f[2]))
d=np.nonzero(q)
d[0][0] = 1
b= grid[k]
h = grid[f]

l=set()
l.add((1,1,1))
l.add((0,2,2))
l.add((0,2,0))
l.add((0,2,1))
l.add((1,2,1))

ll = list(zip(*l))
lar = (np.asarray(ll[0]), np.asarray(ll[1]), np.asarray(ll[2]))
sar = sample[lar[0]+1, lar[1], lar[2]]
a = np.take(sample,[1,1])
b=sample[a]

dd=np.ones((1,3), dtype=int)
surf =set()
# for k in range(grid.shape[0]):
#     for i in range(grid.shape[1]):
#         for j in range(grid.shape[2]):
#             if grid[k,i,j] > 4:
#                 surf.add(np.s_[k+1,i+1,j+1])
ss = np.argwhere(grid>0)
for s in ss:
    surf.add(np.s_[s[0], s[1], s[2]])
    dd= np.append(dd, [np.s_[s[0], s[1], s[2]]], axis=0)
# mesh = np.array([(0,1,1), (2,1,1), (1,1,0), (1,1,2), (1,0,1), (1,0,1)], dtype=tuple)

# @jit(nopython=False, parallel=True)
def laplace_term_(grid, surf):
    # sub_grid = np.pad(grid, 1, mode='symmetric')
    sub_grid = np.copy(grid)
    # imax=grid.shape[0]
    # jmax=grid.shape[1]
    # for i in range(1,imax):
    #     for j in range(1,jmax):
    #         sub_grid[i,j] += grid[i, (j-1)] - 2 * (grid[i, j]) + grid[i, (j+1)]
    #         sub_grid[i,j] += grid[(i-1), j] - 2 * (grid[i, j]) + grid[(i+1), j]
            # ddz = (z_ - 2 * (substrate_alias[cell.z, cell.y, cell.x, 0] + addon) + _z)
    convolute(grid, sub_grid, dd)
    # temp=sub_grid[1:-1,1:-1, 1:-1]
    numexpr.evaluate("grid*D*dt+sub_grid", out=grid)
    pp=0
    # grid = sub_grid[1:-1,1:-1]*D*dt


# @jit(nopython=True)
def convolute(grid_out, grid, coords):
    grid_out *= -6
    for pos in coords:
        # temp = grid[pos.z-1:pos.z+2,pos.y-1:pos.y+2, pos.x-1:pos.x+2] # maybe only the four neighbors can be selected?
        # temp = grid[pos[0],pos[1], pos[2]-1:pos[2]+2]
        temp = grid[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2, pos[2]-1:pos[2]+2]
        # temp = np.delete(temp, [[0,0,0], [1,1,1], [2,2,2], [2,2,0], [2,0,2], [0,2,2], [1,0,0], [0,1,0], [0,0,1]])
        # a=np.argwhere(temp==0)
        # for cell in a:
        #     temp[cell] = grid[pos]
        grid_out[pos[0],pos[1],pos[2]] += kernel_convolution(temp)
        # grid_out[i-1, j-1] += grid[i, (j - 1)] + grid[i, (j + 1)]
        # grid_out[i-1, j-1] += grid[(i - 1), j] + grid[(i + 1), j]
        # grid_out[i-1,j-1]=rk4(temp)
        # for cell in a:
        #     temp[cell[0], cell[1]] = 0


# @jit(nopython=True)
def rk4(grid):
    k1=kernel_convolution(grid)
    k2=kernel_convolution(grid, k1/2)
    k3=kernel_convolution(grid, k2/2)
    k4=kernel_convolution(grid, k3)
    return (k1+k4)/6 +(k2+k3)/3

# mesh = np.array([(0, 1, 1), (2,1,1), (1,1,0), (1,1,2), (1,0,1), (1,0,1)], dtype=tuple)
# @jit(nopython=True, cache=True)
def kernel_convolution(grid, add=0):
    mesh = [(0,1,1), (2,1,1), (1,1,0), (1,1,2), (1,0,1), (1,0,1)]
    sum=0

    for cell in mesh:
        y=(1,1,1)
        b=grid[y]
        if grid[cell]>0:
            sum += grid[cell]
        else:
            sum += grid[1,1,1]
    return sum - add*6

t=0

def plot_it(grid):
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(grid.copy(), cmap=plt.get_cmap('hot'), vmin=0, vmax=10 )
    ax.set_axis_off()
    ax.set_title('{:.1f} ms'.format(t * 1000))
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Tests")
    g = np.copy(grid[500, :, :])
    # plot_it(g)
    # with timebudget("Lap"):
    #     while t<0.01:
    #         laplace_term_(grid, surf)
    #         t+=dt
    cProfile.runctx('laplace_term_(grid, dd)',globals(),locals())
    # exec("Tests3D")
    g = np.copy(grid[500, :, :])
    plot_it(g)
    qq=0
