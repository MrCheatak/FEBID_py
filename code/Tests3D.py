import numpy as np
import matplotlib.pyplot as plt
import cProfile
import numexpr
from line_profiler import LineProfiler

D = 10000
dt = 0.00001

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

gg=np.mgrid[0:5, 0:5]
gh = (gg[0,:].flatten(), gg[1,:].flatten())
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

ghosts=set()
ghosts_index = ()
test_grid = np.full((2, 50, 50), 0.83, dtype=np.float32)
for i in range(test_grid.shape[1]):
    for j in range(test_grid.shape[2]):
        ghosts.add((1,i,j))

test = zip(*ghosts)

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


# @profile
def laplace_term_rolling(grid, D,  dt):
    """
    Calculates diffusion term for all surface cells using rolling

    :param grid: 3D precursor density array
    :param D: diffusion coefficient
    :param dt: time step
    :return: to grid array
    """
    # for i in range(3000):
    grid_out = np.copy(grid)
    grid_out *= -6
    temp = tuple(zip(*ghosts)) # casting a set of coordinates to a list of index sequences for every dimension
    ghosts_index = np.asarray([np.asarray(temp[0]), np.asarray(temp[1]), np.asarray(temp[2])]) # constructing a tuple of ndarray sequences
    # X axis:
    # No need to have a separate array of values, when whe can conveniently call them from the origin:
    grid[ghosts_index[0], ghosts_index[1], ghosts_index[2]] = grid[ghosts_index[0], ghosts_index[1], ghosts_index[2]-1] # assinging precursor density values to ghost cells along the rolling axis and direction
    grid_out[:,:, :-1]+=grid[:,:, 1:] #rolling forward
    grid_out[:,:,-1] += grid[:,:,-1] #taking care of edge values
    grid[ghosts_index] = 0 # flushing ghost cells
    # While Numpy allows negative indicies, indicies that are greater than the given dimention cause IndexiError and thus has to be taken care of
    temp = np.where(ghosts_index[2] > grid.shape[2] - 2, ghosts_index[2] - 1, ghosts_index[2]) # decreasing all the edge indices by one to exclude falling out of the array
    grid[ghosts_index] = grid[ghosts_index[0], ghosts_index[1], temp+1]
    grid_out[:,:,1:] += grid[:,:,:-1] #rolling backwards
    grid_out[:, :, 0] += grid[:, :, 0]
    grid[ghosts_index] = 0
    # Y axis:
    grid[ghosts_index] = grid[ghosts_index[0], ghosts_index[1]-1, ghosts_index[2]]
    grid_out[:, :-1, :] += grid[:, 1:, :]
    grid_out[:, -1, :] += grid[:, -1, :]
    grid[ghosts_index] = 0
    temp = np.where(ghosts_index[1] > grid.shape[1] - 2, ghosts_index[1] - 1, ghosts_index[1])
    grid[ghosts_index] = grid[ghosts_index[0], temp+1, ghosts_index[2]]
    grid_out[:, 1:, :] += grid[:, :-1, :]
    grid_out[:, 0, :] += grid[:, 0, :]
    grid[ghosts_index] = 0
    # Z-axis:
    grid[ghosts_index] = grid[ghosts_index[0]-1, ghosts_index[1], ghosts_index[2]]
    grid_out[:-1, :, :] += grid[1:, :, :]
    grid_out[-1, :, :] += grid[-1, :, :]
    grid[ghosts_index] = 0
    temp = np.where(ghosts_index[0] > grid.shape[0] - 2, ghosts_index[0] - 1, ghosts_index[0])
    grid[ghosts_index] = grid[temp+1, ghosts_index[1], ghosts_index[2]]
    grid_out[1:, :, :] += grid[:-1, :, :]
    grid_out[0, :, :] += grid[0, :, :]
    grid[ghosts_index] = 0
    grid_out[ghosts_index]=0 # result has to also be cleaned as it has redundant values
    # numexpr.evaluate("grid_out*dt*D", casting='same_kind')
    return numexpr.evaluate("grid_out*dt*D", casting='same_kind')

def test_laplace(substrate, D, dt):
    for i in range(3000):
        laplace_term_rolling(substrate, D, dt)


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

    lp = LineProfiler()
    lp_wrapper = lp(test_laplace(test_grid, D, dt))
    lp_wrapper()
    lp.print_stats()
    print("Tests")
    g = np.copy(grid[500, :, :])
    # plot_it(g)
    # with timebudget("Lap"):
    #     while t<0.01:
    #         laplace_term_(grid, surf)
    #         t+=dt
    # cProfile.runctx('laplace_term_(grid, dd)',globals(),locals())
    cProfile.runctx('test_laplace(test_grid, D, dt)', globals(), locals())
    # exec("Tests3D")
    g = np.copy(grid[500, :, :])
    plot_it(g)
    qq=0
