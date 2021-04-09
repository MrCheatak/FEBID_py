import numpy as np
import math
from timebudget import timebudget
import itertools
import matplotlib.pyplot as plt
import numexpr
import numpy.ma as ma
from scipy import ndimage
import line_profiler

D = 10
dt = 0.0001
diffusion_dt = math.pow(1, 2) / (2 * D * (1 + 1))

class Position:
    def __init__(self, z, y, x, dtype=int):
        if dtype == int:
            self.x = np.int(x)
            self.y = np.int(y)
            self.z = np.int(z)
        else:
            self.x = np.float(x)
            self.y = np.float(y)
            self.z = np.float(z)

system_size = 5
substratee = np.zeros((system_size, system_size, system_size, 2), dtype=np.float32)
grid = np.linspace(0, 0.8, num=125).reshape((5, 5, 5))

ind = np.indices((2, 50))

grid = np.zeros((50,50), dtype=np.float64)
# grid[470:530, 470:530] = 5
# grid[490:510, 490:510] = 20
grid[20:30, 20:30] = 5
# grid[23:27, 23:27] = 20
a=np.array([1,2,3,4,5])
b=np.array([1,2,np.NAN,4,5])
a = a+b
gridm=ma.masked_equal(grid, 0,copy=True)
# gridm = ma.copy(grid)
data = gridm[~gridm.mask]
# gridm[data] = ma.nomask
gg=np.nonzero(gridm)
#substrate_alias = np.roll(substrate_alias,+1,1)
sub=np.zeros((5,5,5))
test_trid = np.ones((10,10))

kernel = np.array([[0, 1,0],
                   [1,-4,1],
                   [0, 1,0]], dtype=np.float64)
kernel *= D*dt

x = np.array([[ 0,0,0,0,0,0 ],
              [ 0,1,0,0,0,0 ],
              [ 0,0,10,5,0,0 ],
              [ 0,0,10,10,0,0 ],
              [ 0,0,0,0,0,0 ],
              [ 0,0,0,0,0,0 ]], dtype=np.float64)

ss = np.s_[1,1]
tt = x[ss]
ss = np.s_[3,3]
tt = x[ss]
tt=ss[0]
# ss[0] += 1

hh=np.full((5,5), 1)
hh[[2,2,3,3], [2,3,2,3]] =2,3,4,5
hg= np.roll(hh, -1, axis=1)
pp = hh>1
hh[pp]=hh[np.roll(pp, 1, axis=1)]
oo=np.nonzero(pp)[0].max()

X = np.arange(-5, 5, 0.25*1)
Y = np.arange(-5, 5, 0.25*1)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
x, y, z = np.indices((8, 8, 8))
cube1 = (x < 3) & (y < 3) & (z < 3)

from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))




# prepare some coordinates
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between
# them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# combine the objects into a single boolean array
voxels = cube1 | cube2 | link

# set the colors of each object
colors = np.empty(voxels.shape, dtype=object)
colors[link] = 'red'
colors[cube1] = 'blue'
colors[cube2] = 'green'

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolor='k')

plt.show()

x, y, z = np.random.normal(0, 1, (3, 1000))
d, f,g = np.random.randint(-50, 50, (3,2500))

# Plot the surface
ax.plot_surface(x, y, z, color='b')

plt.show()

def profile_each_line(func, *args, **kwargs):
    profiler = line_profiler.LineProfiler()
    profiled_func = profiler(func)
    try:
        profiled_func(*args, **kwargs)
    finally:
        profiler.print_stats()


def threads_test(*args):
    for i in range(1, 8, 1):
        numexpr.set_num_threads(i)
        numexpr.use_vml = False
        with timebudget(f'Process, Cores: {i}'):
            for k in range(1000):
                numexpr.evaluate("")
    print('\n')
    return numexpr.evaluate("")

def conv_scipy(grid):
    grid_out = np.pad(grid, (1), 'constant')
    ndimage.convolve(grid_out, kernel, output=grid_out, mode='reflect')
    grid += grid_out[1:-1,1:-1]

# @jit(nopython=True, parallel=True)
def laplace_roll(grid):
    grid_out = np.copy(grid)
    grid_out *= -4
    grid_out += np.roll(grid, +1, 0)
    grid_out += np.roll(grid, -1, 0)
    grid_out += np.roll(grid, +1, 1)
    grid_out += np.roll(grid, -1, 1)
    grid +=grid_out*dt*D


# @njit()
def laplace_roll_ev(grid):
    grid_out = np.copy(grid)
    grid_out *= -4
    grid_out += np.roll(grid, +1, 0)
    grid_out += np.roll(grid, -1, 0)
    grid_out += np.roll(grid, +1, 1)
    grid_out += np.roll(grid, -1, 1)
    numexpr.evaluate("grid_out*dt*D+grid", out=grid)

a=0
def loops_test():
    n=1
    with timebudget("Time 1"):
        for i in itertools.count(2, 2):
            n+=2.5*i
            if i>10000:
                break
    with timebudget("Time 2"):
        for i in range(2, 10000, 2):
            n += 2.5 * i
    with timebudget("Time 3"):
        for i in np.arange(2, 10000, 2):
            n += 2.5 * i
    q=0

# @jit(nopython=True)
def rk4_roll(grid, n=numexpr.get_num_threads()):
    numexpr.set_num_threads(n)
    grid_op=np.copy(grid)
    k1=roll_add_2d(dt, grid, n=n)
    k2=roll_add_2d(dt, grid, k1/2, n=n)
    k3=roll_add_2d(dt, grid, k2/2, n=n)
    k4=roll_add_2d(dt, grid, k3, n=n)
    z=numexpr.get_num_threads()
    return numexpr.evaluate("(k1+k4)/6 +(k2+k3)/3+grid")


# @jit( parallel=True, forceobj=True)
def roll_add_2d(dt, grid_a, add=0, n=numexpr.get_num_threads()):
    # a=np.amin(np.argwhere(grid>0.000001))-1
    # b=np.amax(np.argwhere(grid>0.000001))+2
    numexpr.set_num_threads(n)
    grid_a = grid_a+add
    grid_out = np.copy(grid_a)
    grid_out *= -4
    # grid_out+=add
    grid_out[1:, :] += grid_a[:-1, :]
    # grid_out[0, :] += grid_a[-1, :]
    grid_out[:-1, :] += grid_a[1:, :]
    # grid_out[-1, :] += grid_a[0, :]
    grid_out[:, 1:] += grid_a[:, :-1]
    # grid_out[:, 0] += grid_a[:, -1]
    grid_out[:, :-1] += grid_a[:, 1:]
    # grid_out[:, -1] += grid_a[:, 0]

    return numexpr.evaluate("grid_out*dt*D", casting='same_kind')
    # rk4(grid_a)
    # grid += grid_out * dt * D

def update_shell(grid):
    data = np.argwhere(grid>0)
    for i,j in data:
        grid[i-1:i+1,j-1:j+1] = ma.nomask

def rollm_add_2d(gridm):
    grid_out = ma.copy(gridm)
    grid_out *= -4
    grid_out[1:, :] += gridm[:-1, :]
    # grid_out[0, :] += gridm[-1, :]
    grid_out[:-1, :] += gridm[1:, :]
    # grid_out[-1, :] += gridm[0, :]
    grid_out[:, 1:] += gridm[:, :-1]
    # grid_out[:, 0] += gridm[:, -1]
    grid_out[:, :-1] += gridm[:, 1:]
    # grid_out[:, -1] += gridm[:, 0]
    d=np.argwhere(grid_out>0)
    numexpr.evaluate("grid_out*dt*D+grid", out=grid)
    # gridm += grid_out * dt * D
    update_shell(gridm)

def roll_add_3d(out, shift, axis):
    rollee = np.copy.deepcopy(out)
    if shift == 1 and axis == 0:
        out[1:, :, :] += rollee[:-1, :, :]
        out[0, :, :] += rollee[-1, :, :]
    elif shift == -1 and axis == 0:
        out[:-1, :, :] += rollee[1:, :, :]
        out[-1, :, :] += rollee[0, :, :]
    elif shift == 1 and axis == 1:
        out[:, 1:, :] += rollee[:, :-1, :]
        out[:, 0, :] += rollee[:, -1, :]
    elif shift == -1 and axis == 1:
        out[:, :-1, :] += rollee[:, 1:, :]
        out[:, -1, :] += rollee[:, 0, :]
    elif shift == 1 and axis == 2:
        out[:, :, 1:] += rollee[:, :, :-1]
        out[:, :, 0] += rollee[:, :, -1]
    elif shift == -1 and axis == 2:
        out[:, :, :-1] += rollee[:, :, 1:]
        out[:, :, -1] += rollee[:, :, 0]

def laplace(cell, addon=0):
    """
    Calculates the Laplace operator for the given position

    :param cell: Coordinates of the current cell in the matrix, relative
    :param addon: Coefficient for Runge-Kutta method
    :return:
    """
    global substrate_alias
    np.copyto(substrate_alias, substrate_alias)
    substrate_alias *= -6
    substrate_alias +=np.roll(substrate_alias[:, :, :, 0], +1, 0)
    substrate_alias +=np.roll(substrate_alias[:, :, :, 0], -1, 0)
    substrate_alias +=np.roll(substrate_alias[:, :, :, 0], +1, 1)
    substrate_alias +=np.roll(substrate_alias[:, :, :, 0], -1, 1)
    substrate_alias +=np.roll(substrate_alias[:, :, :, 0], +1, 2)
    substrate_alias +=np.roll(substrate_alias[:, :, :, 0], -1, 2)

# @jit(nopython=False, parallel=True)
def laplace_term(grid):
    sub_grid = np.pad(grid, 1, mode='symmetric')
    # imax=grid.shape[0]
    # jmax=grid.shape[1]
    # for i in range(1,imax):
    #     for j in range(1,jmax):
    #         sub_grid[i,j] += grid[i, (j-1)] - 2 * (grid[i, j]) + grid[i, (j+1)]
    #         sub_grid[i,j] += grid[(i-1), j] - 2 * (grid[i, j]) + grid[(i+1), j]
            # ddz = (z_ - 2 * (substrate_alias[cell.z, cell.y, cell.x, 0] + addon) + _z)
    convolute(grid, sub_grid)
    temp=sub_grid[1:-1,1:-1]
    numexpr.evaluate("grid*D*dt+temp", out=grid)
    # grid = sub_grid[1:-1,1:-1]*D*dt


# @jit(nopython=True)
def convolute(grid_out, grid):
    imax = grid_out.shape[0]
    jmax = grid_out.shape[1]
    grid_out *= -4
    for i in range(1, imax):
        for j in range(1, jmax):
            if grid[i,j]==0:
                continue
            temp = grid[i-1:i+2, j-1:j+2] # maybe only the four neighbors can be selected?
            # temp = [grid[i, (j - 1)], grid[i, (j + 1)],grid[i, (j - 1)], grid[i, (j + 1)]]
            a=np.argwhere(temp==0)
            for cell in a:
                temp[cell[0], cell[1]] = grid[i,j]
            grid_out[i-1, j-1] += grid[i, (j - 1)] + grid[i, (j + 1)]
            grid_out[i-1, j-1] += grid[(i - 1), j] + grid[(i + 1), j]
            # grid_out[i-1,j-1]=rk4(temp)
            for cell in a:
                temp[cell[0], cell[1]] = 0


# @jit(nopython=True)
def rk4(grid):
    k1=kernel_convolution(grid)
    k2=kernel_convolution(grid, k1/2)
    k3=kernel_convolution(grid, k2/2)
    k4=kernel_convolution(grid, k3)
    return (k1+k4)/6 +(k2+k3)/3


# @jit(nopython=True)
def kernel_convolution(grid, add=0):
    return grid[0,1]+grid[1,0] + grid[2,1]+grid[1,2] - add*4




def plot_it(grid):
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(grid.copy(), cmap=plt.get_cmap('hot'), vmin=0, vmax=10 )
    ax.set_axis_off()
    ax.set_title('{:.1f} ms'.format(t * 1000))
    fig.tight_layout()
    plt.show()

y=np.copy(x)
# conv_scipy(x)
# roll_add_2d(y)

if __name__ == '__main__':

    # laplace_term(x)
    # roll_add_2d(test_trid)
    # t=0
    # g=np.copy(grid)
    # with timebudget("Time Term +JIT"):
    #     while t<3:
    #         laplace_term(g)
    #         t+=dt
    #     plot_it(g)
    # t=0
    # g=np.copy(grid)
    # with timebudget("Time Roll 1"):
    #     while t<1:
    #         laplace_roll(g)
    #         t+=dt
    #     plot_it(g)
    # t=0
    # g=np.copy(grid)
    # with timebudget("Time Roll 2"):
    #     while t<1:
    #         laplace_roll_ev(g)
    #         t+=dt
    #     plot_it(g)
    t=0
    g=np.copy(grid)
    z=numexpr.get_num_threads()
    # plot_it(g)
    for i in range (8, 0, -1):
        numexpr.set_num_threads(i)
        numexpr.use_vml = False
        t=0
        g = np.copy(grid)
        with timebudget("Time Roll 3"):
            while t<0.5:
                g=rk4_roll(g, i)
                # g+=roll_add_2d(dt, g)
                t+=dt
    plot_it(g)
    # t=0
    # g=np.copy(grid)
    # with timebudget("Time Convolve Scipy"):
    #     while t<0.5:
    #         conv_scipy(g)
    #         t+=dt
    #     plot_it(g)
    # g=np.copy(grid)
    # g1=np.copy(grid)
    # with timebudget("Time Scipy Laplace"):
    #     while t<1:
    #         # g1 *=-4
    #         g1 = ndimage.laplace(g, mode='nearest')
    #         numexpr.evaluate('g+g1*D*dt', out=g)
    #         # g1, g = g,g1
    #         t+=dt
    #     plot_it(g)
    q=0
