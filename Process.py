import numpy as np
#import scipy
import scipy.constants as scpc
import math
#import array
import matplotlib.pyplot as plt
import numexpr
from timebudget import timebudget
import cProfile
from numba import jit,typed, typeof
from numba.experimental import jitclass
from recordclass import recordclass
from collections import namedtuple
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

zz = 0
yy=0
xx=0

@jitclass([('z', typeof(zz)), ('y', typeof(yy)), ('x', typeof(xx)) ])
class Position:
    def __init__(self, z, y, x):
        self.x = x
        self.y = y
        self.z = z

zz=0.
yy=0.
xx=0.

@jitclass([('z', typeof(zz)), ('y', typeof(yy)), ('x', typeof(xx)) ])
class Positionb:
    def __init__(self, z, y, x):
        self.x = x
        self.y = y
        self.z = z

# Position = namedtuple('Position', 'z,y,x')


td = 1E-6  # dwell time of a beam, s
Ie = 1E-10  # beam current, A
beam_d = 10  # electron beam diameter, nm
effective_radius = beam_d * 3.3
f = Ie / scpc.elementary_charge / (math.pi * beam_d * beam_d / 4)  # electron flux at the surface, 1/(nm^2*s)
F = 3000  # precursor flux at the surface, 1/(nm^2*s)   here assumed a constant, but may be dependent on time and position
tau = 500E-6  # average residence time, s may be dependent on temperature

# Precursor properties
sigma = 2.2E-2  # dissociation cross section, nm^2 is averaged from cross sections of all electron types (PE,BSE, SE1, SE2)
n0 = 1.9  # inversed molecule size, Me3PtCpMe, 1/nm^2
M_Me3PtCpMe = 305  # molar mass of the precursor Me3Pt(IV)CpMe, g/mole
p_Me3PtCpMe = 1.5E-20  # density of the precursor Me3Pt(IV)CpMe, g/nm^3
V = 4 / 3 * math.pi * math.pow(0.139, 3)  # atomic volume of the deposited atom (Pt), nm^3
D = np.float32(1E5)  # diffusion coefficient, nm^2/s

###
dt = np.float32(1E-6)  # time step, s
# t = 2E-6  # absolute time, s

kd = F / n0 + 1 / tau + sigma * f  # depletion rate
kr = F / n0 + 1 / tau  # replenishment rate
nr = F / kr  # absolute density after long time
nd = F / kd  # depleted absolute density
t_out = 1 / (1 / tau + F / n0)  # effective residence time
p_out = 2 * math.sqrt(D * t_out) / beam_d
cell_dimension = 5  # side length of a square cell, nm
diffusion_dt = math.pow(cell_dimension * cell_dimension, 2) / (2 * D * (cell_dimension * cell_dimension + cell_dimension * cell_dimension))  # maximum stability lime of the diffusion solution

# Main cell matrix.
# Substrate[z,x,y,0] holds precursor density,
# substrate[z,y,x,1] holds deposit density.
# </summary>
effective_radius_relative = math.floor(effective_radius/cell_dimension/2)
system_size = 50
substrate = np.zeros((system_size, system_size, system_size), dtype=np.float32)
deposit = np.full((system_size, system_size, system_size),0 , dtype=np.float32)
substrate[0, :, :] = nr
deposit[0, :, :] = 0.97
# deposit[0, 20:40, 20:40] = 0.95
zmax, ymax, xmax = substrate.shape

# Holds z coordinates of the surface cells that are allowed to produce deposit.
# Indices of every cell correspond to their x&y position
# The idea is to avoid iterating through the whole 3D matrix and address only surface cells
# It is assumed, that surface cell for this array is a cell with a fully deposited cell(or substrate) under it and thus can produce deposit.
# Thus the number of surface cells is fixed.
# TODO: is it possible to create a non-contiguous and evolvable view to the 3D array?
surface = np.zeros((system_size, system_size), dtype=np.int16) # TODO: since lists allow sending a list of indices without a loop, surface matrix should be untilizing such data type for speed, but preserve addressability or quick search
surf =[]
# ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb = [], [], [], [], [], []
ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb = set(), set(), set(), set(), set(), set()
ghosts = set()
val_zf, val_zb, val_yf, val_yb, val_xf, val_xb = [], [], [], [], [], []
# Semi-surface cells are cells that have precursor density but do not have deposit right under them
# Thus they cannot produce deposit and their precursor density is calculated without disosisiation term.
# They are introduced to allow diffusion on the walls of the deposit.
semi_surface = set() # TODO: Numba doesn't really like sets (listed for deprication) –> Numpy arrays?
# TODO: maybe combine surface and semi-surface by utilizing subsets?
# semi_surface.clear()


@jit(nopython=True)
def pe_flux(r):
    """Calculates PE flux at the given radius according to Gaussian distribution.

    :param r: Radius from the center of the beam.
    """
    #with timebudget("Flux time"):
    return f*math.exp(-r*r/(2*beam_d*beam_d))

# @jit(nopython=False, forceobj=True)
def make_tuple(arr):
    x,y,z =[],[],[]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            x.append(j)
            y.append(i)
            z.append(arr[i,j])
    return ([z, y ,x])


# @jit(nopython=True, parallel=True)
def deposition(flux_matrix, surf, substrate, deposit, dt):
    '''
    Calculates deposition on the surface for a given time step dt (outer loop).

    Manages surface cells if a cell gets fully deposited

    :param flux_matrix: matrix of electron flux distribution
    :param substrate: main 3D array
    :param dt: time step
    :return:
    '''
    #with timebudget("Deposition time"):
    deposit[surf]+=substrate[surf]*sigma*flux_matrix.view().reshape(flux_matrix.shape[0]**2)*V*dt


# @jit(nopython=True)

def update_surface(deposit, substrate, surface, surf, semi_surface, init_y=0, init_x=0):
    with np.nditer(surface, flags=['multi_index'], op_flags=['readwrite'], casting='safe') as it: # TODO: no need to iterate through the whole matrix, it could be quicker to just find overfilled cells and address only them
        for z in it:
            if deposit[z, it.multi_index[0],it.multi_index[1]] >= 1:  # if the cell is fully deposited
                semi_surface.add((0, 0, 0))
                ghosts.add((np.int16(z), it.multi_index[0]+init_y,it.multi_index[1]+init_x))
                z +=1  # rising the surface one cell up (new cell)
                refresh(deposit, substrate, semi_surface, np.int16(z),it.multi_index[0],it.multi_index[1], init_y, init_x)
    surf = make_tuple(surface)


def refresh(deposit, substrate, semi_surface, z,y,x, init_y=0, init_x=0):  # TODO: implement and include evolution of a ghost "shell" here, that should proceed along with the evolution of surface
    semi_surface.discard((0,0,0))
    semi_surface.discard((z, y+init_y, x+init_x))  # removing the new cell from the semi_surface list, if it is present there
    deposit[z, y, x] = deposit[z - 1, y, x] - 1  # if the cell was fulled above unity, transferring that surplus to the cell above
    deposit[z - 1, y, x] = 1  # a fully deposited cell is always a unity
    xx = x + 2
    yy = y + 2
    substrate[z, yy, xx] += substrate[z - 1, yy, xx]
    substrate[z - 1, yy, xx] = 0  # precursor density is zero in the fully deposited cells
    # Adding neighbors(in x-y plane) of the new cell to semi_surface list
    # Checking if the cell falls out of the array and then if it is already has precursor in it
    # Simple check on precursor density also covers if the cell is already in the semi_surface list
    # for ghost in [ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb]:
    #     ghost.discard((z, y+init_y, x+init_x))
    ghosts.discard((z, y + init_y, x + init_x))
    if substrate[z+1, yy, xx] == 0:
        ghosts.add((z+1, y+init_y, x+init_x))
    if substrate[z, yy - 1, xx] == 0:
        semi_surface.add((z, y - 1+init_y, x+init_x))  # adding cell to the list
        substrate[z, yy - 1, xx] += 1E-7
        refresh_ghosts(substrate, x + init_x, xx, y-1 + init_y, yy-1, z)
    if substrate[z, yy + 1, xx] == 0:
        semi_surface.add((z, y + 1+init_y, x+init_x))
        substrate[z, yy + 1, xx] += 1E-7
        refresh_ghosts(substrate, x + init_x, xx, y+1 + init_y, yy+1, z)
    if substrate[z, yy, xx - 1] == 0:
        semi_surface.add((z, y+init_y, x - 1+init_x))
        substrate[z, yy, xx - 1] += 1E-7
        refresh_ghosts(substrate, x -1 + init_x, xx-1, y + init_y, yy, z)
    if substrate[z, yy, xx + 1] == 0:
        semi_surface.add((z, y+init_y, x + 1+init_x))
        substrate[z, yy, xx + 1] += 1E-7
        refresh_ghosts(substrate, x + 1 + init_x, xx+1, y + init_y, yy, z)
    # <editor-fold desc="Lists pipeline">
    # for ghost, val in zip([ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb], [val_zf, val_zb, val_yf, val_yb, val_xf, val_xb]):
    #     try:
    #         remove_index = ghost.index((z, y+init_y, x+init_x))
    #         ghost.pop(remove_index)
    #         val.pop(remove_index)
    #         val.append(substrate[z, yy, xx])
    #     except ValueError: pass
    # </editor-fold>



def refresh_ghosts(substrate, x, xx, y, yy, z):
    global ghosts
    ghosts.discard((z, y, x))
    if substrate[z - 1, yy, xx] == 0:
        ghosts.add((z - 1, y, x))
    if substrate[z + 1, yy, xx] == 0:
        ghosts.add((z + 1, y, x))
    if substrate[z, yy - 1, xx] == 0:
        ghosts.add((z, y - 1, x))
    if substrate[z, yy + 1, xx] == 0:
        ghosts.add((z, y + 1, x))
    if substrate[z, yy, xx - 1] == 0:
        ghosts.add((z, y, x - 1))
    if substrate[z, yy, xx + 1] == 0:
        ghosts.add((z, y, x + 1))
def refresh_ghosts_old(substrate, x, xx, y, yy, z):
    for ghost in [ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb]:
        ghost.discard((z, y, x))
    if substrate[z - 1, yy , xx] == 0:
        ghost_zb.add((z - 1, y, x))
    if substrate[z + 1, yy, xx] == 0:
        ghost_zf.add((z + 1, y, xx))
    if substrate[z, yy - 1, xx] == 0:
        ghost_yb.add((z, y - 1, x))
    if substrate[z, yy + 1, xx] == 0:
        ghost_yf.add((z, y + 1, x))
    if substrate[z, yy, xx - 1] == 0:
        ghost_xb.add((z, y, x - 1))
    if substrate[z, yy, xx + 1] == 0:
        ghost_xf.add((z, y, x + 1))

        # <editor-fold desc="Lists pipeline">
        # for ghost, val in zip([ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb], [val_zf, val_zb, val_yf, val_yb, val_xf, val_xb]):
        #     try:
        #         ghost = list(set(ghost))
        #         remove_index = ghost.index((z, y - 1 + init_y, x + init_x))
        #         ghost.pop(remove_index)
        #         val.pop(remove_index)
        #         val.append(substrate[z, yy, xx])
        #     except ValueError:
        #         pass
        # if substrate[z-1, yy - 1, xx] == 0:
        #     ghost_zb.append((z-1, y - 1 + init_y, x + init_x))
        # if substrate[z+1, yy - 1, xx] == 0:
        #     ghost_zf.append((z+1, y - 1 + init_y, x + init_x))
        # if substrate[z, yy - 1-1, xx] == 0:
        #     ghost_yb.append((z, y - 1 -1 + init_y, x + init_x))
        # if substrate[z, yy - 1+1, xx] == 0:
        #     ghost_yf.append((z, y - 1 +1 + init_y, x + init_x))
        # if substrate[z, yy - 1, xx-1] == 0:
        #     ghost_xb.append((z, y - 1 + init_y, x  - 1 + init_x))
        # if substrate[z, yy - 1, xx+1] == 0:
        #     ghost_xf.append((z, y - 1 + init_y, x  + 1 + init_x))


# @jit(nopython=True, parallel=True)
def precursor_density(flux_matrix, substrate, dt):
    '''
    Recalculates precursor density for all surface cells

    :param flux_matrix: matrix of electron flux distribution
    :param substrate: main 3D array
    :param dt: time step
    :return:
    '''
    sub = np.zeros([ymax, xmax])  # array for surface sells
    semi_sub = np.zeros((len(semi_surface)))
    # An increment is calculated for every cell first and
    # added only in the next loop not to influence diffusion
    with np.nditer(surface, flags=['multi_index'], op_flags=['readonly']) as it:
        for z in it:
            sub[it.multi_index] = substrate[z, it.multi_index[0],it.multi_index[1]]
    # diffusion_matrix = laplace_term(substrate, surface, semi_surface, D, dt)
    diffusion_matrix = laplace_term_rolling(substrate, surface, semi_surface, D, dt)
    rk4(dt, sub, flux_matrix)
    with np.nditer(surface, flags=['multi_index'], op_flags=['readonly']) as it:
        for z in it:
             substrate[z, it.multi_index[0],it.multi_index[1]] += sub[it.multi_index]
    if any(semi_surface):
        i=0
        for cell in semi_surface:  # calculating increment for semi_surface cells (without disosisiation term)
            semi_sub[i] = substrate[cell]
            i+=1
        rk4(dt,semi_sub)
        i=0
        for cell in semi_surface:  # adding increment to semi_surface cells
            substrate[cell] += semi_sub[i]
            i+=1
    substrate+=diffusion_matrix


# @jit(nopython=False, parallel=True)
def rk4(dt, sub, flux_matrix=0):
    '''
    Calculates increment of precursor density by Runge-Kutta method

    :param dt: time step
    :param sub: array of surface cells (2D for surface cells, 1D dor semi-surface cells)
    :param flux_matrix: matrix of electron flux distribution
    :return:
    '''
    k1 = precursor_density_increment(dt, sub, flux_matrix)
    k2 = precursor_density_increment(dt/2, sub, flux_matrix, k1 / 2)
    k3 = precursor_density_increment(dt/2, sub, flux_matrix, k2 / 2)
    k4 = precursor_density_increment(dt, sub, flux_matrix, k3)
    numexpr.evaluate("(k1+k4)/6 +(k2+k3)/3", out=sub)


# @jit(nopython=False, parallel=True)
def precursor_density_increment(dt, sub, flux_matrix, addon=0):
    '''
    Calculates increment of the precursor density without a diffusion term

    :param dt: time step
    :param sub: array of surface cells (2D for surface cells, 1D dor semi-surface cells)
    :param flux_matrix: matrix of electron flux distribution
    :param addon: Runge Kutta term
    :return:
    '''
    #with timebudget("Precursor density time"):
    # a=(F * (1 - (sub[18,18] + addon) / n0) - (sub[18,18] + addon) / tau - (sub[18,18] + addon) * sigma * flux_matrix[18,18])*dt
    return numexpr.evaluate("(F * (1 - (sub + addon) / n0) - (sub + addon) / tau - (sub + addon) * sigma * flux_matrix)*dt")


# @jit(nopython=False, parallel=True)
def laplace_term(grid, surfa, semi_surf, D, dt):
    '''
    Calculates diffusion term for all surface cells using convolution

    :param dt:
    :param grid: 3D array of precursor densities
    :param surf: surface cells coordinates
    :param semi_surf: semi-surface cells coordinates
    :return:
    '''
    p_grid = np.pad(grid, 1, mode='constant', constant_values=0)
    sub_grid = np.copy(grid)
    convolute(sub_grid, p_grid, surfa, semi_surf)
    return numexpr.evaluate("sub_grid*D*dt")


# @jit(nopython=False, parallel=True)
def convolute(grid_out, grid, coords, coords1):
    '''
    Selectively applies convolution operator to the cells in grid with provided coordinates

    :param grid_out: 3D array with convolution results
    :param grid: original 3D array
    :param coords: coordinates of surface cells
    :param coords1: coordinates of semi-surface cells
    :return:
    '''
    grid_out *= -6
    with np.nditer(coords, flags=['multi_index']) as it:
        for z in it:
            kernel=grid[z:z+3, it.multi_index[0]:it.multi_index[0]+3, it.multi_index[1]:it.multi_index[1]+3]
            # kernel = grid[z - 1:z + 2, it.multi_index[0] - 1:it.multi_index[0] + 2, it.multi_index[1] - 1:it.multi_index[1]+2]
            grid_out[z,it.multi_index[0],it.multi_index[1]] += kernel_convolution(kernel)
    if any(coords1):
        for pos in coords1:
            kernel = grid[pos[0]:pos[0] + 3, pos[1]:pos[1] + 3, pos[2]:pos[2] + 3]
            grid_out[pos] += kernel_convolution(kernel)


mesh = [(0,1,1), (2,1,1), (1,1,0), (1,1,2), (1,0,1), (1,2,1)]
@jit(nopython=False)
def kernel_convolution(grid):
    '''
    Convolves the central cell in the given array.

    If one of the neighbors is Zero, it is replaced by the central value.

    :param grid: array to convolute
    :return:
    '''
    summ=0
    # for i in range(6):
    # # for cell in mesh:
    #     if grid[mesh[i]]>0:
    #         summ += grid[mesh[i]]
    #     else:
    #         summ += grid[1,1,1]
    if 1 > grid[0,1,1] > 0:
        summ += grid[0,1,1]
    else:
        summ += grid[1, 1, 1]
    if 1 > grid[2,1,1] > 0:
        summ += grid[2,1,1]
    else:
        summ += grid[1, 1, 1]
    if 1 > grid[1,1,0] > 0:
        summ += grid[1,1,0]
    else:
        summ += grid[1, 1, 1]
    if 1 > grid[1,1,2] > 0:
        summ += grid[1,1,2]
    else:
        summ += grid[1, 1, 1]
    if 1 > grid[1,0,1] > 0:
        summ += grid[1,0,1]
    else:
        summ += grid[1, 1, 1]
    if 1 > grid[1,2,1] > 0:
        summ += grid[1,2,1]
    else:
        summ += grid[1, 1, 1]

    return summ


def define_ghosts(substrate, surface, semi_surface = [] ): # TODO: find a mutable data structure(tuples are immutable), that would allow evolution of ghost cells "shell", rather than finding them every time. Such type must be quickly searchable or indexed and preferably not allow duplicates. Record class? Named tuples?
    '''
    Defines ghost cells for every axis and direction.

    :param surface: Surface cells matrix
    :param semi_surface: Sem-surface cells list(set)
    :param substrate: Main 3D array
    :return: Six lists with coordinates of ghost cells
    '''
    xxf, xyf, xzf, yxf, yyf, yzf, zxf,zyf,zzf =[],[],[],[],[],[],[],[],[]
    xxb, xyb, xzb, yxb, yyb, yzb, zxb, zyb, zzb = [], [], [], [], [], [], [], [], []
    gxf, gxb, gyf, gyb, gzf, gzb = [],[],[],[],[],[]
    for y in range(surface.shape[0]):
        for x in range(surface.shape[1]):
            try:
                if substrate[surface[y,x]-1,y,x] == 0:
                    zxb.append(x)
                    zyb.append(y)
                    zzb.append(surface[y,x]-1)
                    gzb.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass
            try:
                if substrate[surface[y,x]+1,y,x] == 0:
                    zxf.append(x)
                    zyf.append(y)
                    zzf.append(surface[y,x]+1)
                    gzf.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass
            try:
                if substrate[surface[y,x],y-1,x] == 0:
                    yxb.append(x)
                    yyb.append(y-1)
                    yzb.append(surface[y,x])
                    gyb.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass
            try:
                if substrate[surface[y,x],y+1,x] == 0:
                    yxf.append(x)
                    yyf.append(y+1)
                    yzf.append(surface[y,x])
                    gyf.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass
            try:
                if substrate[surface[y,x],y,x-1] == 0:
                    xxb.append(x-1)
                    xyb.append(y)
                    xzb.append(surface[y,x])
                    gxb.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass
            try:
                if substrate[surface[y,x],y,x+1] == 0:
                    xxf.append(x+1)
                    xyf.append(y)
                    xzf.append(surface[y,x])
                    gxf.append(substrate[surface[y,x], y,x])
            except IndexError:
                pass

    for cell in semi_surface:
        try:
            if substrate[cell[0]-1, cell[1], cell[2]] == 0:
                zzb.append(cell[0]-1)
                zyb.append(cell[1])
                zxb.append(cell[2])
                gzb.append(substrate[cell])
        except IndexError:
            pass
        try:
            if substrate[cell[0]+1, cell[1], cell[2]] == 0:
                zzf.append(cell[0]+1)
                zyf.append(cell[1])
                zxf.append(cell[2])
                gzf.append(substrate[cell])
        except IndexError:
            pass
        try:
            if substrate[cell[0], cell[1]-1, cell[2]] == 0:
                yzb.append(cell[0])
                yyb.append(cell[1]-1)
                yxb.append(cell[2])
                gyb.append(substrate[cell])
        except IndexError:
            pass
        try:
            if substrate[cell[0], cell[1]+1, cell[2]] == 0:
                yzf.append(cell[0])
                yyf.append(cell[1]+1)
                yxf.append(cell[2])
                gyf.append(substrate[cell])
        except IndexError:
            pass
        try:
            if substrate[cell[0], cell[1], cell[2]-1] == 0:
                xzb.append(cell[0])
                xyb.append(cell[1])
                xxb.append(cell[2]+1)
                gxb.append(substrate[cell])
        except IndexError:
            pass
        try:
            if substrate[cell[0], cell[1], cell[2]+1] == 0:
                xzf.append(cell[0])
                xyf.append(cell[1])
                xxf.append(cell[2]+1)
                gxf.append(substrate[cell])
        except IndexError:
            pass

    return ([zzf,zyf,zxf], [zzb,zyb,zxb], [yzf,yyf,yxf], [yzb, yyb, yxb], [xzf, xyf, xxf], [xzb, xyb, xxb], gzf, gzb, gyf, gyb, gxf, gxb)


def laplace_term_rolling(grid, surf, semi_surf, D,  dt):
    '''
    Calculates diffusion term for all surface cells using rolling

    :param grid:
    :param surf:
    :param semi_surf:
    :param D:
    :param dt:
    :return:
    '''
    # unli
    grid_out = np.copy(grid)
    grid_out *= -6
    # ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb, val_zf, val_zb, val_yf, val_yb, val_xf, val_xb = define_ghosts(grid, surf, semi_surf)
    temp = list(zip(*ghosts))
    ghosts_index = (np.asarray(temp[0]), np.asarray(temp[1]), np.asarray(temp[2]))
    # grid[ghost_xf] = val_xf
    grid[ghosts_index] = grid[ghosts_index[0], ghosts_index[1], ghosts_index[2]-1]
    grid_out[:,:, :-1]+=grid[:,:, 1:]
    grid_out[:,:,-1] += grid[:,:,-1]
    # grid[ghost_xf]=0
    grid[ghosts_index] = 0
    # grid[ghost_xb] = val_xb
    temp = np.where(ghosts_index[2] > system_size - 2, ghosts_index[2] - 1, ghosts_index[2])
    grid[ghosts_index] = grid[ghosts_index[0], ghosts_index[1], temp+1]
    grid_out[:,:,1:] += grid[:,:,:-1]
    grid_out[:, :, 0] += grid[:, :, 0]
    # grid[ghost_xb] = 0
    grid[ghosts_index] = 0

    # grid[ghost_yf] = val_yf
    grid[ghosts_index] = grid[ghosts_index[0], ghosts_index[1]-1, ghosts_index[2]]
    grid_out[:, :-1, :] += grid[:, 1:, :]
    grid_out[:, -1, :] += grid[:, -1, :]
    # grid[ghost_yf] = 0
    grid[ghosts_index] = 0
    # grid[ghost_yb] = val_yb
    temp = np.where(ghosts_index[1] > system_size - 2, ghosts_index[1] - 1, ghosts_index[1])
    grid[ghosts_index] = grid[ghosts_index[0], temp+1, ghosts_index[2]]
    grid_out[:, 1:, :] += grid[:, :-1, :]
    grid_out[:, 0, :] += grid[:, 0, :]
    grid[ghosts_index] = 0

    # grid[ghost_zf] = val_zf
    grid[ghosts_index] = grid[ghosts_index[0]-1, ghosts_index[1], ghosts_index[2]]
    grid_out[:-1, :, :] += grid[1:, :, :]
    grid_out[-1, :, :] += grid[-1, :, :]
    # grid[ghost_zf] = 0
    grid[ghosts_index] = 0
    # grid[ghost_zb] = val_zb
    temp = np.where(ghosts_index[0] > system_size - 2, ghosts_index[0] - 1, ghosts_index[0])
    grid[ghosts_index] = grid[temp+1, ghosts_index[1], ghosts_index[2]]
    grid_out[1:, :, :] += grid[:-1, :, :]
    grid_out[0, :, :] += grid[0, :, :]
    # grid[ghost_zb] = 0
    grid[ghosts_index] = 0
    grid_out[ghosts_index]=0
    # for ghost in [ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb]:
    #     grid_out[ghost] = 0

    return numexpr.evaluate("grid_out*dt*D", casting='same_kind')


# @jit(nopython=False, parallel=True)
def flux_matrix(matrix, y1, y2, x1, x2):
    '''
    Calculates a matrix with electron flux distribution

    :param matrix: output matrix
    :param y1: y2, x1 – x2 – boundaries of the effectively irradiated area
    :return:
    '''
    matrix[:,:]=0
    irradiated_area = matrix[y1:y2, x1:x2]
    # irradiated_area = np.zeros((7,7), dtype=int)
    center = irradiated_area.shape[0]/2*cell_dimension
    with np.nditer(irradiated_area, flags=['multi_index'], op_flags=['readwrite']) as it:
        for x in it:
            r = pythagor(it.multi_index[0]*cell_dimension-center, it.multi_index[1]*cell_dimension-center)
            if r<effective_radius:
                x[...]=pe_flux(r)


@jit(nopython=True)
def pythagor(a,b):
    return math.sqrt(a*a+b*b)


@jit(nopython=True)
def define_irradiated_area(beam, effective_radius): # TODO: this function will have most of its math redundant if coordinates system will be switched to array-based
    '''
    Defines boundaries of the effectively irradiated area

    :param beam: beam position
    :param effective_radius: a distance at which intensity is lower than 99% of the distribution
    :return:
    '''
    norm_y_start = 0
    norm_y_end = math.floor((beam.y + effective_radius) / cell_dimension)
    norm_x_start = 0
    norm_x_end = math.floor((beam.x + effective_radius) / cell_dimension)
    temp = math.ceil((beam.y - effective_radius) / cell_dimension)
    if temp > 0:
        norm_y_start = temp
    if norm_y_end > ymax:
        norm_y_end = ymax
    temp = math.ceil((beam.x - effective_radius) / cell_dimension)
    if temp > 0:
        norm_x_start = temp
    if norm_x_end > xmax:
        norm_x_end = xmax
    return  norm_y_start, norm_y_end, norm_x_start, norm_x_end


@jit(nopython=True, parallel=True, cache=True)
def show_yeld(deposit, summ, summ1, res):
    summ1 = np.sum(deposit)
    res = summ1-summ
    return summ, summ1, res


# /The printing loop.
# @jit(nopython=True)
def printing(loops=1): # TODO: maybe it could be a good idea to switch to an array-based iteration, rather than absolute distances(dwell_step). Precicision will be lost, but complexity of the code and coordinates management will be greatly simplified
    t = 2E-5
    dwell_step = beam_d / 2
    x_offset = 90  # offset on the X-axis on both sides
    y_offset = 90  # offset on the Y-axis on both sides
    x_limit = xmax * cell_dimension - x_offset
    y_limit = ymax * cell_dimension - y_offset
    beam = Positionb(0, y_offset, x_offset)  # coordinates of the beam
    beam_matrix = np.zeros((system_size,system_size), dtype=int)
    beam_exposure = np.zeros((system_size,system_size), dtype=int)
    summ1,summ, result=0,0,0
    refresh_dt = dt*2
    global ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb, val_zf, val_zb, val_yf, val_yb, val_xf, val_xb, ghosts
    ghost_zf, ghost_zb, ghost_yf, ghost_yb, ghost_xf, ghost_xb, val_zf, val_zb, val_yf, val_yb, val_xf, val_xb = define_ghosts(substrate, surface)
    ghost_zf = set(zip(*ghost_zf))
    # ghost_zb = set(zip(*ghost_zb))
    # ghost_yf = set(zip(*ghost_yf))
    # ghost_yb = set(zip(*ghost_yb))
    # ghost_xf = set(zip(*ghost_xf))
    # ghost_xb = set(zip(*ghost_xb))
    ghosts = set.copy(ghost_zf)
    # ghost_zf = set(zip(*ghost_zf))
    for l in range(loops):  # loop repeats
        # summ1, summ, result = show_yeld(deposit, summ, summ1,result)
        # print(f'Deposit yield:{result}  Loop:{l}')
        print(f'Loop:{l}')
        for beam.y in np.arange(y_offset, y_limit, dwell_step):  # beam travel along Y-axis
            for beam.x in np.arange(x_offset, x_limit, dwell_step):  # beam travel alon X-axis
                # Determining the area around the beam that is effectively irradiated
                norm_y_start, norm_y_end, norm_x_start, norm_x_end = define_irradiated_area(beam,effective_radius)
                irradiated_area_2D=np.s_[norm_y_start:norm_y_end, norm_x_start:norm_x_end]
                irradiated_area_3D=np.s_[:, norm_y_start:norm_y_end, norm_x_start:norm_x_end]
                flux_matrix(beam_matrix, norm_y_start, norm_y_end, norm_x_start, norm_x_end)
                beam_exposure += beam_matrix
                surf = make_tuple(surface[irradiated_area_2D])
                section = substrate[:, 20, :]
                while True:  # iterating through every cell position in the selected area and depositing
                    deposition(beam_matrix[irradiated_area_2D], surf, substrate[irradiated_area_3D], deposit[irradiated_area_3D], dt)
                    update_surface(deposit[irradiated_area_3D], substrate[:, norm_y_start-2:norm_y_end+2, norm_x_start-2:norm_x_end+2], surface[irradiated_area_2D], surf, semi_surface, norm_y_start, norm_x_start)
                    if t % refresh_dt < 1E-6:  # adsorption and desorption is run with a max timestep that allows stable diffusion
                        if l == 3:
                            cProfile.runctx('precursor_density(beam_matrix, substrate, refresh_dt)', globals(), locals())
                            gg=0
                        precursor_density(beam_matrix, substrate, refresh_dt) # TODO: add tracking of the surface's highest point and send only a view to reduce unecessary operations on empty volume
                        beam_exposure = 0
                    t += dt

                    if not t % td > 1E-6:
                        break


if __name__ == '__main__':
    # flux_matrix([1, 2, 3], 34)
    printing(5)
    # cProfile.runctx('printing(5)',globals(),locals())
    # section = np.zeros((substrate.shape[0], ymax), dtype=float)
    fig, (ax0) = plt.subplots(1)
    pos = np.int(xmax/2)
    for i in range(0, substrate.shape[0]):
        for j in range(0, substrate.shape[1]):
            section[i,j] = deposit[i, j,pos]
    ax0.pcolor(section)
    # if substrate[i,j,pos,0] ==0:
    #     ax0.pcolor(substrate[i,:,pos,0], color="white")
    # else:
    #     if substrate[i,j,pos,1] == 1:
    #         ax0.pcolor([i,j], substrate[i,j,pos,1], cmap='RdBu')
    #     else:
    #         ax0.pcolor([i,j], substrate[i,j,pos,0], color="orange")
    fig.tight_layout()
    plt.show()
    q=0

