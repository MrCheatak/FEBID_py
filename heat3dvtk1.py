import numpy as np
import pyvista as pv
from tqdm import tqdm
import sys

def find_monitor_point(grid):
    dims = np.array(grid.dimensions) - 1
    state = np.reshape(grid.cell_arrays['state'], dims, order='F')
    for k in range(dims[2]-1, 0, -1):
        for j in range(dims[1]):
            for i in range(dims[0]):
                if state[i,j,k] == 2:
                    return (i, j, k)

def init_bc(grid, state, T, T0):
    '''Setup Neumann boundary conditions and set all non-void cells to normalized temperature 1.
       grid: vtk grid
       T: numpy array of normalized temperatures
       T0: start temperature = temperature for surface points with Dirichlet boundary condition
       
       Returns: ghost point index lists gx, gy, gz, partner point index lists px, py, pz needed for fulfilling
       Neumann boundary conditions (not gradient perpendicular to surface).
    '''
    dims = T.shape
    gx0, gx1, gx2 = [], [], []
    gy0, gy1, gy2 = [], [], []
    gz0, gz1, gz2 = [], [], []
    px0, px1, px2 = [], [], []
    py0, py1, py2 = [], [], []
    pz0, pz1, pz2 = [], [], []
    # set temperature to start value and find ghost and partner point indices
    for k in range(dims[2]):
        for j in range(dims[1]):
            for i in range(dims[0]):
                if state[i,j,k] == 1:
                    T[i,j,k] = T0
                    if state[i-1,j,k] == 0:
                        gx0.append(i-1)
                        if state[i+1,j,k] != 0:
                            px0.append(i+1)
                        else:
                            px0.append(i)
                        gx1.append(j)
                        px1.append(j)
                        gx2.append(k)
                        px2.append(k)
                    if state[i+1,j,k] == 0:
                        gx0.append(i+1)
                        if state[i-1,j,k] != 0:
                            px0.append(i-1)
                        else:
                            px0.append(i)
                        gx1.append(j)
                        px1.append(j)
                        gx2.append(k)
                        px2.append(k)
                    if state[i,j-1,k] == 0:
                        gy0.append(i)
                        py0.append(i)
                        gy1.append(j-1)
                        if state[i,j+1,k] != 0:
                            py1.append(j+1)
                        else:
                            py1.append(j)
                        gy2.append(k)
                        py2.append(k)
                    if state[i,j+1,k] == 0:
                        gy0.append(i)
                        py0.append(i)
                        gy1.append(j+1)
                        if state[i,j-1,k] != 0:
                            py1.append(j-1)
                        else:
                            py1.append(j)
                        gy2.append(k)
                        py2.append(k)
                    if state[i,j,k-1] == 0:
                        gz0.append(i)
                        pz0.append(i)
                        gz1.append(j)
                        pz1.append(j)
                        gz2.append(k-1)
                        if state[i,j,k+1] != 0:
                            pz2.append(k+1)
                        else:
                            pz2.append(k)
                    if state[i,j,k+1] == 0:
                        gz0.append(i)
                        pz0.append(i)
                        gz1.append(j)
                        pz1.append(j)
                        gz2.append(k+1)
                        if state[i,j,k-1] != 0:
                            pz2.append(k-1)
                        else:
                            pz2.append(k)
                elif state[i,j,k] == 2:
                    T[i,j,k] = T0
    return ([gx0, gx1, gx2], [gy0, gy1, gy2], [gz0, gz1, gz2], [px0, px1, px2], [py0, py1, py2], [pz0, pz1, pz2])

def run_sim(params):
    '''Solve 3d time-dependent heat equation by forward Euler finite-difference scheme.
       params: dictionary containing the following obligatory parameters:
          vti file name: vtk file with mesh containing cell arrays 'state', 'powerdensity' or 'energy', 'dirichlet'
          kappa: heat conduction coefficient in W/mK
          spec_heat: heat capacity in J/kgK
          rho: density in kg/m^3
          T0: temperature for Dirichlet boundary condition in K
       optional parameters:
          e2p: energy (J) to power density (W/m^3) factor for grids with cell array 'energy'
          mon: index triple for point where temperature is to be monitored; if None, will be selected during simulation
          tol: if relative temperature change during one time step is below this value, simulation stops
          N: maximum number of iteration steps
          dt: time increment; if None, will be calculated during run of simulation and updated

       Returns: grid with added cell array 'temperature', time trace of monitored temperature as list of (t, Tmon) pairs
    '''
    gx, gy, gz = [], [], [] # list of indices for ghost points (sup surface points) for Neumann boundary condition
    px, py, pz = [], [], [] # list of indices for partner points (sup surface points) for Neumann boundary condition

    # setup grid
    grid = pv.read('./' + params['vti_file'])
    dims = np.array(grid.dimensions) - 1
    dx, dy, dz = grid.spacing
    dx2, dy2, dz2 = dx**2, dy**2, dz**2
    # setup needed arrays and lists
    T = np.zeros(dims)
    dT = np.zeros(dims) # temperature changes during one iteration
    T0 = params['T0']
    state = np.reshape(grid.cell_arrays['state'], dims, order='F')
    dc = np.nonzero(np.reshape(grid.cell_arrays['dirichlet'], dims, order='F')) # cell indices with Dirichlet boundary conditions
    gx, gy, gz, px, py, pz = init_bc(grid, state, T, T0)
    if 'e2p' in params:
        fT = np.reshape(grid.cell_arrays['energy']*params[e2p]/params['rho']/params['spec_heat'], dims, order='F')
    else:
        fT = np.reshape(grid.cell_arrays['powerdensity']/params['rho']/params['spec_heat'], dims, order='F')
    # complete setup before simulation
    if 'N' in params:
        N = params['N']
    else:
        N = 10000
    if 'mon' in params:
        mon = (int(params['mon'][0]), int(params['mon'][1]), int(params['mon'][2]))
    else:
        mon = find_monitor_point(grid)
    if 'tol' in params:
        tol = params['tol']
    else:
        tol = 1.0E-7
    pre = params['kappa']/params['rho']/params['spec_heat']
    t_sum = 0.0
    Tmon = T0
    if 'dt' in params:
        dt = params['dt']
    else:
        T[gx] = T[px]
        D1 = np.roll(T, -1, 0) + np.roll(T, 1, 0) - 2*T
        T[gy] = T[py]
        D2 = np.roll(T, -1, 1) + np.roll(T, 1, 1) - 2*T
        T[gz] = T[pz]
        D3 = np.roll(T, -1, 2) + np.roll(T, 1, 2) - 2*T
        D1[state==0] = 0.0
        D2[state==0] = 0.0
        D3[state==0] = 0.0
        delta = pre*(D1/dx2 + D2/dy2 + D3/dz2) + fT
        dt = 1/np.max(np.abs(delta))
        T[dc] = T0
        T[state == 0] = 0.0
    # simulation run
    print('Start iterations with time increment dt = {:f}'.format(dt))
    timetrace = []
    for cnt in tqdm(range(N)):
        # calc curvature
        T[gx] = T[px]
        D1 = np.roll(T, -1, 0) + np.roll(T, 1, 0) - 2*T
        T[gy] = T[py]
        D2 = np.roll(T, -1, 1) + np.roll(T, 1, 1) - 2*T
        T[gz] = T[pz]
        D3 = np.roll(T, -1, 2) + np.roll(T, 1, 2) - 2*T
        #D1[state==0] = 0.0
        #D2[state==0] = 0.0
        #D3[state==0] = 0.0
        dT = (pre*(D1/dx2 + D2/dy2 + D3/dz2) + fT)*dt
        T += dT
        # apply Dirichlet boundary condition and reset void cells to 0.0 temperature
        T[dc] = T0
        T[state == 0] = 0.0
        Tmon_new = T[mon]
        rel_change = (Tmon_new - Tmon)/Tmon
        if abs(rel_change) < tol and cnt > int(N/10):
            break
        Tmon = Tmon_new
        t_sum += dt
        if cnt/100 == cnt//100:
            timetrace.append((t_sum, Tmon))
            print('t = {:6.3g}  T_mon = {:06.3f}  dT/T = {:06.3g}'.format(t_sum, Tmon, rel_change))
            #delta = pre*(D1/dx2 + D2/dy2 + D3/dz2) + fT
            #dt = 1/np.max(np.abs(delta))*5E-6
            #print(dt)
    # wrap up
    grid.cell_arrays['T'] = T.flatten(order='F')
    return (grid, timetrace)


def read_cfg(fname):
    '''Reads configuration file for initialization of simulation.

    Returns: dictionary with parameter values grid file name, kappa, spec_heat, rho, T0, e2p_factor
    '''
    f = open(fname, 'r')
    params = {}
    for line in f:
        if line.find('#') == 0:
            continue
        l = line.strip('\n').split(':')
        k = l[0].strip(' ')
        v = l[1].strip(' ')
        if k == 'vti_file':
            params[k] = v
        elif k == 'kappa':
            params[k] = float(v)
        elif k == 'spec_heat':
            params[k] = float(v)
        elif k == 'rho':
            params[k] = float(v)
        elif k == 'T0':
            params[k] = float(v)
        elif k == 'e2p':
            params[k] = float(v)
        elif k == 'mon':
            params[k] = v.split(',')
        elif k == 'tol':
            params[k] = float(v)
        elif k == 'N':
            params[k] = int(v)
        elif k == 'dt':
            params[k] = float(v)
    f.close()
    return params


if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print('Usage: python3 heat3dvtk1.py <cfg-file name> <save-to file name (.vti)>')
    #     exit(0)
    # cfg_file, ofile = sys.argv[1], sys.argv[2]
    cfg_file = "/Users/sandrik1742/PycharmProjects/FEBID/wall.cfg"
    ofile = "/Users/sandrik1742/PycharmProjects/FEBID/wall_long_corner_p.vti", "r"
    params = read_cfg(cfg_file)
    grid, timetrace = run_sim(params)
    timetrace = np.array(timetrace)
    np.savetxt('timetrace.txt', timetrace)
    grid.save(ofile)