import sys
import random as rnd
from math import *
import numpy as np
import pyvista as pv
import etracjectory as et
import etrajmap3d as map3d
from tqdm import tqdm
import pickle

def rnd_gauss_xy(sigma, n):
    '''Gauss-distributed (x, y) positions.
       sigma: standard deviation 
       n: number of positions
       Returns lists of n x and y coordinates which are gauss-distributed.
    '''
    x, y = [], []
    rnd.seed()
    for i in range(n):
        r = rnd.gauss(0.0, sigma)
        phi = rnd.uniform(0, 2*pi - np.finfo(float).tiny)
        x.append(r*cos(phi))
        y.append(r*sin(phi))
    return (x, y)

def read_cfg(fname):
    '''Read configuration file.
       fname: name of configuration file containing key:value pairs
       Returns dictionary of parameter values.
    '''
    f = open(fname, 'r')
    params = {}
    for line in f:
        if line.find('#') == 0:
            continue
        l = line.strip('\n').split(':')
        k = l[0].strip(' ')
        v = l[1].strip(' ')
        if k == 'N':
            params[k] = int(v)
        elif k == 'name':
            params[k] = v
        else:
            params[k] = float(v)
    f.close()
    return params

def plot(grid, trajs): # plot energy loss and all trajectories
    pv.set_plot_theme('paraview')
    sargs = dict(
        title_font_size=24,
        label_font_size=20,
        shadow=False,
        n_labels=5,
        italic=True,
        fmt='%.0f',
        font_family='arial',
    )
    grid.set_active_scalar('state')
    grid_vol = grid.threshold([1, 2])
    grid.set_active_scalar('energy')
    grid_eloss = grid.threshold([0.01, np.max(grid.cell_arrays['energy'])])
    p = pv.Plotter()
    #_ = p.add_mesh(grid_vol, color='white', style='wireframe')
    _ = p.add_mesh(grid_vol, color='white', style='surface', opacity=0.5)
    #_ = p.add_mesh(grid_eloss, show_scalar_bar=False, opacity=0.5, clim=[0.0, 150.0])
    _ = p.add_mesh(grid_eloss, show_scalar_bar=False, opacity=0.5)
    _ = p.add_scalar_bar('Energy (eV)', title_font_size=44, label_font_size=40, fmt='%4.0f', interactive=True, width=0.01)
    for traj in trajs:
        lines = []
        lines.append([traj[0][0], traj[0][1], grid.origin[2] + grid.dimensions[2]*grid.spacing[2]])
        for i in range(0, len(traj), 2):
            lines.append([traj[i][0], traj[i][1], traj[i][2]])
            lines.append([traj[i+1][0], traj[i+1][1], traj[i+1][2]])
        _ = p.add_lines(np.array(lines), color='yellow', width=2)
    p.show()

def run_simulation(fn_vti, fn_cfg, show, pickle_traj):
    '''Run simulation.
       fn_vti: name of vti-file to read geometry from
       fn_cfg: name of config-file to read parameters for simulation from
       show: either 'y' ('Y') or 'n' ('N'); if 'y', result will be shown
       pickle_traj: either 'y' ('Y') or 'n' ('N'); if 'y', trajectors will be pickled for later use
    '''
    params = read_cfg(fn_cfg)
    sim_params = {}
    sim_params['E0'] = params['E0']
    sim_params['Z'] = params['Z']
    sim_params['A'] = params['A']
    sim_params['rho'] = params['rho']
    sim_params['x0'] = 0.0
    sim_params['y0'] = 0.0
    sim_params['z0'] = 0.0
    sim = et.ETrajectory(name=params['name'])
    sim.setParameters(sim_params)
    sim.run(passes=params['N'])

    m3d = map3d.ETrajMap3d()
    m3d.read_vtk(fn_vti)
    x, y = rnd_gauss_xy(params['sigma'], params['N']) # generate gauss-distributed beam positions
    x0 = [xx + params['xb'] for xx in x]
    y0 = [yy + params['yb'] for yy in y]
    for i in tqdm(range(params['N'])):
        points, energies = sim.passes[i][0], sim.passes[i][1]
        m3d.map_trajectory(points, energies, x0[i], y0[i])
    m3d.grid.cell_arrays['energy'] = m3d.DE.flatten(order='F')

    index = fn_vti.find('.vti')
    m3d.grid.save(fn_vti[:index] + '_E' + '.vti')
    if pickle_traj == 'y' or pickle_traj == 'Y':
        f_traj = open(fn_vti[:index] + '_traj' + '.pck', 'wb')
        pickle.dump(m3d.trajectories, f_traj)
        f_traj.close()
    if show == 'y' or show == 'Y':
        print('Preparing plot ...')
        plot(m3d.grid, m3d.trajectories)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: python3 etraj3d.py <vti file> <cfg file> <show plot (y/n)> <pickle trajectories (y/n)>')
        print('Exit.')
        exit(0)

    run_simulation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])