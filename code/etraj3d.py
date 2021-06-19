import copy
import sys, os
import time
import random as rnd
import timeit
from math import *
import numpy as np
import pyvista as pv
import etrajectory as et
import etrajmap3d as map3d
from tqdm import tqdm
import pickle
import line_profiler
import VTK_Rendering as vr
from timebudget import timebudget

# TODO: implement a global flag for collecting data(Se trajes) for plotting


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


def plot(m3d:map3d.ETrajMap3d, sim:et.ETrajectory=nan): # plot energy loss and all trajectories
    struct = m3d.grid
    trajs = m3d.trajectories
    pe_energies = 0
    if sim:
        pe_energies = np.asarray(sim.passes)[:, 1]
        trajs = np.asarray(sim.passes)[:, 0]
    energies = m3d.DE
    se_trajes = m3d.se_traj
    render = vr.Render()
    # Deposited structure
    render.add_3Darray(m3d.grid, m3d.cell_dim, -3, 0, button_name="Structure", color='white', invert=True)
    # Deposited energies
    render.add_3Darray(m3d.DE, m3d.cell_dim, 1, scalar_name='Deposited energy, eV', button_name="Deposited energy", cmap='coolwarm', log_scale=True)
    # SE flux at the surface
    render.add_3Darray(m3d.flux, m3d.cell_dim, 1, scalar_name='Flux, 1/(nm^2*s)', button_name='SE surface flux', cmap='plasma', log_scale=True)
    # PE trajectories
    render.add_trajectory(trajs, pe_energies, 0.5, step=10, scalar_name='PE Energy, keV', button_name='PEs', cmap='viridis')
    # SEs
    render.add_trajectory(se_trajes, radius=0.2, step=25, button_name='SEs', color='red')

    render.p.camera_position = [(463.14450307610286, 271.1171723376318, 156.56895424388603),
                                (225.90027381807235, 164.9577775224395, 71.42188811921902),
                                (-0.27787912231751677, -0.1411181984824172, 0.950194110399093)]
    camera_pos = render.show()


def cache_params(fn_cfg, deposit, surface, cell_dim, beam_ef_rad, dt):
    """
    Creates an instance of simulation class and fetches necessary parameters

    :param fn_cfg: dictionary with simulation parameters
    :param deposit: initial structure
    :param surface: array pointing to surface cells
    :param cell_dim: dimensions of a cell
    :param beam_ef_rad: effective radius of the beam
    :param dt: time step of the simulation
    :return:
    """

    params = fn_cfg # getting configurations
    sim = et.ETrajectory(name=params['name']) # creating an instance of Monte-Carlo simulation class
    sim.setParameters(params, deposit, surface, cell_dim, beam_ef_rad, dt) # setting parameters
    return sim


def run_simulation(sim: et.ETrajectory, deposit:np.ndarray, surface: np.ndarray, i0=25, j0=25):
    """
    Run simulation

    :param sim: Instance of the class implementing simulation
    :param deposit: initial structure
    :param surface: array pointing to surface cells
    :param i0: X beam position in array coordinates
    :param j0: Y beam position in array coordinates
    :return:
    """

    # sim.test_run(i0,j0)
    sim.run(i0, j0)
    m3d = map3d.ETrajMap3d()
    m3d.get_structure(deposit, surface, sim.cell_dim)
    m3d.map_trajectory(sim.passes) # mapping all cached MC trajectories for every generated point[x0, y0]
    m3d.flux /= ((sim.dt*sim.cell_dim*sim.cell_dim)/sim.norm_factor)
    plot(m3d, sim)
    return sim


def rerun_simulation(y0, x0, deposit, surface, sim):
    """
    Rerun simulation using existing instance

    :param y0:
    :param x0:
    :param deposit:
    :param surface:
    :param sim:
    :return:
    """
    # picks = []
    # for i in range(2, 50):
    #     start = timeit.default_timer()
    #     sim.run(y0, x0, i*100)
    #     t = timeit.default_timer()-start
    #     print(f'Run {i} with {i*100} iters took {t}')
    #     picks.append(copy.deepcopy(sim.passes))
    # file = open(f'{sys.path[0]}{os.sep}Trajes_200-4900.txt', 'wb')
    # pickle.dump(picks, file)
    start = timeit.default_timer()
    sim.run(y0, x0)
    t = timeit.default_timer() - start
    print(f'Took {t} s')
    # print(f'Run with {sim.N} iters took {t}')
    sim.ztop = np.nonzero(surface)[0].max()+1
    m3d = map3d.ETrajMap3d()
    m3d.get_structure(deposit, surface, sim.cell_dim)
    start = timeit.default_timer()
    m3d.map_trajectory(sim.passes)
    t = timeit.default_timer() - start
    print(f'Took {t} s')
    # print("Loading trajectories file...")
    # file = open(f'{sys.path[0]}{os.sep}Trajes_200-4900.txt', 'rb')
    # passes = pickle.load(file)
    # file.close()
    # print("Done.")
    # file = open(f'{sys.path[0]}{os.sep}Test_map_follow{time.ctime()}.txt', 'w')
    # file.write("Initial version\n")
    # for pas in passes:
    #     with timebudget(f'Simulation:{i}'):
    #     start = timeit.default_timer()
    #     m3d.map_trajectory(pas)
    #     t = timeit.default_timer()-start
    #     file.write(f'{len(pas)}\t{t}\n')
    #     print(f'Run {len(pas)/100} with {len(pas)} iters took {t}')
    # file.close()
    plot(m3d, sim)
    return np.int32(m3d.flux*sim.norm_factor/(sim.dt*sim.cell_dim*sim.cell_dim))


if __name__ == '__main__':
    sys.argv=input("Input: ").split(' ')
    # if len(sys.argv) != 5:
    #     print('Usage: python3 etraj3d.py <vti file> <cfg file> <show plot (y/n)> <pickle trajectories (y/n)>')
    #     hockeystick.vti hockeystick.cfg y y
    #     print('Exit.')
    #     exit(0)

    run_simulation(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3])