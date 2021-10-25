# Default packages
import os, sys
import random as rnd
import math

# Core packages
import numpy as np

# Axillary packeges
from tkinter import filedialog as fd
import pickle
import timeit
import line_profiler

# Local packages
import VTK_Rendering as vr
import etrajectory as et
import etrajmap3d as map3d

# TODO: implement a global flag for collecting data(Se trajes) for plotting

def plot(m3d:map3d.ETrajMap3d, sim:et.ETrajectory=math.nan): # plot energy loss and all trajectories

    render = vr.Render(sim.cell_dim)
    pe_trajectories = np.asarray(sim.passes)
    render.show_mc_result(sim.grid, pe_trajectories, m3d.DE, m3d.flux, m3d.coords_all)

def cache_params(params, deposit, surface):
    """
    Creates an instance of simulation class and fetches necessary parameters

    :param fn_cfg: dictionary with simulation parameters
    :param deposit: initial structure
    :param surface: array pointing to surface cells
    :param cell_dim: dimensions of a cell
    :param dt: time step of the simulation
    :return:
    """

    sim = et.ETrajectory(name=params['name']) # creating an instance of Monte-Carlo simulation class
    sim.setParameters(params, deposit, surface) # setting parameters
    return sim


def rerun_simulation(y0, x0, deposit, surface, sim:et.ETrajectory, dt):
    """
    Rerun simulation using existing MC simulation instance

    :param y0: beam y-position
    :param x0: beam x-position
    :param deposit: array representing solid structure
    :param surface: array representing surface shape
    :param sim: MC simulation instance
    :return:
    """
    start = timeit.default_timer()
    sim.map_wrapper(y0, x0)
    # sim.save_passes(f'{sim.N} passes', 'pickle')
    t = timeit.default_timer() - start
    print(f'\n{sim.N} trajectories took {t} s')
    print(f'Energy deposition took: \t SE preparation took: \t Flux counting took:')
    m3d = map3d.ETrajMap3d(deposit, surface, sim)
    start = timeit.default_timer()
    # profiler = line_profiler.LineProfiler()
    # profiled_func = profiler(m3d.map_follow)
    # try:
    #     profiled_func(sim.passes, 1)
    # finally:
    #     profiler.print_stats()
    m3d.map_follow(sim.passes, 1)
    t = timeit.default_timer() - start
    print(f' =  {t} s')
    # plot(m3d, sim)
    return np.int32(m3d.flux/m3d.amplifying_factor*sim.norm_factor/(dt*sim.cell_dim*sim.cell_dim))




if __name__ == '__main__':
    sys.argv=input("Input: ").split(' ')
    # if len(sys.argv) != 5:
    #     print('Usage: python3 etraj3d.py <vti file> <cfg file> <show plot (y/n)> <pickle trajectories (y/n)>')
    #     hockeystick.vti hockeystick.cfg y y
    #     print('Exit.')
    #     exit(0)