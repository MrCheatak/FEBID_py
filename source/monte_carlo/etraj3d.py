# Default packages
import os, sys
import random as rnd
import math
import warnings

# Core packages
import numpy as np
import pyvista as pv

# Axillary packeges
from tkinter import filedialog as fd
import pickle
import timeit
import line_profiler

# Local packages
from Structure import Structure
import VTK_Rendering as vr
from monte_carlo import etrajectory as et
from monte_carlo import etrajmap3d as map3d

# TODO: implement a global flag for collecting data(Se trajes) for plotting

def run_mc_simulation(vtk_obj, E0=20, sigma=5, N=100, pos='center', material='Au', Emin=0.1):
    """
    Create necessary objects and run the MC simulation

    :param vtk_obj:
    :param E0:
    :param sigma:
    :param N:
    :param pos:
    :param material:
    :param Emin:
    :return:
    """
    structure = Structure()
    structure.load_from_vtk(vtk_obj)
    params={'E0': E0, 'sigma':sigma, 'N':N, 'material':material, 'Emin':Emin}
    sim = et.ETrajectory()
    sim.setParams_MC_test(structure, params)
    x, y = 0, 0
    if pos == 'center':
        x = structure.shape[2]/2
        y = structure.shape[1]/2
    else:
        x, y = pos
    cam_pos = None
    while True:
        print(f'{N} PE trajectories took:   \t Energy deposition took:   \t SE preparation took:   \t Flux counting took:')
        start = timeit.default_timer()
        sim.map_wrapper(x,y)
        print(f'{timeit.default_timer() - start}', end='\t\t')
        m3d = map3d.ETrajMap3d(structure.deposit, structure.surface_bool, sim)
        m3d.map_follow(sim.passes, 1)
        pe_trajectories = np.asarray(sim.passes)
        render = vr.Render(structure.cell_dimension)
        cam_pos = render.show_mc_result(sim.grid, pe_trajectories, m3d.DE, m3d.flux, cam_pos=cam_pos)


def mc_simulation():
    """
    Fetch necessary data and start the simulation

    :return:
    """
    #TODO: This standalone Monte Carlo module should provide more data than in FEBID simulation

    print(f'Monte-Carlo electron beam - matter interaction module.\n'
          f'First load desired structure from vtk file, then enter parameters.')
    print(f'Select .vtk file....')
    while True:
        try:
            file = fd.askopenfilename()
            # file = '/Users/sandrik1742/Documents/PycharmProjects/FEBID/_source_samples/electron_beam_monte-carlo/hockeystick.vti'
            vtk_obj = pv.read(file)
        except Exception as e:
            print(f'Unable to read vtk file. {e.args} \n'
                  f'Try again:')
        else:
            print(f'Got file!\n')
            break

    print(f'Input parameters: Beam energy(keV), gauss st. deviation, number of electrons to emit, (beam x position, beam y position) , structure material(i.e. Au)\n'
          f'Note: \'center\' can be used instead of a coordinates pair (x,y) to set the beam to the center')
    E0=read_param('Beam energy', [int, float])
    sigma = read_param('Gauss standard deviation', [int, float])
    N = read_param('Number of electrons', [int])
    pos = read_param('Beam position', [tuple, str], check_string=['center'])
    material = read_param('Structure material', [str], check_string=['Au'])
    print(f'Got paramers!\n')
    run_mc_simulation(vtk_obj, E0, sigma, N, pos, material)
    # run_mc_simulation(vtk_obj, 20, 15, 1000, (12,17), 'Au')


def read_param(name, expected_type, message="Inappropriate input for ", check_string=None):
    """
    Read and parse a parameter from the input

    :param name: name of teh parameter
    :param expected_type: data type
    :param message: error text
    :param check_string: look for a specific string
    :return:
    """
    while True:
        item = input(f'Enter {name}:')
        try:
            result = eval(item)
        except:
            result = item
        if type(result) in expected_type:
            if type(result) is str:
                if check_string is not None:
                    if result in check_string:
                        return result
                    else:
                        warnings.warn("Input does not match any text choice")
                        print(f'Try again')
                        continue
            return result
        else:
            # unlike 'raise Warning()', this does not interrupt code execution
            warnings.warn(message+name)
            print(f'Try again')
            continue


def plot(m3d:map3d.ETrajMap3d, sim:et.ETrajectory): # plot energy loss and all trajectories
    """
    Show the structure with surface electron flux and electron trajectories

    :param m3d:
    :param sim:
    :return:
    """
    render = vr.Render(sim.cell_dim)
    pe_trajectories = np.asarray(sim.passes)
    render.show_mc_result(sim.grid, pe_trajectories, m3d.DE, m3d.flux, m3d.coords_all)


def cache_params(params, deposit, surface, surface_neighbors):
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
    sim.setParameters(params, deposit, surface, surface_neighbors) # setting parameters
    return sim


def rerun_simulation(y0, x0, sim:et.ETrajectory, dt):
    """
    Rerun simulation using existing MC simulation instance

    :param y0: beam y-position, absolute
    :param x0: beam x-position, absolute
    :param deposit: array representing solid structure
    :param surface: array representing surface shape
    :param sim: MC simulation instance
    :return:
    """
    start = timeit.default_timer()
    # sim.map_wrapper(y0, x0)
    sim.map_wrapper_cy(y0, x0)
    t = timeit.default_timer() - start
    m3d = sim.m3d
    start = timeit.default_timer()
    m3d.map_follow(sim.passes)
    t = timeit.default_timer() - start
    # plot(m3d, sim)
    if m3d.flux.max() > 10000*m3d.amplifying_factor:
        print(f' Encountered infinity in the beam matrix: {np.nonzero(m3d.flux>10000*m3d.amplifying_factor)}')
        m3d.flux[m3d.flux>10000*m3d.amplifying_factor] = 0
    if m3d.flux.min() < 0:
        print(f'Encountered negative in beam matrix: {np.nonzero(m3d.flux<0)}')
        m3d.flux[m3d.flux<0] = 0
    const = sim.norm_factor/(dt*sim.cell_dim*sim.cell_dim)/m3d.amplifying_factor
    return np.int32(m3d.flux*const)


if __name__ == '__main__':
    print("Current script does not have an entry point.....")
    input('Press Enter to exit.')