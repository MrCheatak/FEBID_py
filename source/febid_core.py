###################################################################
#
#  FEBID Simulation
#
#  Version 0.9
#
####################################################################
# Default packages
import ast
import datetime
import math
import os, sys
import random as rnd
import time
import warnings
from typing import Optional
import timeit
from threading import Thread
from tkinter import filedialog as fd

# Core packages
import numpy as np
import pyvista as pv

# Auxillary packages
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import cm
import scipy.constants as scpc
import yaml
from tqdm import tqdm
import line_profiler

# Local packages
from Process import Structure, Process
import VTK_Rendering as vr
import etraj3d, etrajectory, etrajmap3d

# It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus able produce deposit under irradiation.

# Semi-surface cells are cells that have precursor density but do not have a neighboring deposit cell
# Thus concept serves an alternative diffusion channel

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
    sim = etrajectory.ETrajectory()
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
        m3d = etrajmap3d.ETrajMap3d(structure.deposit, structure.surface_bool, sim)
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

################################

def initialize_framework(from_file=False):
    """
    Prepare data framework and fetch parameters from input configuration files

    :param from_file: load structure from vtk file
    :return:
    """
    print(f'Select three files: precursor parameters, beam parameters and optionally initial geometry of the simulation volume.')
    precursor = None
    settings = None
    sim_params = None
    try:
        precursor = yaml.load(open(fd.askopenfilename(), 'r'), Loader=yaml.Loader) # Precursor and substrate properties(substrate here is the top layer)
        settings = yaml.load(open(fd.askopenfilename(), 'r'), Loader=yaml.Loader) # Parameters of the beam, dwell time and precursor flux
        sim_params = yaml.load(open(fd.askopenfilename(), 'r'),Loader=yaml.Loader)  # Size of the chamber, cell size and time step
        # precursor = yaml.load(open(f'{sys.path[0]}{os.sep}Me3PtCpMe.yml', 'r'), Loader=yaml.Loader)
        # settings = yaml.load(open(f'{sys.path[0]}{os.sep}Parameters.yml', 'r'),Loader=yaml.Loader)
        # sim_params = yaml.load(open(f'{sys.path[0]}{os.sep}Simulation.yml', 'r'),Loader=yaml.Loader)
    except:
        sys.exit('An error occurred while reading one of the parameter files')

    #TODO: remove dt from the input parameters and move timings calculation to Process class
    mc_config, equation_values, timings, nr = buffer_constants(precursor, settings, sim_params)

    structure = Structure()
    vtk_file = None
    if from_file:
        try:
            print(f'Specify .vtk structure file:')
            vtk_file = fd.askopenfilename()
            # vtk_file = '/Users/sandrik1742/Documents/PycharmProjects/FEBID/source/Profiling/Pillar_s.vtk'
            vtk_obj = pv.read(vtk_file)
            if not vtk_obj:
                raise RuntimeError
        except:
            sys.exit("An error occurred while reading the VTK file")
        structure.load_from_vtk(vtk_obj=vtk_obj)
    else:
        try:
            # sim_params = yaml.load(open(fd.askopenfilename(),'r'), Loader=yaml.Loader)  # Size of the chamber, cell size and time step
            # sim_params = yaml.load(open(f'{sys.path[0]}{os.sep}Simulation.yml', 'r'),Loader=yaml.Loader) # Size of the chamber, cell size and time step
            pass
        except:
            sys.exit("An error occurred while reading initial geometry parameters file")
        structure.create_from_parameters(sim_params['cell_dimension'], sim_params['width'], sim_params['length'],
                              sim_params['height'], sim_params['substrate_height'])

    return structure, mc_config, equation_values, timings


def buffer_constants(precursor: dict, settings: dict, sim_params: dict):
    """
    Calculate necessary constants and prepare parameters for modules

    :param precursor: precursor properties
    :param settings: simulation conditions
    :param sim_params: parameters of the simulation
    :return:
    """
    td = settings["dwell_time"]  # dwell time of a beam, s
    Ie = settings["beam_current"]  # beam current, A
    beam_FWHM = 2.36 * settings["gauss_dev"]  # electron beam diameter, nm
    F = settings[
        "precursor_flux"]  # precursor flux at the surface, 1/(nm^2*s)   here assumed a constant, but may be dependent on time and position
    effective_diameter = beam_FWHM * 3.3  # radius of an area which gets 99% of the electron beam
    f = Ie / scpc.elementary_charge / (
                math.pi * beam_FWHM * beam_FWHM / 4)  # electron flux at the surface, 1/(nm^2*s)
    e = precursor["SE_emission_activation_energy"]
    l = precursor["SE_mean_free_path"]

    # Precursor properties
    sigma = precursor[
        "cross_section"]  # dissociation cross section, nm^2; is averaged from cross sections of all electron types (PE,BSE, SE1, SE2)
    n0 = precursor["max_density"]  # inversed molecule size, Me3PtCpMe, 1/nm^2
    molar = precursor["molar_mass_precursor"]  # molar mass of the precursor Me3Pt(IV)CpMe, g/mole
    # density = 1.5E-20  # density of the precursor Me3Pt(IV)CpMe, g/nm^3
    V = precursor["dissociated_volume"]  # atomic volume of the deposited atom (Pt), nm^3
    D = precursor["diffusion_coefficient"]  # diffusion coefficient, nm^2/s
    tau = precursor["residence_time"] * 1E-6  # average residence time, s; may be dependent on temperature

    kd = F / n0 + 1 / tau + sigma * f  # depletion rate
    kr = F / n0 + 1 / tau  # replenishment rate
    nr = F / kr  # absolute density after long time
    nd = F / kd  # depleted absolute density
    t_out = 1 / (1 / tau + F / n0)  # effective residence time
    p_out = 2 * math.sqrt(D * t_out) / beam_FWHM

    # Initializing framework
    dt = sim_params["time_step"]
    cell_dimension = sim_params["cell_dimension"]  # side length of a square cell, nm

    t_flux = 1 / (sigma + f)  # dissociation event time
    diffusion_dt = math.pow(cell_dimension * cell_dimension, 2) / (2 * D * (
            cell_dimension * cell_dimension + cell_dimension * cell_dimension))  # maximum stability

    # Parameters for Monte-Carlo simulation
    mc_config = {'name': precursor["deposit"], 'E0': settings["beam_energy"],
                 'Emin': settings["minimum_energy"],
                 'Z': precursor["average_element_number"],
                 'A': precursor["average_element_mol_mass"], 'rho': precursor["average_density"],
                 'I0': settings["beam_current"], 'sigma': settings["gauss_dev"],
                 'N': Ie * dt / scpc.elementary_charge, 'sub': settings["substrate_element"],
                 'cell_dim': sim_params["cell_dimension"],
                 'e': precursor["SE_emission_activation_energy"], 'l': precursor["SE_mean_free_path"]}
    # Parameters for reaction-equation solver
    equation_values = {'F': settings["precursor_flux"], 'n0': precursor["max_density"],
                       'sigma': precursor["cross_section"], 'tau': precursor["residence_time"] * 1E-6,
                       'V': precursor["dissociated_volume"], 'D': precursor["diffusion_coefficient"],
                       'dt': sim_params["time_step"]}
    # Stability time steps
    timings = {'t_diff': diffusion_dt, 't_flux': t_flux, 't_desorption': tau, 'dt': dt}

# effective_radius_relative = math.floor(effective_diameter / cell_dimension / 2)
    return mc_config, equation_values, timings, nr


def run_febid():
    """
    Create necessary objects and start the FEBID process

    :param structure: structure object
    :param mc_config: parameters for MC simulation
    :param equation_values: parameters for Process
    :param timings: stability times
    :param path: stream file filename
    :return:
    """
    from_file = input("Open predefined structure? [y/n] Otherwise a clear substrate will be created.")
    from_file = from_file.strip()
    if from_file not in ['y', 'n']:
        raise RuntimeError('Unacceptable input! Exiting.')
    elif from_file == 'y':
        from_file = True
    else:
        from_file = False

    structure, mc_config, equation_values, timings = initialize_framework(from_file)
    # print(f'Specify a stream file. If no file is specified, beam will be stationed in the center.')
    path = '' # fd.askopenfilename()
    if not path:
        _, ydim, xdim = structure.deposit.shape
        path = np.array([ydim/2, xdim/2, 1])
        if path.ndim<2:
            path = path.reshape(1, path.shape[0])
    path = generate_line(10000, 0.00001, path[0,0], path[0,1], 15, 1)
    process_obj = Process(structure, mc_config, equation_values, timings, deposition_scaling=16)
    sim = etraj3d.cache_params(mc_config, process_obj.deposit, process_obj.surface)
    total_iters = int(np.sum(path[:,2])/process_obj.dt)
    monitor = Thread(target=monitoring, args=[process_obj, total_iters])
    # monitor.setDaemon(True)
    # monitor.start()
    for y, x, step in path:
        process_obj.beam_matrix[:,:,:] = etraj3d.rerun_simulation(y, x, process_obj.deposit, process_obj.surface, sim, process_obj.dt)
        process_obj.update_helper_arrays()
        print_step(y ,x, step, process_obj, sim)
    # monitor.join()


def print_step(y, x, dwell_time, pr:Process, sim):
    """
    Run deposition on a single spot

    :param x: spot x-coordinate
    :param y: spot y-coordinate
    :param dwell_time: time of the exposure
    :param pr: Process object
    :param sim: MC simulation object
    :return:
    """
    loops = int(dwell_time/pr.dt)
    for l in tqdm(range(0, loops)):  # loop repeats
        pr.deposition()  # depositing on a selected area
        if pr.check_cells_filled():
            pr.update_surface() # updating surface on a selected area
            pr.beam_matrix = etraj3d.rerun_simulation(y,x, pr.deposit, pr.surface, sim, pr.dt)
            pr.update_helper_arrays()
            a = 1
        pr.precursor_density()
        pr.t += pr.dt


def monitoring(pr:Process, l, dump = None, refresh_rate=0.5):
    """
    A daemon process function to manage statistics gathering and graphics update from a separate thread
    :param pr: process object
    :param l: approximate number of iterations
    :param dump: weather to periodically save the structure to a file, None or interval in min
    :return:
    """

    time_step = 60
    frame_rate = 1
    dump_interval = 60*60
    start_time = datetime.datetime.now()
    pr.start_time = start_time
    time_spent = frame = dump_time = timeit.default_timer()
    redraw = True
    rn = vr.Render(pr.structure.cell_dimension)
    rn._add_3Darray(pr.precursor, opacity=1, nan_opacity=1, scalar_name='Precursor',
                        button_name='precursor', cmap='plasma')
    rn.show(interactive_update=True, cam_pos=[(206.34055818793468, 197.6510638707941, 100.47106597548205),
                                                  (0.0, 0.0, 0.0),
                                                  (-0.23307751464125356, -0.236197909312718, 0.9433373838690787)])
    while True:
        now = timeit.default_timer()
        if now > frame:
            frame += frame_rate
            update_graphical(rn,pr,time_step, time_spent)
        if now > time_spent:
            time_spent += time_step
            print(f'Time passed: {time_spent}, Av.speed: {l/time_spent}')
        if dump:
            if now > dump_time*60:
                dump_time += dump_interval
                dump_structure(pr.structure, dump)
        time.sleep(refresh_rate)

    
def update_graphical(rn:vr.Render, pr:Process, time_step, time_spent):
    """
    Update the visual representation of the current process state

    :param rn: visual scene object
    :param pr: process object
    :param time_step:
    :param time_spent:
    :return:
    """
    if pr.redraw:
        rn.p.clear()
        pr.n_filled_cells.append(np.count_nonzero(pr.deposit[pr.deposit < 0]) - pr.n_substrate_cells)
        i = len(pr.n_filled_cells)-1
        pr.growth_rate.append((pr.n_filled_cells[i] - pr.n_filled_cells[i - 1]) / (time_spent - time_step + 0.001) * 60 * 60)
        rn._add_3Darray(pr.precursor, 0.00000001, 1, opacity=0.5, show_edges=True, exclude_zeros=False,
                            scalar_name='Precursor',
                            button_name='precursor', cmap='plasma')
        rn.meshes_count += 1
        # rn.add_3Darray(deposit, structure.cell_dimension, -2, -0.5, 0.7, scalar_name='Deposit',
        #            button_name='Deposit', color='white', show_scalar_bar=False)
        rn.p.add_text(f'Time: {str(datetime.timedelta(seconds=int(time_spent)))} \n'
                          f'Sim. time: {(pr.t):.8f} s \n'
                          f'Speed: {(pr.t / time_spent):.8f} \n'  # showing time passed
                          f'Relative growth rate: {int(pr.n_filled_cells[i] / time_spent * 60 * 60)} cell/h \n'  # showing average growth rate
                          f'Real growth rate: {int(pr.n_filled_cells[i] / pr.t * 60)} cell/min \n',
                          position='upper_left',
                          font_size=12)  # showing average growth rate
        rn.p.add_text(f'Cells: {pr.n_filled_cells[i]} \n'  # showing total number of deposited cells
                          f'Height: {(pr.max_z-pr.substrate_height) * pr.structure.cell_dimension} nm \n', position='upper_right',
                          font_size=12)  # showing current height of the structure
        pr.redraw = False
    else:
        # rn.p.mesh['precursor'] = precursor[precursor!=0]
        rn.p.update_scalars(pr.precursor[pr.precursor > 0])
        rn.p.update_scalar_bar_range(clim=[pr.precursor[pr.precursor>0].min(), pr.precursor.max()])
    rn.p.update()
    return pr.redraw


def dump_structure(structure:Structure, filename='FEBID_result'):
    vr.save_deposited_structure(structure, filename)


def generate_circle(loops, dwell_time, x, y, radius, step=1):
    path = None
    angle_step = step/radius
    n = int(np.pi*2 // angle_step)
    loop = np.zeros((n, 3))
    stub = np.arange(angle_step, np.pi*2, angle_step)
    loop[:,0] = radius * np.sin(stub) + y
    loop[:,1] = radius * np.cos(stub) + x
    loop[:,2] = dwell_time
    path = np.tile(loop, (loops, 1))
    a = 0
    return path

def generate_square(loops, dwell_time, x, y, side_a, side_b:Optional, step=1):
    path = None
    if side_b is None:
        side_b = side_a
    top_left = (y+side_b/2, x-side_a/2)
    top_right = (y+side_b/2, x+side_a/2)
    low_right = (y-side_b/2, x+side_a/2)
    low_left = (y-side_b/2, x-side_a/2)
    steps_a = int(side_a/step)
    steps_b = int(side_b/step)
    edge_top = np.empty((steps_a,3))
    edge_top[:,1] = np.arange(top_left[1]+step, top_right[1], step)
    edge_top[:,0] = top_left[0]
    edge_right = np.empty((steps_b,3))
    edge_right = np.arange(top_right[0]-step, low_right[0], -step)
    edge_right[:,1] = top_right[1]
    edge_bottom = np.empty((steps_a,3))
    edge_bottom[:,1] = np.arange(low_left[1]-step, low_left[1], -step)
    edge_bottom[:,0] = low_right[0]
    edge_left = np.empty((steps_b,3))
    edge_left[:,0] = np.arange(low_left[0]+step, top_left[0], step)
    edge_left[:,1] = low_left[1]
    path = np.concatenate([edge_top, edge_right, edge_bottom, edge_left])
    path[:,2] = dwell_time
    path = np.tile(path, (loops,1))
    return path

def generate_line(loops, dwell_time, x, y, line, step=1):
    path = None
    start = x - line/2
    end = x + line/2
    path = np.empty((int(line/step*2)-2,3))
    loop1 = np.arange(start + step, end, step)
    loop2 = np.arange(end-step, start, -step)
    loop = np.concatenate([loop1, loop2])
    path[:,0] = y
    path[:,1] = loop
    path[:,2] = dwell_time
    path = np.tile(path, (loops,1))
    return path

if __name__== '__main__':
    print(f'##################### FEBID Simulator ###################### \n\n'
          f'Two modes are available: deposition process and Monte Carlo electron beam-matter simulation\n'
          f'Type \'febid\' to enter deposition mode or \'mc\' for electron trajectory simulation:')
    mode_choice = input("Enter 'dp', 'mc' or press Enter to exit:")
    if mode_choice not in ['dp', 'mc']:
        sys.exit("Exit program.")
    if mode_choice == 'mc':
        print(f'Entering Monte Carlo electron beam-matter simulation module')
        mc_simulation()


    run_febid()

