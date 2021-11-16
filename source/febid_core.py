###################################################################
#
#  FEBID Simulation
#
#  Version 0.9
#
####################################################################
import ast
import datetime
import math
import os
import sys
import warnings
import timeit
from contextlib import suppress
from tkinter import filedialog as fd

import line_profiler
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
import pyvista as pv
import scipy.constants as scpc
import yaml
# import ipyvolume as ipv
from numexpr_mod import evaluate_cached, cache_expression
from numpy import zeros, copy, s_
from tqdm import tqdm

from Process import Structure, Process
import VTK_Rendering as vr
import etraj3d, etrajectory, etrajmap3d
from libraries.rolling import roll

# It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus able produce deposit under irradiation.

# Semi-surface cells are cells that have precursor density but do not have a neighboring deposit cell
# Thus concept serves an alternative diffusion channel



def initialize_framework(from_file=False):
    """
    Prepare simulation framework and parameters from input configuration files
    :param from_file: load structure from vtk file
    :return:
    """
    precursor = yaml.load(fd.askopenfilename(), Loader=yaml.Loader ) #open(f'{sys.path[0]}{os.sep}Me3PtCpMe.yml', 'r'), Loader=yaml.Loader)  # Precursor and substrate properties(substrate here is the top layer)
    settings = yaml.load(fd.askopenfilename(), Loader=yaml.Loader) # (open(f'{sys.path[0]}{os.sep}Parameters.yml', 'r'),Loader=yaml.Loader)  # Parameters of the beam, dwell time and precursor flux
    sim_params = yaml.load(fd.askopenfilename(), Loader=yaml.Loader) # (open(f'{sys.path[0]}{os.sep}Simulation.yml', 'r'),Loader=yaml.Loader)  # Size of the chamber, cell size and time step
    mc_config, equation_values, timings, nr = buffer_constants(precursor, settings, sim_params)

    structure = Structure()
    if from_file:
        vtk_file = fd.askopenfilename()  # '/Users/sandrik1742/Documents/PycharmProjects/FEBID/code/New Folder With Items/35171.372k_of_1000000.0_loops_k_gr4 15/02/42.vtk'#
        vtk_obj = pv.read(vtk_file)
        structure.load_from_vtk(vtk_obj=vtk_obj)
    else:
        structure.create_from_parameters(sim_params['cell_dimension'], sim_params['width'], sim_params['length'],
                              sim_params['height'], sim_params['substrate_height'], nr)

    return structure, mc_config, equation_values, timings

def run_mc_simulation(vtk_obj, E0=20, sigma=5, N=100, pos='center', material='Au', Emin=0.1):


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
    print(f'Monte-Carlo electron beam - matter interaction module.\n'
          f'First load desired structure from vtk file, then enter parameters.')
    print(f'Select .vtk file....')
    while True:
        try:
            file = '/Users/sandrik1742/Documents/PycharmProjects/FEBID/source_samples/electron_beam_monte-carlo/hockeystick.vti' # fd.askopenfilename()
            vtk_obj = pv.read(file)
        except Exception as e:
            print(f'Unable to read vtk file. {e.args} \n'
                  f'Try again:')
        else:
            print(f'Got file!\n')
            break

    print(f'Input parameters: Beam energy(keV), gauss st. deviation, number of electrons to emit, (beam x position, beam y position) , structure material(i.e. Au)\n'
          f'Note: \'center\' can be used instead of a coordinates pair (x,y) to set the beam to the center')
    # E0=read_param('Beam energy', [int, float])
    # sigma = read_param('Gauss standard deviation', [int, float])
    # N = read_param('Number of electrons', [int])
    # pos = read_param('Beam position', [tuple, str], check_string=['center'])
    # material = read_param('Structure material', [str], check_string=['Au'])
    print(f'Got paramers!\n')
    # run_mc_simulation(vtk_obj, E0, sigma, N, pos, material)
    run_mc_simulation(vtk_obj, 20, 15, 1000, (12,17), 'Au')



def read_param(name, expected_type, message="Inappropriate input for ", check_string=None):
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



if __name__== '__main__':
    print(f'##################### FEBID Simulator ###################### \n'
          f'Two functions are available: deposition process and Monte Carlo electron trajectory simulation\n'
          f'Type \'febid\' to enter deposition mode or \'mc\' for electron trajectory simulation:')
    mode_choice = 'mc' #input("Enter 'dp', 'mc' or press Enter to exit:")
    if mode_choice not in ['dp', 'mc']:
        sys.exit("Exit program.")
    if mode_choice == 'mc':
        mc_simulation()

    structure, mc_config, equation_values, timings = initialize_framework(True)
    process_obj = Process(structure, mc_config, equation_values, timings)
