"""
Scripting template for running series of simulations
"""
import os
import re

import numpy as np
import yaml
from ruamel.yaml import YAML
import pyvista as pv

from febid.ui import ui_shell
import febid.simple_patterns as sp
from febid import febid_core
from febid.Structure import Structure


def start_ui(config_f=None):
    ui_shell.start(config_f)


def start_no_ui(config_f=None):
    if not config_f:
        config_f = input('Specify configuration file:')

    try:
        with open(config_f, mode='rb') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print('An error occurred while opening configuration file.')
        print(e.args)
        return

    # Creating a simulation volume
    structure = Structure()
    if params['structure_source'] == 'vtk':  # opening from a .vtk file
        try:
            structure.load_from_vtk(pv.read(params['vtk_filename']))
        except Exception as e:
            print(f'Failed to open vtk.file: {e.args}')
            return
        cell_dimension = structure.cell_dimension
        substrate_height = structure.substrate_height
    if params['structure_source'] == 'geom':  # creating from geometry parameters
        try:
            cell_dimension = int(params['cell_size'])
            xdim = int(float(params['width'])) // cell_dimension  # array length
            ydim = int(float(params['length'])) // cell_dimension  # array length
            zdim = int(float(params['height'])) // cell_dimension  # array length
            substrate_height = int(float(params['substrate_height'])) // cell_dimension  # array length
            structure.create_from_parameters(cell_dimension, xdim, ydim, zdim, substrate_height)
        except Exception as e:
            print('An error occurred while fetching geometry parameters for the simulation volume. \n '
                  'Check values and try again.')
            print(e.args)
            return
    if params['structure_source'] == 'auto':  # defining it later based on a stream-file
        cell_dimension = int(params['cell_size'])
        substrate_height = int(float(params['substrate_height'])) // cell_dimension  # array length

        # Defining printing path
    dwell_time_units = 1E-6  # input units are in microseconds, internally seconds are used
    printing_path = None
    if params['pattern_source'] == 'simple':  # creating printing path based on the figure and parameters
        try:
            pattern = params['pattern']
            p1 = float(params['param1'])  # nm
            p2 = float(params['param1']) if pattern in ['Point', 'Rectangle', 'Square'] else 0  # nm
            dwell_time = float(params['dwell_time']) * dwell_time_units  # s
            pitch = float(params['pitch'])  # nm
            repeats = int(float(params['repeats']))
            x = structure.shape[2] // 2 * cell_dimension  # nm
            y = structure.shape[1] // 2 * cell_dimension  # nm
            if pattern == 'Point':
                x, y = p1, p2
            printing_path = sp.generate_pattern(pattern, repeats, dwell_time, x, y, (p1, p2), pitch)
        except Exception as e:
            print('Error occurred while creating a printing path. \n Check values and try again.')
            print(e.args)
            return
    if params['pattern_source'] == 'stream_file':  # importing printing path from stream_file
        try:
            hfw = params['hfw']
            printing_path, shape = sp.open_stream_file(params['stream_file_filename'], hfw, collapse=True)
        except Exception as e:
            print(f'Failed to open stream-file: {e.args}')
            return
        if params['structure_source'] == 'auto':
            shape = shape[::-1] // cell_dimension
            structure.create_from_parameters(cell_dimension, *shape, substrate_height)

    # Opening beam and precursor files
    try:
        with open(params['settings_filename'], mode='rb') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
        factor = settings.get('deposition_scaling', 1)
        if factor:
            printing_path[:, 2] /= factor
    except Exception as e:
        print(f'An error occurred while opening a settings file')
        print(e.args)
        return
    try:
        with open(params['precursor_filename'], mode='rb') as f:
            precursor_params = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print(f'An error occurred while opening a stream-file')
        print(e.args)
        return

    sim_volume_params = {
        'width': structure.shape[2],
        'length': structure.shape[1],
        'height': structure.shape[0],
        'cell_dimension': cell_dimension,
        'substrate_height': substrate_height}  # array length

    # Collecting parameters of file saving
    saving_params = {'monitoring': None, 'snapshot': None, 'filename': None}
    flag1, flag2 = params['save_simulation_data'], params['save_structure_snapshot']
    if flag1:
        saving_params['monitoring'] = float(params['simulation_data_interval'])
    if flag2:
        saving_params['snapshot'] = float(params['structure_snapshot_interval'])
    if flag1 or flag2:
        saving_params['filename'] = os.path.join(params['save_directory'], params['unique_name'])
        try:
            os.makedirs(saving_params['filename'])
        except FileExistsError as e:
            pass
        saving_params['filename'] = os.path.join(saving_params['filename'], params['unique_name'])

    temperature_tracking = params.get('temperature_tracking', False)

    rendering = {'show_process': params['show_process'], 'frame_rate': 0.5}
    # Starting the process
    process_obj, sim = febid_core.run_febid_interface(structure, precursor_params, settings, sim_volume_params,
                                                      printing_path, temperature_tracking, saving_params, rendering)

    return process_obj, sim


def start_default(config_f=None):
    start_ui(config_f)
    # start_no_ui(config_f)


def extr_number(text):
    r = re.split('(\d+)', text)
    return [atoi(c) for c in r]


def atoi(text):
    a = int(text) if text.isdigit() else text
    return a


def write_param(file, param_name, val):
    """
    Write a value to a parameter in a configuration file.

    :param file: path to configuration file
    :param param_name: name of the parameter
    :param val: value to write
    :return:
    """
    yml = YAML()
    with open(file, 'r+b') as f:
        params = yml.load(f)
    try:
        a = params[param_name]
    except KeyError:
        raise KeyError(f'Failed to overwrite parameter. The parameter not present in the file!')
    params[param_name] = val
    with open(file, mode='wb') as f:
        yml.dump(params, f)


def read_param(file, param_name):
    """
    Read a parameter value from a configuration file.

    :param file: path to configuration file
    :param param_name: name of the parameter
    :return: value of the parameter
    """
    yml = YAML()
    with open(file, 'r+b') as f:
        params = yml.load(f)
    try:
        return params[param_name]
    except KeyError:
        raise KeyError(f'Failed to read parameter. The parameter not present in the file!')


def scan_stream_files(session_file, directory):
    """
    Launch a series of simulations using multiple patterning files

    The files are named after the patterning file
    :param session_file: YAML file with session configuration
    :param directory: folder with stream files
    :return:
    """
    files_orig = os.listdir(directory)
    files_orig.sort(key=extr_number, reverse=True)
    files = [os.path.join(directory, f) for f in files_orig]
    init_stream = read_param(session_file, 'stream_file_filename')
    init_name = read_param(session_file, 'unique_name')
    for stream_file, name in zip(files, files_orig):
        write_param(session_file, 'stream_file_filename', stream_file)
        write_param(session_file, 'unique_name', name)
        start_no_ui(session_file)
    write_param(session_file, 'stream_file_filename', init_stream)
    write_param(session_file, 'unique_name', init_name)
    print(f'Successfully finished {len(files)} simulations with all {len(files_orig)} pattering files in {directory}')


def scan_settings(session_file, param_name, scan, base_name=''):
    """
    Launch a series of simulations by changing a single parameter

    :param session_file: YAML file with session configuration
    :param param_name: the name of the parameter, refer to settings and precursor parameters
    :param scan: a collection of values to use in consequent runs
    :param base_name: a common name for simulation files
    :return:
    """
    yml = YAML()
    # Looking for the parameter in settings and precursor parameters
    settings_file = read_param(session_file, 'settings_filename')
    precursor_params_file = read_param(session_file, 'precursor_filename')
    with open(settings_file, 'r+b') as s:
        settings: dict = yml.load(s)
        settings_keys = settings.keys()
    with open(precursor_params_file, 'r+b') as p:
        precursor_params = yml.load(p)
        precursor_keys = precursor_params.keys()
    if param_name in settings_keys:
        file = settings_file
    elif param_name in precursor_keys:
        file = precursor_params_file
    else:
        raise RuntimeError(f'Parameter {param_name} not found!')
    # Scanning
    initial_val = read_param(file, param_name)
    vals = np.asarray(scan)
    for i, val in enumerate(vals):
        write_param(file, param_name, val)
        name = base_name + '_' + param_name + '_' + 'scan' + '_' + f'{i:0>3d}'
        write_param(session_file, 'unique_name', name)
        start_no_ui(session_file)
    # Restoring initial state
    write_param(file, param_name, initial_val)
    print(
        f'Successfully finished {vals.shape[0]} simulations, scanning \'{param_name}\' from {vals.amin()} to {vals.max()}')


if __name__ == '__main__':
    # Here is an example of how to set up a several series of simulations, that change
    # various parameters for consequent runs for autonomous simulation execution.
    # It is advised to refrain from setting structure snapshot saving frequency below 0.01 s.
    # .vtk-files take up from 5 to 400 MB depending on structure size and resolution
    # and will very quickly occupy dozens of GB of space.

    # Initially, a session configuration file has to be specified.
    # This file, along settings and precursor parameters files specified in it, is to be modified
    # and then used to run a simulation. This routine is repeated until the desired parameter
    # has taken a given number of values.
    # The routine only changes a single parameter. All other parameters have to be preset forehand.
    session_file = '/home/kuprava/simulations/last_session.yml'

    # The first parameter change or scan modifies the Gaussian deviation parameter of the beam.
    # The file that will be modified in this case is the settings file.
    # Set up a folder (it will be created automatically) for simulation save files
    directory = '/home/kuprava/simulations/gauss_dev_scan/'
    write_param(session_file, 'save_directory', directory)
    # Specify parameter name
    param = 'gauss_dev'
    # Specify values that the parameter will take during consequent simulations
    vals = [2, 3, 4, 5, 6, 7, 8]
    # Launch the scan
    scan_settings(session_file, param, vals, 'hs')
    # Files that are saved during the simulation are named after the specified common name (here i.e. 'hs')
    # and the parameter name.

    # The second parameter scan modifies the thermal conductivity of the deposit.
    # The routine is the same as in the example above, although the file that will be
    # modified is the precursor parameters file.
    directory = '/home/kuprava/simulations/gauss_dev_scan/'
    write_param(session_file, 'save_directory', directory)
    param = 'thermal_conductivity'
    vals = np.arange(2e-10, 11e-10, 2e-10)  # [2e-10, 4e-10, 6e-10, 8e-10, 10e-10]
    scan_settings(session_file, param, vals, 'hs')

    directory = '/home/kuprava/simulations/ads.act.energy_scan/'
    write_param(session_file, 'save_directory', directory)
    param = 'desorption_activation_energy'
    vals = [0.67, 0.64, 0.61, 0.58, 0.55, 0.52, 0.49]
    scan_settings(session_file, param, vals, 'hs')

    # The third series runs simulations using several patterning files.
    # Again, specify a desired location for simulation save files
    directory = '/home/kuprava/simulations/longs/'
    # Optionally, an initial structure can be specified. This will 'continue' deposition
    # onto a structure obtained in one of the earlier simulations.
    # It can be used i.e. when all planned structures share a same initial feature such as a pillar.
    # Keep in mind that it can be used only for patterning files with the same patterning area.
    # To that, the patterning area must correspond to one that is defined by the simulation for the current
    # pattern including margins.
    initial_structure = '/home/kuprava/simulations/hockey_stick_therm_050_5_01_15:12:31.vtk'
    write_param(session_file, 'structure_source', 'vtk')
    write_param(session_file, 'vtk_filename', initial_structure)
    write_param(session_file, 'save_directory', directory)
    # Specifying a folder with patterning files
    stream_files = '/home/kuprava/simulations/steam_files_long_s'
    # Launching the series
    scan_stream_files(session_file, stream_files)
