"""
A Collection of utils for launching series of simulations
"""
import os
import re

import numpy as np
from ruamel.yaml import YAML

from febid.__main__ import start_no_ui


def extr_number(text):
    """
    Extract numbers from a string.

    :param text: string
    :return: list of numbers
    """
    r = re.split('(\d+)', text)
    return [atoi(c) for c in r]


def atoi(text):
    """
    Convert a string to an integer.

    :param text: string
    :return: integer
    """
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
