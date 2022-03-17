import os

import yaml
import pyvista as pv

from ui import ui_shell
import febid_core
from Structure import Structure
import simple_patterns as sp

def start_ui():
    ui_shell.start()

def start_no_ui(config_f=None):
    if not config_f:
        config_f = input('Specify configuration file:')

    try:
        params = yaml.load(open(config_f, 'r'), Loader=yaml.Loader)
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
            printing_path, shape = sp.open_stream_file(params['stream_file_filename'])
        except Exception as e:
            print(f'Failed to open stream-file: {e.args}')
            return
        shape = shape[::-1] // cell_dimension
        structure.create_from_parameters(cell_dimension, *shape, substrate_height)

    # Opening beam and precursor files
    try:
        beam_params = yaml.load(open(params['settings_filename']), Loader=yaml.Loader)
        factor = beam_params.get('deposition_scaling', 1)
        if factor:
            printing_path[:, 2] /= factor
    except Exception as e:
        print(f'An error occurred while opening a settings file')
        print(e.args)
        return
    try:
        precursor_params = yaml.load(open(params['precursor_filename'], 'r', encoding='UTF-8'), Loader=yaml.Loader)
    except Exception as e:
        print(f'An error occurred while opening a stream-file')
        print(e.args)
        return

    sim_volume_params = {}  # array length
    sim_volume_params['width'] = structure.shape[2]
    sim_volume_params['length'] = structure.shape[1]
    sim_volume_params['height'] = structure.shape[0]
    sim_volume_params['cell_dimension'] = cell_dimension
    sim_volume_params['substrate_height'] = substrate_height

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

    rendering = {'show_process': params['show_process'], 'frame_rate': 0.2}
    # Starting the process
    febid_core.run_febid_interface(structure, precursor_params, beam_params, sim_volume_params, printing_path,
                                   saving_params, rendering)

    return

if __name__ == '__main__':
    start_no_ui('/home/kuprava/febid/source/last_session.yml')
    # start_ui()



