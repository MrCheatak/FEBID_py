"""
Interface class for binding GUI and simulation together
"""
import os
import random
import warnings

import yaml
import pyvista as pv

import febid.simple_patterns as sp
from febid import febid_core
from febid.Structure import Structure
from febid.monte_carlo import etraj3d as e3d


class Starter:
    """
    This class serves as an interface between the interface and the simulation code.
    """
    def __init__(self, params=None):
        self._params = params
        self.structure = Structure()
        self.dwell_time_units = 1E-6
        self.printing_path = None
        self.settings = None
        self.precursor_params = None
        self.process_obj = None
        self.sim = None

    def start(self):
        """
        Compile simulation parameters and start the simulation
        """
        self._create_simulation_volume()
        self._create_printing_path()
        self._open_settings_and_precursor_params()
        self._run_febid_interface()
        return self.process_obj, self.sim

    def start_mc(self, **kwargs):
        self._create_simulation_volume()
        self.precursor_params = self._open_precursor_params()
        self._run_mc_interface(**kwargs)

    def _create_simulation_volume(self):
        """
        Create simulation volume
        """
        structure_sources = {
            'vtk': self._create_from_vtk,
            'geom': self._create_from_geometry,
            'auto': self._create_auto
        }
        structure_sources.get(self._params['structure_source'], self._create_auto)()

    def _create_from_vtk(self):
        """
        Create simulation volume from VTK file
        """
        try:
            self.structure.load_from_vtk(pv.read(self._params['vtk_filename']))
        except FileNotFoundError as e:
            e.errno = 1
            e.args = ('VTK file not specified. Please choose the file and try again.',)
            raise e
        except Exception as e:
            print(f'Unexpected error occurred while opening VTK file: {e.args}')
            raise e

    def _create_from_geometry(self):
        """
        Create simulation volume from geometry parameters
        """
        try:
            cell_size = int(self._params['cell_size'])
            xdim = int(float(self._params['width'])) // cell_size
            ydim = int(float(self._params['length'])) // cell_size
            zdim = int(float(self._params['height'])) // cell_size
            substrate_height = int(float(self._params['substrate_height'])) // cell_size
            self.structure.create_from_parameters(cell_size, xdim, ydim, zdim, substrate_height)
        except KeyError as e:
            e.errno = 2
            e.args = ('An error occurred while fetching geometry parameters for the simulation volume. \n '
                      'Check values and try again.',)
            raise e
        except Exception as e:
            print(f'Unexpected error occurred while creating simulation volume: {e.args}')
            raise e

    def _create_auto(self):
        """
        Set only cell size and substrate height and leave simulation volume creation for later
        """
        cell_size = int(self._params['cell_size'])
        substrate_height = int(float(self._params['substrate_height'])) // cell_size
        self.structure.cell_size = cell_size
        self.structure.substrate_height = substrate_height

    def _create_printing_path(self):
        """
        Create printing path either from simple shape or stream file
        """
        if self._params['pattern_source'] == 'simple':
            self.printing_path = self._create_from_simple_shape()
        if self._params['pattern_source'] == 'stream_file':
            self.printing_path = self._create_from_stream_file()

    def _create_from_simple_shape(self):
        """
        Create printing path from simple shape
        """
        try:
            if self._params['structure_source'] == 'auto':
                raise AttributeError('Not allowed to choose \'simple  pattern\' together with \'auto\'.')
            pattern = self._params['pattern']
            p1 = float(self._params['param1'])
            p2 = float(self._params['param1']) if pattern in ['Point', 'Rectangle', 'Square'] else 0
            dwell_time = float(self._params['dwell_time']) * self.dwell_time_units
            pitch = float(self._params['pitch'])
            repeats = int(float(self._params['repeats']))
            x = self.structure.shape[2] // 2 * self.structure.cell_size
            y = self.structure.shape[1] // 2 * self.structure.cell_size
            if pattern == 'Point':
                x, y = p1, p2
            printing_path = sp.generate_pattern(pattern, repeats, dwell_time, x, y, (p1, p2), pitch)
            return printing_path
        except AttributeError as e:
            e.errno = 3
            raise e
        except Exception as e:
            print(f'Unexpected error occurred while creating simple printing path: {e.args}')
            raise e

    def _create_from_stream_file(self):
        """
        Create printing path from stream file
        """
        try:
            hfw = self._params['hfw']
            printing_path, shape = sp.open_stream_file(self._params['stream_file_filename'], hfw, collapse=True)
            if self._params['structure_source'] != 'auto':
                if printing_path[:, 0].max() > self.structure.xdim_abs or printing_path[:,
                                                                          1].max() > self.structure.ydim_abs:
                    raise ValueError('Printing path is out of simulation volume', f'Required dimensions: {shape[2]} * {shape[1]}')
                return printing_path
            shape = shape[::-1] // self.structure.cell_size
            self.structure.create_from_parameters(self.structure.cell_size, *shape, self.structure.substrate_height)
            return printing_path
        except FileNotFoundError as e:
            e.errno = 4
            e.args = ('Stream file not specified. Please choose the file and try again.',)
            raise e
        except ValueError as e:
            e.errno = 5
            raise e
        except Exception as e:
            print(f'Unexpected error occurred while opening a stream-file: {e.args}')
            raise e

    def _open_settings_and_precursor_params(self):
        """
        Open settings and precursor parameters
        """
        self.settings = self._open_settings()
        self.precursor_params = self._open_precursor_params()

    def _open_settings(self):
        """
        Open settings from a file
        """
        try:
            with open(self._params['settings_filename'], mode='rb') as f:
                settings = yaml.load(f, Loader=yaml.FullLoader)
            factor = settings.get('deposition_scaling', 1)
            if factor:
                self.printing_path[:, 2] /= factor
            return settings
        except FileNotFoundError as e:
            e.errno = 6
            e.args = ('Beam parameters file not specified. Please choose the file and try again.',)
            raise e
        except Exception as e:
            print(f'Unexpected error occurred while opening a settings file')
            print(e.args)
            raise e

    def _open_precursor_params(self):
        """
        Open precursor parameters from a file
        """
        try:
            with open(self._params['precursor_filename'], mode='rb') as f:
                precursor_params = yaml.load(f, Loader=yaml.FullLoader)
            return precursor_params
        except FileNotFoundError as e:
            e.errno = 7
            e.args = ('Precursor parameters file not specified. Please choose the file and try again.',)
            raise e
        except Exception as e:
            print(f'Unexpected error occurred while opening a precursor parameters file')
            print(e.args)
            raise e

    def _run_febid_interface(self):
        """
        Compile simulation parameters and start the simulation
        """
        sim_volume_params = self.get_simulation_volume_parameters()
        saving_params = {'gather_stats': False, 'gather_stats_interval': None, 'save_snapshot': False, 'save_snapshot_interval': None,
                         'filename': ''}
        flag1, flag2 = self._params['save_simulation_data'], self._params['save_structure_snapshot']
        gather_stats = saving_params['gather_stats'] = self._params['save_simulation_data']
        if gather_stats:
            saving_params['gather_stats_interval'] = float(self._params['simulation_data_interval'])
        save_snapshot = saving_params['save_snapshot'] = self._params['save_structure_snapshot']
        if save_snapshot:
            saving_params['save_snapshot_interval'] = float(self._params['structure_snapshot_interval'])
        flag1, flag2, flag3 = self._params['save_simulation_data'], self._params['save_structure_snapshot'], self._params['show_process']
        if flag1 or flag2:
            try:
                if not self._params['unique_name']:
                    random_name = 'simulation_' + str(random.randint(10000, 99999))
                    self._params['unique_name'] = random_name
                    raise AttributeError('Unique name is not specified')
            except AttributeError as e:
                e.errno = 8
                raise e
            saving_params['filename'] = os.path.join(self._params['save_directory'], self._params['unique_name'])
            try:
                os.makedirs(saving_params['filename'])
            except FileExistsError as e:
                print('Saving directory already exists.')
            saving_params['filename'] = os.path.join(saving_params['filename'], self._params['unique_name'])
        if not (flag1 or flag2 or flag3):
            warnings.warn('No simulation information is interfaced. Neither saving statistics, nor snapshots, nor showing process.')
        temperature_tracking = self._params.get('temperature_tracking', False)
        try:
            if not self.check_for_temperature_tracking_consistency():
                raise AttributeError('Temperature tracking cannot be enabled without specifying all of the following parameters: '
                                 'Desorption activation energy, Desorption attempt frequency, Diffusion activation energy, '
                                 'Diffusion prefactor, Thermal conductivity')
        except AttributeError as e:
            e.errno = 9
            raise e
        rendering = {'show_process': self._params['show_process'], 'frame_rate': 0.5}
        self.process_obj, self.sim = febid_core.run_febid_interface(self.structure, self.precursor_params,
                                                                    self.settings,
                                                                    sim_volume_params, self.printing_path,
                                                                    temperature_tracking, saving_params, rendering)

    def _run_mc_interface(self, **kwargs):
        """
        Run Monte Carlo simulation
        """
        sim_volume_params = self.get_simulation_volume_parameters()
        return e3d.run_mc_simulation(self.structure, precursor=self.precursor_params, **kwargs)

    def get_simulation_volume_parameters(self):
        sim_volume_params = {
            'width': self.structure.shape_abs[2],
            'length': self.structure.shape_abs[1],
            'height': self.structure.shape_abs[0],
            'cell_size': self.structure.cell_size,
            'substrate_height': self.structure.substrate_height_abs
        }
        return sim_volume_params

    def check_for_temperature_tracking_consistency(self):
        if self._params['temperature_tracking']:
            precursor_params = self.precursor_params
            keys = ['desorption_activation_energy', 'desorption_attempt_frequency',
                    'diffusion_activation_energy', 'diffusion_prefactor',
                    'thermal_conductivity']
            keys_present = all(key in precursor_params for key in keys)
            if not keys_present:
                return False
        return True

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params


