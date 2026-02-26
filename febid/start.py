"""
Interface class for binding GUI and simulation together
"""
import os
import random
import warnings

import yaml
import pyvista as pv

import febid.simple_patterns as sp
from febid.Structure import Structure
from febid.monte_carlo import etraj3d as e3d
from febid.simulation_context import SimulationContext, SimulationParameters
from febid.febid_core import SimulationManager
from febid.logging_config import setup_logger
# Setup logger
logger = setup_logger(__name__)


class Starter:
    """
    This class serves as an interface between the interface and the simulation code.
    """
    def __init__(self, params: SimulationParameters = None):
        """Initialize starter with optional simulation-parameter dataclass.
        If params argument is not provided, it has to be set before starting the simulation.

        :param params: Simulation parameters object; default values are used when omitted.
        :type params: SimulationParameters
        :return: None
        """
        self._params: SimulationParameters = params or SimulationParameters()
        self.context = SimulationContext()
        self.dwell_time_units = 1E-6
        self.context.structure = Structure()
        self.simulation_manager: SimulationManager = None
        self.syncHelper = None


    def start(self):
        """
        Compile simulation parameters and start the simulation
        """
        self._params.validate()
        self._create_simulation_volume()
        self._create_printing_path()
        self._open_settings_and_precursor_params()
        self._run_febid_interface()
        return self.simulation_manager

    def start_mc(self, **kwargs):
        """Prepare context and execute Monte Carlo-only workflow.

        :param kwargs: Extra Monte Carlo runtime arguments.
        :type kwargs: dict
        :return: None
        """
        self._params.validate()
        self._create_simulation_volume()
        self.context.precursorParams  = self._open_precursor_params()
        self._run_mc_interface(**kwargs)

    def stop(self):
        """
        Stop the simulation
        """
        if not self.syncHelper:
            self.simulation_manager.stop()

    def _create_simulation_volume(self):
        """
        Create simulation volume
        """
        structure_sources = {
            'vtk': self._create_from_vtk,
            'geom': self._create_from_geometry,
            'auto': self._create_auto
        }
        structure_sources.get(self._params.structure_source, self._create_auto)()

    def _create_from_vtk(self):
        """
        Create simulation volume from VTK file
        """
        try:
            self.context.structure.load_from_vtk(pv.read(self._params.vtk_filename))
        except FileNotFoundError as e:
            msg = 'VTK file not specified. Please choose the file and try again.'
            logger.exception(msg)
            e.errno = 1
            e.args = (msg,)
            raise e
        except Exception as e:
            logger.exception(f'Unexpected error occurred while opening VTK file: {e.args}')
            raise e

    def _create_from_geometry(self):
        """
        Create simulation volume from geometry parameters
        """
        try:
            cell_size = int(self._params.cell_size)
            xdim = int(float(self._params.width)) // cell_size
            ydim = int(float(self._params.length)) // cell_size
            zdim = int(float(self._params.height)) // cell_size
            substrate_height = int(float(self._params.substrate_height)) // cell_size
            self.context.structure.create_from_parameters(cell_size, xdim, ydim, zdim, substrate_height)
        except KeyError as e:
            msg = (f'An error occurred while fetching geometry parameters for the simulation volume. \n '
                   f'Check values and try again.')
            logger.exception(msg)
            e.errno = 2
            e.args = (msg,)
            raise
        except Exception as e:
            msg = 'Unexpected error occurred while creating simulation volume from geometry parameters.'
            logger.exception(msg)
            raise
    def _create_auto(self):
        """
        Set only cell size and substrate height and leave simulation volume creation for later
        """
        cell_size = int(self._params.cell_size)
        substrate_height = int(float(self._params.substrate_height)) // cell_size
        self.context.structure.cell_size = cell_size
        self.context.structure.substrate_height = substrate_height

    def _create_printing_path(self):
        """
        Create printing path either from simple shape or stream file
        """
        if self._params.pattern_source == 'simple':
            self.context.printingPath = self._create_from_simple_shape()
        if self._params.pattern_source == 'stream_file':
            self.context.printingPath = self._create_from_stream_file()

    def _create_from_simple_shape(self):
        """
        Create printing path from simple shape
        """
        try:
            if self._params.structure_source == 'auto':
                raise AttributeError('Not allowed to choose \'simple  pattern\' together with \'auto\'.')
            pattern = self._params.pattern
            p1 = float(self._params.param1)
            p2 = float(self._params.param2) if pattern in ['Point', 'Rectangle', 'Square'] else 0
            dwell_time = float(self._params.dwell_time) * self.dwell_time_units
            pitch = float(self._params.pitch)
            repeats = int(float(self._params.repeats))
            x = self.context.structure.shape[2] // 2 * self.context.structure.cell_size
            y = self.context.structure.shape[1] // 2 * self.context.structure.cell_size
            if pattern == 'Point':
                x, y = p1, p2
            printing_path = sp.generate_pattern(pattern, repeats, dwell_time, x, y, (p1, p2), pitch)
            return printing_path
        except AttributeError as e:
            msg = 'An error occurred while creating simple printing path.'
            logger.exception(msg)
            e.errno = 3
            raise
        except Exception as e:
            msg = 'Unexpected error occurred while creating simple printing path.'
            logger.exception(msg)
            raise

    def _create_from_stream_file(self):
        """
        Create printing path from stream file
        """
        try:
            hfw = self._params.hfw
            printing_path, shape = sp.open_stream_file(self._params.stream_file_filename, hfw, collapse=True)
            if self._params.structure_source != 'auto':
                if printing_path[:, 0].max() > self.context.structure.xdim_abs or printing_path[:,
                                                                          1].max() > self.context.structure.ydim_abs:
                    raise ValueError('Printing path is out of simulation volume', f'Required dimensions: {shape[2]} * {shape[1]}')
                return printing_path
            shape = shape[::-1] // self.context.structure.cell_size
            self.context.structure.create_from_parameters(self.context.structure.cell_size, *shape, self.context.structure.substrate_height)
            return printing_path
        except FileNotFoundError as e:
            msg = 'User input error: stream file not specified or not found. Please choose the file and try again.'
            logger.exception(msg)
            e.errno = 4
            e.args = ('Stream file not specified. Please choose the file and try again.',)
            raise
        except ValueError as e:
            logger.error('User input error: printing path is out of simulation volume', exc_info=True)
            e.errno = 5
            raise

    def _open_settings_and_precursor_params(self):
        """
        Open settings and precursor parameters
        """
        self.context.settings = self._open_settings()
        self.context.precursorParams  = self._open_precursor_params()

    def _open_settings(self):
        """
        Open settings from a file
        """
        try:
            with open(self._params.settings_filename, mode='rb') as f:
                settings = yaml.load(f, Loader=yaml.FullLoader)
            factor = settings.get('deposition_scaling', 1)
            if factor:
                self.context.printingPath[:, 2] /= factor
            return settings
        except FileNotFoundError as e:
            msg = 'User input error: beam parameters file not specified or not found.'
            logger.exception(msg)
            e.errno = 6
            e.args = ('Beam parameters file not specified. Please choose the file and try again.',)
            raise
        except Exception as e:
            msg = 'Unexpected error occurred while opening a settings file.'
            logger.exception(msg)
            raise

    def _open_precursor_params(self):
        """
        Open precursor parameters from a file
        """
        try:
            with open(self._params.precursor_filename, mode='rb') as f:
                precursor_params = yaml.load(f, Loader=yaml.FullLoader)
            return precursor_params
        except FileNotFoundError as e:
            msg = 'User input error: precursor parameters file not specified or not found.'
            logger.exception(msg)
            e.errno = 7
            e.args = ('Precursor parameters file not specified. Please choose the file and try again.',)
            raise
        except Exception as e:
            msg = 'Unexpected error occurred while opening a precursor parameters file.'
            logger.exception(msg)
            raise

    def _run_febid_interface(self):
        """
        Compile simulation parameters and start the simulation
        """
        sim_volume_params = self.get_simulation_volume_parameters()
        saving_params = {'gather_stats': False, 'gather_stats_interval': 1000, 'save_snapshot': False, 'save_snapshot_interval': 1000,
                         'filename': ''}
        flag1, flag2 = self._params.save_simulation_data, self._params.save_structure_snapshot
        gather_stats = saving_params['gather_stats'] = self._params.save_simulation_data
        if gather_stats:
            saving_params['gather_stats_interval'] = float(self._params.simulation_data_interval)
        save_snapshot = saving_params['save_snapshot'] = self._params.save_structure_snapshot
        if save_snapshot:
            saving_params['save_snapshot_interval'] = float(self._params.structure_snapshot_interval)
        flag1, flag2 = self._params.save_simulation_data, self._params.save_structure_snapshot
        if flag1 or flag2:
            try:
                if not self._params.unique_name:
                    random_name = 'simulation_' + str(random.randint(10000, 99999))
                    self._params.unique_name = random_name
                    logger.warning(f'Unique name is not specified. Setting random name: {random_name}')
                    raise AttributeError('Unique name is not specified')
            except AttributeError as e:
                e.errno = 8
                raise
            saving_params['filename'] = os.path.join(self._params.save_directory, self._params.unique_name)
            try:
                os.makedirs(saving_params['filename'])
            except FileExistsError as e:
                logger.info('Saving directory already exists. Using it for saving simulation data.')
            saving_params['filename'] = os.path.join(saving_params['filename'], self._params.unique_name)
        if not (flag1 or flag2):
            logger.warn('No simulation information is saved. Neither saving statistics, nor snapshots.')
        temperature_tracking = self._params.temperature_tracking
        try:
            if not self.check_for_temperature_tracking_consistency():
                msg = ('Temperature tracking cannot be enabled without specifying all of the following parameters: '
                                 'Desorption activation energy, Desorption attempt frequency, Diffusion activation energy, '
                                 'Diffusion prefactor, Thermal conductivity')
                logger.error(msg, exc_info=True)
                raise AttributeError(msg)
        except AttributeError as e:
            e.errno = 9
            raise e
        gpu_param = self._params.gpu
        gpu_flag = (4,0) if gpu_param else False

        self.context.savingParams = saving_params
        self.context.simParams = sim_volume_params
        self.context.temperatureTracking = temperature_tracking
        self.context.device = gpu_flag

        self.simulation_manager = SimulationManager(self.context)
        self.simulation_manager.initialize()
        self.simulation_manager.run()

        self.syncHelper = self.simulation_manager.syncHelper

    def _run_mc_interface(self, **kwargs):
        """
        Run Monte Carlo simulation
        """
        sim_volume_params = self.get_simulation_volume_parameters()
        return e3d.run_mc_simulation(self.context.structure, precursor=self.context.precursorParams , **kwargs)

    def get_simulation_volume_parameters(self):
        """Return physical simulation-volume parameters derived from the structure.

        :return: Dictionary with width, length, height, cell size, and substrate height.
        """
        sim_volume_params = {
            'width': self.context.structure.shape_abs[2],
            'length': self.context.structure.shape_abs[1],
            'height': self.context.structure.shape_abs[0],
            'cell_size': self.context.structure.cell_size,
            'substrate_height': self.context.structure.substrate_height_abs
        }
        return sim_volume_params

    def check_for_temperature_tracking_consistency(self):
        """Verify that required precursor keys exist when temperature tracking is enabled.

        :return: True when configuration is consistent, otherwise False.
        """
        if self._params.temperature_tracking:
            precursor_params = self.context.precursorParams 
            keys = ['desorption_activation_energy', 'desorption_attempt_frequency',
                    'diffusion_activation_energy', 'diffusion_prefactor',
                    'thermal_conductivity']
            keys_present = all(key in precursor_params for key in keys)
            if not keys_present:
                return False
        return True

    @property
    def params(self) -> SimulationParameters:
        """Return active simulation parameters.

        :return: Current simulation-parameter dataclass.
        """
        return self._params

    @params.setter
    def params(self, params: SimulationParameters):
        """Replace active simulation parameters.

        :param params: New simulation-parameter dataclass.
        :type params: SimulationParameters
        :return: None
        """
        self._params = params


