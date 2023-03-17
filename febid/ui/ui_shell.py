import math
import os, sys
import random
from typing import Union
from contextlib import suppress
import faulthandler

faulthandler.enable(file=sys.stderr)

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import QtWidgets, QtGui

from febid.ui.main_window import Ui_MainWindow as UI_MainPanel

import pyvista as pv
import yaml
from ruamel.yaml import YAML

from febid import febid_core, simple_patterns as sp
from febid.Structure import Structure
from febid.monte_carlo import etraj3d as e3d
from febid.libraries.vtk_rendering.VTK_Rendering import read_field_data


class UI_Group(set):
    """
    A collection of UI elements.
    """
    def __init__(self, *args):
        super().__init__()
        if type(args[0]) is set:
            self.update(args[0])
        else:
            for arg in args:
                self.add(arg)

    def disable(self):
        """
        Disable UI element.

        :return:
        """
        for element in self:
            element.setEnabled(False)

    def enable(self):
        """
        Enable UI element.
        :return:
        """
        for element in self:
            element.setEnabled(True)


class MainPannel(QMainWindow, UI_MainPanel):
    def __init__(self, config_filename=None, parent=None):
        super().__init__(parent)
        self.initialized = False
        self.setupUi(self)
        self.show()
        self.tab_switched(self.tabWidget.currentIndex())
        self.__prepare()
        # Parameters
        if config_filename is not None:
            self.last_session_filename = config_filename
        else:
            self.last_session_filename = 'last_session.yml'
        self.session = None
        self.save_flag = False
        self.structure_source = 'vtk'  # vtk, geom or auto
        self.pattern_source = 'simple'  # simple or stream_file
        self.pattern = 'Point'
        self.vtk_filename = ''
        self.geom_parameters_filename = ''
        self.stream_file_filename = ''
        self.settings_filename = ''
        self.precursor_parameters_filename = ''
        self.temperature_tracking = False
        self.save_directory = ''
        self.show_process = False
        self.cam_pos = None

        self.open_last_session()
        self.initialized = True

    def open_last_session(self, filename=''):
        """
        Open last session settings from the file or create a new file.

        File is saved in the current working directory.

        :param filename: name of the file
        :return:
        """
        if not filename:
            filename = self.last_session_filename
        print('Trying to load last session...', end='')
        if os.path.exists(filename):
            self.open_new_session(True, filename)
        else:
            yml = YAML()
            input_stub = ''.join(self.last_session_stub)
            self.session = yml.load(input_stub)
            self.__gather_session_config()
        print('done!')

    def open_new_session(self, triggered, file=''):
        """
        Load a session from a config file.

        :param triggered:
        :return:
        """
        if not file:
            file, _ = QtWidgets.QFileDialog.getOpenFileName()
        if not file:
            return
        try:
            with open(file, 'r+b') as f:
                yml = YAML()
                self.session = params = yml.load(f)
                input = ''.join(self.last_session_stub)
                session = yml.load(input)
                if params.keys() != session.keys():
                    raise ImportError('Failed to open session file.')
        except ImportError as e:
            self.view_message('Session file', *e.args)
            return
        except Exception as e:
            raise e
        with open(file, 'r+b') as f:
            self.__load_config(self.session)
            self.last_session_filename = f.name
            self.checkbox_load_last_session.setToolTip(f.name)

    def change_state_load_last_session(self, param=None):
        switch = True if param else False
        self.checkbox_load_last_session.setChecked(switch)
        self.save_flag = switch
        if switch and self.initialized:
            self.open_last_session()
        self.save_parameter('load_last_session', switch)

    def vtk_chosen(self):
        self.structure_source = 'vtk'
        # Changing FEBID tab interface
        self.choice_vtk_file.setChecked(True)
        self.ui_sim_volume.disable()
        self.ui_vtk_choice.enable()

        # Changing MC tab interface
        self.choice_vtk_file_mc.setChecked(True)
        self.ui_sim_volume_mc.disable()
        self.ui_vtk_choice_mc.enable()

        self.save_parameter('structure_source', self.structure_source)

    def geom_parameters_chosen(self):
        self.structure_source = 'geom'
        # Changing FEBID tab interface
        self.choice_geom_parameters_file.setChecked(True)
        self.ui_sim_volume.disable()
        self.ui_geom_choice.enable()

        # Changing MC tab interface
        self.choice_geom_parameters_file_mc.setChecked(True)
        self.ui_sim_volume_mc.disable()
        self.ui_geom_choice_mc.enable()

        # Auto-switching to Simple patterns option
        self.choice_simple_pattern.setChecked(True)
        self.simple_pattern_chosen()

        self.save_parameter('structure_source', self.structure_source)

    def auto_chosen(self):
        self.structure_source = 'auto'
        # Changing FEBID tab interface
        self.choice_auto.setChecked(True)
        self.ui_sim_volume.disable()
        self.ui_auto_choice.enable()

        # Changing MC tab interface
        self.choice_geom_parameters_file_mc.setAutoExclusive(False)
        self.choice_geom_parameters_file_mc.setChecked(False)
        self.choice_geom_parameters_file_mc.setAutoExclusive(True)
        self.choice_vtk_file_mc.setAutoExclusive(False)
        self.choice_vtk_file_mc.setChecked(False)
        self.choice_vtk_file_mc.setAutoExclusive(True)

        # Auto-switching to Stream-file option
        self.choice_stream_file.setChecked(True)
        self.stream_file_chosen()

        self.save_parameter('structure_source', self.structure_source)

    def simple_pattern_chosen(self):
        self.pattern_source = 'simple'
        # Changing FEBID tab interface
        self.choice_simple_pattern.setChecked(True)
        self.ui_pattern.disable()
        self.ui_simple_patterns.enable()

        self.save_parameter('pattern_source', self.pattern_source)

    def stream_file_chosen(self):
        self.pattern_source = 'stream_file'
        # Changing FEBID tab interface
        self.choice_stream_file.setChecked(True)
        self.ui_pattern.disable()
        self.ui_stream_file.enable()

        self.save_parameter('pattern_source', self.pattern_source)

    def pattern_selection_changed(self, current=''):
        if current == 'Point':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.enable()
        if current == 'Line':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.disable()
        if current == 'Rectangle':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.enable()
        if current == 'Square':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.disable()
        if current == 'Triangle':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.disable()
        if current == 'Circle':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.disable()
        self.pattern = current
        self.save_parameter('pattern', current)

    def open_vtk_file(self, file=''):
        # For both tabs:
        #  Check if the specified file is a valid .vtk file
        #  Insert parameters into fields
        if not file:
            file, _ = QtWidgets.QFileDialog.getOpenFileName()
            if not file:
                return
        try:
            deposit = Structure()
            vtk_obj = pv.read(file)
            deposit.load_from_vtk(vtk_obj)
            structure = deposit.deposit
            cell_dim = deposit.cell_dimension
            zdim, ydim, xdim = [str(int(dim * cell_dim)) for dim in list(structure.shape)]
            cell_dim = str(int(cell_dim))
            try:
                substrate_height = str(deposit.substrate_height * deposit.cell_dimension)
            except:
                substrate_height = 'nan'
            self.vtk_filename = file
            self.input_width.setText(xdim)
            self.input_width_mc.setText(xdim)
            self.input_length.setText(ydim)
            self.input_length_mc.setText(ydim)
            self.input_height.setText(zdim)
            self.input_height_mc.setText(zdim)
            self.input_cell_size.setText(cell_dim)
            self.input_cell_size_mc.setText(cell_dim)
            self.input_substrate_height.setText(substrate_height)
            self.input_substrate_height_mc.setText(substrate_height)
            self.vtk_filename_display.setText(file)
            self.vtk_filename_display_mc.setText(file)
            params = read_field_data(vtk_obj)
            if params[2] is not None:
                self.x_pos.setText(str(params[2][0]))
                self.y_pos.setText(str(params[2][1]))
            self.save_parameter('vtk_filename', file)
        except Exception as e:
            self.view_message('File read error',
                              'Specified file is not a valid VTK file. Please choose a valid .vtk file.')
            print("Was unable to open .vtk file. Following error occurred:")
            print(e.args)

    def open_geom_parameters_file(self, file=''):
        if not file:
            file, _ = QtWidgets.QFileDialog.getOpenFileName()
        if not file:
            return
        try:
            with open(file, mode='rb') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
            cell_dim = str(params['cell_dimension'])
            xdim = str(params['width'])
            ydim = str(params['length'])
            zdim = str(params['height'])
            substrate_height = str(params['substrate_height'])
            self.geom_parameters_filename = file
            self.input_width.setText(xdim)
            self.input_width_mc.setText(xdim)
            self.input_length.setText(ydim)
            self.input_length_mc.setText(ydim)
            self.input_height.setText(zdim)
            self.input_height_mc.setText(zdim)
            self.input_cell_size.setText(cell_dim)
            self.input_cell_size_mc.setText(cell_dim)
            self.input_substrate_height.setText(substrate_height)
            self.input_substrate_height_mc.setText(substrate_height)
            self.save_parameter('geom_parameters_filename', file)
        except Exception as e:
            print("Was unable to open .yml geometry parameters file. Following error occurred:")
            print(e.args)

    def open_stream_file(self, file=''):
        if not file:
            file, _ = QtWidgets.QFileDialog.getOpenFileName()
            if not file:
                return
        self.stream_file_filename = file
        self.stream_file_filename_display.setText(file)
        self.save_parameter('stream_file_filename', file)

    def open_settings_file(self, file=''):
        if not file:
            file, _ = QtWidgets.QFileDialog.getOpenFileName()
            if not file:
                return
        ### Read and insert parameters in Monte Carlo tab
        try:
            with open(file, mode='rb') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
            self.beam_energy.setText(str(params['beam_energy']))
            self.energy_cutoff.setText(str(params['minimum_energy']))
            self.gauss_dev.setText(str(params['gauss_dev']))
            self.settings_filename_display.setText(file)
            self.settings_filename_display_mc.setText(file)
            self.settings_filename = file
            self.save_parameter('settings_filename', file)
        except Exception as e:
            print("Was unable to open .yaml beam parameters file. Following error occurred:")
            print(e.args)

    def open_precursor_parameters_file(self, file=''):
        if not file:
            file, _ = QtWidgets.QFileDialog.getOpenFileName()
            if not file:
                return
        try:
            with open(file, mode='rb') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
            self.precursor_parameters_filename_display.setText(file)
            self.precursor_parameters_filename_display_mc.setText(file)
            self.precursor_parameters_filename = file
            self.save_parameter('precursor_filename', file)
        except Exception as e:
            self.view_message('File read error',
                              'Specified file is not a valid parameters file. Please choose a valid .yml file.')
            print("Was unable to open .yml precursor parameters file. Following error occurred:")
            print(e.args)

    def change_state_save_sim_data(self, param=None):
        switch = True if param else False
        self.checkbox_save_simulation_data.setChecked(switch)
        if switch:
            self.ui_sim_data_interval.enable()
        else:
            self.ui_sim_data_interval.disable()
        if switch or self.checkbox_save_snapshots.isChecked():
            self.ui_unique_name.enable()
            self.ui_save_folder.enable()
        else:
            self.ui_unique_name.disable()
            self.ui_save_folder.disable()
        self.save_parameter('save_simulation_data', switch)

    def change_state_save_snapshots(self, param=None):
        switch = True if param else False
        self.checkbox_save_snapshots.setChecked(switch)
        if switch:
            self.ui_snapshot.enable()
        else:
            self.ui_snapshot.disable()
        if switch or self.checkbox_save_snapshots.isChecked():
            self.ui_unique_name.enable()
            self.ui_save_folder.enable()
        else:
            self.ui_unique_name.disable()
            self.ui_save_folder.disable()
        self.save_parameter('save_structure_snapshot', switch)

    def change_state_show_process(self, param=None):
        switch = True if param else False
        self.checkbox_show.setChecked(switch)
        self.show_process = switch
        self.save_parameter('show_process', switch)

    def change_state_temperature_tracking(self, param):
        switch = True if param else False
        self.checkbox_temperature_tracking.setChecked(switch)
        self.temperature_tracking = switch
        self.save_parameter('temperature_tracking', switch)

    def unique_name_changed(self):
        self.save_parameter('unique_name', self.input_unique_name.text())

    def open_save_directory(self, directory=''):
        if not directory:
            directory = QtWidgets.QFileDialog.getExistingDirectory()
            if not directory:
                return
        if directory:
            self.save_folder_display.setText(directory)
        self.save_directory = directory
        self.save_parameter('save_directory', directory)

    def tab_switched(self, current):
        if current == 0:
            self.resize(self.width(), 684)
        if current == 1:
            self.resize(self.width(), 500)

    def start_febid(self):
        # Creating a simulation volume
        structure = Structure()
        if self.structure_source == 'vtk':  # opening from a .vtk file
            try:
                structure.load_from_vtk(pv.read(self.vtk_filename))
            except:
                if not self.vtk_filename:
                    self.view_message('File not specified',
                                      'VTK file not specified. Please choose the file and try again.')
                else:
                    self.view_message(additional_message='VTK file not found')
                return
            cell_dimension = structure.cell_dimension
            substrate_height = structure.substrate_height
        if self.structure_source == 'geom':  # creating from geometry parameters
            try:
                cell_dimension = int(self.input_cell_size.text())
                xdim = int(float(self.input_width.text())) // cell_dimension  # array length
                ydim = int(float(self.input_length.text())) // cell_dimension  # array length
                zdim = int(float(self.input_height.text())) // cell_dimension  # array length
                substrate_height = math.ceil(int(float(self.input_substrate_height.text())) / cell_dimension)
                structure.create_from_parameters(cell_dimension, xdim, ydim, zdim, substrate_height)
            except Exception as e:
                self.view_message('Input error',
                                  'An error occurred while fetching geometry parameters for the simulation volume. \n '
                                  'Check values and try again.')
                print(e.args)
                return
        if self.structure_source == 'auto':  # defining it later based on a stream-file
            cell_dimension = int(float(self.input_cell_size.text()))
            substrate_height = math.ceil(int(float(self.input_substrate_height.text())) / cell_dimension)

        # Defining printing path
        dwell_time_units = 1E-6  # input units are in microseconds, internally seconds are used
        printing_path = None
        if self.pattern_source == 'simple':  # creating printing path based on the figure and parameters
            if self.structure_source == 'auto':
                self.view_message(f'Input warning',
                                  f'Not allowed to choose \'Auto\' and \'Simple pattern\' together! \n'
                                  f'Ambiguous required simulation volume.')
                return
            try:
                print(f'Creating a simple pattern...', end='')
                pattern = self.pattern
                p1 = float(self.input_param1.text())  # nm
                p2 = float(self.input_param2.text()) if pattern in ['Point', 'Rectangle', 'Square'] else 0  # nm
                dwell_time = int(self.input_dwell_time.text()) * dwell_time_units  # s
                pitch = float(self.input_pitch.text())  # nm
                repeats = int(float(self.input_repeats.text()))
                x = structure.shape[2] // 2 * cell_dimension  # nm
                y = structure.shape[1] // 2 * cell_dimension  # nm
                if pattern == 'Point':
                    x, y = p1, p2
                printing_path = sp.generate_pattern(pattern, repeats, dwell_time, x, y, (p1, p2), pitch)
            except Exception as e:
                self.view_message('Error occurred while creating a printing path. \n Check values and try again.')
                print(e.args)
                return
        if self.pattern_source == 'stream_file':  # importing printing path from stream_file
            try:
                hfw = float(self.input_hfw.text())
            except ValueError:
                self.view_message('Invalid HFW input',
                                  'Half field width can not be read. \nPlease enter a valid number')
                return
            try:
                printing_path, shape = sp.open_stream_file(self.stream_file_filename, hfw, 200, True)
            except Exception as e:
                if not self.stream_file_filename:
                    self.view_message('File not specified',
                                      'Stream-file not specified. Please choose the file and try again.')
                else:
                    self.view_message(additional_message='Stream-file not found')
                return
            if self.structure_source != 'auto':
                if printing_path[:, 0].max() > structure.xdim_abs or printing_path[:, 1].max() > structure.ydim_abs:
                    self.view_message('Incompatible dimensions',
                                      f'The specified simulation volume does not enclose the printing path from '
                                      f'the stream-file. Increase base size or choose \'Auto\' \n'
                                      f'Specified stream-file uses '
                                      f'{printing_path[:, 0].max():.1f} x {printing_path[:, 1].max():.1f} nm area.')
                    return
            else:
                shape = shape[::-1] // cell_dimension
                structure.create_from_parameters(cell_dimension, *shape, substrate_height)
                print(f'Pattern base area with margin: {shape[0] * cell_dimension} x {shape[1] * cell_dimension} nm')
        t = printing_path[:, 2].sum()
        print(f'Total patterning time: {t:.3f} s')

        # Opening beam and precursor files
        try:
            with open(self.settings_filename, mode='rb') as f:
                settings = yaml.load(f, Loader=yaml.FullLoader)
            factor = settings.get('deposition_scaling', 1)
            if factor:
                printing_path[:, 2] /= factor
        except Exception as e:
            if not self.settings_filename:
                self.view_message('File not specified',
                                  'Beam parameters file not specified. Please choose the file and try again.')
            else:
                self.view_message(additional_message='Beam parameters file not found')
                print(e.args)
            return
        try:
            with open(self.precursor_parameters_filename, mode='rb') as f:
                precursor_params = yaml.load(f, Loader=yaml.FullLoader)
        except:
            if not self.precursor_parameters_filename:
                self.view_message('File not specified',
                                  'Precursor parameters file not specified. Please choose the file and try again.')
            else:
                self.view_message(additional_message='Precursor parameters file not found')
            return

        sim_volume_params = {}  # array length
        sim_volume_params['width'] = structure.shape[2]
        sim_volume_params['length'] = structure.shape[1]
        sim_volume_params['height'] = structure.shape[0]
        sim_volume_params['cell_dimension'] = cell_dimension
        sim_volume_params['substrate_height'] = substrate_height

        # Collecting parameters of file saving
        saving_params = {'monitoring': None, 'snapshot': None, 'filename': None}
        flag1, flag2 = self.checkbox_save_simulation_data.isChecked(), self.checkbox_save_snapshots.isChecked()
        if flag1:
            saving_params['monitoring'] = float(self.input_sim_data_interval.text())
        if flag2:
            saving_params['snapshot'] = float(self.input_snapshot_interval.text())
        if flag1 or flag2:
            if not self.input_unique_name.text():
                random_name = 'simulation_' + str(random.randint(10000, 99999))
                self.view_message('No name specified',
                                  f'Unique name is not specified, assigning {random_name}', icon='Info')
                self.input_unique_name.setText(random_name)
                return
            if not self.save_directory:
                self.view_message('No directory specified', 'The directory to save the files to is not specified.')
                return
            saving_params['filename'] = os.path.join(self.save_directory, self.input_unique_name.text())
            try:
                os.makedirs(saving_params['filename'])
            except FileExistsError as e:
                pass
            saving_params['filename'] = os.path.join(saving_params['filename'], self.input_unique_name.text())

        rendering = {'show_process': self.show_process, 'frame_rate': 0.5}
        # Starting the process
        febid_core.run_febid_interface(structure, precursor_params, settings, sim_volume_params, printing_path,
                                       self.temperature_tracking, saving_params, rendering)

        return

    def start_mc(self):
        # Creating a simulation volume
        structure = Structure()
        params = None
        if self.structure_source == 'auto':
            self.view_message('Structure source not set', 'Seems like \'Auto\' option was chosen previously '
                                                           'in the FEBID tab...')
            return
        if self.structure_source == 'vtk':  # opening from a .vtk file
            try:
                vtk_obj = pv.read(self.vtk_filename)
                structure.load_from_vtk(vtk_obj)
                params = read_field_data(vtk_obj)
            except Exception as e:
                if not self.vtk_filename:
                    self.view_message('File not specified',
                                      'VTK file not specified. Please choose the file and try again.')
                else:
                    self.view_message(additional_message='VTK file not found')
                return
            cell_dimension = structure.cell_dimension
            substrate_height = structure.substrate_height

        if self.structure_source == 'geom':  # creating from geometry parameters
            try:
                cell_dimension = int(self.input_cell_size_mc.text())
                xdim = int(float(self.input_width_mc.text())) // cell_dimension  # array length
                ydim = int(float(self.input_length_mc.text())) // cell_dimension  # array length
                zdim = int(float(self.input_height_mc.text())) // cell_dimension  # array length
                substrate_height = math.ceil(int(float(self.input_substrate_height_mc.text())) / cell_dimension)
                structure.create_from_parameters(cell_dimension, xdim, ydim, zdim, substrate_height)
            except Exception as e:
                self.view_message('Input error',
                                  'An error occurred while fetching geometry parameters for the simulation volume. \n '
                                  'Check values and try again.')
                print(e.args)
                return

        # Opening precursor file
        try:
            with open(self.precursor_parameters_filename, mode='rb') as f:
                precursor_params = yaml.load(f, Loader=yaml.FullLoader)
        except:
            if not self.precursor_parameters_filename:
                self.view_message('File not specified',
                                  'Precursor parameters file not specified. Please choose the file and try again.')
            else:
                self.view_message(additional_message='Precursor parameters file not found')
            return
        E0 = float(self.beam_energy.text())
        Emin = float(self.energy_cutoff.text())
        gauss_dev = float(self.gauss_dev.text())
        x0 = float(self.x_pos.text())
        y0 = float(self.y_pos.text())
        N = int(self.number_of_e.text())
        n = int(self.gauss_order.text())
        heating = self.checkbox_beam_heating.isChecked()
        self.cam_pos = e3d.run_mc_simulation(structure, E0, gauss_dev, n, N, (x0, y0), precursor_params, Emin, 0.6,
                                             heating, params, self.cam_pos)
        return 1

    # Utilities
    def __prepare(self):
        """
        Group interface elements for easier enabling/disabling.

        :return:
        """
        # Groups of controls on the panel for easier Enabling/Disabling

        # Inputs and their labels
        self.ui_dimensions = UI_Group(self.input_width, self.input_length, self.input_height,
                                      self.l_width, self.l_height, self.l_length,
                                      self.l_dimensions_units)
        self.ui_dimensions_mc = UI_Group(self.input_width_mc, self.input_length_mc, self.input_height_mc,
                                         self.l_width_mc, self.l_height_mc, self.l_length_mc,
                                         self.l_dimensions_units_mc)
        self.ui_cell_size = UI_Group(self.l_cell_size, self.input_cell_size, self.l_cell_size_units)
        self.ui_cell_size_mc = UI_Group(self.l_cell_size_mc, self.input_cell_size_mc, self.l_cell_size_units_mc)
        self.ui_substrate_height = UI_Group(self.l_substrate_height, self.input_substrate_height,
                                            self.l_substrate_height_units)
        self.ui_substrate_height_mc = UI_Group(self.l_substrate_height_mc, self.input_substrate_height_mc,
                                               self.l_substrate_height_units_mc)

        self.ui_pattern_param1 = UI_Group(self.l_param1, self.input_param1, self.l_param1_units)
        self.ui_pattern_param2 = UI_Group(self.l_param2, self.input_param2, self.l_param2_units)
        self.ui_dwell_time = UI_Group(self.l_dwell_time, self.input_dwell_time, self.l_dwell_time_units)
        self.ui_pitch = UI_Group(self.l_pitch, self.input_pitch, self.l_pitc_units)
        self.ui_repeats = UI_Group(self.l_repeats, self.input_repeats)

        self.ui_hfw = UI_Group(self.l_hfw, self.input_hfw, self.l_hfw_units)

        self.ui_sim_data_interval = UI_Group(self.l_sim_data_interval, self.input_sim_data_interval,
                                             self.l_sim_data_interval_units)
        self.ui_snapshot = UI_Group(self.l_snapshot_interval, self.input_snapshot_interval,
                                    self.l_snapshot_interval_units)
        self.ui_unique_name = UI_Group(self.l_unique_name, self.input_unique_name)
        self.ui_save_folder = UI_Group(self.open_save_folder_button, self.save_folder_display)

        # Grouping elements by their designation
        self.ui_vtk_choice = UI_Group(self.open_vtk_file_button, self.vtk_filename_display)
        self.ui_vtk_choice_mc = UI_Group(self.open_vtk_file_button_mc, self.vtk_filename_display_mc)

        self.ui_geom_choice = UI_Group(
            {self.open_geom_parameters_file_button} | self.ui_dimensions | self.ui_cell_size | \
            self.ui_substrate_height)
        self.ui_geom_choice_mc = UI_Group({self.open_geom_parameters_file_button_mc} | self.ui_dimensions_mc | \
                                          self.ui_cell_size_mc | self.ui_substrate_height_mc)

        self.ui_auto_choice = UI_Group(self.ui_cell_size | self.ui_substrate_height)

        self.ui_simple_patterns = UI_Group({self.pattern_selection} | self.ui_pattern_param1 | self.ui_pattern_param2 | \
                                           self.ui_dwell_time | self.ui_pitch | self.ui_repeats)

        self.ui_stream_file = UI_Group({self.open_stream_file_button} | self.ui_hfw)

        # Grouping by the groupBoxes
        self.ui_sim_volume = UI_Group(self.ui_vtk_choice | self.ui_geom_choice | self.ui_auto_choice)
        self.ui_sim_volume_mc = UI_Group(self.ui_vtk_choice_mc | self.ui_geom_choice_mc)
        self.ui_pattern = UI_Group(self.ui_simple_patterns | self.ui_stream_file)

    def __gather_session_config(self):
        """
        Compile a session config from interface elements and class attributes

        :return:
        """
        self.session['load_last_session'] = self.save_flag
        self.session['structure_source'] = self.structure_source
        self.session['vtk_filename'] = self.vtk_filename
        self.session['geom_parameters_filename'] = self.geom_parameters_filename
        self.session['width'] = int(self.input_width.text())
        self.session['length'] = int(self.input_length.text())
        self.session['height'] = int(self.input_height.text())
        self.session['cell_size'] = int(self.input_cell_size.text())
        self.session['substrate_height'] = int(self.input_substrate_height.text())
        self.session['pattern_source'] = self.pattern_source
        self.session['pattern'] = self.pattern_selection.currentText()
        self.session['param1'] = float(self.input_param1.text())
        self.session['param2'] = float(self.input_param2.text())
        self.session['dwell_time'] = int(self.input_dwell_time.text())
        self.session['pitch'] = int(self.input_pitch.text())
        self.session['repeats'] = int(self.input_repeats.text())
        self.session['stream_file_filename'] = self.stream_file_filename
        self.session['hfw'] = float(self.input_hfw.text())
        self.session['settings_filename'] = self.settings_filename
        self.session['precursor_filename'] = self.precursor_parameters_filename
        self.session['temperature_tracking'] = self.temperature_tracking
        self.session['save_simulation_data'] = self.checkbox_save_simulation_data.isChecked()
        self.session['save_structure_snapshot'] = self.checkbox_save_snapshots.isChecked()
        self.session['simulation_data_interval'] = float(self.input_sim_data_interval.text())
        self.session['structure_snapshot_interval'] = float(self.input_snapshot_interval.text())
        self.session['unique_name'] = self.input_unique_name.text()
        self.session['save_directory'] = self.save_directory
        self.session['show_process'] = self.checkbox_show.isChecked()

    def __load_config(self, params):
        """
        Set session parameters and update interface elements from a dict.

        :param params: collection of parameters
        :return:
        """
        try:
            self.save_flag = True
            self.structure_source = params['structure_source']
            self.vtk_filename = params['vtk_filename']
            self.geom_parameters_filename = params['geom_parameters_filename']
            self.pattern_source = params['pattern_source']
            self.pattern = params['pattern']
            self.stream_file_filename = params['stream_file_filename']
            self.settings_filename = params['settings_filename']
            self.precursor_parameters_filename = params['precursor_filename']
            self.temperature_tracking = params['temperature_tracking']
            self.save_directory = params['save_directory']
            # Setting FEBID tab interface
            self.checkbox_load_last_session.setChecked(True)
            if self.structure_source == 'vtk':
                self.vtk_chosen()
                if self.vtk_filename:
                    self.open_vtk_file(self.vtk_filename)
            elif self.structure_source == 'geom':
                self.geom_parameters_chosen()
            elif self.structure_source == 'auto':
                self.auto_chosen()
            self.vtk_filename_display.setText(self.vtk_filename)
            self.input_width.setText(str(params['width']))
            self.input_length.setText(str(params['length']))
            self.input_height.setText(str(params['height']))
            self.input_cell_size.setText(str(params['cell_size']))
            self.input_substrate_height.setText(str(params['substrate_height']))
            self.pattern_selection.setCurrentText(str.title(self.pattern))
            self.pattern_selection_changed(str.title(self.pattern))
            if self.pattern_source == 'simple':
                self.simple_pattern_chosen()
            elif self.pattern_source == 'stream_file':
                self.stream_file_chosen()
            self.input_param1.setText(str(params['param1']))
            self.input_param2.setText(str(params['param2']))
            self.input_dwell_time.setText(str(params['dwell_time']))
            self.input_pitch.setText(str(params['pitch']))
            self.input_repeats.setText(str(params['repeats']))
            self.stream_file_filename_display.setText(self.stream_file_filename)
            self.input_hfw.setText(str(params['hfw']))
            self.settings_filename_display.setText(self.settings_filename)
            self.precursor_parameters_filename_display.setText(self.precursor_parameters_filename)
            self.checkbox_temperature_tracking.setChecked(self.temperature_tracking)
            self.change_state_save_sim_data(params['save_simulation_data'])
            self.change_state_save_snapshots(params['save_structure_snapshot'])
            self.input_sim_data_interval.setText(str(params['simulation_data_interval']))
            self.input_snapshot_interval.setText(str(params['structure_snapshot_interval']))
            self.input_unique_name.setText(params['unique_name'])
            self.save_folder_display.setText(self.save_directory)
            self.change_state_show_process(params['show_process'])
            # Setting MC interface
            if self.structure_source == 'vtk':
                self.vtk_chosen()
            elif self.structure_source == 'geom':
                self.geom_parameters_chosen()
            elif self.structure_source == 'auto':
                self.auto_chosen()
            self.vtk_filename_display_mc.setText(self.vtk_filename)
            self.input_width_mc.setText(str(params['width']))
            self.input_length_mc.setText(str(params['length']))
            self.input_height_mc.setText(str(params['height']))
            self.input_cell_size_mc.setText(str(params['cell_size']))
            self.input_substrate_height_mc.setText(str(params['substrate_height']))
            if self.settings_filename:
                self.open_settings_file(self.settings_filename)
            self.settings_filename_display_mc.setText(self.settings_filename)
            self.precursor_parameters_filename_display_mc.setText(self.precursor_parameters_filename)
            self.checkbox_beam_heating.setChecked(self.temperature_tracking)
        except KeyError as e:
            print('Was not able to read a configuration file. Following error occurred.')
            raise e

    def save_parameter(self, param_name, value):
        """
        Change the specified parameter and write it to the file
        :param param_name:
        :param value:
        :return:
        """
        self.session[param_name] = value
        yml = YAML()
        if self.save_flag:
            with open(self.last_session_filename, mode='wb') as f:
                yml.dump(self.session, f)

    def change_color(self, labels: Union[list, QtWidgets.QLabel], color='gray'):
        if type(labels) is list:
            for label in labels:
                pallet = label.palette()
                pallet.setColor(QtGui.QPalette.WindowText, QtGui.QColor(color))
                label.setPalette(pallet)
        elif labels is QtWidgets.QLabel:
            pallet = labels.palette()
            pallet.setColor(QtGui.QPalette.WindowText, QtGui.QColor(color))
            labels.setPalette(pallet)

    def check_input(self):
        """
        Check if the input line is numeric and not negative and save the parameter to the file
        :return:
        """
        lineEdit = self.sender()
        text = lineEdit.text()
        if self.is_float(text):
            val = float(text)
            if float(text) >= 0:
                if self.save_flag:
                    # Here the parameter is saved to the file
                    # A naming convention is made: lineEdit objects are named the same
                    # as the parameters in the file only with 'input_' in the beginning
                    name = lineEdit.objectName()[6:]  # stripping 'input_'
                    if int(val) > 0 and val / int(val) == 1:
                        self.save_parameter(name, int(val))
                    else:
                        self.save_parameter(name, val)
                return
            else:
                self.view_message("Value cannot be negative.")
                lineEdit.clear()
        else:
            self.view_message('Input is invalid.',
                              f'The value entered is not numerical.')
            lineEdit.clear()
        a = 0

    def is_float(self, element) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    def view_message(self, message="An error occurred", additional_message='', icon='Warning'):
        if icon not in ['Warning', 'Question', 'Information', 'Critical']:
            icon = QMessageBox.NoIcon
        if icon == 'Warning':
            icon = QMessageBox.Warning
        if icon == 'Question':
            icon = QMessageBox.Question
        if icon == 'Information':
            icon = QMessageBox.Information
        if icon == 'Critical':
            icon = QMessageBox.Critical
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText(
            message + ' ' * len(additional_message))  # QMessageBox resizes only with the length of the main text
        msgBox.setInformativeText(additional_message)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.setIcon(icon)
        msgBox.exec()

    def read_yaml(self, file):
        """
        Read YAML file with units

        :param file: full path to YAML file
        :return: values dict, units dict
        """
        values = {}
        units = {}
        with open(file, mode='rb') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            for key, entry in config.items():
                if type(entry) is str:
                    entry_splitted = entry.split(' ')
                    try:
                        value, unit = entry_splitted
                    except ValueError:
                        value = entry_splitted[0]
                        unit = ''
                    except Exception as e:
                        print(f'An error occurred while reading units from YAML file:')
                        raise e
                    try:
                        val = int(value)
                    except ValueError:
                        try:
                            val = float(value)
                        except ValueError:
                            val = value
                    values[key] = val
                    units[key] = unit
                else:
                    values[key] = entry
                    units[key] = ''
        return values, units

    @property
    def last_session_stub(self):
        """
        Configuration file template

        :return:
        """
        stub = ['# Settings from the last session are saved here and loaded\n',
                "# upon the launch if the parameter 'load_last_session' is True\n",
                '\n',
                'load_last_session: True\n',
                'structure_source: geom # vtk – load from a vtk_file, geom – create from parameters, auto – create from stream-file\n',
                'vtk_filename: ''\n',
                "geom_parameters_filename: ''\n",
                'width: ''\n',
                'length: ''\n',
                'height: ''\n',
                'cell_size: ''\n',
                'substrate_height: ''\n',
                '\n',
                'pattern_source: '' # simple - print a simple figure, stream_file - load printing path from file\n',
                'pattern: '' # available: point, line, square, circle, rectangle\n',
                '# For the point these parameters are position coordinates, while other patterns are automatically\n',
                '# positioned in the center and these parameters define the figures.\n',
                'param1: '' # nm\n',
                'param2: '' # nm\n',
                'dwell_time: '' # µs\n',
                'pitch: '' # nm\n',
                'repeats: ''\n',
                'stream_file_filename: '' # stream-file path\n',
                'hfw:'' # µm\n',
                '\n',
                'settings_filename: ''\n',
                'precursor_filename: ''\n',
                'temperature_tracking: ''\n',
                '\n',
                'save_simulation_data: ''\n',
                'save_structure_snapshot: ''\n',
                '\n',
                'simulation_data_interval: ''\n',
                'structure_snapshot_interval: ''\n',
                'unique_name: ''\n',
                'save_directory: ''\n',
                'show_process: ''\n']
        return stub


def start(config_filename=None):
    app = QApplication(sys.argv)
    win1 = MainPannel(config_filename)
    sys.exit(app.exec())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win1 = MainPannel()
    sys.exit(app.exec())
