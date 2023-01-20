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


class MainPannel(QMainWindow, UI_MainPanel):
    def __init__(self, config_filename=None, test_kwargs=None, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.show()
        # TODO: variables and corresponding interface elements can be grouped into objects with a getter and a setter
        # Parameters
        if config_filename is not None:
            self.last_session_filename = config_filename
        else:
            self.last_session_filename = 'last_session.yml'
        self.session = None
        self.save_flag = False
        self.structure_source = 'vtk' # vtk, geom or auto
        self.pattern_source = 'simple' # simple or stream_file
        self.pattern = 'Point'
        self.vtk_filename = ''
        self.geom_parameters_filename = ''
        self.stream_file_filename = ''
        self.settings_filename = ''
        self.precursor_parameters_filename = ''
        self.save_directory = ''
        self.show_process = True
        # Groups of controls on the panel for easier Enabling/Disabling
        self.vtk_choice_to_gray = [self.l_width, self.l_width_mc, self.l_height,self.l_height_mc,
                                   self.l_length,self.l_length_mc, self.l_cell_size, self.l_cell_size_mc,
                                   self.l_substrate_height, self.l_substrate_height_mc,
                                   self.l_dimensions_units, self.l_cell_size_units, self.l_substrate_height_units,
                                   self.l_dimensions_units_mc, self.l_cell_size_units_mc, self.l_substrate_height_units_mc,]
        self.vtk_choice_to_disable = [self.input_width, self.input_width_mc,
                                      self.input_length, self.input_length_mc,
                                      self.input_height, self.input_height_mc,
                                      self.input_cell_size, self.input_cell_size_mc,
                                      self.input_substrate_height, self.input_substrate_height_mc]
        self.simple_pattern_controls = [self.pattern_selection,
                                        self.l_param1, self.input_param1, self.l_param1_units,
                                        self.l_param2, self.input_param2, self.l_param2_units,
                                        self.l_dwell_time, self.input_dwell_time, self.l_dwell_time_units,
                                        self.l_pitch, self.input_pitch, self.l_pitc_units,
                                        self.l_repeats, self.input_repeats]
        self.cell_size_controls = [self.l_cell_size, self.input_cell_size, self.l_cell_size_units]
        self.cell_size_controls_mc = [self.l_cell_size_mc, self.input_cell_size_mc, self.l_cell_size_units_mc]
        self.substrate_height_controls = [self.l_substrate_height, self.input_substrate_height, self.l_substrate_height_units]
        self.substrate_height_controls_mc = [self.l_substrate_height_mc, self.input_substrate_height_mc, self.l_substrate_height_units_mc]
        self.pattern_param1_controls = self.simple_pattern_controls[1:4]
        self.pattern_param2_controls = self.simple_pattern_controls[4:7]
        self.open_geom_parameters_file_button.setDisabled(True)
        self.open_geom_parameters_file_button_mc.setDisabled(True)
        for obj in self.vtk_choice_to_disable:
            obj.setDisabled(True)

        self.open_last_session()

        if test_kwargs:
            self.inject_parameters(test_kwargs)

    def open_last_session(self, filename=''):
        """
        Open last session settings from the file or create a new file.

        File is saved in the same directory as the script.

        :param filename: name of the file
        :return:
        """
        if filename:
            pass
        else:
            filename = self.last_session_filename
        try:
            print('Trying to load last session...', end='')
            if not os.path.exists(filename):
                raise FileNotFoundError
            with open(filename, 'r+b') as f:
                yml = YAML()
                self.session = params = yml.load(f)
                if params is None:
                    raise RuntimeError('YAML error: was unable to read the .yml configuration file.')
                if params['load_last_session'] is True:
                    with suppress(KeyError):
                        self.save_flag = True
                        self.checkbox_load_last_session.setChecked(True)
                        self.structure_source = params['structure_source']
                        if self.structure_source == 'vtk':
                            self.vtk_chosen()
                        elif self.structure_source == 'geom':
                            self.geom_parameters_chosen()
                        elif self.structure_source == 'auto':
                            self.auto_chosen()
                        self.vtk_filename = params['vtk_filename']
                        self.vtk_filename_display.setText(self.vtk_filename)
                        self.geom_parameters_filename = params['geom_parameters_filename']
                        self.input_width.setText(str(params['width']))
                        self.input_length.setText(str(params['length']))
                        self.input_height.setText(str(params['height']))
                        self.input_cell_size.setText(str(params['cell_size']))
                        self.input_substrate_height.setText(str(params['substrate_height']))

                        self.pattern_source = params['pattern_source']
                        self.pattern = params['pattern']
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
                        self.stream_file_filename = params['stream_file_filename']
                        self.stream_file_filename_display.setText(self.stream_file_filename)

                        self.settings_filename = params['settings_filename']
                        self.settings_filename_display.setText(self.settings_filename)
                        self.precursor_parameters_filename = params['precursor_filename']
                        self.precursor_parameters_filename_display.setText(self.precursor_parameters_filename)

                        self.change_state_save_sim_data(params['save_simulation_data'])
                        self.change_state_save_snapshots(params['save_structure_snapshot'])
                        self.input_simulation_data_interval.setText(str(params['simulation_data_interval']))
                        self.input_structure_snapshot_interval.setText(str(params['structure_snapshot_interval']))
                        self.input_unique_name.setText(params['unique_name'])
                        self.save_directory = params['save_directory']
                        self.save_folder_display.setText(self.save_directory)

                        self.change_state_show_process(params['show_process'])
                print('done!')
        except FileNotFoundError:
                yml = YAML()
                input = ''.join(self.last_session_stub)
                self.session = yml.load(input)
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

                self.session['settings_filename'] = self.settings_filename
                self.session['precursor_filename'] = self.precursor_parameters_filename

                self.session['save_simulation_data'] = self.checkbox_save_simulation_data.isChecked()
                self.session['save_structure_snapshot'] = self.checkbox_save_snapshots.isChecked()
                self.session['simulation_data_interval'] = float(self.input_simulation_data_interval.text())
                self.session['structure_snapshot_interval'] = float(self.input_structure_snapshot_interval.text())
                self.session['unique_name'] = self.input_unique_name.text()
                self.session['save_directory'] = self.save_directory

                self.session['show_process'] = self.checkbox_show.isChecked()

                if self.save_flag:
                    print('Last session file not found, creating a new one.')
                    with open(filename, "x") as f:
                        self.save_parameter('show_process', self.checkbox_show.isChecked())

    def change_state_load_last_session(self, param=None):
        switch = True if param else False
        self.checkbox_load_last_session.setChecked(switch)
        self.save_flag = switch
        if switch:
            self.open_last_session()

    def vtk_chosen(self):
        # For both tabs:
        #  Gray out labels for other two options
        #  Disable input fields for other options
        self.choice_vtk_file.setChecked(True)
        self.structure_source = 'vtk'
        self.choice_vtk_file.setChecked(True)
        self.choice_vtk_file_mc.setChecked(True)
        self.open_vtk_file_button.setEnabled(True)
        self.open_vtk_file_button_mc.setEnabled(True)
        self.vtk_filename_display.setEnabled(True)
        self.vtk_filename_display_mc.setEnabled(True)
        self.open_geom_parameters_file_button.setDisabled(True)
        self.open_geom_parameters_file_button_mc.setDisabled(True)
        for obj in self.vtk_choice_to_disable:
            obj.setReadOnly(True)
            obj.setDisabled(True)
        for obj in self.vtk_choice_to_gray:
            obj.setEnabled(True)
        self.save_parameter('structure_source', 'vtk')

    def geom_parameters_chosen(self):
        self.choice_geom_parameters_file.setChecked(True)
        self.structure_source = 'geom'
        self.choice_geom_parameters_file.setChecked(True)
        self.choice_geom_parameters_file_mc.setChecked(True)
        self.open_vtk_file_button.setDisabled(True)
        self.open_vtk_file_button_mc.setDisabled(True)
        self.vtk_filename_display.setDisabled(True)
        self.vtk_filename_display_mc.setDisabled(True)
        self.open_geom_parameters_file_button.setEnabled(True)
        self.open_geom_parameters_file_button_mc.setEnabled(True)
        for obj in self.vtk_choice_to_disable:
            obj.setReadOnly(False)
            obj.setEnabled(True)
        for obj in self.vtk_choice_to_gray:
            obj.setEnabled(True)
        self.choice_simple_pattern.setChecked(True)
        self.save_parameter('structure_source', 'geom')
        self.simple_pattern_chosen()

    def auto_chosen(self):
        self.choice_auto.setChecked(True)
        self.structure_source = 'auto'
        self.open_vtk_file_button.setDisabled(True)
        self.open_vtk_file_button_mc.setDisabled(True)
        self.vtk_filename_display.setDisabled(True)
        self.vtk_filename_display_mc.setDisabled(True)
        self.open_geom_parameters_file_button.setDisabled(True)
        self.open_geom_parameters_file_button_mc.setDisabled(True)
        for obj in self.vtk_choice_to_disable:
            # obj.setReadOnly(True)
            obj.setDisabled(True)
        for obj in self.vtk_choice_to_gray:
            obj.setDisabled(True)
        self.set_params_ui(self.cell_size_controls, enable=True)
        self.set_params_ui(self.substrate_height_controls, enable=True)
        self.choice_stream_file.setChecked(True)
        self.save_parameter('structure_source', 'auto')
        self.stream_file_chosen()

    def simple_pattern_chosen(self):
        self.choice_simple_pattern.setChecked(True)
        self.pattern_source = 'simple'
        for obj in self.simple_pattern_controls:
            obj.setEnabled(True)
        self.open_stream_file_button.setDisabled(True)
        self.stream_file_filename_display.setDisabled(True)
        self.save_parameter('pattern_source', 'simple')

    def stream_file_chosen(self):
        self.choice_stream_file.setChecked(True)
        self.pattern_source = 'stream_file'
        for obj in self.simple_pattern_controls:
            obj.setEnabled(False)
        self.open_stream_file_button.setEnabled(True)
        self.stream_file_filename_display.setEnabled(True)
        self.save_parameter('pattern_source', 'stream_file')

    def pattern_selection_changed(self, current=''):
        if current == 'Point':
            self.set_params_ui(self.pattern_param1_controls, 'x:', True)
            self.set_params_ui(self.pattern_param2_controls, 'y:', True)
        if current == 'Line':
            self.set_params_ui(self.pattern_param1_controls, 'l:', True)
            self.set_params_ui(self.pattern_param2_controls, ' ', False)
        if current == 'Rectangle':
            self.set_params_ui(self.pattern_param1_controls, 'a:', True)
            self.set_params_ui(self.pattern_param2_controls, 'b:', True)
        if current == 'Square':
            self.set_params_ui(self.pattern_param1_controls, 'a:', True)
            self.set_params_ui(self.pattern_param2_controls, 'b:', False)
        if current == 'Triangle':
            self.set_params_ui(self.pattern_param1_controls, 'a:', True)
            self.set_params_ui(self.pattern_param2_controls, ' ', False)
        if current == 'Circle':
            self.set_params_ui(self.pattern_param1_controls, 'd:', True)
            self.set_params_ui(self.pattern_param2_controls, ' ', False)
        self.pattern = current
        self.save_parameter('pattern', current)

    def open_vtk_file(self, file=''):
        # For both tabs:
        #  Check if the specified file is a valid .vtk file
        #  Insert parameters into fields
        if not file:
            file,_ = QtWidgets.QFileDialog.getOpenFileName()
        if not file:
            return
        try:
            deposit = Structure()
            vtk_obj = pv.read(file)
            deposit.load_from_vtk(vtk_obj)
            structure = deposit.deposit
            cell_dim = deposit.cell_dimension
            zdim, ydim, xdim = [str(int(dim*cell_dim)) for dim in list(structure.shape)]
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
            self.save_parameter('vtk_filename', file)
        except Exception as e:
            self.view_message('File read error', 'Specified file is not a valid VTK file. Please choose a valid .vtk file.')
            print("Was unable to open .vtk file. Following error occurred:")
            print(e.args)
    def open_geom_parameters_file(self, file=''):
        if not file:
            file,_ = QtWidgets.QFileDialog.getOpenFileName()
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
            print("Was unable to open .yaml geometry parameters file. Following error occurred:")
            print(e.args)
        ### Read and insert parameters
    def open_stream_file(self, file=''):
        file,_ = QtWidgets.QFileDialog.getOpenFileName()
        if not file:
            return
        self.stream_file_filename = file
        self.stream_file_filename_display.setText(file)
        self.save_parameter('stream_file_filename', file)
    def open_settings_file(self, file=''):
        file,_ = QtWidgets.QFileDialog.getOpenFileName()
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
            self.beam_parameters_filename_display_mc.setText(file)
            self.settings_filename = file
            self.save_parameter('settings_filename', file)
        except Exception as e:
            print("Was unable to open .yaml beam parameters file. Following error occurred:")
            print(e.args)
    def open_precursor_parameters_file(self, file=''):
        file,_ = QtWidgets.QFileDialog.getOpenFileName()
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
            self.view_message('File read error', 'Specified file is not a valid parameters file. Please choose a valid .yml file.')
            print("Was unable to open .yml precursor parameters file. Following error occurred:")
            print(e.args)

    def change_state_save_sim_data(self, param=None):
        switch = True if param else False
        self.checkbox_save_simulation_data.setChecked(switch)
        self.input_simulation_data_interval.setEnabled(switch)
        self.l_sim_data_interval.setEnabled(switch)
        self.l_sim_data_interval_units.setEnabled(switch)
        if switch or self.checkbox_save_snapshots.isChecked():
            self.input_unique_name.setEnabled(True)
            self.l_unique_name.setEnabled(True)
            self.open_save_folder_button.setEnabled(True)
            self.save_folder_display.setEnabled(True)
        else:
            self.input_unique_name.setEnabled(False)
            self.l_unique_name.setEnabled(False)
            self.open_save_folder_button.setEnabled(False)
            self.save_folder_display.setEnabled(False)
        self.save_parameter('save_simulation_data', switch)

    def change_state_save_snapshots(self, param=None):
        switch = True if param else False
        self.input_structure_snapshot_interval.setEnabled(switch)
        self.checkbox_save_snapshots.setChecked(switch)
        self.l_snapshot_interval.setEnabled(switch)
        self.l_snapshot_interval_units.setEnabled(switch)
        if switch or self.checkbox_save_simulation_data.isChecked():
            self.input_unique_name.setEnabled(True)
            self.l_unique_name.setEnabled(True)
            self.open_save_folder_button.setEnabled(True)
            self.save_folder_display.setEnabled(True)
        else:
            self.input_unique_name.setEnabled(False)
            self.l_unique_name.setEnabled(False)
            self.open_save_folder_button.setEnabled(False)
            self.save_folder_display.setEnabled(False)
        self.save_parameter('save_structure_snapshot', switch)

    def change_state_show_process(self, param=None):
        switch = True if param else False
        self.checkbox_show.setChecked(switch)
        self.show_process = switch
        self.save_parameter('show_process', switch)

    def unique_name_changed(self):
        self.save_parameter('unique_name', self.input_unique_name.text())

    def open_save_directory(self): #implement
        directory = QtWidgets.QFileDialog.getExistingDirectory()
        if directory:
            self.save_folder_display.setText(directory)
        self.save_directory = directory
        self.save_parameter('save_directory', directory)

    def tab_switched(self, current):
        if current == 0:
            self.resize(self.width(), 630)
        if current == 1:
            self.resize(self.width(), 500)

    def start_febid(self):
        # Creating a simulation volume
        structure = Structure()
        if self.structure_source == 'vtk': # opening from a .vtk file
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
        if self.structure_source == 'geom': # creating from geometry parameters
            try:
                cell_dimension = int(self.input_cell_size.text())
                xdim = int(float(self.input_width.text()))//cell_dimension # array length
                ydim = int(float(self.input_length.text()))//cell_dimension # array length
                zdim = int(float(self.input_height.text()))//cell_dimension # array length
                substrate_height = math.ceil(int(float(self.input_substrate_height.text())) / cell_dimension)
                structure.create_from_parameters(cell_dimension, xdim, ydim, zdim, substrate_height)
            except Exception as e:
                self.view_message('Input error',
                                  'An error occurred while fetching geometry parameters for the simulation volume. \n '
                                  'Check values and try again.')
                print(e.args)
                return
        if self.structure_source == 'auto': # defining it later based on a stream-file
            cell_dimension = int(float(self.input_cell_size.text()))
            substrate_height = math.ceil(int(float(self.input_substrate_height.text())) / cell_dimension)

        # Defining printing path
        dwell_time_units = 1E-6 # input units are in microseconds, internally seconds are used
        printing_path = None
        if self.pattern_source == 'simple': # creating printing path based on the figure and parameters
            if self.structure_source == 'auto':
                self.view_message(f'Input warning',
                                  f'Not allowed to choose \'Auto\' and \'Simple pattern\' together! \n'
                                  f'Ambiguous required simulation volume.')
                return
            try:
                pattern = self.pattern
                p1 = float(self.input_param1.text()) # nm
                p2 = float(self.input_param2.text()) if pattern in ['Point', 'Rectangle', 'Square'] else 0 # nm
                dwell_time = int(self.input_dwell_time.text()) * dwell_time_units # s
                pitch = float(self.input_pitch.text()) # nm
                repeats = int(float(self.input_repeats.text()))
                x = structure.shape[2]//2 * cell_dimension # nm
                y = structure.shape[1]//2 * cell_dimension # nm
                if pattern == 'Point':
                    x, y = p1, p2
                printing_path = sp.generate_pattern(pattern, repeats, dwell_time, x, y, (p1, p2), pitch)
            except Exception as e:
                self.view_message('Error occurred while creating a printing path. \n Check values and try again.')
                print(e.args)
                return
        if self.pattern_source == 'stream_file': # importing printing path from stream_file
            try:
                printing_path, shape = sp.open_stream_file(self.stream_file_filename, 200, True)
            except Exception as e:
                if not self.stream_file_filename:
                    self.view_message('File not specified',
                                      'Stream-file not specified. Please choose the file and try again.')
                else:
                    self.view_message(additional_message='Stream-file not found')
                return
            if self.structure_source != 'auto':
                if printing_path[:,0].max() > structure.xdim_abs or printing_path[:,1].max() > structure.ydim_abs:
                    self.view_message('Incompatible dimensions',
                                      f'The specified simulation volume does not enclose the printing path from '
                                      f'the stream-file. Increase base size or choose \'Auto\' \n'
                                      f'Specified stream-file uses '
                                      f'{printing_path[:,0].max()} x {printing_path[:,1].max()} nm area.')
                    return
            else:
                shape = shape[::-1]//cell_dimension
                structure.create_from_parameters(cell_dimension, *shape, substrate_height)

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

        sim_volume_params = {} # array length
        sim_volume_params['width'] = structure.shape[2]
        sim_volume_params['length'] = structure.shape[1]
        sim_volume_params['height'] = structure.shape[0]
        sim_volume_params['cell_dimension'] = cell_dimension
        sim_volume_params['substrate_height'] = substrate_height

        # Collecting parameters of file saving
        saving_params = {'monitoring': None, 'snapshot': None, 'filename': None}
        flag1, flag2 = self.checkbox_save_simulation_data.isChecked(), self.checkbox_save_snapshots.isChecked()
        if flag1:
            saving_params['monitoring'] = float(self.input_simulation_data_interval.text())
        if flag2:
            saving_params['snapshot'] = float(self.input_structure_snapshot_interval.text())
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
        febid_core.run_febid_interface(structure, precursor_params, settings, sim_volume_params, printing_path, saving_params, rendering)

        return

    def start_mc(self):
        self.view_message('Not implemented yet...', icon='Information')
        pass

    # Utilities
    def set_params_ui(self, param, name=None, enable=True):
        if name:
            param[0].setText(name)
        for element in param:
            element.setEnabled(enable)
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
    def change_color(self, labels: Union[list,QtWidgets.QLabel], color='gray'):
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
                    # Here the the parameter is saved to the file
                    # A naming convention is made: lineEdit objects are named the same
                    # as the parameters in the file only with 'input_' in the beginning
                    name = lineEdit.objectName()[6:] # stripping 'input_'
                    if int(val) > 0 and val/int(val) == 1:
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
        a=0
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
        msgBox.setText(message + ' '*len(additional_message)) # QMessageBox resizes only with the length of the main text
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
                'pattern_source: '' # simple - print a simple figure, stream-file - load printing path from file\n',
                'pattern: '' # available: point, line, square, circle, rectangle\n',
                '# For the point these parameters are position coordinates, while other patterns are automatically\n',
                '# positioned in the center and these parameters define the figures.\n',
                'param1: ''\n',
                'param2: ''\n',
                'dwell_time: ''\n',
                'pitch: ''\n',
                'repeats: ''\n',
                'stream_file_filename: ''\n',
                '\n',
                'settings_filename: ''\n',
                'precursor_filename: ''\n',
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

    # Testing
    def inject_parameters(self, kwargs):
        # TODO: testing function is undone
        attrs = vars(self).keys()
        for attr_name, attr_value in kwargs.items():
            try:
                if attr_name in attrs:
                    self.__setattr__(attr_name, attr_value)
            except Exception as e:
                print(f'Error occurred while setting attribute {attr_name}')
        # Insert all other arguments into the corresponding fields
        if kwargs['structure_source'] == 'vtk':
            self.choice_vtk_file.setChecked(True)
        if kwargs['structure_source'] == 'geom':
            self.choice_geom_parameters_file.setChecked(True)
        if kwargs['structure_source'] == 'auto':
            self.choice_auto.setChecked(True)

        if kwargs['pattern_source'] == 'simple':
            self.choice_simple_pattern.setChecked(True)
        if kwargs['pattern_source'] == 'stream_file':
            self.choice_stream_file.setChecked(True)

        self.vtk_filename_display.setText(kwargs['vtk_filename'])
        self.input_width.setText(str(kwargs['width']))
        self.input_length.setText(str(kwargs['length']))
        self.input_height.setText(str(kwargs['height']))
        self.input_cell_size.setText(str(kwargs['cell_dim']))
        self.input_substrate_height.setText(str(kwargs['substrate_height']))
        self.pattern_selection.setCurrentText(kwargs['pattern'])
        self.input_param1.setText(str(kwargs['p1']))
        self.input_param2.setText(str(kwargs['p2']))
        self.input_dwell_time.setText(str(kwargs['dwell_time']))
        self.input_pitch.setText(str(kwargs['pitch']))
        self.input_repeats.setText(str(kwargs['repeats']))
        self.stream_file_filename_display.setText(kwargs['stream_file_filename'])
        self.settings_filename_display.setText(kwargs['settings_filename'])
        self.precursor_parameters_filename_display.setText(kwargs['precursor_parameters_filename'])
        self.input_simulation_data_interval.setText(kwargs['stats_interval'])
        self.input_structure_snapshot_interval.setText(kwargs['snapshot_interval'])
        self.save_folder_display.setText(kwargs['save_directory'])
        self.input_unique_name.setText(kwargs['unique_name'])


def start(config_filename=None):
    app = QApplication(sys.argv)
    win1 = MainPannel(config_filename)
    sys.exit(app.exec())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win1 = MainPannel()
    sys.exit(app.exec())

