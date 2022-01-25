import os, sys
import copy
import random
from typing import Union

from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox
from PyQt5 import QtWidgets, QtGui

from main_window import Ui_MainWindow as UI_MainPanel
from warning_ui import  Ui_Dialog as UI_Warning

import pyvista as pv
import yaml

import febid_core
import simple_patterns as sp
from VTK_Rendering import open_deposited_structure
from Structure import Structure


class MainPannel(QMainWindow, UI_MainPanel):
    def __init__(self, test_kwargs=None, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.show()

        # Parameters
        self.structure_source = 'vtk' # vtk, geom or auto
        self.pattern_source = 'simple' # simple or stream_file
        self.vtk_filename = ''
        self.geom_parameters_filename = ''
        self.stream_file_filename = ''
        self.beam_parameters_filename = ''
        self.precursor_parameters_filename = ''
        self.save_directory = ''
        # Groups of controls on the panel for quick Enabling/Disabling
        self.vtk_choice_to_gray = [self.l_width, self.l_width_mc, self.l_height,self.l_height_mc,
                                   self.l_length,self.l_length_mc, self.l_cell_size, self.l_cell_size_mc,
                                   self.l_substrate_height, self.l_substrate_height_mc,
                                   self.l_dimensions_units, self.l_cell_size_units, self.l_substrate_height_units,
                                   self.l_dimensions_units_mc, self.l_cell_size_units_mc, self.l_substrate_height_units_mc,]
        self.vtk_choice_to_disable = [self.input_geom_param_width, self.input_geom_param_width_mc,
                                      self.input_geom_param_length, self.input_geom_param_length_mc,
                                      self.input_geom_param_height, self.input_geom_param_height_mc,
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
        if test_kwargs:
            self.inject_parameters(test_kwargs)

    def vtk_chosen(self):
        # For both tabs:
        #  Gray out labels for other two options
        #  Disable input fields for other options
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

    def geom_parameters_chosen(self):
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
        self.simple_pattern_chosen()

    def auto_chosen(self):
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
        self.stream_file_chosen()

    def simple_pattern_chosen(self):
        self.pattern_source = 'simple'
        for obj in self.simple_pattern_controls:
            obj.setEnabled(True)
        self.open_stream_file_button.setDisabled(True)
        self.stream_file_filename_display.setDisabled(True)

    def stream_file_chosen(self):
        self.pattern_source = 'stream_file'
        for obj in self.simple_pattern_controls:
            obj.setEnabled(False)
        self.open_stream_file_button.setEnabled(True)
        self.stream_file_filename_display.setEnabled(True)

    def pattern_selection_changed(self, current=False):
        if current == 'Point':
            self.set_params_ui(self.pattern_param1_controls, 'x:', True)
            self.set_params_ui(self.pattern_param2_controls, 'y:', True)
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
                substrate_height = str((structure == -2).nonzero()[0].max()+1)
            except:
                substrate_height = 'nan'
            self.input_geom_param_width.setText(xdim)
            self.input_geom_param_width_mc.setText(xdim)
            self.input_geom_param_length.setText(ydim)
            self.input_geom_param_length_mc.setText(ydim)
            self.input_geom_param_height.setText(zdim)
            self.input_geom_param_height_mc.setText(zdim)
            self.input_cell_size.setText(cell_dim)
            self.input_cell_size_mc.setText(cell_dim)
            self.input_substrate_height.setText(substrate_height)
            self.input_substrate_height_mc.setText(substrate_height)
            self.vtk_filename_display.setText(file)
            self.vtk_filename_display_mc.setText(file)
        except Exception as e:
            print("Was unable to open .vtk file. Following error occurred:")
            print(e.args)

    def open_geom_parameters_file(self, file=''):
        if not file:
            file,_ = QtWidgets.QFileDialog.getOpenFileName()
        if not file:
            return
        try:
            params = yaml.load(open(file), Loader=yaml.Loader)
            cell_dim = str(params['cell_dimension'])
            xdim = str(params['width'])
            ydim = str(params['length'])
            zdim = str(params['height'])
            substrate_height = str(params['substrate_height'])
            self.input_geom_param_width.setText(xdim)
            self.input_geom_param_width_mc.setText(xdim)
            self.input_geom_param_length.setText(ydim)
            self.input_geom_param_length_mc.setText(ydim)
            self.input_geom_param_height.setText(zdim)
            self.input_geom_param_height_mc.setText(zdim)
            self.input_cell_size.setText(cell_dim)
            self.input_cell_size_mc.setText(cell_dim)
            self.input_substrate_height.setText(substrate_height)
            self.input_substrate_height_mc.setText(substrate_height)
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
    def open_beam_parameters_file(self, file=''):
        file,_ = QtWidgets.QFileDialog.getOpenFileName()
        if not file:
            return
        ### Read and insert parameters in Monte Carlo tab
        try:
            params = yaml.load(open(file), Loader=yaml.Loader)
            self.beam_energy.setText(str(params['beam_energy']))
            self.energy_cutoff.setText(str(params['minimum_energy']))
            self.gauss_dev.setText(str(params['gauss_dev']))
            self.beam_parameters_filename_display.setText(file)
            self.beam_parameters_filename_display_mc.setText(file)
            self.beam_parameters_filename = file
        except Exception as e:
            print("Was unable to open .yaml beam parameters file. Following error occurred:")
            print(e.args)
    def open_precursor_parameters_file(self, file=''):
        file,_ = QtWidgets.QFileDialog.getOpenFileName()
        if not file:
            return
        try:
            params = yaml.load(open(file), Loader=yaml.Loader)
            self.precursor_parameters_filename_display.setText(file)
            self.precursor_parameters_filename_display_mc.setText(file)
            self.precursor_parameters_filename = file
        except Exception as e:
            print("Was unable to open .yaml precursor parameters file. Following error occurred:")
            print(e.args)

    def change_state_save_sim_data(self, param=None):
        switch = True if param else False
        self.input_sim_data_interval.setEnabled(switch)
        self.l_sim_data_interval.setEnabled(switch)
        self.l_sim_data_interval_units.setEnabled(switch)
        if switch or self.checkbox_save_snapshots.isChecked():
            self.input_unique_filename.setEnabled(True)
            self.l_unique_name.setEnabled(True)
            self.open_save_folder_button.setEnabled(True)
            self.save_folder_display.setEnabled(True)
        else:
            self.input_unique_filename.setEnabled(False)
            self.l_unique_name.setEnabled(False)
            self.open_save_folder_button.setEnabled(False)
            self.save_folder_display.setEnabled(False)

    def change_state_save_snapshots(self, param=None):
        switch = True if param else False
        self.input_snapshot_interval.setEnabled(switch)
        self.l_snapshot_interval.setEnabled(switch)
        self.l_snapshot_interval_units.setEnabled(switch)
        if switch or self.checkbox_save_snapshots.isChecked():
            self.input_unique_filename.setEnabled(True)
            self.l_unique_name.setEnabled(True)
            self.open_save_folder_button.setEnabled(True)
            self.save_folder_display.setEnabled(True)
        else:
            self.input_unique_filename.setEnabled(False)
            self.l_unique_name.setEnabled(False)
            self.open_save_folder_button.setEnabled(False)
            self.save_folder_display.setEnabled(False)

    def open_save_directory(self): #implement
        directory = QtWidgets.QFileDialog.getExistingDirectory()
        if directory:
            self.save_folder_display.setText(directory)

    def tab_switched(self, current):
        if current == 0:
            self.resize(self.width(), 630)
        if current == 1:
            self.resize(self.width(), 500)

    def start_febid(self):
        # Creating a simulation volume
        structure = Structure()
        if self.structure_source == 'vtk': # opening from a .vtk file
            structure.load_from_vtk(pv.read(self.vtk_filename))
            cell_dimension = structure.cell_dimension
            substrate_height = structure.substrate_height
        if self.structure_source == 'geom': # creating from geometry parameters
            try:
                cell_dimension = int(self.input_cell_size.text())
                xdim = int(self.input_geom_param_width.text())//cell_dimension # array length
                ydim = int(self.input_geom_param_length.text())//cell_dimension # array length
                zdim = int(self.input_geom_param_height.text())//cell_dimension # array length
                substrate_height = int(self.input_substrate_height.text())//cell_dimension # array length
                structure.create_from_parameters(cell_dimension, xdim, ydim, zdim, substrate_height)
            except Exception as e:
                self.view_message('Input error.',
                                  'An Error occurred while fetching geometry parameters for the simulation volume. \n '
                                  'Check values and try again.')
                print(e.args)
                return
        if self.structure_source == 'auto': # defining it later based on a stream-file
            cell_dimension = int(self.input_cell_size.text())
            substrate_height = int(self.input_substrate_height.text()) // cell_dimension

        # Defining printing path
        dwell_time_units = 1E-6 # input units are in microseconds, internally seconds are used
        printing_path = None
        if self.pattern_source == 'simple': # creating printing path based on the figure and parameters
            if self.structure_source == 'auto':
                self.view_message(f'Input warning.',
                                  f'Not allowed to choose \'Auto\' and \'Simple pattern\' together! \n'
                                  f'Ambiguous required simulation volume.')
                return
            try:
                pattern = self.pattern_selection.currentText()
                p1 = float(self.input_param1.text()) # nm
                p2 = float(self.input_param2.text()) if self.pattern_selection.currentIndex() in [0, 1, 2] else 0 # nm
                dwell_time = int(self.input_dwell_time.text()) * dwell_time_units # s
                pitch = int(self.input_pitch.text()) # nm
                repeats = int(self.input_repeats.text())
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
            printing_path, shape = sp.open_stream_file(self.stream_file_filename)
            if self.structure_source != 'auto':
                if printing_path[:,0].max() > structure.xdim_abs or printing_path[:,1].max() > structure.ydim_abs:
                    self.view_message('Incompatible dimensions.',
                                      f'The specified simulation volume does not enclose the printing path from '
                                      f'the stream-file. Increase base size or choose \'Auto\' \n'
                                      f'Specified stream-file uses '
                                      f'{printing_path[:,0].max()} x {printing_path[:,1].max()} nm area.')
                    return
            else:
                shape = shape[::-1]//cell_dimension
                structure.create_from_parameters(cell_dimension, *shape, substrate_height)

        # Opening beam and precursor files
        beam_params = yaml.load(open(self.beam_parameters_filename), Loader=yaml.Loader)
        precursor_params = yaml.load(open(self.precursor_parameters_filename, 'r', encoding='UTF-8'), Loader=yaml.Loader)

        sim_volume_params = {} # array length
        sim_volume_params['width'] = structure.shape[2]
        sim_volume_params['length'] = structure.shape[1]
        sim_volume_params['height'] = structure.shape[0]
        sim_volume_params['cell_dimension'] = cell_dimension
        sim_volume_params['substrate_height'] = substrate_height

        # Collecting parameters of file saving
        saving_params = {'monitoring': None, 'snapshot': None, 'filename': None}
        flag1, flag2 = self.checkbox_save_simulation_data.isChecked(), self.checkbox_save_simulation_data.isChecked()
        if flag1:
            saving_params['monitoring'] = float(self.input_sim_data_interval.text())
        if flag2:
            saving_params['snapshot'] = float(self.input_snapshot_interval.text())
        if flag1 or flag2:
            saving_params['filename'] = os.path.join(self.save_directory, self.input_unique_filename.text())

        # Starting the process
        febid_core.run_febid_interface(structure, precursor_params, beam_params, sim_volume_params, printing_path, saving_params)

        return

    def start_mc(self):
        pass

    # Utilities
    def set_params_ui(self, param, name=None, enable=True):
        if name:
            param[0].setText(name)
        for element in param:
            element.setEnabled(enable)
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
        lineEdit = self.sender()
        text = lineEdit.text()
        if self.is_float(text):
            if float(text) >= 0:
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
    def view_message(self, message="An error occurred.", additional_message='', icon='Warning'):
        if icon == 'Warning':
            icon = QMessageBox.Warning
        if icon == 'Question':
            icon = QMessageBox.Question
        if icon == 'Information':
            icon = QMessageBox.Information
        if icon == 'Critical':
            icon = QMessageBox.Critical
        if icon not in ['Warning', 'Question', 'Information', 'Critical']:
            icon = QMessageBox.NoIcon
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText(message + ' '*len(additional_message)) # QMessageBox resizes only with the length of the main text
        msgBox.setInformativeText(additional_message)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.exec()

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
        self.input_geom_param_width.setText(str(kwargs['width']))
        self.input_geom_param_length.setText(str(kwargs['length']))
        self.input_geom_param_height.setText(str(kwargs['height']))
        self.input_cell_size.setText(str(kwargs['cell_dim']))
        self.input_substrate_height.setText(str(kwargs['substrate_height']))
        self.input_param1.setText(str(kwargs['p1']))
        self.input_param2.setText(str(kwargs['p2']))
        self.input_dwell_time.setText(str(kwargs['dwell_time']))
        self.input_pitch.setText(str(kwargs['pitch']))
        self.input_repeats.setText(str(kwargs['repeats']))
        self.stream_file_filename_display.setText(kwargs['stream_file_filename'])
        self.beam_parameters_filename_display.setText(kwargs['beam_parameters_filename'])
        self.precursor_parameters_filename_display.setText(kwargs['precursor_parameters_filename'])
        self.input_sim_data_interval.setText(kwargs['stats_interval'])
        self.input_snapshot_interval.setText(kwargs['snapshot_interval'])
        self.save_folder_display.setText(kwargs['save_directory'])
        self.input_unique_filename.setText(kwargs['unique_filename'])



class Warning(QDialog, UI_Warning):
    def __init__(self, message="An error occurred.", parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.label.setText(message)
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win1 = MainPannel()
    sys.exit(app.exec())

