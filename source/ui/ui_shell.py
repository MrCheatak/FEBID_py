import os, sys
import copy
import random
from typing import Union

from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox
from PyQt5 import QtWidgets, QtGui
from PyQt5.uic import loadUi

from main_window import Ui_MainWindow as UI_MainPanel
from warning_ui import  Ui_Dialog as UI_Warning

import pyvista as pv
import yaml

from VTK_Rendering import open_deposited_structure
from Structure import Structure

class MainPannel(QMainWindow, UI_MainPanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.show()

        # Parameters
        self.structure_source = 'vtk'
        self.pattern_source = 'simple'
        self.vtk_filename = ''
        self.geom_parameters_filename = ''
        self.vtk_choice_to_gray = [self.l_width, self.l_width_mc, self.l_height,self.l_height_mc,
                                   self.l_length,self.l_length_mc, self.l_cell_size, self.l_cell_size_mc,
                                   self.l_substrate_height, self.l_substrate_height_mc,
                                   self.l_nm_1, self.l_nm_2, self.l_nm_3,
                                   self.l_nm_1_mc, self.l_nm_2_mc, self.l_nm_3_mc]
        self.vtk_choice_to_disable = [self.input_geom_param_width, self.input_geom_param_width_mc,
                                      self.input_geom_param_length, self.input_geom_param_length_mc,
                                      self.input_geom_param_height, self.input_geom_param_height_mc,
                                      self.input_cell_size, self.input_cell_size_mc,
                                      self.input_substrate_height, self.input_substrate_height_mc]
        self.open_geom_parameters_file_button.setDisabled(True)
        self.open_geom_parameters_file_button_mc.setDisabled(True)
        for obj in self.vtk_choice_to_disable:
            obj.setDisabled(True)
        # filename = '/Users/sandrik1742/Documents/PycharmProjects/FEBID/Experiment runs/gr=0/0k_of_500_loops_k_gr0 14:10:52.vtk'
        # self.open_vtk_file(filename)
        # g_p_filename =

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

    def auto_chosen(self):
        self.structure_source = 'auto'
        self.open_vtk_file_button.setDisabled(True)
        self.open_vtk_file_button_mc.setDisabled(True)
        self.vtk_filename_display.setDisabled(True)
        self.vtk_filename_display_mc.setDisabled(True)
        self.open_geom_parameters_file_button.setDisabled(True)
        self.open_geom_parameters_file_button_mc.setDisabled(True)
        for obj in self.vtk_choice_to_disable:
            obj.setReadOnly(True)
            obj.setDisabled(True)
        for obj in self.vtk_choice_to_gray:
            obj.setDisabled(True)

    def simple_pattern_chosen(self):
        self.pattern_source = 'simple'
        self.pattern_selection.setEnabled(True)
        self.l_param1.setEnabled(True)
        self.input_param1.setEnabled(True)
        self.pattern_selection_changed()
        self.open_stream_file_button.setDisabled(True)
        self.stream_file_filename_display.setDisabled(True)

    def stream_file_chosen(self):
        self.pattern_source = 'stream_file'
        self.open_stream_file_button.setEnabled(True)
        self.stream_file_filename_display.setEnabled(True)
        self.pattern_selection_changed(disable_all=True)
        self.pattern_selection.setDisabled(True)

    def pattern_selection_changed(self, disable_all=False):
        if self.pattern_selection.currentIndex() == 0: # Point
            self.l_param1.setText('x:')
            self.l_param2.setText('y:')
            self.l_param2.setEnabled(True)
            self.input_param2.setEnabled(True)
        if self.pattern_selection.currentIndex() == 1: # Rectangle
            self.l_param1.setText('a:')
            self.l_param2.setText('b:')
            self.l_param2.setEnabled(True)
            self.input_param2.setEnabled(True)
        if self.pattern_selection.currentIndex() == 2: # Square
            self.l_param1.setText('a:')
            self.l_param2.setText('b:')
            self.l_param2.setDisabled(True)
            self.input_param2.setDisabled(True)
        if self.pattern_selection.currentIndex() == 3: # Triangle
            self.l_param1.setText('a:')
            self.l_param2.setText('')
            self.l_param2.setDisabled(True)
            self.input_param2.setDisabled(True)
        if self.pattern_selection.currentIndex() == 4: # Circle
            self.l_param1.setText('r:')
            self.l_param2.setText('')
            self.l_param2.setDisabled(True)
            self.input_param2.setDisabled(True)
        if disable_all:
            self.l_param1.setDisabled(True)
            self.l_param2.setDisabled(True)
            self.input_param1.setDisabled(True)
            self.input_param2.setDisabled(True)

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
            cell_dim = params['cell_dimension']
            xdim = str(int(params['width']*cell_dim))
            ydim = str(int(params['length']*cell_dim))
            zdim = str(int(params['height']*cell_dim))
            cell_dim = str(cell_dim)
            substrate_height = str(int(params['substrate_height']))
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
        except Exception as e:
            print("Was unable to open .yaml precursor parameters file. Following error occurred:")
            print(e.args)

    def start_febid(self):
        pass
    def start_mc(self):
        pass

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
            self.view_message('Input is invalid.')
            lineEdit.clear()
        a=0

    def is_float(self, element) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    def view_message(self, message="An error occurred."):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText(message)
        msgBox.exec()

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

