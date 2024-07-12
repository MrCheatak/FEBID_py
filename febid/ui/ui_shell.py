import math
import os, sys
import faulthandler
import traceback

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLineEdit
from PyQt5 import QtWidgets

from febid.ui.main_window import Ui_MainWindow as UI_MainPanel

import pyvista as pv
import yaml
from ruamel.yaml import YAML, CommentedMap

from febid.start import Starter
from febid.Structure import Structure
from febid.monte_carlo import etraj3d as e3d
from febid.libraries.vtk_rendering.VTK_Rendering import read_field_data


class SessionHandler:
    """
    Class for creating, loading and saving sessions (interface configuration)
    """
    def __init__(self):
        self.params = CommentedMap()
        self.starter = Starter()
        self.starter.params = self.params
        self.default_config_stub = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'last_session_stub.yml')

    def load_session(self, filename):
        """
        Load session configuration from a file

        :param filename: full file name
        """
        try:
            with open(filename, mode='rb') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
                self.params.update(params)
        except FileNotFoundError as e:
            print('Session file not found')
            raise e

    def create_session(self, params):
        """
        Create new session configuration and set it up

        :param params: session configuration parameters
        """
        self.load_empty_config()
        for param in params:
            self.set_parameter(param, params[param])

    def load_empty_config(self):
        """
        Configuration file template

        :return:
        """
        self.load_session(self.default_config_stub)

    def save_session(self, filename):
        """
        Save current configuration to a file

        :param filename: full file name
        """
        # self.filename = filename
        yml = YAML()
        with open(filename, mode='wb') as f:
            yml.dump(self.params, f, )

    def open_vtk_file(self, filename):
        """
        Open VTK file and load a structure from it.

        :param filename: full file name
        :return:
        """
        structure = Structure()
        vtk_obj = pv.read(filename)
        structure.load_from_vtk(vtk_obj)
        params = read_field_data(vtk_obj)
        return structure, params

    def set_parameter(self, name, value):
        """
        Set the value of a parameter in the session configuration.

        :param name: parameter name
        :param value: parameter value
        """
        if name in self.params:
            self.params[name] = value

    def start(self, module='febid', **kwargs):
        """
        Start the simulation

        :return:
        """
        if module == 'febid':
            return self.starter.start()
        elif module == 'monte_carlo':
            self.starter.start_mc(**kwargs)
        else:
            raise ValueError(f'Unknown module: {module}')


class UI_Group(list):
    """
    A collection of UI elements.
    """

    def __init__(self, *args):
        super().__init__()
        if type(args[0]) in [set, list, tuple]:
            self.extend(args[0])
        else:
            for arg in args:
                self.append(arg)

    def set(self):
        """
        Convert to set
        """
        return set(self)

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


class RadioButtonGroup(UI_Group):
    """
    A collection of bound radio buttons.
    """
    def __init__(self, *args, names=None):
        super().__init__(*args)
        self.names = names

    def setChecked(self, param):
        """
        Check one of the radio buttons based on its name.

        :param param: name of the button to check
        :return:
        """
        index = self.names.index(param)
        self[index].setChecked(True)

    def getChecked(self):
        """
        Get the name of the checked button.

        :return: name of checked button
        """
        for i, button in enumerate(self):
            if button.isChecked():
                return self.names[i]


class MainPanel(QMainWindow, UI_MainPanel):
    """
    Main control panel window class
    """
    def __init__(self, config_filename=None, parent=None):
        super().__init__(parent)
        self.initialized = False
        self.setupUi(self)
        self.show()
        self.tab_switched(self.tabWidget.currentIndex())
        self.__group_interface_elements()
        self.__aggregate_radio_buttons()
        # Parameters
        if config_filename is not None:
            self.last_session_filename = config_filename
        else:
            self.last_session_filename = 'last_session.yml'
        self.session_handler = SessionHandler()
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

        self.load_last_session()
        self.initialized = True

    # Slots
    def change_state_load_last_session(self, param=None):
        switch = True if param else False
        self.checkbox_load_last_session.setChecked(switch)
        self.save_flag = switch
        if switch and self.initialized:
            self.load_last_session()
        self.__save_parameter('load_last_session', switch)

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

        self.__save_parameter('structure_source', self.structure_source)

    def geom_parameters_chosen(self):
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

        self.__save_parameter('structure_source', 'geom')

    def auto_chosen(self):
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

        self.__save_parameter('structure_source', 'auto')

    def simple_pattern_chosen(self):
        # Changing FEBID tab interface
        self.choice_simple_pattern.setChecked(True)
        self.ui_pattern.disable()
        self.ui_simple_patterns.enable()
        self.__save_parameter('pattern_source', 'simple')

    def stream_file_chosen(self):
        # Changing FEBID tab interface
        self.choice_stream_file.setChecked(True)
        self.ui_pattern.disable()
        self.ui_stream_file.enable()
        self.__save_parameter('pattern_source', 'stream_file')

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
        self.__save_parameter('pattern', current)

    def open_vtk_file(self, file=''):
        # For both tabs:
        #  Check if the specified file is a valid .vtk file
        #  Insert parameters into fields
        if not file:
            file = self.__get_file_name_from_dialog()
            if not file:
                return
        try:
            self.__set_dimensions_from_vtk(file)
            self.__save_parameter('vtk_filename', file)
        except Exception as e:
            self.__view_message('File read error',
                                'Specified file is not a valid VTK file. Please choose a valid .vtk file.')
            print("Was unable to open .vtk file. Following error occurred:")
            print(e.args)

    def open_geom_parameters_file(self, file=''):
        if not file:
            file = self.__get_file_name_from_dialog()
            if not file:
                return
        try:
            with open(file, mode='rb') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
            self.__save_parameter('geom_parameters_filename', file)
            # Setting FEBID panel
            self.set_interface_from_config(params)
        except Exception as e:
            print("Was unable to open .yml geometry parameters file. Following error occurred:")
            print(e.args)

    def open_stream_file(self, file=''):
        if not file:
            file = self.__get_file_name_from_dialog()
            if not file:
                return
        self.stream_file_filename = file
        self.stream_file_filename_display.setText(file)
        self.__save_parameter('stream_file_filename', file)

    def open_settings_file(self, file=''):
        if not file:
            file = self.__get_file_name_from_dialog()
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
            self.__save_parameter('settings_filename', file)
        except Exception as e:
            self.__view_message('File read error',
                                'Specified file is not a valid settings file. Please choose a valid .yml file.')
            print("Was unable to open .yaml settings file. Following error occurred:")
            print(e.args)

    def open_precursor_parameters_file(self, file=''):
        if not file:
            file = self.__get_file_name_from_dialog()
            if not file:
                return
        try:
            with open(file, mode='rb') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
            self.precursor_parameters_filename_display.setText(file)
            self.precursor_parameters_filename_display_mc.setText(file)
            self.precursor_parameters_filename = file
            self.__save_parameter('precursor_filename', file)
        except Exception as e:
            self.__view_message('File read error',
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
        self.__save_parameter('save_simulation_data', switch)

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
        self.__save_parameter('save_structure_snapshot', switch)

    def change_state_show_process(self, param=None):
        switch = True if param else False
        self.checkbox_show.setChecked(switch)
        self.__save_parameter('show_process', switch)

    def change_state_temperature_tracking(self, param):
        switch = True if param else False
        self.checkbox_temperature_tracking.setChecked(switch)
        self.__save_parameter('temperature_tracking', switch)

    def unique_name_changed(self):
        self.__save_parameter('unique_name', self.input_unique_name.text())

    def open_save_directory(self, directory=''):
        if not directory:
            directory = QtWidgets.QFileDialog.getExistingDirectory()
            if not directory:
                return
        if directory:
            self.save_folder_display.setText(directory)
        self.__save_parameter('save_directory', directory)

    def tab_switched(self, current):
        if current == 0:
            self.resize(self.width(), 684)
        if current == 1:
            self.resize(self.width(), 500)

    def check_input(self):
        """
        Check if the input line is numeric and not negative and save the parameter to the file
        :return:
        """
        lineEdit: QLineEdit = self.sender()
        text = lineEdit.text()
        try:
            val = float(text)
        except ValueError:
            self.__view_message('Input is invalid.', f'The value entered is not numerical.')
            lineEdit.clear()
            return
        if val >= 0:
            # Here the parameter is saved to the file
            # A naming convention is made: lineEdit objects are named the same
            # as the parameters in the file only with 'input_' in the beginning
            name = lineEdit.objectName()[6:]  # stripping 'input_'
            name = name.replace('_mc', '')  # stripping '_mc' if present
            if self.__is_int(val) / val == 1:
                val = int(val)
            self.__save_parameter(name, val)
        else:
            self.__view_message("Value cannot be negative.")
            lineEdit.clear()

    def open_new_session(self, file=''):
        """
        Load a session from a config file.

        :return:
        """
        if not file:
            file = self.__get_file_name_from_dialog()
            if not file:
                return
        self.load_last_session(file)
        self.last_session_filename = file
        self.checkbox_load_last_session.setToolTip(file)
        self.change_state_load_last_session(True)

    def start_febid(self):
        # Creating a simulation volume
        try:
            self.session_handler.start()
        except Exception as e:
            self.__exception_handler(e)

    def start_mc(self):
        E0 = float(self.beam_energy.text())
        Emin = float(self.energy_cutoff.text())
        gauss_dev = float(self.gauss_dev.text())
        x0 = float(self.x_pos.text())
        y0 = float(self.y_pos.text())
        N = int(self.number_of_e.text())
        n = int(self.gauss_order.text())
        heating = self.checkbox_beam_heating.isChecked()
        params = {'E0': E0, 'Emin': Emin, 'sigma': gauss_dev, 'pos': (x0, y0), 'N': N, 'n': n, 'heating': heating, 'cam_pos': self.cam_pos}
        self.session_handler.start(module='monte_carlo', **params)
        return 1


    # Supporting functions
    def load_last_session(self, filename=''):
        """
        Load session configuration from the file or create a new one.

        File is saved in the current working directory.

        :param filename: name of the file
        :return:
        """
        if not filename:
            filename = self.last_session_filename
        print('Trying to load last session...', end='')
        if os.path.exists(filename):
            self.session_handler.load_session(filename)
            self.set_interface_from_config(self.session_handler.params)
            self.__set_structure_source(self.session_handler.params['structure_source'])
            self.__set_pattern_source(self.session_handler.params['pattern_source'])
        else:
            params = self.get_config_from_ui()
            self.session_handler.create_session(params)
            if self.save_flag:
                self.session_handler.save_session(filename)

        print('done!')

    def ui_to_parameters_mapping(self):
        """
        Mapping of interface elements to session configuration parameters.
        :return: mapping dictionary
        """
        mapping_of_interface_elements_to_parameters = {
            'load_last_session': self.checkbox_load_last_session,
            'structure_source': self.radio_buttons_structure_source,
            'vtk_filename': self.vtk_filename_display,
            'geom_parameters_filename': self.geom_parameters_filename,
            'width': self.input_width,
            'length': self.input_length,
            'height': self.input_height,
            'cell_size': self.input_cell_size,
            'substrate_height': self.input_substrate_height,
            'pattern_source': self.radio_buttons_pattern_source,
            'pattern': self.pattern_selection,
            'param1': self.input_param1,
            'param2': self.input_param2,
            'dwell_time': self.input_dwell_time,
            'pitch': self.input_pitch,
            'repeats': self.input_repeats,
            'stream_file_filename': self.stream_file_filename_display,
            'hfw': self.input_hfw,
            'settings_filename': self.settings_filename_display,
            'precursor_filename': self.precursor_parameters_filename_display,
            'temperature_tracking': self.checkbox_temperature_tracking,
            'save_simulation_data': self.checkbox_save_simulation_data,
            'save_structure_snapshot': self.checkbox_save_snapshots,
            'simulation_data_interval': self.input_simulation_data_interval,
            'structure_snapshot_interval': self.input_structure_snapshot_interval,
            'unique_name': self.input_unique_name,
            'save_directory': self.save_folder_display,
            'show_process': self.checkbox_show
        }
        return mapping_of_interface_elements_to_parameters

    def set_interface_from_config(self, parameters=None):
        """
        Insert values from session configuration to UI.
        If parameters is None, use all parameters from current session configuration.
        :param parameters: dictionary with parameters
        """
        mapping = self.ui_to_parameters_mapping()
        if parameters is None:
            param_source = self.session_handler.params
        else:
            param_source = parameters
            mapping = {key: value for key, value in mapping.items() if key in parameters.keys()}
        for parameter, element in mapping.items():
            val = param_source[parameter]
            if element.__class__ == QtWidgets.QCheckBox:
                element.setChecked(val)
            elif element.__class__ == QtWidgets.QLineEdit:
                element.setText(str(val))
            elif element.__class__ == QtWidgets.QComboBox:
                element.setCurrentText(str(val))
            elif element.__class__ == RadioButtonGroup:
                element.setChecked(val)

    def get_config_from_ui(self):
        """
        Retrieve values from UI to session configuration.
        :return: dictionary with parameters
        """
        mapping = self.ui_to_parameters_mapping()
        params = dict()
        for parameter, element in mapping.items():
            if element.__class__ == QtWidgets.QCheckBox:
                params[parameter] = element.isChecked()
            elif element.__class__ == QtWidgets.QLineEdit:
                text = element.text()
                if self.__is_float(text):
                    val = float(text)
                    if int(val) - val == 0:
                        val = int(text)
                else:
                    val = text
                params[parameter] = val
            elif element.__class__ == QtWidgets.QComboBox:
                params[parameter] = element.currentText()
            elif element.__class__ == RadioButtonGroup:
                params[parameter] = element.getChecked()
        return params


    # Helper interface functions
    def __group_interface_elements(self):
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

        self.ui_sim_data_interval = UI_Group(self.l_sim_data_interval, self.input_simulation_data_interval,
                                             self.l_sim_data_interval_units)
        self.ui_snapshot = UI_Group(self.l_snapshot_interval, self.input_structure_snapshot_interval,
                                    self.l_snapshot_interval_units)
        self.ui_unique_name = UI_Group(self.l_unique_name, self.input_unique_name)
        self.ui_save_folder = UI_Group(self.open_save_folder_button, self.save_folder_display)

        # Grouping elements by their designation
        self.ui_vtk_choice = UI_Group(self.open_vtk_file_button, self.vtk_filename_display)
        self.ui_vtk_choice_mc = UI_Group(self.open_vtk_file_button_mc, self.vtk_filename_display_mc)

        self.ui_geom_choice = UI_Group(
            {self.open_geom_parameters_file_button} | self.ui_dimensions.set() | self.ui_cell_size.set() | \
            self.ui_substrate_height.set())
        self.ui_geom_choice_mc = UI_Group({self.open_geom_parameters_file_button_mc} | self.ui_dimensions_mc.set() | \
                                          self.ui_cell_size_mc.set() | self.ui_substrate_height_mc.set())

        self.ui_auto_choice = UI_Group(self.ui_cell_size.set() | self.ui_substrate_height.set())

        self.ui_simple_patterns = UI_Group(
            {self.pattern_selection} | self.ui_pattern_param1.set() | self.ui_pattern_param2.set() | \
            self.ui_dwell_time.set() | self.ui_pitch.set() | self.ui_repeats.set())

        self.ui_stream_file = UI_Group({self.open_stream_file_button} | self.ui_hfw.set())

        # Grouping by the groupBoxes
        self.ui_sim_volume = UI_Group(self.ui_vtk_choice.set() | self.ui_geom_choice.set() | self.ui_auto_choice.set())
        self.ui_sim_volume_mc = UI_Group(self.ui_vtk_choice_mc.set() | self.ui_geom_choice_mc.set())
        self.ui_pattern = UI_Group(self.ui_simple_patterns.set() | self.ui_stream_file.set())

    def __aggregate_radio_buttons(self):
        """
        Aggregate radio buttons into a group.

        :return:
        """
        self.radio_buttons_structure_source = RadioButtonGroup(self.choice_vtk_file, self.choice_geom_parameters_file,
                                                               self.choice_auto, names=['vtk', 'geom', 'auto'])
        self.radio_buttons_pattern_source = RadioButtonGroup(self.choice_simple_pattern, self.choice_stream_file,
                                                             names=['simple', 'stream_file'])

    def __get_file_name_from_dialog(self):
        """
        Get file name from a file selection window

        :return: full name of the selected file
        """
        file, _ = QtWidgets.QFileDialog.getOpenFileName()
        return file

    def __set_structure_source(self, source_name):
        """
        Select the radio button of the specified structure source

        :param source_name:
        :return:
        """
        self.radio_buttons_structure_source.setChecked(source_name)

    def __set_pattern_source(self, source_name):
        """
        Select the radio button of the specified pattern source

        :param source_name:
        :return:
        """
        self.radio_buttons_pattern_source.setChecked(source_name)

    def __set_dimensions_from_vtk(self, file):
        """
        Set displayed structure dimensions from VTK file

        :param params: parameters
        :param structure: structure
        :param file: vtk file name
        """
        structure, params = self.__load_vtk_file(file)
        cell_size = structure.cell_size
        zdim, ydim, xdim = structure.shape_abs
        try:
            substrate_height = str(structure.substrate_height * structure.cell_size)
        except:
            substrate_height = 'nan'
        # Setting FEBID panel
        interface_set_config = {
            'vtk_filename': file,
            'width': xdim,
            'length': ydim,
            'height': zdim,
            'cell_size': cell_size,
            'substrate_height': substrate_height
        }
        self.set_interface_from_config(interface_set_config)
        # Setting MC panel
        self.input_width_mc.setText(str(xdim))
        self.input_length_mc.setText(str(ydim))
        self.input_height_mc.setText(str(zdim))
        self.input_cell_size_mc.setText(str(cell_size))
        self.input_substrate_height_mc.setText(str(substrate_height))
        self.vtk_filename_display_mc.setText(file)
        if params[2] is not None:
            self.x_pos.setText(str(params[2][0]))
            self.y_pos.setText(str(params[2][1]))


    # Utilities
    def __save_parameter(self, param_name, value):
        """
        Change the specified parameter and write current configuration to the file
        :param param_name: name of the parameter
        :param value: value of the parameter
        :return:
        """
        self.session_handler.set_parameter(param_name, value)
        if self.save_flag:
            self.session_handler.save_session(self.last_session_filename)

    @staticmethod
    def __is_float(element) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    @staticmethod
    def __is_int(element) -> bool:
        try:
            int(element)
            return True
        except ValueError:
            return False

    @staticmethod
    def __view_message(message="An error occurred", additional_message='', icon='Warning'):
        icon_mapping = {
            'Warning': QMessageBox.Warning,
            'Question': QMessageBox.Question,
            'Information': QMessageBox.Information,
            'Critical': QMessageBox.Critical
        }
        if icon not in icon_mapping:
            icon = QMessageBox.NoIcon
        else:
            icon = icon_mapping[icon]
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

    def __load_vtk_file(self, file):
        structure = Structure()
        vtk_obj = pv.read(file)
        structure.load_from_vtk(vtk_obj)
        params = read_field_data(vtk_obj)
        return structure, params

    def __exception_handler(self, e):
        known_errnos = {
                        1: 'VTK file not specified. Please choose the file and try again.',
                        2: 'An error occurred while fetching geometry parameters for the simulation volume.',
                        3: "Not allowed to choose 'Auto' and 'Simple pattern' together! \nAmbiguous required area.",
                        4: 'Stream-file not specified. Please choose the file and try again.',
                        5: 'Printing path is out of simulation volume',
                        6: 'Beam parameters file not specified. Please choose the file and try again.',
                        7: 'Precursor parameters file not specified. Please choose the file and try again.',
                        8: 'Unique name not given. Please enter a name and try again.',
                        9: 'Inconsistent temperature tracking setup. Missing parameters.'
                       }
        if hasattr(e, 'errno'):
            errno = e.errno
        else:
            errno = -1
        if errno in known_errnos:
            self.__view_message('An error occurred', e.args[0], icon='Warning')
        else:
            traceback_output = traceback.format_exc()
            self.__view_message('An unknownerror occurred', traceback_output, icon='Critical')
            # raise e

    # def interface_to_config_map(self):
    #     """
    #     Mapping between UI and config file
    #
    #     :return:
    #     """
    #     self.session['load_last_session'] = self.save_flag
    #     self.session['structure_source'] = self.structure_source
    #     self.session['vtk_filename'] = self.vtk_filename
    #     self.session['geom_parameters_filename'] = self.geom_parameters_filename
    #     self.session['width'] = int(self.input_width.text())
    #     self.session['length'] = int(self.input_length.text())
    #     self.session['height'] = int(self.input_height.text())
    #     self.session['cell_size'] = int(self.input_cell_size.text())
    #     self.session['substrate_height'] = int(self.input_substrate_height.text())
    #     self.session['pattern_source'] = self.pattern_source
    #     self.session['pattern'] = self.pattern_selection.currentText()
    #     self.session['param1'] = float(self.input_param1.text())
    #     self.session['param2'] = float(self.input_param2.text())
    #     self.session['dwell_time'] = int(self.input_dwell_time.text())
    #     self.session['pitch'] = int(self.input_pitch.text())
    #     self.session['repeats'] = int(self.input_repeats.text())
    #     self.session['stream_file_filename'] = self.stream_file_filename
    #     self.session['hfw'] = float(self.input_hfw.text())
    #     self.session['settings_filename'] = self.settings_filename
    #     self.session['precursor_filename'] = self.precursor_parameters_filename
    #     self.session['temperature_tracking'] = self.temperature_tracking
    #     self.session['save_simulation_data'] = self.checkbox_save_simulation_data.isChecked()
    #     self.session['save_structure_snapshot'] = self.checkbox_save_snapshots.isChecked()
    #     self.session['simulation_data_interval'] = float(self.input_sim_data_interval.text())
    #     self.session['structure_snapshot_interval'] = float(self.input_snapshot_interval.text())
    #     self.session['unique_name'] = self.input_unique_name.text()
    #     self.session['save_directory'] = self.save_directory
    #     self.session['show_process'] = self.checkbox_show.isChecked()


def start(config_filename=None):
    app = QApplication(sys.argv)
    win1 = MainPanel(config_filename)
    sys.exit(app.exec())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win1 = MainPanel()
    sys.exit(app.exec())
