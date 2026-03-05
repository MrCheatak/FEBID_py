import os, sys
import traceback
from importlib.metadata import PackageNotFoundError, version

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLineEdit
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot

from febid.ui.main_window import Ui_MainWindow as UI_MainPanel

import pyvista as pv
import yaml

from febid.Structure import Structure
from febid.libraries.vtk_rendering.VTK_Rendering import read_field_data

from febid.ui.process_viz import RenderWindow
from febid.logging_config import setup_logger
from febid.ui.session_manager import SessionManager
from febid.ui.app_controller import ApplicationController
from febid.ui.ui_helper import UIHelper, RadioButtonGroup
# Setup logger
logger = setup_logger(__name__)


class MainPanel(QMainWindow, UI_MainPanel):
    """
    Main control panel window class
    """
    # Define signals that the view can emit
    start_simulation_requested = pyqtSignal(dict)
    stop_simulation_requested = pyqtSignal()

    def __init__(self, app=None, config_filename=None, parent=None):
        """Initialize main control panel, session state, and UI bindings.

        :param app: Optional Qt application instance.
        :type app: QApplication
        :param config_filename: Optional session file loaded on startup.
        :type config_filename: str
        :param parent: Optional parent widget.
        :type parent: QWidget
        :return: None
        """
        super().__init__(parent)
        self.app = app
        self.controller = None  # Will be set when controller registers
        self.initialized = False
        self.setupUi(self)
        self.setWindowTitle('FEBID Control Panel')
        self.window_size_febid = self.size().width(), self.size().height()
        self.window_size_mc = self.size().width(), 530
        self.stop_febid_button.setVisible(False)
        self.groupBox_visualization.setVisible(False)
        self.show()
        # self.tab_switched(self.tabWidget.currentIndex())
        self.ui_helper = UIHelper(self)
        # Getting radio buttons for compatability with UIConfigMapper
        self.radio_buttons_structure_source = self.ui_helper.radio_buttons_structure_source
        self.radio_buttons_pattern_source = self.ui_helper.radio_buttons_pattern_source

        # Parameters
        if config_filename is not None:
            self.last_session_filename = config_filename
        else:
            self.last_session_filename = 'last_session.yml'
        self.session_handler: SessionManager = SessionManager()
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
        self.cam_pos = None
        self.viz = None
        self.frame_rate = 1
        self.displayed_data = 'precursor' # Name of the data tp be visualized
        self.frame_rate_control_tick_size = 0.1  # s

        self.config_mapper = UIConfigMapper(self)  # Pass self (the UI) to the mapper
        self.load_last_session()
        self.update_ui()

        self.initialized = True
        self.statusBar().showMessage('Ready')

    def update_ui(self):
        """
        Update UI appearance based on the current configuration
        """
        structure_source = self.session_handler.params['structure_source']
        if structure_source == 'vtk':
            self.vtk_chosen()
        elif structure_source == 'geom':
            self.geom_parameters_chosen()
        elif structure_source == 'auto':
            self.auto_chosen()
        pattern_source = self.session_handler.params['pattern_source']
        if pattern_source == 'simple':
            self.simple_pattern_chosen()
        elif pattern_source == 'stream_file':
            self.stream_file_chosen()

    # Slots
    def change_state_load_last_session(self, param=None):
        """Toggle auto-loading of the last session file.

        :param param: Truthy value enables loading at startup.
        :type param: bool
        :return: None
        """
        switch = True if param else False
        self.checkbox_load_last_session.setChecked(switch)
        self.save_flag = switch
        if switch and self.initialized:
            self.load_last_session()
        self.__save_parameter('load_last_session', switch)

    def vtk_chosen(self):
        """Switch structure-source mode to VTK input.

        :return: None
        """
        self.structure_source = 'vtk'
        self.ui_helper.set_vtk_chosen()
        self.__save_parameter('structure_source', self.structure_source)

    def geom_parameters_chosen(self):
        """Switch structure-source mode to geometry parameters.

        :return: None
        """
        self.structure_source = 'geom'
        self.ui_helper.set_geom_chosen()

        # Auto-switching to Simple patterns option
        self.choice_simple_pattern.setChecked(True)
        self.simple_pattern_chosen()

        self.__save_parameter('structure_source', 'geom')

    def auto_chosen(self):
        """Switch structure-source mode to auto volume generation.

        :return: None
        """
        self.structure_source = 'auto'
        self.ui_helper.set_auto_chosen()

        # Auto-switching to Stream-file option
        self.choice_stream_file.setChecked(True)
        self.stream_file_chosen()

        self.__save_parameter('structure_source', 'auto')

    def simple_pattern_chosen(self):
        """Switch path-source mode to built-in simple patterns.

        :return: None
        """
        self.ui_helper.set_simple_pattern_chosen()
        self.__save_parameter('pattern_source', 'simple')

    def stream_file_chosen(self):
        """Switch path-source mode to stream-file input.

        :return: None
        """
        self.ui_helper.set_stream_file_chosen()
        self.__save_parameter('pattern_source', 'stream_file')

    def pattern_selection_changed(self, current=''):
        """Handle pattern selection changes and persist the selected pattern.

        :param current: Newly selected pattern name.
        :type current: str
        :return: None
        """
        self.ui_helper.set_simple_pattern_change(current)
        self.pattern = current
        self.__save_parameter('pattern', current)

    def open_vtk_file(self, file=''):
        """Load VTK file, extract dimensions, and update related UI/session fields.

        :param file: Path to VTK file; dialog is used when empty.
        :type file: str
        :return: None
        """
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
            logger.exception("Was unable to open .vtk file.")
        self.statusBar().showMessage('VTK file loaded')

    def open_geom_parameters_file(self, file=''):
        """Load geometry-parameter YAML and apply values to FEBID controls.

        :param file: Path to geometry YAML file; dialog is used when empty.
        :type file: str
        :return: None
        """
        if not file:
            file = self.__get_file_name_from_dialog()
            if not file:
                return
        try:
            with open(file, mode='rb') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
            self.__save_parameter('geom_parameters_filename', file)
            # Setting FEBID panel
            self.config_mapper.apply_config_to_ui(params)
        except Exception as e:
            logger.exception("Was unable to open .yml geometry parameters file.")

    def open_stream_file(self, file=''):
        """Set stream-file path used for pattern import.

        :param file: Path to stream file; dialog is used when empty.
        :type file: str
        :return: None
        """
        if not file:
            file = self.__get_file_name_from_dialog()
            if not file:
                return
        self.stream_file_filename = file
        self.stream_file_filename_display.setText(file)
        self.__save_parameter('stream_file_filename', file)

    def open_settings_file(self, file=''):
        """Load beam/settings YAML and mirror key values in the UI.

        :param file: Path to settings YAML; dialog is used when empty.
        :type file: str
        :return: None
        """
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
            logger.exception("Was unable to open .yaml settings file.")
        self.statusBar().showMessage('Settings loaded')

    def open_precursor_parameters_file(self, file=''):
        """Load precursor-parameter YAML and persist selected file path.

        :param file: Path to precursor YAML; dialog is used when empty.
        :type file: str
        :return: None
        """
        if not file:
            file = self.__get_file_name_from_dialog()
            if not file:
                return
        try:
            with open(file, mode='rb') as f:
                _ = yaml.load(f, Loader=yaml.FullLoader)
            self.precursor_parameters_filename_display.setText(file)
            self.precursor_parameters_filename_display_mc.setText(file)
            self.precursor_parameters_filename = file
            self.__save_parameter('precursor_filename', file)
        except Exception as e:
            self.__view_message('File read error',
                                'Specified file is not a valid parameters file. Please choose a valid .yml file.')
            logger.exception("Was unable to open .yml precursor parameters file.")
        self.statusBar().showMessage('Precursor parameters loaded')

    def change_state_save_sim_data(self, param=None):
        """Toggle periodic simulation-data export controls and state.

        :param param: Truthy value enables simulation-data export.
        :type param: bool
        :return: None
        """
        switch = bool(param)
        self.ui_helper.set_state_save_sim_data(switch)
        self.__save_parameter('save_simulation_data', switch)

    def change_state_save_snapshots(self, param=None):
        """Toggle structure-snapshot export controls and state.

        :param param: Truthy value enables snapshot export.
        :type param: bool
        :return: None
        """
        switch = bool(param)
        self.ui_helper.set_state_save_snapshots(switch)
        self.__save_parameter('save_structure_snapshot', switch)

    def change_state_temperature_tracking(self, param):
        """Toggle temperature-tracking option in UI and session state.

        :param param: Truthy value enables temperature tracking.
        :type param: bool
        :return: None
        """
        switch = bool(param)
        self.checkbox_temperature_tracking.setChecked(switch)
        self.__save_parameter('temperature_tracking', switch)

    def unique_name_changed(self):
        """Persist current run-name value from the UI.

        :return: None
        """
        self.__save_parameter('unique_name', self.input_unique_name.text())

    def change_state_gpu(self, param):
        """Toggle GPU execution flag in UI and session state.

        :param param: Truthy value enables GPU mode.
        :type param: bool
        :return: None
        """
        switch = bool(param)
        self.checkbox_gpu.setChecked(switch)
        self.__save_parameter('gpu', switch)

    def open_save_directory(self, directory=''):
        """Select and persist directory used for simulation outputs.

        :param directory: Target directory; chooser dialog is used when empty.
        :type directory: str
        :return: None
        """
        if not directory:
            directory = QtWidgets.QFileDialog.getExistingDirectory()
            if not directory:
                return
        if directory:
            self.save_folder_display.setText(directory)
        self.__save_parameter('save_directory', directory)

    def tab_switched(self, current):
        """Resize window for FEBID or Monte Carlo tab layouts.

        :param current: Active tab index.
        :type current: int
        :return: None
        """
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

    @pyqtSlot(bool)
    def on_actionAbout_triggered(self, checked=False):
        """Show About dialog with application and runtime details."""
        app_version = self.__get_app_version()
        about_text = (
            "<h3>3D FEBID Simulation</h3>"
            "<p>Direct-write nano- and microscale chemical vapor deposition simulator.</p>"
            f"<p><b>Version:</b> {app_version}<br>"
            f"<b>Python:</b> {sys.version.split()[0]}<br>"
            "<b>Developers:</b> Alexander Kuprava, Michael Huth<br>"
            "<b>Affiliation:</b> Institute of Physics, Goethe University, "
            "Frankfurt am Main, Germany</p>"
            "<p><b>Source:</b> https://github.com/MrCheatak/FEBID_py</p>"
        )

        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("About FEBID")
        dialog.setIcon(QMessageBox.Information)
        dialog.setTextFormat(QtCore.Qt.RichText)
        dialog.setText(about_text)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec()

    def start_febid(self):
        """
        Start FEBID simulation
        """
        # Collect parameters from UI and update session config (dict-like)
        params = self.config_mapper.get_config_from_ui()
        try:
            self.start_simulation_requested.emit(params)
            self.groupBox_visualization.setVisible(True)
        except Exception as e:
            self.__exception_handler(e)
            return
        self.start_febid_button.setVisible(False)
        self.stop_febid_button.setVisible(True)


        self.statusBar().showMessage('Simulation running')

    def stop_febid(self):
        """
        Stop FEBID simulation
        """
        self.stop_simulation_requested.emit()
        self.on_finish('Simulation stopped')

    def start_mc(self):
        """
        Start Monte Carlo simulation
        """
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

    def reopen_viz(self):
        """
        Reopen process visualization window
        """
        if type(self.viz) is RenderWindow:
            if self.viz.isVisible():
                return
        self.session_handler.starter.context.process.displayed_data = self.displayed_data # enables data acquisition from GPU
        self.viz = RenderWindow(self.session_handler.starter.context.process, self.session_handler.starter.syncHelper,
                                displayed_data=self.displayed_data, show=True, app=self.app)
        self.viz.start(frame_rate=self.frame_rate)

    def precursor_coverage_viz_chosen(self):
        """Set visualization mode to precursor coverage.

        :return: None
        """
        self.choice_precursor_coverage_viz.setChecked(True)
        self.displayed_data = 'precursor'

    def surface_deposit_viz_chosen(self):
        """Set visualization mode to deposited volume map.

        :return: None
        """
        self.choice_surface_deposit_viz.setChecked(True)
        self.displayed_data = 'deposit'

    def frame_rate_slider_moved(self, tick):
        """Update visualization frame rate from slider ticks.

        :param tick: Slider tick value.
        :type tick: int
        :return: None
        """
        frame_rate = tick * self.frame_rate_control_tick_size
        self.display_frame_rate.setText(str(f'{frame_rate:.1f}'))
        self.frame_rate = frame_rate
        try:
            self.viz.frame_rate = frame_rate
        except AttributeError:
            pass

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
        if os.path.exists(filename):
            self.session_handler.load(filename)
            # Use dict-like config for UI population (preserves comments)
            self.config_mapper.apply_config_to_ui(self.session_handler.params)
        else:
            params = self.config_mapper.get_config_from_ui()
            self.session_handler.create(params)
            if self.save_flag:
                self.session_handler.save(filename)
        logger.info('Loaded last session.')

    @pyqtSlot(str)
    def on_finish(self, message=''):
        """Restore idle UI state after simulation completion.

        :param message: Status-bar message shown after completion.
        :type message: str
        :return: None
        """
        self.start_febid_button.setVisible(True)
        self.stop_febid_button.setVisible(False)
        self.groupBox_visualization.setVisible(False)
        self.statusBar().showMessage(message)

    def on_close(self):
        """Stop active tasks and close visualization resources.

        :return: None
        """
        self.session_handler.stop()
        if self.viz is not None:
            self.viz.close()

    def closeEvent(self, event):
        """Handle Qt window-close event and perform cleanup.

        :param event: Qt close event object.
        :type event: QCloseEvent
        :return: None
        """
        self.on_close()
        event.accept()

    def __get_file_name_from_dialog(self):
        """
        Get file name from a file selection window

        :return: full name of the selected file
        """
        file, _ = QtWidgets.QFileDialog.getOpenFileName()
        return file

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
        self.config_mapper.apply_config_to_ui(interface_set_config)
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
        self.session_handler.set_param(param_name, value)
        if self.save_flag:
            self.session_handler.save(self.last_session_filename)

    @staticmethod
    def __is_int(element) -> bool:
        """Check whether a value can be converted to integer.

        :param element: Value to test.
        :type element: object
        :return: True when conversion to int succeeds.
        """
        try:
            int(element)
            return True
        except ValueError:
            return False

    @staticmethod
    def __view_message(message="An error occurred", additional_message='', icon='Warning'):
        """Show a QMessageBox with mapped icon and optional details.

        :param message: Main message text.
        :type message: str
        :param additional_message: Secondary informative text.
        :type additional_message: str
        :param icon: Icon key (`Warning`, `Question`, `Information`, `Critical`).
        :type icon: str
        :return: None
        """
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

    @staticmethod
    def __get_app_version():
        """Return installed package version or fallback string."""
        try:
            return version("febid")
        except PackageNotFoundError:
            return "development"

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
                        logger.exception(f'An error occurred while reading units from YAML file.')
                        raise
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
        """Load a VTK file and return populated structure plus stored field metadata.

        :param file: Path to VTK file.
        :type file: str
        :return: Tuple of structure instance and parsed field-data payload.
        """
        structure = Structure()
        vtk_obj = pv.read(file)
        structure.load_from_vtk(vtk_obj)
        params = read_field_data(vtk_obj)
        return structure, params

    def __exception_handler(self, e):
        """Display a user-facing message for known and unexpected startup errors.

        :param e: Caught exception.
        :type e: Exception
        :return: None
        """
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

    def validate_config(self):
        """
        Validate the current configuration using the dataclass. Show errors to the user if any.
        """
        params = self.config_mapper.get_config_from_ui()
        self.session_handler.set_all_params(params)
        try:
            self.session_handler.validate()
            self.__view_message('Validation successful', 'The configuration is valid.', icon='Information')
        except Exception as e:
            self.__exception_handler(e)

    def register_ApplicationController(self, controller: ApplicationController):
        """
        Register the application controller to handle application-level events.

        :param controller: ApplicationController instance
        """
        self.controller = controller
        self.start_simulation_requested.connect(controller.on_start_simulation_requested)
        self.stop_simulation_requested.connect(controller.on_stop_simulation_requested)
        self.controller.register_view(self)


def start(config_filename=None):
    """Launch the Qt application with main panel and application controller.

    :param config_filename: Optional startup session file.
    :type config_filename: str
    :return: None
    """
    app = QApplication(sys.argv)
    win1 = MainPanel(config_filename)
    controller = ApplicationController(win1.session_handler)
    win1.register_ApplicationController(controller)
    sys.exit(app.exec())


class UIConfigMapper:
    """
    Maps UI elements to configuration parameters.
    This class is used to simplify the process of updating UI elements based on configuration parameters
    and vice versa.
    """

    def __init__(self, ui: MainPanel):
        """Initialize mapper for synchronizing UI widgets and config parameters.

        :param ui: MainPanel instance containing mapped widgets.
        :type ui: MainPanel
        :return: None
        """
        self.ui = ui
        self._mapping = self._get_mapping()

    def _get_mapping(self):
        """
        Mapping of interface elements to session configuration parameters.
        :return: mapping dictionary
        """
        mapping_of_interface_elements_to_parameters = {
            'load_last_session': self.ui.checkbox_load_last_session,
            'structure_source': self.ui.radio_buttons_structure_source,
            'vtk_filename': self.ui.vtk_filename_display,
            'geom_parameters_filename': self.ui.geom_parameters_filename,
            'width': self.ui.input_width,
            'length': self.ui.input_length,
            'height': self.ui.input_height,
            'cell_size': self.ui.input_cell_size,
            'substrate_height': self.ui.input_substrate_height,
            'pattern_source': self.ui.radio_buttons_pattern_source,
            'pattern': self.ui.pattern_selection,
            'param1': self.ui.input_param1,
            'param2': self.ui.input_param2,
            'dwell_time': self.ui.input_dwell_time,
            'pitch': self.ui.input_pitch,
            'repeats': self.ui.input_repeats,
            'stream_file_filename': self.ui.stream_file_filename_display,
            'hfw': self.ui.input_hfw,
            'settings_filename': self.ui.settings_filename_display,
            'precursor_filename': self.ui.precursor_parameters_filename_display,
            'temperature_tracking': self.ui.checkbox_temperature_tracking,
            'save_simulation_data': self.ui.checkbox_save_simulation_data,
            'save_structure_snapshot': self.ui.checkbox_save_snapshots,
            'simulation_data_interval': self.ui.input_simulation_data_interval,
            'structure_snapshot_interval': self.ui.input_structure_snapshot_interval,
            'unique_name': self.ui.input_unique_name,
            'save_directory': self.ui.save_folder_display,
            'gpu': self.ui.checkbox_gpu
        }
        return mapping_of_interface_elements_to_parameters

    def apply_config_to_ui(self, config: dict):
        """
        Insert values from session configuration into UI widgets.

        :param config: dictionary with widget names and a value to set
        """
        for param_name, widget in self._mapping.items():
            if param_name not in config:
                continue
            value = config[param_name]
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, RadioButtonGroup):
                widget.setChecked(value)

    def get_config_from_ui(self) -> dict:
        """Reads widget values and returns them as a configuration dictionary."""
        """
        Retrieve values from UI to session configuration.
        :return: dictionary with parameters
        """
        mapping = self._mapping
        params = {}
        for parameter, element in mapping.items():
            if element.__class__ == QtWidgets.QCheckBox:
                params[parameter] = element.isChecked()
            elif element.__class__ == QtWidgets.QLineEdit:
                val = self.__infer_type(element)
                params[parameter] = val
            elif element.__class__ == QtWidgets.QComboBox:
                params[parameter] = element.currentText()
            elif element.__class__ == RadioButtonGroup:
                params[parameter] = element.getChecked()
        return params

    def __infer_type(self, element: QtWidgets.QLineEdit):
        """
        Infer the type of the value from the UI element text and convert to that type.

        :param element: UI element to read text from
        :return: value converted to the appropriate type (int, float, or str)
        """
        text = element.text()
        if self.__is_float(text):
            val = float(text)
            if int(val) - val == 0:
                val = int(val)
        else:
            val = text
        return val

    @staticmethod
    def __is_float(element) -> bool:
        """Check whether a value can be parsed as float.

        :param element: Value to test.
        :type element: object
        :return: True when conversion to float succeeds.
        """
        try:
            float(element)
            return True
        except ValueError:
            return False

        
if __name__ == "__main__":
    start()
