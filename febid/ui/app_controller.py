from threading import Thread

from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
from .session_manager import SessionManager
from febid.logging_config import setup_logger

logger = setup_logger(__name__)

class ApplicationController(QObject):
    """
    Application Controller for the FEBID simulation UI.
    This class acts as the main controller for the application, handling user requests.
    It manages application's lifecycle and listens for signals from the UI.
    """
    simulation_finished = pyqtSignal(str)  # Signal to emit when simulation finishes

    def __init__(self, session_manager):
        super().__init__()
        self.session_manager = session_manager

    @pyqtSlot(dict)
    def on_start_simulation_requested(self, config_params):
        """This slot is triggered when the UI's start button is clicked."""
        logger.debug("Application Controller received start request.")
        try:
            # The logic from the old start_febid method now lives here.
            self.session_manager.set_all_params(config_params)
            self.session_manager.start() # This returns the SimulationManager

            def wait_for_success():
                flag = self.session_manager.starter.syncHelper
                flag.event.wait()
                if flag.is_success:
                    self.simulation_finished.emit("Simulation completed successfully.")

            thread = Thread(target=wait_for_success)
            thread.start()

        except Exception as e:
            logger.error("Failed to start simulation from controller.", exc_info=True)

    @pyqtSlot()
    def on_stop_simulation_requested(self):
        """This slot is triggered by the UI's stop button."""
        logger.debug("Application Controller received stop request.")
        if self.session_manager and self.session_manager.starter:
            self.session_manager.stop()

    def register_view(self, view):
        """
        Register a view with the controller.
        This allows the controller to communicate with the view.
        """
        self.simulation_finished.connect(view.on_finish)