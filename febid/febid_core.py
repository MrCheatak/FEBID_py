###################################################################
#
#  FEBID Simulation
#
#  Version 0.9
#
####################################################################
# Default packages
import math
import warnings
import timeit
import threading
from threading import Thread

# Core packages
import numpy as np

from febid.Statistics import Statistics, StructureSaver, SynchronizationHelper
# Local packages
from febid.Process import Process
from febid.monte_carlo.etraj3d import MC_Simulation
from febid.simulation_context import SimulationContext
from febid.parameter_utils import prepare_equation_values, prepare_ms_config
from engine import print_all

# It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus able produce deposit under irradiation.

# Semi-surface cells are cells that have precursor density but do not have a neighboring deposit cell
# Thus concept serves an alternative diffusion channel

# The program uses multithreading to run the simulation itself, statistics gathering, structure snapshot dumping
# and visualization all in parallel to the main thread that is reserved for the UI
# These flags are used to synchronize the threads and stop them if needed.
warnings.simplefilter('always')

class SimulationManager:
    def __init__(self, context: SimulationContext):
        self.context = context
        self._running = threading.Event()
        self._lock = threading.Lock()
        self.process = None
        self.mc_sim = None
        self.stats_thread = None
        self.snapshot_thread = None
        self.printing = None
        self.syncHelper = SynchronizationHelper(False)
        self.context.syncHelper = self.syncHelper

    def initialize(self):
        eq_vals = prepare_equation_values(self.context.precursorParams, self.context.settings)
        mc_conf = prepare_ms_config(self.context.precursorParams, self.context.settings, self.context.structure)

        self.process = Process(self.context.structure, eq_vals, temp_tracking=self.context.temperatureTracking, device=self.context.device)
        self.mc_sim = MC_Simulation(self.context.structure, mc_conf)
        self.context.process = self.process
        self.context.mcSimulation = self.mc_sim

        self.process.max_neib = math.ceil(
            np.max([self.mc_sim.deponat.lambda_escape, self.mc_sim.substrate.lambda_escape])
        )
        self.context.structure.define_surface_neighbors(self.process.max_neib)

        gather_stats = self.context.savingParams.get('gather_stats', False)
        save_snapshot = self.context.savingParams.get('save_snapshot', False)
        if gather_stats or save_snapshot:
            if gather_stats:
                self.stats_thread = Statistics(self.process, self.syncHelper,
                                               refresh_rate=self.context.savingParams.get('gather_stats_interval'),
                                               filename=self.context.savingParams.get('filename'))
                self.stats_thread.get_params(self.context.precursorParams, 'Precursor parameters')
                self.stats_thread.get_params(self.context.settings, 'Beam parameters and settings')
                self.stats_thread.get_params(self.context.simParams, 'Simulation volume parameters')
            if save_snapshot:
                self.snapshot_thread = StructureSaver(self.process, self.syncHelper,
                                                      refresh_rate=self.context.savingParams.get('save_snapshot_interval'),
                                                      filename=self.context.savingParams.get('filename'))
            stats_interval = self.context.savingParams.get('gather_stats_interval', 1)
            self.process.stats_frequency = min(self.context.savingParams.get('gather_stats_interval', 1),
                                              self.context.savingParams.get('save_snapshot_interval', 1))
            self.process.stats_gathering = True

        self.printing = Thread(target=print_all, args=[self.context],)

    def run(self):
        self._running.set()
        if self.stats_thread is not None:
            self.stats_thread.start()
        if self.snapshot_thread is not None:
            self.snapshot_thread.start()
        self.printing.start()
        Thread(target=self._monitor_completion, daemon=True).start()

    def stop(self):
        self.syncHelper.is_stopped = True
        self.syncHelper.run_flag = True
        self._running.clear()
        self.printing.join()
        self._join_threads()

    def _monitor_completion(self):
        self.syncHelper.event.wait()  # Wait until the printing thread finishes
        self.printing.join()
        self._join_threads()
        self._running.clear()

    def _join_threads(self):
        """
        Wait for all threads to finish.
        This is used to ensure that all threads are properly joined before exiting the program.
        """
        if self.stats_thread:
            self.stats_thread.join()
        if self.snapshot_thread:
            self.snapshot_thread.join()
        self.syncHelper.reset()


if __name__ == '__main__':
    print('##################### FEBID Simulator ###################### \n')
    print('Please use `python -m febid` for launching')
