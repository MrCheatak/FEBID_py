import datetime
import timeit
import warnings

from tqdm import tqdm

from febid.Process import Process
from febid.monte_carlo.etraj3d import MC_Simulation
from febid.Statistics import SynchronizationHelper
from febid.simulation_context import SimulationContext
from febid.logging_config import setup_logger
# Setup logger
logger = setup_logger(__name__)


class MonteCarloExecutor:
    """Run Monte Carlo transport and propagate beam-matrix updates to the process."""

    def __init__(self, process: Process, mc_sim: MC_Simulation):
        """Store process and Monte Carlo simulation references.

        :param process: Process instance receiving updated beam matrices.
        :type process: Process
        :param mc_sim: Monte Carlo simulator used to compute electron flux.
        :type mc_sim: MC_Simulation
        :return: None
        """
        self.process = process
        self.sim = mc_sim
        self.beam_matrix = None

    def step(self, y, x):
        """Run one Monte Carlo simulation at the given beam position.

        :param y: Y coordinate in simulation grid space.
        :type y: float
        :param x: X coordinate in simulation grid space.
        :type x: float
        :return: None
        """
        start = timeit.default_timer()
        self.beam_matrix = self.sim.run_simulation(y, x, self.process.request_temp_recalc)
        logger.info(f'Finished MC in {(timeit.default_timer() - start):.3f} s')
        if self.beam_matrix.max() <= 1:
            logger.warning("No surface flux!", RuntimeWarning)
            self.process.set_beam_matrix(1)
        else:
            self.process.set_beam_matrix(self.beam_matrix)

    def update_structure(self):
        """
        Update the structure in the MC simulation with the current process structure.
        """
        self.sim.update_structure(self.process.structure)


class HeatSolverExecutor:
    """Trigger heat-transfer updates when temperature recomputation is pending."""

    def __init__(self, process: Process, mc_sim: MC_Simulation):
        """Initialize solver dependencies.

        :param process: Process object owning the temperature manager.
        :type process: Process
        :param mc_sim: Monte Carlo simulator providing beam heating fields.
        :type mc_sim: MC_Simulation
        :return: None
        """
        self.process = process
        self.mc_sim = mc_sim

    def step(self):
        """Update structure temperature if recalculation was requested.

        :return: None
        """
        if self.process.request_temp_recalc:
            # Process heat transfer through the temperature manager pipeline.
            self.process.heat_transfer(self.mc_sim.beam_heating)
            self.process.request_temp_recalc = False


class TimeStepper:
    """Track simulation time, per-dwell progress, and periodic synchronization events."""

    def __init__(self, process, printingPath, syncHelper):
        """Create a helper that manages time-step adaptation and progress reporting.

        :param process: Process object providing the adaptive time step.
        :type process: Process
        :param printingPath: Stream-file path with x, y, dwell-time rows.
        :type printingPath: numpy.ndarray
        :param syncHelper: Shared synchronization state for auxiliary threads.
        :type syncHelper: SynchronizationHelper
        :return: None
        """
        self.pr = process
        self.printingPath = printingPath
        self.scaling = process.deposition_scaling
        self.sync = syncHelper
        self.time_passed_dt_loop = 0.0  # time passed during progressing through a single dwell time
        self.time_passed_total = 0.0
        self.start_time = datetime.datetime.now()
        self.real_timer_start = timeit.default_timer()
        self.real_time_passed = 0.0
        self._dt = self.pr.dt
        self.last_loop = False
        self.progress_bar = None
        self.__setup_progress_bar()

    def get_dt(self, dwell_time):
        """Select the next integration step without exceeding current dwell-time limits.

        :param dwell_time: Remaining dwell duration for the active beam coordinate.
        :type dwell_time: float
        :return: None
        """
        self._dt = self.pr.dt
        if self._dt > dwell_time:   # reducing time step to the dwell time if it is larger
            warnings.warn('Dwell time is smaller than the requested deposition range!')
        # Next condition plays two roles:
        # 1. It ensures that the time step is not larger than the dwell time. It is caught at the first iteration and time_passed_loop=0
        # 2. It ensures that the time step is not larger than the remaining dwell time in the last iteration.
        if self.time_passed_dt_loop + self._dt > dwell_time:  # stepping only for remaining dwell time to avoid accumulating of excess deposit
            self._dt = dwell_time - self.time_passed_dt_loop
            self.last_loop = True
        self.pr.dt = self._dt

    def update_timer(self):
        """Advance timers, trigger periodic callbacks, and notify waiting daemons.

        :return: None
        """
        self.time_passed_dt_loop += self._dt
        self.time_passed_total += self._dt * self.scaling
        self.pr.t = self.time_passed_total
        self.real_time_passed = timeit.default_timer() - self.real_timer_start

        # update progress
        if self.progress_bar:
            d_it = self._dt * self.scaling * 1e6
            self.progress_bar.update(min(d_it, self.progress_bar.total - self.progress_bar.n))

        # Trigger periodic statistics collection.
        if self.pr.stats_gathering and self.time_passed_total % self.pr.stats_frequency < self._dt * 1.5:
            self.pr.gather_stats()

        # tick threads
        self.sync.timer = self.time_passed_total
        # Allow only one tick of the loop for daemons per one tick of simulation
        self.sync.notify()
        self.pr.reset_dt()  # reset the time step for the next iteration

    def __setup_progress_bar(self):
        """Create a progress bar scaled to total simulated patterning time.

        :return: None
        """
        total_time = int(self.printingPath[:, 2].sum() * self.scaling * 1e6)
        bar_format = "{desc}: {percentage:.1f}%|{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        self.progress_bar = tqdm(total=total_time, desc='Patterning', position=0, unit='µs',
                                 bar_format=bar_format)  # the execution speed is shown in µs of simulation time per s of real time

    def reset_dt_loop(self):
        """
        Reset the loop parameters for a new iteration.
        """
        self.time_passed_dt_loop = 0.0
        self.last_loop = False
        self._dt = self.pr.dt


class SimulationPipeline:
    """Coordinate Monte Carlo, heat transfer, and deposition loops for a full run."""

    def __init__(self, context: SimulationContext):
        """Assemble executors and keep shared simulation context.

        :param context: Container with structure, process, and synchronization objects.
        :type context: SimulationContext
        :return: None
        """
        self.logger = setup_logger("febid.simulation")
        self.context = context
        self.mc_executor = MonteCarloExecutor(context.process, context.mcSimulation)
        self.heat_solver = HeatSolverExecutor(context.process, context.mcSimulation)
        self.stepper: TimeStepper = None
        self.initialized = False

    def initialize(self):
        """Prepare process coordinates and time-step controller before execution.

        :return: None
        """
        self.context.process.start_time = datetime.datetime.now()
        self.context.process.x0, self.context.process.y0 = self.context.printingPath[0, 0:2]
        self.stepper = TimeStepper(self.context.process, self.context.printingPath, self.context.syncHelper)
        self.initialized = True
        self.logger.info("SimulationPipeline Initialization complete.")

    def run(self):
        """Execute the full patterning path until completion or stop request.

        :return: None
        """
        if not self.initialized:
            self.initialize()
        pr = self.context.process
        path = self.context.printingPath
        run_flag = self.context.syncHelper

        self.logger.info("Simulation started.")
        for i, (x, y, dwell_time) in enumerate(path):
            if run_flag.is_stopped:
                self.logger.warning(f"Simulation stopped at step {i}/{len(path)}.")
                break
            self.mc_executor.step(y, x)
            self.heat_solver.step()
            self.run_step(x, y, dwell_time)

        run_flag.is_success = not run_flag.is_stopped
        run_flag.run_flag = True
        run_flag.notify()
        run_flag.event.set()

        self.logger.info("Simulation finished successfully." if run_flag.is_success else "Simulation ended early.")

    def run_step(self, x, y, dwell_time):
        """
        Equivalent to print_step() function
        """
        pr = self.context.process
        run_flag = self.context.syncHelper
        stepper = self.stepper

        pr.x0, pr.y0 = x, y
        # THE core loop.
        # Any changes to the events sequence are defined by or stem from this loop.
        # The FEBID process is 'constructed' here by arranging events like deposition(dissociated volume calculation),
        # precursor coverage recalculation, execution of the MC simulation, temperature profile recalculation and other.
        # If any additional calculations and to be included, they shall be run from this loop
        while not stepper.last_loop and not run_flag.run_flag:
            stepper.get_dt(dwell_time)  # get the time step for the current iteration
            pr.deposition()  # depositing on a selected area
            if pr.check_cells_filled():
                self._handle_cell_filled(y, x)
                if pr.state.temperature_tracking:
                    self.heat_solver.step()
            pr.precursor_density()  # recalculate precursor coverage
            stepper.update_timer()  # update timer, progress bar and trigger timed events
        stepper.reset_dt_loop()  # reset the loop parameters for the next iteration

    def _handle_cell_filled(self, y, x):
        """Handle topology changes after cell filling and refresh coupled subsystems.

        :param y: Y coordinate currently being processed.
        :type y: float
        :param x: X coordinate currently being processed.
        :type x: float
        :return: None
        """
        pr = self.context.process
        sim = self.context.mcSimulation
        if pr.device:
            pr.gpu_facade.retrieve_array('deposit', blocking=False)
            pr.gpu_facade.retrieve_array('precursor', blocking=True)
        flag_resize = pr.cell_filled_routine()  # updating surface on a selected area

        if flag_resize:  # update references if the allocated simulation volume was increased
            sim.update_structure(pr.structure)
        else:
            if pr.device:
                pr.gpu_facade.update_structure_partial(cells=pr.last_full_cells, blocking=True)

        self.mc_executor.step(y, x)  # run MC sim. and retrieve SE surface flux and update beam matrix

    def stop(self):
        """Request graceful termination of the simulation loop.

        :return: None
        """
        self.logger.info("Stop requested.")
        self.context.syncHelper.is_stopped = True

    def is_running(self):
        """Report whether the simulation loop is currently active.

        :return: True when running and not stopped, otherwise False.
        """
        return not self.context.syncHelper.run_flag and not self.context.syncHelper.is_stopped


def print_all(context: SimulationContext):
    """
    Main event loop, that iterates through consequent points in a stream-file.

    :param context: SimulationContext object that has all necessary data to start a simulation.
    :return:
    """
    sim = SimulationPipeline(context)
    sim.initialize()
    sim.run()
