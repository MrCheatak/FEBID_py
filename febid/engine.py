import datetime
import timeit
import warnings

from tqdm import tqdm

from febid.Process import Process
from febid.monte_carlo.etraj3d import MC_Simulation
from febid.Statistics import SynchronizationHelper
from febid.simulation_context import SimulationContext


def print_all(context: SimulationContext, stats, struc):
    """
    Main event loop, that iterates through consequent points in a stream-file.

    :param path: patterning path from a stream file
    :param pr: Process class instance
    :param sim: Monte Carlo simulation object
    :param stats: Statistics object, responsible for recording process statistics
    :param struc: StructureSaver object, responsible for saving structure snapshots
    :param run_flag:
    :return:
    """
    path = context.printingPath
    pr = context.process
    sim = context.mcSimulation
    run_flag = context.syncHelper
    pr.start_time = datetime.datetime.now()
    pr.x0, pr.y0 = path[0, 0:2]
    start = 0
    total_time = int(path[:, 2].sum() * pr.deposition_scaling * 1e6)
    bar_format = "{desc}: {percentage:.1f}%|{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    t = tqdm(total=total_time, desc='Patterning', position=0, unit='µs',
             bar_format=bar_format)  # the execution speed is shown in µs of simulation time per s of real time
    for x, y, step in path[start:]:
        pr.x0, pr.y0 = x, y
        beam_matrix = sim.run_simulation(y, x, pr.request_temp_recalc)
        if beam_matrix.max() <= 1:
            warnings.warn('No surface flux!', RuntimeWarning)
            pr.set_beam_matrix(1)
        else:
            pr.set_beam_matrix(beam_matrix)
        if pr.temperature_tracking:
            pr.heat_transfer(sim.beam_heating)
            pr.request_temp_recalc = False
        if pr.device:
            pr.knl.load_beam_matrix(beam_matrix, blocking=False)
            print_step_GPU(y, x, step, pr, sim, t, run_flag)
        else:
            print_step(y, x, step, pr, sim, t, run_flag)
        if run_flag.is_stopped:
            print('Stopping simulation...')
            break
    if not run_flag.is_stopped:
        run_flag.is_success = True
        message = 'Simulation finished!'
    else:
        message = 'Simulation stopped!'
    run_flag.run_flag = True
    run_flag.loop_tick.acquire()
    run_flag.loop_tick.notify_all()
    run_flag.loop_tick.release()
    print(message)
    run_flag.event.set()


def print_step(y, x, dwell_time, pr: Process, sim: MC_Simulation, t, run_flag: SynchronizationHelper):
    """
    Sub-loop, that iterates through the dwell time by a time step

    :param y: spot y-coordinate
    :param x: spot x-coordinate
    :param dwell_time: time of the exposure
    :param pr: Process object
    :param sim: MC simulation object
    :param t: tqdm progress bar
    :param run_flag: Thread synchronization object

    :return:
    """
    if dwell_time < pr.dt:
        warnings.warn('Dwell time is smaller that the time step!')
        pr.dt = dwell_time
    time_passed = 0
    flag_dt = True
    flag_resize = True
    # THE core loop.
    # Any changes to the events sequence are defined by or stem from this loop.
    # The FEBID process is 'constructed' here by arranging events like deposition(dissociated volume calculation),
    # precursor coverage recalculation, execution of the MC simulation, temperature profile recalculation and other.
    # If any additional calculations and to be included, they shall be run from this loop
    while flag_dt and not run_flag.run_flag:
        if time_passed + pr.dt > dwell_time:  # stepping only for remaining dwell time to avoid accumulating of excess deposit
            pr.dt = dwell_time - time_passed
            flag_dt = False
        pr.deposition()  # depositing on a selected area
        if pr.check_cells_filled():
            flag_resize = pr.cell_filled_routine()  # updating surface on a selected area
            if flag_resize:  # update references if the allocated simulation volume was increased
                sim.update_structure(pr.structure)
            start = timeit.default_timer()
            beam_matrix = sim.run_simulation(y, x, pr.request_temp_recalc)  # run MC sim. and retrieve SE surface flux
            print(f'Finished MC in {timeit.default_timer() - start} s')
            if beam_matrix.max() <= 1:
                warnings.warn('No surface flux!', RuntimeWarning)
                pr.set_beam_matrix(1)
            else:
                pr.set_beam_matrix(beam_matrix)
            if pr.temperature_tracking:
                pr.heat_transfer(sim.beam_heating)
                pr.request_temp_recalc = False
            cell_filling_routine(y, x, pr, sim)  # cell configuration update
        pr.precursor_density()  # recalculate precursor coverage
        pr.t += pr.dt * pr.deposition_scaling
        time_passed += pr.dt
        run_flag.timer = pr.t
        # Advancing the progress bar
        # Making sure the last iteration does not overflow the counter
        d_it = pr.dt * pr.deposition_scaling * 1e6
        if t.n + d_it > t.total:
            d_it = t.total - t.n
        t.update(d_it)
        # Collecting process stats
        if time_passed % pr.stats_frequency < pr.dt * 1.5:
            pr._gather_stats()
        pr.reset_dt()
        # Allow only one tick of the loop for daemons per one tick of simulation
        run_flag.loop_tick.acquire()
        run_flag.loop_tick.notify_all()
        run_flag.loop_tick.release()


def cell_filling_routine(y, x, pr: Process, sim: MC_Simulation):
    """
    Run the full set of operations on a filled cell event.

    :param y: spot y-coordinate
    :param x: spot x-coordinate
    :param pr: Process object
    :param sim: MC simulation object
    """
    flag_resize = pr.cell_filled_routine()  # updating surface on a selected area
    if flag_resize:  # update references if the allocated simulation volume was increased
        sim.update_structure(pr.structure)
    start = timeit.default_timer()
    beam_matrix = sim.run_simulation(y, x, pr.request_temp_recalc)  # run MC sim. and retrieve SE surface flux
    print(f'Finished MC in {timeit.default_timer() - start} s')
    if beam_matrix.max() <= 1:
        warnings.warn('No surface flux!', RuntimeWarning)
        pr.set_beam_matrix(1)
    else:
        pr.set_beam_matrix(beam_matrix)
    if pr.temperature_tracking:
        pr.heat_transfer(sim.beam_heating)
        pr.request_temp_recalc = False


def cell_filling_routine(y, x, pr: Process, sim: MC_Simulation):
    """
    Run the full set of operations on a filled cell event.

    :param y: spot y-coordinate
    :param x: spot x-coordinate
    :param pr: Process object
    :param sim: MC simulation object
    """
    flag_resize = pr.cell_filled_routine()  # updating surface on a selected area
    if flag_resize:  # update references if the allocated simulation volume was increased
        sim.update_structure(pr.structure)
    start = timeit.default_timer()
    beam_matrix = sim.run_simulation(y, x, pr.request_temp_recalc)  # run MC sim. and retrieve SE surface flux
    print(f'Finished MC in {timeit.default_timer() - start} s')
    if beam_matrix.max() <= 1:
        warnings.warn('No surface flux!', RuntimeWarning)
        pr.set_beam_matrix(1)
    else:
        pr.set_beam_matrix(beam_matrix)
    if pr.temperature_tracking:
        pr.heat_transfer(sim.beam_heating)
        pr.request_temp_recalc = False


def print_step_GPU(y, x, dwell_time, pr: Process, sim: MC_Simulation, t, run_flag: SynchronizationHelper):
    """
    Run deposition on a single spot using GPU.

    :param x: spot x-coordinate
    :param y: spot y-coordinate
    :param dwell_time: time of the exposure
    :param pr: Process object
    :param sim: MC simulation object
    :param t: tqdm progress bar
    :param run_flag: Thread synchronization object
    :return:
    """
    if dwell_time < pr.dt:
        warnings.warn('Dwell time is smaller that the time step!')
        pr.dt = dwell_time
    time_passed = 0
    flag_dt = True
    while flag_dt and not run_flag.run_flag:
        if time_passed + pr.dt > dwell_time:  # stepping only for remaining dwell time to avoid accumulating of excess deposit
            pr.dt = dwell_time - time_passed
            flag_dt = False
        pr.knl.queue.finish() # acts as a memory barrier that the precursor coverage operation is done
        full = pr.deposition_gpu(blocking=True)  # depositing on a selected area
        if full:
            # cell_filling_routine_GPU(y, x, pr, sim) # cell configuration update done on on GPU
            cell_filling_routine_CPU(y, x, pr, sim)  # cell configuration update done on CPU
        pr.precursor_density_gpu(blocking=False)
        pr.t += pr.dt * pr.deposition_scaling
        time_passed += pr.dt
        run_flag.timer = pr.t
        # Advancing the progress bar
        # Making sure the last iteration does not overflow the counter
        d_it = pr.dt * pr.deposition_scaling * 1e6
        if t.n + d_it > t.total:
            d_it = t.total - t.n
        t.update(d_it)
        # Collecting prcess stats
        if time_passed % pr.stats_frequency < pr.dt * 1.5:
            pr._gather_stats()
            pr.get_data()
            # pr.structure.offload_partial(pr.knl, 'surface_bool')
        pr.reset_dt()
        # Allow only one tick of the loop for daemons per one tick of simulation
        run_flag.loop_tick.acquire()
        run_flag.loop_tick.notify_all()
        run_flag.loop_tick.release()


def cell_filling_routine_GPU(y, x, pr: Process, sim: MC_Simulation):
    """
    Run the full set of operations on a filled cell event.
    Cell configuration update is performed on the GPU.

    :param y: spot y-coordinate
    :param x: spot x-coordinate
    :param pr: Process object
    :param sim: MC simulation object
    """
    flag = pr.update_surface_GPU()  # updating surface on a selected area
    # If the structure was resized, the actual data is already in local memory
    # But if it was not, the actual data is on the GPU and needs to be offloaded for MC simulation
    if not flag:
        pr.offload_from_gpu_partial('deposit')
        pr.offload_from_gpu_partial('surface_bool')
    sim.update_structure(pr.structure)
    start = timeit.default_timer()
    beam_matrix = sim.run_simulation(y, x, pr.request_temp_recalc) # run MC sim. and retrieve SE surface flux
    pr.set_beam_matrix(beam_matrix)
    print(f'Finished MC in {timeit.default_timer() - start} s')
    if beam_matrix.max() <= 1:
        warnings.warn('No surface flux!', RuntimeWarning)
        beam_matrix = 1
    if flag:
        try:
            pr.onload_structure_to_gpu(beam_matrix)
        except Exception as e:
            print("Error during structure resizing: " + repr(e))
            return False
        print("Resize successfull")
    else:
        pr.knl.update_beam_matrix(beam_matrix)
    if pr.temperature_tracking:
        pr.heat_transfer(sim.beam_heating)
        pr.request_temp_recalc = False


def cell_filling_routine_CPU(y, x, pr: Process, sim: MC_Simulation):
    """
    Run the full set of operations on a filled cell event.
    Cell configuration update is performed on the CPU.

    :param y: spot y-coordinate
    :param x: spot x-coordinate
    :param pr: Process object
    :param sim: MC simulation object
    """
    pr.offload_from_gpu_partial('deposit', blocking=False)
    pr.offload_from_gpu_partial('precursor', blocking=True)
    flag_resize = pr.cell_filled_routine()  # updating surface on a selected area
    # pr.onload_structure_to_gpu(blocking=False)
    pr.update_structure_to_gpu(blocking=True)
    sim.update_structure(pr.structure)
    start = timeit.default_timer()
    beam_matrix = sim.run_simulation(y, x, pr.request_temp_recalc)  # run MC sim. and retrieve SE surface flux
    if flag_resize:
        pr.knl.reload_beam_matrix(beam_matrix, blocking=False)
    else:
        pr.knl.update_beam_matrix(beam_matrix, blocking=False)
    print(f'Finished MC in {timeit.default_timer() - start} s')
    if beam_matrix.max() <= 1:
        warnings.warn('No surface flux!', RuntimeWarning)
        pr.set_beam_matrix(1)
    else:
        pr.set_beam_matrix(beam_matrix)
    if pr.temperature_tracking:
        pr.heat_transfer(sim.beam_heating)
        pr.request_temp_recalc = False