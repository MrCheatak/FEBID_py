###################################################################
#
#  FEBID Simulation
#
#  Version 0.9
#
####################################################################
# Default packages
import datetime
import math
import warnings
import timeit
from threading import Thread

# Core packages
import numpy as np

# Auxiliary packages
from tqdm import tqdm

from febid.Statistics import Statistics, StructureSaver, SynchronizationHelper
# Local packages
from febid.Structure import Structure
from febid.Process import Process
from febid.monte_carlo.etraj3d import MC_Simulation

# It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus able produce deposit under irradiation.

# Semi-surface cells are cells that have precursor density but do not have a neighboring deposit cell
# Thus concept serves an alternative diffusion channel

# The program uses multithreading to run the simulation itself, statistics gathering, structure snapshot dumping
# and visualization all in parallel to the main thread that is reserved for the UI
# These flags are used to synchronize the threads and stop them if needed.
flag = SynchronizationHelper(False)
x_pos, y_pos = 0., 0.
warnings.simplefilter('always')


def prepare_equation_values(precursor: dict, settings: dict):
    """
    Prepare equation values for the reaction-equation solver.

    :param precursor: dictionary containing precursor properties
    :param settings: dictionary containing beam and precursor flux settings
    :return: dictionary containing equation values for the solver
    """
    equation_values = {}
    try:
        equation_values['F'] = settings.get("precursor_flux")
        equation_values['n0'] = precursor.get("max_density")
        equation_values['sigma'] = precursor.get("cross_section")
        equation_values['tau'] = precursor.get("residence_time") * 1E-6
        equation_values['Ea'] = precursor.get('desorption_activation_energy')
        equation_values['k0'] = precursor.get('desorption_attempt_frequency')
        equation_values['V'] = precursor.get("dissociated_volume")
        equation_values['D'] = precursor.get("diffusion_coefficient")
        equation_values['Ed'] = precursor.get('diffusion_activation_energy')
        equation_values['D0'] = precursor.get('diffusion_prefactor')
        equation_values['rho'] = precursor.get('average_density')
        equation_values['heat_cond'] = precursor.get('thermal_conductivity')
        equation_values['cp'] = precursor.get('heat_capacity')
        equation_values['deposition_scaling'] = settings.get('deposition_scaling')
    except KeyError as e:
        raise KeyError(f"Missing key in precursor or settings dictionary: {str(e)}")
    return equation_values


def prepare_ms_config(precursor: dict, settings: dict, structure: Structure):
    """
    Prepare the configuration for Monte-Carlo simulation.

    :param precursor: dictionary containing precursor information
    :param settings: dictionary containing simulation settings
    :param structure: Structure object representing the simulation volume
    :return: dictionary containing the Monte-Carlo simulation configuration
    :raises TypeError: if the 'structure' parameter is not an instance of the 'Structure' class
    :raises KeyError: if any key is missing in the precursor or settings dictionaries
    """
    if not isinstance(structure, Structure):
        raise TypeError("The 'structure' parameter must be an instance of the 'Structure' class.")
    # Parameters for Monte-Carlo simulation
    try:
        mc_config = {'name': precursor["deposit"], 'E0': settings["beam_energy"],
                     'Emin': settings["minimum_energy"],
                     'Z': precursor["average_element_number"],
                     'A': precursor["average_element_mol_mass"], 'rho': precursor["average_density"],
                     'I0': settings["beam_current"], 'sigma': settings["gauss_dev"], 'n': settings['n'],
                     'substrate_element': settings["substrate_element"],
                     'cell_size': structure.cell_size,
                     'e': precursor["SE_emission_activation_energy"], 'l': precursor["SE_mean_free_path"],
                     'emission_fraction': settings['emission_fraction']}
    except KeyError as e:
        raise KeyError(f"Missing key in precursor or settings dictionary: {str(e)}")
    return mc_config


def setup_stats_collection(observed_obj, run_flag, config):
    stats = Statistics(observed_obj, run_flag, config['gather_stats_interval'], config['filename'])
    return stats


def setup_structure_saving(process_obj, run_flag, saving_params):
    struc = StructureSaver(process_obj, flag, saving_params['save_snapshots'], saving_params['filename'])
    return struc


def run_febid_interface(*args, **kwargs):
    process_obj, sim, printing_thread = run_febid(*args, **kwargs)
    return process_obj, sim, printing_thread


def run_febid(structure, precursor_params, settings, sim_params, path, temperature_tracking,
              saving_params=None, device=None):
    """
        Create necessary objects and start the FEBID process.

    :param structure: structure object
    :param precursor_params: precursor properties
    :param settings: beam and precursor flux settings
    :param sim_params: simulation volume properties
    :param path: printing path
    :param temperature_tracking: if True, enable temperature tracking
    :param saving_params: settings for the monitoring function
    :return:
    """
    equation_values = prepare_equation_values(precursor_params, settings)
    mc_config = prepare_ms_config(precursor_params, settings, structure)

    flag.reset()
    process_obj = Process(structure, equation_values, temp_tracking=temperature_tracking)

    sim = MC_Simulation(structure, mc_config)
    process_obj.max_neib = math.ceil(
        np.max([sim.deponat.lambda_escape, sim.substrate.lambda_escape]) / process_obj.cell_size)
    process_obj.structure.define_surface_neighbors(process_obj.max_neib)
    if saving_params['gather_stats']:
        stats = setup_stats_collection(process_obj, flag, saving_params)
        stats.get_params(precursor_params, 'Precursor parameters')
        stats.get_params(settings, 'Beam parameters and settings')
        stats.get_params(sim_params, 'Simulation volume parameters')
        process_obj.stats_frequency = min(saving_params.get('gather_stats_interval', 1),
                                          saving_params.get('save_snapshot_interval', 1))
    else:
        stats = None
    if saving_params['save_snapshot']:
        struc = StructureSaver(process_obj, flag, saving_params['save_snapshot_interval'], saving_params['filename'])
    else:
        struc = None
    printing = Thread(target=print_all, args=[path, process_obj, sim, stats, struc])
    printing.start()
    return process_obj, sim, printing


def print_all(path, pr: Process, sim: MC_Simulation, stats: Statistics=None, struc: StructureSaver=None):
    """
    Main event loop, that iterates through consequent points in a stream-file.

    :param path: patterning path from a stream file
    :param pr: Process class instance
    :param sim: Monte Carlo simulation object
    :param run_flag:
    :return:
    """
    run_flag = flag
    run_flag.run_flag = False
    if stats:
        stats.start()
    if struc:
        struc.start()
    pr.start_time = datetime.datetime.now()
    pr.x0, pr.y0 = path[0, 0:2]
    start = 0
    first_it = True
    total_time = int(path[:, 2].sum() * pr.deposition_scaling * 1e6)
    bar_format = "{desc}: {percentage:.1f}%|{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    t = tqdm(total=total_time, desc='Patterning', position=0, unit='µs',
             bar_format=bar_format)  # the execution speed is shown in µs of simulation time per s of real time
    for x, y, step in path[start:]:
        pr.x0, pr.y0 = x, y
        # beam_matrix = sim.run_simulation(y, x, pr.request_temp_recalc)
        beam_matrix = sample_beam_matrix(sim.pe_sim.sigma, pr.structure)
        if beam_matrix.max() <= 1:
            warnings.warn('No surface flux!', RuntimeWarning)
            pr.set_beam_matrix(1)
        else:
            pr.set_beam_matrix(beam_matrix)
        if pr.temperature_tracking:
            pr.heat_transfer(sim.beam_heating)
            pr.request_temp_recalc = False
        if pr.device:
            if first_it:
                try:
                    pr.load_kernel()
                    first_it = False
                except Exception as e:
                    print('Could not load the kernel!')
                    print(e)
                    run_flag.run_flag = True
            else:
                pr.knl.update_beam_matrix(beam_matrix)
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
    if stats or struc is not None:
        run_flag.loop_tick.acquire()
        run_flag.loop_tick.notify_all()
        run_flag.loop_tick.release()
        if stats:
            stats.join()
        if struc:
            struc.join()
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
    cell_count = 0
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
            cell_count += (pr.structure.deposit > 1).nonzero()[0].size
            flag_resize = pr.cell_filled_routine()  # updating surface on a selected area
            if flag_resize:  # update references if the allocated simulation volume was increased
                sim.update_structure(pr.structure)
            start = timeit.default_timer()
            # beam_matrix = sim.run_simulation(y, x, pr.request_temp_recalc)  # run MC sim. and retrieve SE surface flux
            beam_matrix = sample_beam_matrix(sim.pe_sim.sigma, pr.structure)
            print(f'Finished MC in {timeit.default_timer() - start} s')
            if beam_matrix.max() <= 1:
                warnings.warn('No surface flux!', RuntimeWarning)
                pr.set_beam_matrix(1)
            else:
                pr.set_beam_matrix(beam_matrix)
            if pr.temperature_tracking:
                pr.heat_transfer(sim.beam_heating)
                pr.request_temp_recalc = False
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
        # Collecting prcess stats
        if time_passed % pr.stats_frequency < pr.dt * 1.5:
            pr._gather_stats()
        pr.reset_dt()
        # Allow only one tick of the loop for daemons per one tick of simulation
        run_flag.loop_tick.acquire()
        run_flag.loop_tick.notify_all()
        run_flag.loop_tick.release()


def print_step_GPU(y, x, dwell_time, pr: Process, sim, t, run_flag: SynchronizationHelper):
    """
    Run deposition on a single spot

    :param x: spot x-coordinate
    :param y: spot y-coordinate
    :param dwell_time: time of the exposure
    :param pr: Process object
    :param sim: MC simulation object
    :return:
    """
    if dwell_time < pr.dt:
        warnings.warn('Dwell time is smaller that the time step!')
        pr.dt = dwell_time
    time_passed = 0
    flag_dt = True
    cell_count = 0
    full = pr.deposition_gpu()
    cell_count += full
    if full:
        full_cell_ops(y, x, pr, sim)
    while flag_dt and not run_flag.run_flag:
        if time_passed + pr.dt > dwell_time:  # stepping only for remaining dwell time to avoid accumulating of excess deposit
            pr.dt = dwell_time - time_passed
            flag_dt = False
        full = pr.deposition_gpu()
        cell_count += full
        if full:
            full_cell_ops(y, x, pr, sim)
        pr.precur_density_gpu()
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
            pr.min_precursor_coverage = pr.precursor_min
            pr.dep_vol = pr.deposited_vol
        pr.reset_dt()
        # Allow only one tick of the loop for daemons per one tick of simulation
        run_flag.loop_tick.acquire()
        run_flag.loop_tick.notify_all()
        run_flag.loop_tick.release()


def full_cell_ops(y, x, pr: Process, sim):
    load_count = 0
    flag = pr.update_surface_GPU()  # updating surface on a selected area
    pr.offload_from_gpu()
    if flag: # look into what Monte Carlo reads!!!
        sim.pe_sim.grid = sim.se_sim.grid = pr.structure.deposit
        sim.pe_sim.surface = sim.se_sim.surface = pr.structure.surface_bool
        sim.se_sim.s_neighb = pr.structure.surface_neighbors_bool
    else:
        # start = timeit.default_timer()
        sim.pe_sim.grid = sim.se_sim.grid = pr.structure.deposit
        sim.pe_sim.surface = sim.se_sim.surface = pr.structure.surface_bool
        sim.se_sim.s_neighb = pr.structure.surface_neighbors_bool
        # load_count += timeit.default_timer() - start
    start = timeit.default_timer()
    # beam_matrix = sim.run_simulation(y, x, pr.request_temp_recalc)  # run MC sim. and retrieve SE surface flux
    if flag:
        beam_matrix_buff = None
        filled_cells_init = None
    else:
        beam_matrix_buff = pr.knl.flux_matrix.reshape(pr.structure.shape)
        filled_cells_init = pr.full_cells
    beam_matrix = sample_beam_matrix(sim.pe_sim.sigma, pr.structure,
                                     beam_matrix_buff=beam_matrix_buff,
                                     filled_cells_init=filled_cells_init)
    print(f'Finished MC in {timeit.default_timer() - start} s')
    if beam_matrix.max() <= 1:
        warnings.warn('No surface flux!', RuntimeWarning)
        beam_matrix = 1
    # mont_time = timeit.default_timer() - start
    # pr.get_dt(beam_matrix)
    # pr.update_helper_arrays()
    if flag:
        try:
            # start = timeit.default_timer()
            zero_dim = pr._irradiated_area_2D[0]
            irr_ind_2D = (zero_dim.start, zero_dim.stop)
            pr.structure.onload_all(pr.knl, beam_matrix, irr_ind_2D)
            # load_count += timeit.default_timer() - start
        except Exception as e:
            print("Error during structure resizing: " + repr(e))
            return False
        print("Resize successfull")
    else:
        # start = timeit.default_timer()
        pr.knl.update_beam_matrix(beam_matrix)
        # load_count += timeit.default_timer() - start
    # return mont_time, load_count


def sample_beam_matrix(a, structure, f0=1e6, beam_matrix_buff=None, filled_cells_init=None):
    """
    Sample a gaussian beam matrix for testing purposes. Optimized for quick updates based on filled cells and irradiated area slice.

    :param a: Gaussian standard deviation
    :param structure: structure object of the simulation
    :param f0: Gaussian peak value
    :param beam_matrix_buff: buffer for the beam matrix
    :param filled_cells_init: filled cells, must be an array of tripples
    :param irrad_area_2d: irradiated area slice
    :return: sampled array
    """
    # Generate a 2D Gaussian PDF matrix
    _, ydim, xdim = structure.shape
    _, ydim_abs, xdim_abs = structure.shape_abs
    x = np.linspace(0, xdim_abs, xdim) - xdim_abs / 2
    y = np.linspace(0, ydim_abs, ydim) - ydim_abs / 2
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x * x + y * y)
    gaussian_2d = (np.exp(-(d ** 2 / (2 * a ** 2))) * f0).astype(np.int32)
    # Create a new array if the buffer is not provided
    if beam_matrix_buff is not None:
        beam_matrix = beam_matrix_buff
    else:
        beam_matrix = np.zeros(structure.shape, dtype=np.int32)
    # Clear filled cells and project the values from the 2D Gaussian onto the 3D structure using surface_bool
    if filled_cells_init is not None:
        beam_matrix[filled_cells_init] = 0
        for n in range(len(filled_cells_init)):
            filled_cells_T = filled_cells_init.T
            slice_3d = np.s_[filled_cells_T[0,n] - 1:filled_cells_T[0,n] + 2,
                       filled_cells_T[1,n] - 1:filled_cells_T[1,n] + 2,
                       filled_cells_T[2,n] - 1:filled_cells_T[2,n] + 2]
            slice_2d = np.s_[filled_cells_T[1,n] - 1:filled_cells_T[1,n] + 2,
                       filled_cells_T[2,n] - 1:filled_cells_T[2,n] + 2]
            beam_matrix_view = beam_matrix[slice_3d]
            index = structure.surface_bool[slice_3d]
            gaussian_2d_view = gaussian_2d[slice_2d]
            for i in range(index.shape[1]):
                for j in range(index.shape[2]):
                    beam_matrix_view[index] = gaussian_2d_view[i, j]
    for n in range(ydim):
        for j in range(xdim):
            if gaussian_2d[n, j] < 10:
                continue
            index = structure.surface_bool[:, n, j]
            beam_matrix_view = beam_matrix[:, n, j]
            beam_matrix_view[index] = gaussian_2d[n, j]
    return beam_matrix.astype(np.int32)


if __name__ == '__main__':
    print('##################### FEBID Simulator ###################### \n')
    print('Please use `python -m febid` for launching')
