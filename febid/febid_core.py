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
import sys
import time
import warnings
import timeit
from threading import Thread

# Core packages
import numpy as np
import pyvista as pv

# Auxiliary packages
import yaml
from tqdm import tqdm

from febid.Statistics import Statistics, StructureSaver, SynchronizationHelper
# Local packages
from febid.Structure import Structure
from febid.Process import Process
from febid.libraries.vtk_rendering import VTK_Rendering as vr
from febid.monte_carlo.etraj3d import MC_Simulation

# It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus able produce deposit under irradiation.

# Semi-surface cells are cells that have precursor density but do not have a neighboring deposit cell
# Thus concept serves an alternative diffusion channel

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
    process_obj, sim = run_febid(*args, **kwargs)
    return process_obj, sim


def run_febid(structure, precursor_params, settings, sim_params, path, temperature_tracking,
              saving_params=None, rendering=None):
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

    flag = SynchronizationHelper(False)
    process_obj = Process(structure, equation_values, temp_tracking=temperature_tracking)

    sim = MC_Simulation(structure, mc_config)
    process_obj.max_neib = math.ceil(
        np.max([sim.deponat.lambda_escape, sim.substrate.lambda_escape]) / process_obj.cell_size)
    process_obj.structure.define_surface_neighbors(process_obj.max_neib)
    # Actual simulation runs in a second Thread, because visualization of the process
    # via Pyvista works only from the main Thread
    printing = Thread(target=print_all, args=[path, process_obj, sim, flag])
    if saving_params['gather_stats']:
        stats = setup_stats_collection(process_obj, flag, saving_params)
        stats.get_params(precursor_params, 'Precursor parameters')
        stats.get_params(settings, 'Beam parameters and settings')
        stats.get_params(sim_params, 'Simulation volume parameters')
        process_obj.stats_frequency = min(saving_params.get('gather_stats_interval', 1),
                                          saving_params.get('save_snapshot_interval', 1),
                                          rendering.get('frame_rate', 1))
        stats.start()
    if saving_params['save_snapshot']:
        struc = StructureSaver(process_obj, flag, saving_params['save_snapshot_interval'], saving_params['filename'])
        struc.start()
    printing.start()
    if rendering['show_process']: # running visualization in the main loop
        total_time = visualize_process(process_obj, flag, **rendering)
    printing.join()
    if saving_params['gather_stats']:
        stats.join()
    if saving_params['save_snapshot']:
        struc.join()
    print('Finished path.')
    if rendering['show_process']:
        visualize_result(process_obj, total_time, **rendering)
    return process_obj, sim


def print_all(path, pr: Process, sim: MC_Simulation, run_flag: SynchronizationHelper):
    """
    Main event loop, that iterates through consequent points in a stream-file.

    :param path: patterning path from a stream file
    :param pr: Process class instance
    :param sim: Monte Carlo simulation object
    :param run_flag:
    :return:
    """
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
        print_step(y, x, step, pr, sim, t, run_flag)
    run_flag.run_flag = True


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
    while flag_dt:
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
        pr.precursor_density()  # recalculate precursor coverage
        pr.t += pr.dt * pr.deposition_scaling
        time_passed += pr.dt
        run_flag.timer = pr.t
        t.update(pr.dt * pr.deposition_scaling * 1e6)
        if time_passed % pr.stats_frequency < pr.dt * 1.5:
            pr.min_precursor_coverage = pr.precursor_min
            pr.dep_vol = pr.deposited_vol
        pr.reset_dt()
        # Allow only one tick of the loop for daemons per one tick of simulation
        run_flag.loop_tick.acquire()
        run_flag.loop_tick.notify_all()
        run_flag.loop_tick.release()


def visualize_process(pr: Process, run_flag, frame_rate=1, displayed_data='precursor', **kwargs):
    """
    A daemon process function to manage statistics gathering and graphics update.

    :param pr: object of the core deposition process
    :param run_flag: thread synchronization object, allows to stop visualization when simulation concludes
    :param frame_rate: redrawing delay
    :param displayed_data: name of the displayed data. Options: 'precursor', 'deposit', 'temperature', 'surface_temperature'
    :return:
    """
    start_time = timeit.default_timer()
    # Initializing graphical monitoring

    rn = vr.Render(pr.structure.cell_size)
    rn.p.clear()
    pr.redraw = True
    now = 0
    # Event loop
    while not run_flag:
        now = timeit.default_timer()
        update_graphical(rn, pr, now - start_time, displayed_data)
        time.sleep(frame_rate)
    rn.p.close()
    print('Closing rendering.')
    return now - start_time


def visualize_result(pr, total_time, displayed_data='precursor', **kwargs):
    """
    Rendering the final state of the process.

    :param pr: object of the core deposition process
    :param displayed_data: name of the displayed data
    :param total_time: total time of the simulation
    :return:
    """
    print('Rendering last frame interactively.')
    rn = vr.Render(pr.structure.cell_size)
    pr.redraw = True
    update_graphical(rn, pr, total_time, displayed_data, False)
    rn.show(interactive_update=False)


def update_graphical(rn: vr.Render, pr: Process, time_spent, displayed_data='precursor', update=True):
    """
    Update the visual representation of the current process state

    :param rn: visual scene object
    :param pr: process object
    :param time_spent:
    :param displayed_data:
    :param update:
    :return:
    """
    try:
        if displayed_data == 'precursor':
            data = pr.structure.precursor
            mask = pr.structure.surface_bool
            cmap = 'plasma'
        if displayed_data == 'deposit':
            data = pr.structure.deposit
            mask = pr.structure.surface_bool
            cmap = 'viridis'
        if displayed_data == 'temperature':
            data = pr.structure.temperature
            mask = pr.structure.deposit < 0
            cmap = 'inferno'
        if displayed_data == 'surface_temperature':
            data = pr.surface_temp
            mask = pr.structure.surface_bool
            cmap = 'inferno'
        if displayed_data not in ['precursor', 'deposit', 'temperature', 'surface_temperature']:
            raise AttributeError(f'Dataset {displayed_data} is not available for rendering.')
        if pr.redraw:
            try:
                # Clearing scene
                rn.y_pos = 5
                try:
                    rn.p.button_widgets.clear()
                except Exception as e:
                    print('Something went wrong while clearing widgets from the scene...')
                rn.p.clear()
                # Putting an arrow to indicate beam position
                start = np.array([0, 0, 100]).reshape(1, 3)  # position of the center of the arrow
                end = np.array([0, 0, -100]).reshape(1, 3)  # direction and resulting size
                rn.arrow = rn.p.add_arrows(start, end, color='tomato')
                rn.arrow.SetPosition(pr.x0, pr.y0,
                                     pr.max_z * pr.cell_size + 10)  # relative to the initial position
                # Plotting data
                rn.add_3Darray(data, opacity=1, scalar_name=displayed_data,
                               button_name=displayed_data, show_edges=True, cmap=cmap)
                scalar = rn.p.mesh.active_scalars_name
                rn.p.mesh[scalar] = data.reshape(-1)
                rn.update_mask(mask)
                rn.p.add_text('.', position='upper_left', font_size=12, name='time')
                rn.p.add_text('.', position='upper_right', font_size=12, name='stats')
                rn.show(interactive_update=True, cam_pos=[(206.34055818793468, 197.6510638707941, 100.47106597548205),
                                                          (0.0, 0.0, 0.0),
                                                          (-0.23307751464125356, -0.236197909312718,
                                                           0.9433373838690787)])
            except Exception as e:
                print('An error occurred while redrawing the scene.')
                print(e.args)
            rn.meshes_count += 1
            pr.redraw = False
        # Changing arrow position
        x, y, z = rn.arrow.GetPosition()
        z_pos = pr.structure.deposit[:, int(pr.y0 / pr.cell_size), int(pr.x0 / pr.cell_size)].nonzero()[
                    0].max() * pr.cell_size
        if z_pos != z or pr.y0 != y or pr.x0 != x:
            rn.arrow.SetPosition(pr.x0, pr.y0, z_pos + 30)  # relative to the initial position
        # Calculating values to indicate
        pr.n_filled_cells.append(pr.filled_cells)
        i = len(pr.n_filled_cells) - 1
        time_real = str(datetime.timedelta(seconds=int(time_spent)))
        speed = pr.t / time_spent
        height = (pr.max_z - pr.substrate_height) * pr.structure.cell_size
        total_V = int(pr.dep_vol)
        delta_t = pr.t - pr.t_prev
        delta_V = total_V - pr.vol_prev
        if delta_t == 0 or delta_V == 0:
            growth_rate = pr.growth_rate
        else:
            growth_rate = delta_V / delta_t
            growth_rate = int(growth_rate)
            pr.growth_rate = growth_rate
        pr.t_prev += delta_t
        pr.vol_prev = total_V
        max_T = pr.structure.temperature.max()
        # Updating displayed text
        rn.p.actors['time'].SetText(2,
                                    f'Time: {time_real} \n'  # showing real time passed 
                                    f'Sim. time: {(pr.t):.8f} s \n'  # showing simulation time passed
                                    f'Speed: {speed:.8f} \n'
                                    f'Av. growth rate: {growth_rate} nm^3/s \n'
                                    f'Max. temperature: {max_T:.1f} K')
        rn.p.actors['stats'].SetText(3,
                                     f'Cells: {pr.n_filled_cells[i]} \n'  # showing total number of deposited cells
                                     f'Height: {height} nm \n'
                                     f'Volume: {total_V:.0f} nm^3')
        # Updating scene
        rn.update_mask(mask)
        try:
            _min = data[data > 0.00001].min()
        except ValueError:
            _min = 1e-8
        rn.p.update_scalar_bar_range(clim=[_min, data.max()])

        if update:
            rn.update()
    except Exception as e:
        warnings.warn(f"Failed to redraw the scene.\n"
                      f"{e.args}")
        pr.redraw = True
    return 0


if __name__ == '__main__':
    print('##################### FEBID Simulator ###################### \n')
    print('Please use `python -m febid` for launching')
