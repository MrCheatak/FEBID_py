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
from tkinter import filedialog as fd

# Core packages
import numpy as np
import pyvista as pv

# Auxiliary packages
import yaml
from tqdm import tqdm

from febid.Statistics import Statistics
# Local packages
from febid.Structure import Structure
from febid.Process import Process
from febid.libraries.vtk_rendering import VTK_Rendering as vr
from febid.monte_carlo.etraj3d import MC_Simulation

# It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus able produce deposit under irradiation.

# Semi-surface cells are cells that have precursor density but do not have a neighboring deposit cell
# Thus concept serves an alternative diffusion channel

flag = False
x_pos, y_pos = 0., 0.
warnings.simplefilter('always')


def initialize_framework(from_file=False, precursor=None, settings=None, sim_params=None, vtk_file=None,
                         geom_params=None):
    """
        Open simulation configuration files and prepare data framework

    :param from_file: True to load structure from vtk file
    :param precursor: path to a file with precursor properties
    :param settings: path to a file with beam parameters and settings
    :param sim_params: path to a file with simulation volume parameters
    :param vtk_file: if from_file is True, path to a vtk file to get structure from
    :param geom_params: a list of predetermined simulation volume parameters
    :return:
    """

    print(f'Select three files: precursor parameters, beam parameters and either a predefined structure(.vtk) or a parameters file for the creation:')
    # precursor = None
    # settings = None
    # sim_params = None
    try:
        if precursor is None:
            precursor = yaml.load(open(fd.askopenfilename(), 'r'),
                                  Loader=yaml.Loader)  # Precursor and substrate properties(substrate here is the top layer)
        else:
            precursor = yaml.load(open(precursor, 'r'), Loader=yaml.Loader)
        if settings is None:
            settings = yaml.load(open(fd.askopenfilename(), 'r'),
                                 Loader=yaml.Loader)  # Parameters of the beam, dwell time and precursor flux
        else:
            settings = yaml.load(open(settings, 'r'), Loader=yaml.Loader)
        if not from_file:
            if sim_params is None and geom_params is None:
                sim_params = yaml.load(open(fd.askopenfilename(), 'r'),
                                       Loader=yaml.Loader)  # Size of the chamber, cell size and time step
            elif geom_params is not None:
                sim_params = dict()
                sim_params['width'] = geom_params[0]
                sim_params['length'] = geom_params[1]
                sim_params['height'] = geom_params[2]
                sim_params['cell_dimension'] = geom_params[4]
                sim_params['substrate_height'] = geom_params[3]
            else:
                sim_params = yaml.load(open(sim_params, 'r'), Loader=yaml.Loader)
        # precursor = yaml.load(open(f'{sys.path[0]}{os.sep}Me3PtCpMe.yml', 'r'), Loader=yaml.Loader)
        # settings = yaml.load(open(f'{sys.path[0]}{os.sep}Parameters.yml', 'r'),Loader=yaml.Loader)
        # sim_params = yaml.load(open(f'{sys.path[0]}{os.sep}Simulation.yml', 'r'),Loader=yaml.Loader)
    except Exception as e:
        print(e.args)
        sys.exit('An error occurred while reading one of the parameter files')

    structure = Structure()
    vtk_obj = None
    if from_file:
        try:
            if vtk_file is None:
                print(f'Specify .vtk structure file:')
                vtk_file = fd.askopenfilename()
            vtk_obj = pv.read(vtk_file)
            if not vtk_obj:
                raise RuntimeError
        except Exception as e:
            print(e.args)
            sys.exit("An error occurred while reading the VTK file")
        structure.load_from_vtk(vtk_obj=vtk_obj)
        sim_params = dict()
        sim_params['width'] = structure.shape[2] * structure.cell_dimension
        sim_params['length'] = structure.shape[1] * structure.cell_dimension
        sim_params['height'] = structure.shape[0] * structure.cell_dimension
        sim_params['cell_dimension'] = structure.cell_dimension
        sim_params['substrate_height'] = structure.substrate_height * structure.cell_dimension
    else:
        try:
            structure.create_from_parameters(sim_params['cell_dimension'], sim_params['width'], sim_params['length'],
                                             sim_params['height'], sim_params['substrate_height'])
        except:
            sys.exit("An error occurred while reading initial geometry parameters file")

    return structure, precursor, settings, sim_params


def buffer_constants(precursor: dict, settings: dict, cell_dimension: int):
    """
    Calculate necessary constants and prepare parameters for modules

    :param precursor: precursor properties
    :param settings: simulation conditions
    :param cell_dimension: side length of a square cell, nm
    :return:
    """

    elementary_charge = 1.60217662e-19
    # td = settings["dwell_time"]  # dwell time of a beam, s
    Ie = settings["beam_current"]  # beam current, A
    beam_FWHM = 2.36 * settings["gauss_dev"]  # electron beam diameter, nm
    F = settings[
        "precursor_flux"]  # precursor flux at the surface, 1/(nm^2*s)   here assumed a constant, but may be dependent on time and position
    effective_diameter = beam_FWHM * 3.3  # radius of an area which gets 99% of the electron beam
    f = Ie / elementary_charge / (
            math.pi * beam_FWHM * beam_FWHM / 4)  # electron flux at the surface, 1/(nm^2*s)
    e = precursor["SE_emission_activation_energy"]
    l = precursor["SE_mean_free_path"]

    # Precursor properties
    sigma = precursor[
        "cross_section"]  # dissociation cross section, nm^2; is averaged from cross sections of all electron types (PE,BSE, SE1, SE2)
    n0 = precursor["max_density"]  # inversed molecule size, Me3PtCpMe, 1/nm^2
    molar = precursor["molar_mass_precursor"]  # molar mass of the precursor Me3Pt(IV)CpMe, g/mole
    # density = 1.5E-20  # density of the precursor Me3Pt(IV)CpMe, g/nm^3
    V = precursor["dissociated_volume"]  # atomic volume of the deposited atom (Pt), nm^3
    D = precursor["diffusion_coefficient"]  # diffusion coefficient, nm^2/s
    tau = precursor["residence_time"] * 1E-6  # average residence time, s; may be dependent on temperature

    kd = F / n0 + 1 / tau + sigma * f  # depletion rate
    kr = F / n0 + 1 / tau  # replenishment rate
    nr = F / kr  # absolute density after long time
    nd = F / kd  # depleted absolute density
    t_out = 1 / (1 / tau + F / n0)  # effective residence time
    p_out = 2 * math.sqrt(D * t_out) / beam_FWHM

    # Initializing framework
    t_flux = 1 / (sigma + f)  # dissociation event time
    diffusion_dt = math.pow(cell_dimension * cell_dimension, 2) / (2 * D * (
            cell_dimension * cell_dimension + cell_dimension * cell_dimension))  # maximum stability
    dt = np.min([t_flux, diffusion_dt, tau])

    # Parameters for Monte-Carlo simulation
    mc_config = {'name': precursor["deposit"], 'E0': settings["beam_energy"],
                 'Emin': settings["minimum_energy"],
                 'Z': precursor["average_element_number"],
                 'A': precursor["average_element_mol_mass"], 'rho': precursor["average_density"],
                 'I0': settings["beam_current"], 'sigma': settings["gauss_dev"], 'n': settings['n'],
                 'N': Ie, 'substrate_element': settings["substrate_element"],
                 'cell_dim': cell_dimension,
                 'e': precursor["SE_emission_activation_energy"], 'l': precursor["SE_mean_free_path"],
                  'emission_fraction': settings['emission_fraction']}
    # Parameters for reaction-equation solver
    equation_values = {'F': settings["precursor_flux"], 'n0': precursor["max_density"],
                       'sigma': precursor["cross_section"], 'tau': precursor["residence_time"] * 1E-6,
                       'Ea': precursor['desorption_activation_energy'], 'k0': precursor['desorption_attempt_frequency'],
                       'V': precursor["dissociated_volume"], 'D': precursor["diffusion_coefficient"],
                       'Ed': precursor['diffusion_activation_energy'], 'D0': precursor['diffusion_prefactor'],
                       'rho': precursor['average_density'], 'heat_cond': precursor['thermal_conductivity'],
                       'cp': precursor['heat_capacity'],
                       'dt': dt, 'deposition_scaling': settings['deposition_scaling']}
    # Stability time steps
    timings = {'t_diff': diffusion_dt, 't_flux': t_flux, 't_desorption': tau, 'dt': dt}

    # effective_radius_relative = math.floor(effective_diameter / cell_dimension / 2)
    return mc_config, equation_values, timings, nr


def run_febid_interface(structure, precursor_params, settings, sim_params, path, temperature_tracking, saving_params, rendering):

    if saving_params['monitoring']:
        dump_stats = True
        stats_rate = saving_params['monitoring']
    else:
        dump_stats = False
        stats_rate = sys.maxsize
    if saving_params['snapshot']:
        dump_vtk = True
        dump_rate = saving_params['snapshot']
    else:
        dump_vtk = False
        dump_rate = sys.maxsize

    kwargs = dict(location=saving_params['filename'], stats_rate=stats_rate,
                  dump_rate=dump_rate, render=rendering['show_process'],
                  frame_rate=rendering['frame_rate'], refresh_rate=1e-5)
    process_obj, sim = run_febid(structure, precursor_params, settings, sim_params, path, temperature_tracking, dump_stats, kwargs)
    return process_obj, sim


def run_febid(structure, precursor_params, settings, sim_params, path, temperature_tracking, gather_stats=False, monitor_kwargs=None):
    """
        Create necessary objects and start the FEBID process.

    :param structure: structure object
    :param precursor_params: precursor properties
    :param settings: beam and precursor flux settings
    :param sim_params: simulation volume properties
    :param path: printing path
    :param gather_stats: True enables statistics gathering
    :param monitor_kwargs: settings for the monitoring function
    :return:
    """
    mc_config, equation_values, timings, nr = buffer_constants(precursor_params, settings, sim_params['cell_dimension'])
    stats = None
    if gather_stats:
        stats = Statistics(monitor_kwargs['location'])
        stats.get_params(precursor_params, 'Precursor parameters')
        stats.get_params(settings, 'Beam parameters and settings')
        stats.get_params(sim_params, 'Simulation volume parameters')
    process_obj = Process(structure, equation_values, timings, temp_tracking=temperature_tracking)
    process_obj.stats_freq = min(monitor_kwargs['stats_rate'], monitor_kwargs['dump_rate'], monitor_kwargs['refresh_rate'])
    sim = MC_Simulation(structure, mc_config)
    process_obj.max_neib = math.ceil(np.max([sim.deponat.lambda_escape, sim.substrate.lambda_escape])/process_obj.cell_dimension)
    process_obj.structure.define_surface_neighbors(process_obj.max_neib)
    total_iters = int(np.sum(path[:, 2]) / process_obj.dt)
    # Actual simulation runs in a second Thread, because visualization of the process
    # via Pyvista works only from the main Thread
    printing = Thread(target=print_all, args=[path, process_obj, sim])
    printing.start()
    monitoring(process_obj, stats, **monitor_kwargs)
    printing.join()
    print('Finished path.')
    return process_obj, sim


def print_all(path, process_obj, sim):
    """
    Main event loop, that iterates through consequent points in a stream-file.

    :param path: patterning path from a stream file
    :param process_obj: Process class instance
    :param sim: Monte Carlo simulation object
    :return:
    """
    global flag, x_pos, y_pos
    x_pos, y_pos = path[0, 0:2]
    start = 0
    total_time = int(path[:,2].sum() * process_obj.deposition_scaling * 1e6)
    bar_format = "{desc}: {percentage:.1f}%|{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    t = tqdm(total=total_time, desc='Patterning', position=0, unit='µs',
             bar_format=bar_format) # the displayed execution speed is shown in µs of simulation time per s of real time
    for x, y, step in path[start:]:
        x_pos, y_pos = x, y
        beam_matrix = sim.run_simulation(y, x, process_obj.request_temp_recalc)
        process_obj.beam_matrix[:, :, :] = beam_matrix
        if process_obj.beam_matrix.max() <= 1:
            warnings.warn('No surface flux!', RuntimeWarning)
            process_obj.beam_matrix[...] = 1
        process_obj.get_dt()
        process_obj.update_helper_arrays()
        process_obj.get_dt()
        if process_obj.temperature_tracking:
            process_obj.heat_transfer(sim.beam_heating)
            process_obj.request_temp_recalc = False
        print_step(y, x, step, process_obj, sim, t)
    flag = True


def print_step(y, x, dwell_time, pr: Process, sim, t):
    """
    Sub-loop, that iterates through the dwell time by a time step

    :param y: spot y-coordinate
    :param x: spot x-coordinate
    :param dwell_time: time of the exposure
    :param pr: Process object
    :param sim: MC simulation object
    :param t: tqdm progress bar

    :return:
    """
    loops = int(dwell_time / pr.dt)
    if dwell_time < pr.dt:
        warnings.warn('Dwell time is smaller that the time step!')
        pr.dt = dwell_time
    time = 0
    flag = True
    flag_resize = True
    # THE core loop.
    # Any changes to the events sequence are defined by or stem from this loop.
    # The FEBID process is 'constructed' here by arranging events like deposition(dissociated volume calculation),
    # precursor coverage recalculation, execution of the MC simulation, temperature profile recalculation and other.
    # If any additional calculations and to be included, they shall be run from this loop
    while flag:
        if time+pr.dt>dwell_time: # stepping only for remaining dwell time to avoid accumulating of excessive deposit
            pr.dt = dwell_time - time
            flag = False
        pr.deposition()  # depositing on a selected area
        if pr.check_cells_filled():
            flag_resize = pr.update_surface()  # updating surface on a selected area
            if flag_resize: # update references if the allocated simulation volume was increased
                sim.update_structure(pr.structure)
            start = timeit.default_timer()
            beam_matrix= sim.run_simulation(y, x, pr.request_temp_recalc) # run MC sim. and retrieve SE surface flux
            pr.beam_matrix[...] = beam_matrix
            print(f'Finished MC in {timeit.default_timer()-start} s')
            if pr.beam_matrix.max() <= 1:
                warnings.warn('No surface flux!', RuntimeWarning)
                pr.beam_matrix[...] = 1
                continue
            pr.update_helper_arrays() # auxiliary method that maintains an efficiency increasing infrastructure
            if pr.temperature_tracking:
                pr.heat_transfer(sim.beam_heating)
                pr.request_temp_recalc = False
            if dwell_time >= pr.dt:
                pr.get_dt()
            else:
                pr.dt = dwell_time
        pr.precursor_density() # recalculate precursor coverage
        pr.t += pr.dt*pr.deposition_scaling
        time += pr.dt
        t.update(pr.dt * pr.deposition_scaling * 1e6)
        if time % pr.stats_freq < pr.dt*1.5:
            pr.min_precursor_covearge = pr.precursor_min
            pr.dep_vol = pr.deposited_vol



def monitoring(pr: Process, stats: Statistics = None, location=None, stats_rate=60, dump_rate=60, render=False,
               frame_rate=1, refresh_rate=0.5, displayed_data='precursor'):
    """
    A daemon process function to manage statistics gathering and graphics update.

    :param pr: object of the core deposition process
    :param stats: object for gathering monitoring data
    :param location: file saving directory
    :param stats_rate: statistics recording interval in seconds, None disables statistics recording
    :param dump_rate: dumping interval in seconds, None disables structure dumping
    :param render: True will enable graphical monitoring of the process
    :param frame_rate: redrawing delay
    :param refresh_rate: sleep time
    :return:
    """

    global flag  # This flag variable is used for the communication between current and the process thread.
    # When deposition process thread finishes, it sets flag to False which will finish current thread

    time_step = 60
    pr.start_time = datetime.datetime.now()
    dump_time = stats_time = 0
    time_spent = start_time = frame = timeit.default_timer()
    # Initializing graphical monitoring
    rn = None
    if render:
        rn = vr.Render(pr.structure.cell_dimension)
        pr.redraw = True
    else:
        frame = sys.maxsize  # current time is always less than infinity
    if not stats_rate or stats_rate == sys.maxsize:
        stats_time = sys.maxsize
        stats_rate = sys.maxsize
    else:
        stats_rate = 1e-1 * stats_rate
    if not dump_rate or dump_rate == sys.maxsize:
        dump_time = sys.maxsize
        dump_rate = sys.maxsize
    else:
        dump_rate = 1e-1 * dump_rate
    # Event loop
    while not flag:
        now = timeit.default_timer()
        if now > time_spent:  # overall time and speed
            time_spent += time_step
            # print(f'Time passed: {time_spent}, Av.speed: {l / time_spent}')
        if now > frame:  # graphical
            frame += frame_rate
            redrawed = update_graphical(rn, pr, now - start_time, displayed_data)
        if pr.t > stats_time:
            stats_time += stats_rate
            stats.append(pr.t, pr.min_precursor_covearge, pr.dep_vol, pr.max_T,)
            stats.save_to_file()
        if pr.t > dump_time:
            dump_time += dump_rate
            dump_structure(pr.structure, pr.t, now - start_time, (x_pos, y_pos), f'{location}')
        time.sleep(refresh_rate)
    else:
        if stats_time != sys.maxsize:
            stats.append(pr.t, pr.min_precursor_covearge, pr.dep_vol, pr.max_T,)
            stats.get_growth_rate()
            # stats.add_plots([('Sim.time', 'Min.precursor coverage'),('Sim.time', 'Growth rate')], position=['J1','J23'])
            stats.save_to_file(force=True)
        if dump_time != sys.maxsize:
            dump_structure(pr.structure, pr.t, now - start_time, (x_pos, y_pos), f'{location}')
        if frame != sys.maxsize:
            rn.p.close()
            rn = vr.Render(pr.structure.cell_dimension)
            pr.redraw = True
            update_graphical(rn, pr, now-start_time, displayed_data, False)
            rn.show(interactive_update=False)
    flag = False
    print('Exiting monitoring.')


def update_graphical(rn: vr.Render, pr: Process, time_spent, displayed_data='precursor', update=True):
    """
    Update the visual representation of the current process state

    :param rn: visual scene object
    :param pr: process object
    :param time_step:
    :param time_spent:
    :return:
    """
    try:
        if displayed_data == 'precursor':
            data = pr.precursor
            mask = pr.surface
            cmap = 'plasma'
        if displayed_data == 'deposit':
            data = pr.deposit
            mask = pr.surface
            cmap = 'viridis'
        if displayed_data == 'temperature':
            data = pr.temp
            mask = pr.deposit < 0
            cmap = 'inferno'
        if displayed_data == 'surface_temperature':
            data = pr.surface_temp
            mask = pr.surface
            cmap = 'inferno'
        # data = pr.temp
        redrawed = pr.redraw
        if pr.redraw:
            try:
                # Clearing scene
                rn.y_pos = 5
                try:
                    rn.p.button_widgets.clear()
                except: pass
                rn.p.clear()
                # Putting an arrow to indicate beam position
                start = np.array([0, 0, 100]).reshape(1, 3) # position of the center of the arrow
                end =  np.array([0, 0, -100]).reshape(1, 3) # direction and resulting size
                rn.arrow = rn.p.add_arrows(start, end, color='tomato')
                rn.arrow.SetPosition(x_pos, y_pos, (pr.max_z) * pr.cell_dimension + 10)  # relative to the initial position
                # Plotting data
                rn._add_3Darray(data, opacity=1, scalar_name=displayed_data,
                                button_name=displayed_data, show_edges=True, cmap=cmap)
                scalar = rn.p.mesh.active_scalars_name
                rn.p.mesh[scalar] = data.reshape(-1)
                rn.update_mask(mask)
                rn.p.add_text('.', position='upper_left', font_size=12, name='time')
                rn.p.add_text('.', position='upper_right', font_size=12, name='stats')
                rn.show(interactive_update=True, cam_pos=[(206.34055818793468, 197.6510638707941, 100.47106597548205),
                                                          (0.0, 0.0, 0.0),
                                                          (-0.23307751464125356, -0.236197909312718, 0.9433373838690787)])
            except Exception as e:
                print('An error occurred while redrawing the scene.')
                print(e.args)
                pass
            rn.meshes_count += 1
            pr.redraw = False
        # Changing arrow position
        x, y, z = rn.arrow.GetPosition()
        z_pos = pr.deposit[:, int(y_pos/pr.cell_dimension), int(x_pos/pr.cell_dimension)].nonzero()[0].max() * pr.cell_dimension
        if z_pos != z or y_pos != y or x_pos != x:
            rn.arrow.SetPosition(x_pos, y_pos, z_pos+30) # relative to the initial position
        # Calculating values to indicate
        pr.n_filled_cells.append(pr.filled_cells)
        i = len(pr.n_filled_cells) - 1
        time_real = str(datetime.timedelta(seconds=int(time_spent)))
        speed = pr.t / time_spent
        height = (pr.max_z - pr.substrate_height) * pr.structure.cell_dimension
        total_V = int(pr.dep_vol)
        delta_t = pr.t-pr.t_prev
        delta_V = total_V-pr.vol_prev
        if delta_t == 0 or delta_V == 0:
            growth_rate = pr.growth_rate
        else:
            growth_rate = delta_V / delta_t
            growth_rate = int(growth_rate)
            pr.growth_rate = growth_rate
        pr.t_prev += delta_t
        pr.vol_prev = total_V
        max_T = pr.temp.max()
        # Updating displayed text
        rn.p.actors['time'].SetText(2,
                        f'Time: {time_real} \n' # showing real time passed 
                        f'Sim. time: {(pr.t):.8f} s \n' # showing simulation time passed
                        f'Speed: {speed:.8f} \n'  
                        f'Av. growth rate: {growth_rate} nm^3/s \n'
                        f'Max. temperature: {max_T:.3f} K')
        rn.p.actors['stats'].SetText(3,
                        f'Cells: {pr.n_filled_cells[i]} \n'  # showing total number of deposited cells
                        f'Height: {height} nm \n'
                        f'Volume: {total_V:.0f} nm^3')
        # Updating scene
        rn.update_mask(mask)
        try:
            min = data[data > 0.00001].min()
        except:
            min = 1e-8
        rn.p.update_scalar_bar_range(clim=[min, data.max()])

        if update:
            rn.update()
    except Exception as e:
        warnings.warn(f"Failed to redraw the scene.\n"
                      f"{e.args}")
        pr.redraw = True
        redrawed = False
    return redrawed


def dump_structure(structure: Structure, sim_t=None, t=None, beam_position=None, filename='FEBID_result'):
    vr.save_deposited_structure(structure, sim_t, t, beam_position, filename)


if __name__ == '__main__':
    print(f'##################### FEBID Simulator ###################### \n')
    print('Please use `python -m febid` for launching')
