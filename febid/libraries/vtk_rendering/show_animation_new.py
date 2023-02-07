"""
View series of consequent 3D-Structure files as an animated process.
"""
import os, time
from datetime import datetime
from tkinter import filedialog as fd

import numpy as np
import vtk

import febid.libraries.vtk_rendering.VTK_Rendering as vr
from febid.Structure import Structure

def open_file(directory=''):
    """
    Gather files and timestamps sorted in the order of creation

    :param directory: folder with vtk files
    :return: filenames and timestamps
    """
    # Getting all filenames in the specified directory
    # Getting creation dates of the files
    # Zipping them together and sorting by the creation date
    # Unzipping and returning to the order sorted
    # directory = '/Users/sandrik1742/Documents/PycharmProjects/FEBID/code/Experiment runs/gr=0'
    files = sorted(os.listdir(directory))[:]
    n = 0
    for i in range(len(files)-1, -1, -1):
        if os.path.splitext(files[i])[1] != '.vtk':
            files.pop(i)
            n += 1
    ctimes = [time.ctime(os.path.getmtime(os.path.join(directory, file))) for file in files]
    times = [datetime.strptime(t, '%a %b %d %H:%M:%S %Y') for t in ctimes]
    # occurences = [re.findall("^\d+",file) for file in files]
    # ocurrences = [int(item[0]) for item in occurences]
    ocurrences = zip(times, files)
    ocurrences = sorted(ocurrences)
    files = [item for _, item in ocurrences]
    times.sort()
    # times = [re.findall("\d\d:\d\d:\d\d",file) for file in files]

    return files, times

def show_animation(directory='', show='precursor'):
    """
    Show animated process from series of vtk files.
    Files must have consequent creation dates to align correctly

    :param directory: folder with vtk files
    :param show: which dataset to use for imaging. Accepts 'precursor' for surface precursor density or 'deposit' for surface deposit filling.
    :return:
    """

    # The space where a 3D object appears when rendered is called a Scene
    # The Scene also contains all the rendered texts, scalar bars and buttons

    # Opening files
    if not directory:
        os.chdir('../../..')
        init_dir = os.getcwd()
        directory = fd.askdirectory()
    files, times = open_file(directory)

    # Getting data for initialization
    structure = Structure()
    vtk_obj = vr.pv.read(os.path.join(directory, files[0]))
    structure.load_from_vtk(vtk_obj)
    cell_dim, deposit, precursor, surface_bool, semi_surface_bool, ghosts_bool = structure.cell_dimension, \
                                                                                 structure.deposit, \
                                                                                 structure.precursor, \
                                                                                 structure.surface_bool, \
                                                                                 structure.semi_surface_bool, \
                                                                                 structure.ghosts_bool
    # Determining rendered dataset
    if show not in ['precursor', 'deposit', 'temperature']:
        raise RuntimeError(f'The specified dataset \'{show}\' is not supported.')
    if show == 'precursor':
        data_name = show.capitalize()
        cmap = 'plasma'
        mask_name = 'surface_bool'
    if show == 'deposit':
        data_name = show.capitalize()
        cmap = 'viridis'
        mask_name = 'surface_bool'
    if show == 'temperature':
        data_name = show.capitalize()
        cmap = 'inferno'
        mask_name = 'deposit'
    data = structure.__getattribute__(show)
    mask = structure.__getattribute__(mask_name)
    t, sim_time, beam_position = vr.read_field_data(vtk_obj) # getting deposition process features
    # Preparing left corner text with times
    text = ''
    if t:
        text += f'Time: {t} \n'
    if sim_time:
        text += f'Simulation time: {sim_time:.7f} s \n'

    # Setting the setup scene
    render = vr.Render(cell_dim)
    render._add_3Darray(data)
    render.p.add_text('Adjust the scene for the animation \n and close the window.', position='upper_edge')
    cam_pos = render.show()
    # Setting the first frame
    render = vr.Render(cell_dim)
    # Creating an arrow at beam position
    if beam_position is not None:
        x_pos, y_pos = beam_position
        x, y = int(x_pos / render.cell_dim), int(y_pos / render.cell_dim)
        max_z = structure.deposit[:, y, x].nonzero()[0].max()
        start = np.array([0, 0, 100]).reshape(1, 3)  # position of the center of the arrow
        end = np.array([0, 0, -100]).reshape(1, 3)  # direction and resulting size
        render.arrow = render.p.add_arrows(start, end, color='tomato', name='Beam_position')
        render.arrow.SetPosition(x_pos, y_pos, max_z * render.cell_dim + 30)  # relative to the initial position
    render._add_3Darray(data, opacity=1, show_edges=True,
                        scalar_name=data_name, button_name=data_name, cmap=cmap)
    # Hiding cells
    index = np.zeros_like(data, dtype=np.uint8)
    index[mask == False] = vtk.vtkDataSetAttributes.HIDDENCELL
    render.p.mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = index.ravel()
    render.p.mesh.set_active_scalars(data_name)
    # Adding text
    t, sim_time, beam_position = vr.read_field_data(vtk_obj)
    text = ''
    if t:
        text += f'Time: {t} \n'
    if sim_time:
        text += f'Simulation time: {sim_time:.7f} s \n'
    font_size = 12
    stats = '''Cells:
    Height:
    Volume:
    Frame '''
    render.p.add_text(text, position='upper_left', font_size=font_size, name='time')
    render.p.add_text(stats, font_size=font_size, position='upper_right', name='stats')
    render.show(interactive_update=True, cam_pos=cam_pos)
    init_layer = np.count_nonzero(deposit==-2) # substrate layer
    total_dep_cells = [np.count_nonzero(deposit[deposit<0])-init_layer] # total number of fully deposited cells
    growth_rate=[] # growth rate on each step

    # Animation
    for i in range(1, len(files)):
        # Reading data
        vtk_obj = vr.pv.read(os.path.join(directory, files[i]))
        # Loading the structure
        structure.load_from_vtk(vtk_obj)
        cell_dim = structure.cell_dimension
        surface_bool = structure.surface_bool
        deposit = structure.deposit
        data = structure.__getattribute__(show)
        if show == 'precursor':
            mask = surface_bool
        if show == 'deposit':
            mask = surface_bool
        if show == 'temperature':
            mask = deposit<0
        # Calculating deposition process features
        total_dep_cells.append(np.count_nonzero(deposit[deposit < 0]) - init_layer)
        volume = int((total_dep_cells[i] + deposit[surface_bool].sum())*cell_dim**3)
        delta_t = (times[i] - times[i - 1]).total_seconds()
        if delta_t < 1: delta_t = 1
        growth_rate.append(cell_dim**3 * (total_dep_cells[i] - total_dep_cells[i - 1]) / delta_t * 60 * 60)
        t, sim_time, beam_position = vr.read_field_data(vtk_obj)
        # Setting up text
        # Left corner
        text = ''
        if t:
            text += f'Time: {t} \n'
        if sim_time:
            text += f'Simulation time: {sim_time:.7f} s \n'
        # Right corner
        stats = f'Cells: {total_dep_cells[i]} \n\
                Height: {int(np.nonzero(deposit)[0].max() * cell_dim)} nm \n\
                Volume: {volume} nm^3 \n\
                Frame {i+1}/{len(files)}'
        # Updating arrow position
        if beam_position is not None:
            x_pos, y_pos = beam_position
            x, y = int(x_pos / render.cell_dim), int(y_pos / render.cell_dim)
            max_z = structure.deposit[:, y, x].nonzero()[0].max()
            render.arrow.SetPosition(x_pos, y_pos, max_z * render.cell_dim + 30)  # relative to the initial position
        # Redrawing the 3D object if necessary
        if render.p.mesh.n_cells != data.size: # must redraw if shape changes
            data_visibility = render.p.renderer.actors[data_name].GetVisibility()
            render.p.remove_actor(data_name+'_caption')
            render.y_pos = 5
            render.p.button_widgets.clear()
            render.p.remove_actor(data_name)
            render._add_3Darray(data, opacity=1, show_edges=True,
                            scalar_name=data_name, button_name=data_name, cmap=cmap)
            render.p.renderer.actors[data_name].SetVisibility(data_visibility)
            index = np.zeros_like(data, dtype=np.uint8)
            render.p.mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = index.ravel()
            render.p.mesh.set_active_scalars(data_name)
        else:
            render.p.mesh[data_name] = data.ravel() # new data, ravel() sends a view
        # Updating hidden cells
        index[mask == 0] = vtk.vtkDataSetAttributes.HIDDENCELL
        index[mask == 1] = 0  # surface_bool is not bool type and cannot be used directly as index
        # Updating text
        render.p.actors['time'].SetText(2, text)
        render.p.actors['stats'].SetText(3, stats)

        # render.p.mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = index.ravel()
        p=data[mask]
        # Updating the scalar bar
        try:
            render.p.update_scalar_bar_range([np.partition(p[p!=p.min()], 4)[2], p.max()])
        except: pass
        render.update()
    else: # finishing with a static scene
        # Clearing the scene
        render.y_pos = 5
        render.p.button_widgets.clear()
        render.p.clear()
        # Reading data
        vtk_obj = vr.pv.read(os.path.join(directory, files[-1]))
        # Loading the structure
        structure.load_from_vtk(vtk_obj)
        cell_dim = structure.cell_dimension
        surface_bool = structure.surface_bool
        deposit = structure.deposit
        data = structure.__getattribute__(show)
        total_dep_cells.append(np.count_nonzero(deposit[deposit < 0]) - init_layer)
        render._add_3Darray(data, opacity=1, show_edges=True,
                            scalar_name=data_name, button_name=data_name, cmap=cmap)  # adding structure
        render.p.update_scalar_bar_range([np.partition(p[p != p.min()], 4)[2], p.max()])
        render.p.mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = index.ravel()
        render.p.mesh.set_active_scalars(data_name)
        t, sim_time, beam_position = vr.read_field_data(vtk_obj)
        text = ''
        if t:
            text += f'Time: {t} \n'
        if sim_time:
            text += f'Simulation time: {sim_time:.7f} s \n'
        render.p.add_text(text, position='upper_left', font_size=font_size)  # showing time passed
        render.p.add_text(f'Cells: {total_dep_cells[i-1]} \n'  # showing total number of deposited cells
                          f'Height: {int(np.nonzero(deposit)[0].max() * cell_dim)} nm \n'  # showing current height of the structure
                          f'Volume: {volume} nm^3 \n'
                          f'Growth rate: {int(np.asarray(growth_rate).mean())} cell/h \n'  # showing average growth rate
                          f'Frame {i+1}/{len(files)} \n', position='upper_right', font_size=font_size)
        if beam_position is not None:
            x_pos, y_pos = beam_position
            x, y = int(x_pos / render.cell_dim), int(y_pos / render.cell_dim)
            max_z = structure.deposit[:, y, x].nonzero()[0].max()
            start = np.array([0, 0, 100]).reshape(1, 3)  # position of the center of the arrow
            end = np.array([0, 0, -100]).reshape(1, 3)  # direction and resulting size
            render.arrow = render.p.add_arrows(start, end, color='tomato')
            render.arrow.SetPosition(x_pos, y_pos, max_z * render.cell_dim + 30)  # relative to the initial position
        render.show(interactive_update=False)

if __name__ == '__main__':
    show_animation()