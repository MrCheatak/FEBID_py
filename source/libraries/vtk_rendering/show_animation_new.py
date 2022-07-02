import os, time
from datetime import datetime
from tkinter import filedialog as fd

import numpy as np
import vtk

import VTK_Rendering as vr
from Structure import Structure

def open_file(directory=''):
    """
    Gather files and timestamps sorted in the order of creation

    :param directory: folder with vtk files
    :return: filenames and timestamps
    """
    # Getting all filenames in the specified directory
    # Getting creation dates of the files
    # Zipping them together and sorting by the creation date
    # Unzipping and returning in the order sorted
    # directory = '/Users/sandrik1742/Documents/PycharmProjects/FEBID/code/Experiment runs/gr=0'
    files = sorted(os.listdir(directory))[1:]
    for i in range(len(files)-1):
        if os.path.splitext(files[i])[1] != '.vtk':
            files.pop(i)
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
    Show animated process from series of vtk files
    Files must have consequent creation dates to align correctly

    :param directory: folder with vtk files
    :param show: which dataset to use for imaging. Accepts 'precursor' for surface precursor density or 'deposit' for surface deposit filling.
    :return:
    """
    data = None
    if show not in ['precursor', 'deposit']:
        raise RuntimeError(f'The specified dataset \'{show}\' is not supported.')
    if show == 'precursor':
        data_name = show.capitalize()
        cmap = 'plasma'
    if show == 'deposit':
        data_name = show.capitalize()
        cmap = 'viridis'

    # Opening files
    if not directory:
        directory = fd.askdirectory()
    files, times = open_file(directory)
    # Getting data
    structure = Structure()
    structure.load_from_vtk(vr.pv.read(os.path.join(directory, files[0])))
    cell_dim, deposit, precursor, surface_bool, semi_surface_bool, ghosts_bool = structure.cell_dimension, \
                                                                                 structure.deposit, \
                                                                                 structure.precursor, \
                                                                                 structure.surface_bool, \
                                                                                 structure.semi_surface_bool, \
                                                                                 structure.ghosts_bool
    data = structure.__getattribute__(show)
    # Pre-setting scene
    render = vr.Render(cell_dim)
    render._add_3Darray(data)
    render.p.add_text('Adjust the scene for the animation \n and close the window.', position='upper_edge')
    cam_pos = render.show()
    # Setting scene
    render = vr.Render(cell_dim)
    render._add_3Darray(data, opacity=1, show_edges=True,
                        scalar_name=data_name, button_name=data_name, cmap=cmap)
    # Hiding non-surface cells
    index = np.zeros_like(data, dtype=np.uint8)
    index[surface_bool == False] = vtk.vtkDataSetAttributes.HIDDENCELL
    render.p.mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = index.ravel()
    render.p.mesh.set_active_scalars(data_name)
    # Adding text
    font_size = 12
    stats = '''Cells:
    Height:
    Volume:
    Frame '''
    render.p.add_text(str(times[0] - times[0]), font_size=font_size, position='upper_left', name='time')
    render.p.add_text(stats, font_size=font_size, position='upper_right', name='stats')
    render.show(interactive_update=True, cam_pos=cam_pos)
    init_layer = np.count_nonzero(deposit==-2) # substrate layer
    total_dep_cells = [np.count_nonzero(deposit[deposit<0])-init_layer] # total number of fully deposited cells
    growth_rate=[] # growth rate on each step
    # Animation
    for i in range(1, len(files)):
        structure.load_from_vtk(vr.pv.read(os.path.join(directory, files[i])))
        cell_dim = structure.cell_dimension
        surface_bool = structure.surface_bool
        deposit = structure.deposit
        data = structure.__getattribute__(show)
        total_dep_cells.append(np.count_nonzero(deposit[deposit < 0]) - init_layer)
        volume = int((total_dep_cells[i] + deposit[surface_bool].sum())*cell_dim**3)
        delta_t = (times[i] - times[i - 1]).total_seconds()
        if delta_t < 1: delta_t = 1
        growth_rate.append(cell_dim**3 * (total_dep_cells[i] - total_dep_cells[i - 1]) / delta_t * 60 * 60)
        stats = f'Cells: {total_dep_cells[i]} \n\
                Height: {int(np.nonzero(deposit)[0].max() * cell_dim)} nm \n\
                Volume: {volume} nm^3 \n\
                Frame {i+1}/{len(files)}'
        if render.p.mesh.n_cells != data.size: # must redraw if shape changes
            render.p.remove_actor(data_name)
            render.y_pos = 5
            render.p.button_widgets.clear()
            render._add_3Darray(data, opacity=1, show_edges=True,
                            scalar_name=data_name, button_name=data_name, cmap='plasma')
            index = np.zeros_like(data, dtype=np.uint8)
            render.p.mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = index.ravel()
            render.p.mesh.set_active_scalars(data_name)
            render.p.show_grid()
        else:
            render.p.mesh[data_name] = data.ravel() # new data, ravel() sends a view
        # Updating text
        render.p.textActor.renderer.actors['time'].SetText(2, str(times[i]-times[0]))
        render.p.textActor.renderer.actors['stats'].SetText(3, stats)
        # Updating hidden cells
        index[surface_bool == 0] = vtk.vtkDataSetAttributes.HIDDENCELL
        index[surface_bool == 1] = 0 # surface_bool is not bool type and cannot be used directly as index
        # render.p.mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = index.ravel()
        p=data[surface_bool]
        try:
            render.p.update_scalar_bar_range([np.partition(p[p!=p.min()], 4)[2], p.max()])
        except: pass
        render.update()
    else: # finishing with a static scene
        render.y_pos = 5
        render.p.button_widgets.clear()
        render.p.clear()
        structure.load_from_vtk(vr.pv.read(os.path.join(directory, files[-1])))
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
        render.p.add_text(str(times[-1] - times[0]))  # showing time passed
        render.p.add_text(f'Cells: {total_dep_cells[i-1]} \n'  # showing total number of deposited cells
                          f'Height: {int(np.nonzero(deposit)[0].max() * cell_dim)} nm \n'  # showing current height of the structure
                          f'Volume: {volume} nm^3 \n'
                          f'Growth rate: {int(np.asarray(growth_rate).mean())} cell/h \n'  # showing average growth rate
                          f'Frame {i+1}/{len(files)} \n', position='upper_right', font_size=font_size)
        render.show(interactive_update=False)

if __name__ == '__main__':
    show_animation()