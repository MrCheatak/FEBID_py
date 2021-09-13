import copy

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import pylab as p
from matplotlib import cm
import pyvista as pv
from tqdm import tqdm
import os, sys, time
import pickle

class Render:
    def __init__(self, font=12, button_size=25):
        self.p = pv.Plotter() # object of the plot
        self.font = font # button caption font size
        self.size = button_size # button size
        self.y_pos = 5 # y-position of a button
        self.x_pos = self.size + 5 # x-position of a button
    class SetVisibilityCallback:
        """
        Helper callback to keep a reference to the actor being modified.
        This helps button show and hide plot elements
        """
        def __init__(self, actor):
            self.actor = actor

        def __call__(self, state):
            self.actor.SetVisibility(state)

    def add_trajectory(self, traj, energies=[], radius=0.7, step=1, scalar_name='scalars_t', button_name='1', color='', cmap='plasma'):
        """
        Adds trajectories to the Pyvista plot

        :param traj: collection of trajectories, that are represented as a sequence of points
        :param energies: collection of corresponding energies
        :param radius: line thickness
        :param step: a number of trajectories to skip
        :param scalar_name: name of the scalar bar
        :param button_name: button caption
        :param color: color of the trajectories
        :param cmap: colormap for the trajectories
        :return: adds PolyData() to Plotter()
        """
        obj = self.render_trajectories(traj=traj, energies=energies, radius=radius, step=step, name=scalar_name)
        self.prepare_obj(obj, button_name, cmap, color)


    def add_3Darray(self, arr, cell_dim, lower_t=None, upper_t=1, opacity=0.5, clim=None, nan_opacity=None, scalar_name='scalars_s', button_name='1', color='', cmap='viridis', log_scale=False, invert=False):
        """
        Adds 3D structure from a Numpy array to the Pyvista plot
        :param arr: numpy array
        :param cell_dim: size of a cell
        :param lower_t: lower threshold of values
        :param upper_t: upper threshold of values
        :param scalar_name: name of the scalar bar
        :param button_name: button caption
        :param color: color of the trajectories
        :param cmap: colormap for the trajectories
        :return: adds PolyData() to Plotter()
        """
        if nan_opacity == None:
            nan_opacity = opacity
        self.obj = self.render_3Darray(arr=arr, cell_dim=cell_dim, lower_t=lower_t, upper_t=upper_t, name=scalar_name, invert=invert)
        self.prepare_obj(self.obj, button_name, cmap, color, clim, log_scale=log_scale, opacity=opacity, nan_opacity=nan_opacity)

    def prepare_obj(self, obj, name, cmap, color, clim=None, log_scale=False, opacity=0.5, nan_opacity=None):
        while True:
            try:
                if color:
                    obj_a = self.p.add_mesh(obj, style='surface', opacity=opacity, nan_opacity=nan_opacity, clim=clim, name=name, label='Structure', log_scale=log_scale, color=color, lighting=True, show_edges=True, render=False) # adding data to the plot
                    break
                if cmap:
                    obj_a = self.p.add_mesh(obj, style='surface', opacity=opacity, nan_opacity=nan_opacity, clim=clim, name=name, label='Structure', log_scale=log_scale, cmap=cmap, lighting=True, show_edges=True, render=False)
                    break
            except Exception as e:
                print(f'Error:{e.args}')
                print("Empty mesh, nothing to plot.")
                return
        self.p.add_text(name, font_size=self.font, position=(self.x_pos + 5, self.y_pos)) # captioning button
        obj_aa = self.SetVisibilityCallback(obj_a)
        self.p.add_checkbox_button_widget(obj_aa, value=True, position=(5, self.y_pos), size=self.size, color_on='blue') # adding button
        self.y_pos += self.size


    def render_3Darray(self, arr, cell_dim, lower_t=0, upper_t=1, name='scalars_s', invert=False ):
        """
        Renders a 3D numpy array and trimms values
        Array is plotted as a solid block without value trimming

        :param arr: array
        :param cell_dim: size of a single cell
        :param lower_t: lower cutoff threshold
        :param upper_t: upper cutoff threshold
        :return: pyvista.PolyData object
        """
        if upper_t == 1: upper_t = arr.max()
        if lower_t == 0: lower_t = arr.min()
        grid = numpy_to_vtk(arr, cell_dim, data_name=name)
        if lower_t:
            grid = grid.threshold([lower_t,upper_t], invert=invert) # trimming
        return grid


    def render_trajectories(self, traj, energies=[], radius=0.7, step=1, name='scalars_t'):
        """
        Renders mapped trajectories as splines with the given thickness

        :param traj: collection of trajectories
        :param energies: collection of energies
        :param radius: line width
        :return: pyvista.PolyData object
        """

        mesh = pv.PolyData()
        # If energies are provided, they are gonna be used as scalars to color trajectories
        print("Rendering trajectories:")
        if any(energies):
            for i in tqdm(range(0, len(traj), step)): #
                mesh = mesh + self.render_trajectory(traj[i], energies[i], radius, name)
                # mesh.plot()
        else:
            for i in tqdm(range(0, len(traj), step)):
                mesh = mesh + self.render_trajectory(traj[i], 0, radius, name)
        return mesh.tube(radius=radius) # it is important for color mapping to creaate tubes after all trajectories are added


    def render_trajectory(self, traj, energies=0, radius=0.7, name='scalars'):
        """
        Renders a single trajectory with the given thickness

        :param traj: collection of points
        :param energies: energies for every point
        :param radius: line width
        :return: pyvista.PolyData object
        """
        points = np.asarray([[t[2], t[1], t[0]] for t in traj]) # coordinates are provided in a numpy array manner [z,y,x], but vista implements [x,y,z]
        mesh = pv.PolyData()
        mesh.points = points # assigning points between segments
        line = np.arange(0, len(points), dtype=np.int_)
        line = np.insert(line, 0, len(points))
        mesh.lines = line # assigning lines that connect the points
        if energies:
            mesh[name] = np.asarray(energies) # assigning energies for every point
        return mesh #.tube(radius=radius) # making line thicker

    def save_3Darray(self, filename, arr, cell_dim, data_name='scalar'):
        grid = numpy_to_vtk(arr, cell_dim, data_name)
        print("File is saved in the same directory with current python script. Current time is appended")
        grid.save(f'{sys.path[0]}{os.sep}{filename}{time.strftime("%H:%M:%S", time.localtime())}.vtk')

    def show(self, screenshot=False, show_grid=True, keep_plot=False, interactive_update=False, cam_pos=None):
        if show_grid:
            self.p.show_grid()
        if keep_plot:
            p1 = copy.deepcopy(self.p)
        camera_pos = self.p.show(screenshot=screenshot, interactive_update=interactive_update, cpos=cam_pos)
        if keep_plot:
            self.p = copy.deepcopy(p1)
        self.y_pos = 5
        return camera_pos

    def update(self, time=1, force_redraw=False):
        """
        Update the plot

        :param time: minimum time before each subsequent update
        :param force_redraw: redraw the plot immediately
        :return:
        """
        self.p.update(stime=time, force_redraw=force_redraw)
        self.y_pos -= self.size
        self.p.clear()


def numpy_to_vtk(arr, cell_dim=1, data_name='scalar', grid=None):
        if not grid:
            grid = pv.UniformGrid()
            grid.dimensions = np.asarray([arr.shape[2], arr.shape[1], arr.shape[0]]) + 1  # creating a grid with the size of the array
            grid.spacing = (cell_dim, cell_dim, cell_dim)  # assigning dimensions of a cell
        grid.cell_arrays[data_name] = arr.flatten()  # writing values
        return grid


def save_deposited_structure( structure: Process.Structure, filename):
    """

    :param structure:
    :return:
    """

    vtk_obj = numpy_to_vtk(structure.deposit, structure.cell_dimension, 'deposit')
    vtk_obj = numpy_to_vtk(structure.substrate, data_name='precursor_density', grid=vtk_obj)
    vtk_obj = numpy_to_vtk(structure.surface_bool, data_name='surface_bool', grid=vtk_obj)
    vtk_obj = numpy_to_vtk(structure.semi_surface_bool, data_name='semi_surface_bool', grid=vtk_obj)
    vtk_obj = numpy_to_vtk(structure.ghosts_bool, data_name='ghosts_bool', grid=vtk_obj)
    vtk_obj.__setattr__('features', True) # Availability of this parameter will show if vtk file is either just a structure or a simulation result
    vtk_obj.__setattr__('substrate_val', structure.substrate_val)
    vtk_obj.__setattr__('substrate_height', structure.substrate_height)
    vtk_obj.__setattr__('deposit_val', structure.deposit_val)
    vtk_obj.__setattr__('volume_prefill', structure.vol_prefill)
    a = vtk_obj.features
    b = vtk_obj.cell_arrays['surface_bool']
    c = vtk_obj.cell_arrays['deposit']
    if filename == None:
        filename = "Structure"
    file = open(f'{sys.path[0]}{os.sep}{filename}{time.strftime("%H:%M:%S", time.localtime())}.vtk', 'wb')
    # pickle.dump(vtk_obj, file,protocol=4)
    # Eventually, vtk does not save those new attributes
    vtk_obj.save((f'{sys.path[0]}{os.sep}{filename}{time.strftime("%H:%M:%S", time.localtime())}.vtk'))

def open_deposited_structure(filename=None):
    vtk_obj = pv.read(filename)
    structure = Structure()
    cell_dimension = vtk_obj.spacing[0]
    deposit = np.asarray(vtk_obj.cell_arrays['deposit'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
    substrate = np.asarray(vtk_obj.cell_arrays['precursor_density'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
    surface_bool = np.asarray(vtk_obj.cell_arrays['surface_bool'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
    semi_surface_bool = np.asarray(vtk_obj.cell_arrays['semi_surface_bool'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
    ghosts_bool = np.asarray(vtk_obj.cell_arrays['ghosts_bool'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))

    return (cell_dimension, deposit, substrate, surface_bool, semi_surface_bool, ghosts_bool)


def show_animation(directory=''):
    """
    Show animated process from series of vtk files
    Files must have consequent creation dates to align correctly

    :param directory: folder with vtk files
    :return:
    """
    a=0
    if not directory:
        directory = fd.askdirectory()
    font_size = 12
    files, times = open_file(directory)
    cell_dim, deposit, substrate, surface_bool, semi_surface_bool, ghosts_bool = open_deposited_structure(pv.read(os.path.join(directory, files[0])))
    render = Render()
    substrate[np.isnan(substrate)] = 0 # setting all NAN values to 0
    render.add_3Darray(substrate, cell_dim, 0.0001, 1, opacity=1, nan_opacity=1, clim=[0, 1], scalar_name='Precursor',button_name='Precursor', cmap='plasma')
    render.show(interactive_update=True, cam_pos=[(206.34055818793468, 197.6510638707941, 100.47106597548205),
                                                  (0.0, 0.0, 0.0),
                                                  (-0.23307751464125356, -0.236197909312718, 0.9433373838690787)])
    init_layer = deposit.shape[1] * deposit.shape[2] * np.nonzero(deposit)[0].max() # substrate layer
    total_dep_cells = [np.count_nonzero(deposit[deposit<0])-init_layer] # total number of fully deposited cells
    growth_rate=[] # growth rate on each step
    for i in range(1, len(files)):
        # filename = os.path.join(directory, filename)
        # os.renames(filename, filename.replace('.0',''))
        # with pv.read(os.path.join(directory, filename)) as vtk_obj:
        cell_dim, deposit, substrate, surface_bool, semi_surface_bool, ghosts_bool = open_deposited_structure(pv.read(os.path.join(directory, files[i])))
        total_dep_cells.append(np.count_nonzero(deposit[deposit<0])-init_layer)
        growth_rate.append((total_dep_cells[i]-total_dep_cells[i-1])/((times[i]-times[i-1]).total_seconds())*60*60)
        # render.add_3Darray(deposit, cell_dim,-2, -0.001, 0.7, scalar_name='Solid Deposit', button_name='Deposit')
        # render.add_3Darray(deposit, cell_dim, 0.001, 1, 0.7, scalar_name='Surface Deposit', button_name='Deposit(S)')
        # substrate[substrate < 0] = 0
        # substrate[substrate > 1] = 0
        substrate[np.isnan(substrate)] = 0
        # if i == 0:
        # render.p.clear()
        render.add_3Darray(substrate, cell_dim, 0.0001, 1, opacity=1, nan_opacity=1, clim=[0,1], scalar_name='Precursor', button_name='Precursor', cmap='plasma') # adding structure
        render.p.add_text(str(times[i]-times[0])) # showing time passed
        render.p.add_text(str(f'Cells: {total_dep_cells[i]}'), position='upper_right', font_size=font_size) # showing total number of deposited cells
        render.p.add_text((str(f'Height: {int(np.nonzero(deposit)[0].max()*cell_dim)} nm')), position=(0.85,0.92), viewport=True, font_size=font_size) # showing current height of the structure
        render.p.add_text((str(f'Growth rate: {int(np.asarray(growth_rate).mean())} cell/h')), position=(0, 0.9), viewport=True, font_size=font_size) # showing average growth rate
        render.p.add_text(f'Frame {i}/{len(files)}', position=(0, 0.85), viewport=True, font_size=font_size)
        # else:
        #     render.p.update_scalars(substrate.ravel(), render=True)
        render.update(500, force_redraw=True) # redrawing scene
        # input()



def export_excel(name='gr1'):
    n=10
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = name
    # Creating column headers
    for i in range(n):
        ws.cell(1, n).value = i
    # Inserting data
    for i in range(n):
        j = i - 2
        k = 1
        ws.cell(i, k).value = i

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
    directory = '/Users/sandrik1742/Documents/PycharmProjects/FEBID/code/Experiment runs/gr=0'
    files = sorted(os.listdir(directory))[1:]
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