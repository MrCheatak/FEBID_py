# Default packages
import os, sys, time
import timeit
from datetime import datetime
import copy

# Core packages
import numpy as np
import pyvista as pv
from vtk import vtkDataSetAttributes

# Axillary packeges
from tqdm import tqdm
from tkinter import filedialog as fd
import pickle

# Local packages
from febid.Structure import Structure



#### Some colormap(cmap) names: viridis, inferno, plasma, coolwarm, cool, Spectral

filename = ''

class Render:
    """
    Class implementing rendering utilities for visualizing of Numpy data using Pyvista
    """
    def __init__(self, cell_dim:int, font=12, button_size=25):
        """
        :param cell_dim: cell data spacing for VTK objects
        :param font: button caption font size
        :param button_size: size of the show on/off button
        """
        self.p = pv.Plotter() # main object that keeps the plot
        self.cell_dim = cell_dim
        self.font = font # button caption font size
        self.size = button_size # button size
        self.y_pos = 5 # y-position of a button
        self.x_pos = self.size + 5 # x-position of a button
        self.meshes_count = 0
    class SetVisibilityCallback:
        """
        Helper callback to keep a reference to the actor being modified.
        This helps button show and hide plot elements
        """
        def __init__(self, actor):
            self.actor = actor

        def __call__(self, state):
            self.actor.SetVisibility(state)

    def show_full_structure(self, structure:Structure, struct=True, deposit=True, precursor=True, surface=True, semi_surface=True, ghosts=True):
        """
        Render and plot all the structure components

        :param structure: data object
        :param struct: if True, plot solid structure
        :param deposit: if True, plot deposit on the surface
        :param precursor: if True, plot precursor surface density
        :param surface: if True, color all surface cells
        :param semi_surface: if True, color all semi_surface cells
        :param ghosts: if True, color ghost cells
        :return:
        """
        if struct:
            self._add_3Darray(structure.deposit, structure.deposit.min(), -0.01, False, opacity=1, clim=[-2,-1], below_color='red', show_edges=False, scalar_name='Structure', button_name='Structure', cmap='binary', n_colors=1, show_scalar_bar=False)
        if deposit:
            self._add_3Darray(structure.deposit, 0.00001, 1, False, opacity=1, clim=[0.00001,1], below_color='red', above_color='red', show_edges=True, scalar_name='Surface deposit', button_name='Deposit', cmap='viridis')
        if precursor:
            self._add_3Darray(structure.precursor, 0.00001, 1, False, opacity=1, show_edges=True, scalar_name="Surface precursor density", button_name='Precursor', cmap='plasma')
        if surface:
            self._add_3Darray(structure.surface_bool, 1, 1, False, opacity=0.7, show_edges=True, scalar_name="Semi surface prec. density", button_name='Surface', color='red', show_scalar_bar=False)
        if semi_surface:
            self._add_3Darray(structure.semi_surface_bool, 1, 1, False, opacity=0.7, show_edges=True, scalar_name="Semi surface prec. density", button_name='Semi-surface', color='green', show_scalar_bar=False)
        if ghosts:
            self._add_3Darray(structure.ghosts_bool, 1, 1, False, opacity = 0.7, show_edges=True, scalar_name='ghosts', button_name="Ghosts", color='brown', show_scalar_bar=False)

        init_layer = np.count_nonzero(structure.deposit == -2)  # substrate layer
        total_dep_cells = np.count_nonzero(structure.deposit[structure.deposit < 0]) - init_layer  # total number of fully deposited cells
        self.p.add_text(f'Cells: {total_dep_cells} \n'  # showing total number of deposited cells
                        f'Height: {int(np.nonzero(structure.deposit)[0].max() * structure.cell_dimension)} nm \n'                           # showing current height of the structure
                        f'Deposited volume: {int(total_dep_cells + structure.deposit[structure.deposit>0].sum()) * structure.cell_dimension**3} nm^3\n',
                        position='upper_right', font_size=self.font)
        cam_pos = [(463.14450307610286, 271.1171723376318, 156.56895424388603),
                   (225.90027381807235, 164.9577775224395, 71.42188811921902),
                   (-0.27787912231751677, -0.1411181984824172, 0.950194110399093)]
        return self.show(cam_pos=cam_pos)

    def show_mc_result(self, grid, pe_traj=None, deposited_E=None, surface_flux=None, se_traj=None, cam_pos=None, interactive=True):
        pe_traj = copy.deepcopy(pe_traj)
        se_traj = copy.deepcopy(se_traj)
        if grid is not None:
            self._add_3Darray(grid, -2, -0.01, opacity=0.9, show_edges=True, scalar_name='Structure', button_name='Structure', color='white')
        if pe_traj is not None:
            self._add_trajectory(pe_traj[:,0], pe_traj[:,1], 0.2, step=1, scalar_name='PE Energy, keV', button_name='PEs', cmap='viridis')
        if deposited_E is not None:
            self._add_3Darray(deposited_E, 1, exclude_zeros=False, opacity=0.7, show_edges=False, scalar_name='Deposited energy, eV', button_name="Deposited energy", cmap='coolwarm', log_scale=True)
        if surface_flux is not None:
            self._add_3Darray(surface_flux, 1, exclude_zeros=False, opacity=1, show_edges=False, scalar_name='SE Flux, 1/(nm^2*s)', button_name="SE surface flux", cmap='plasma', log_scale=True)
        if se_traj is not None:
            max_trajes = 4000
            step = int(se_traj.shape[0]/max_trajes)+1
            self._add_trajectory(se_traj, radius=0.1, step=step, button_name='SEs', cmap=None, color='red')
        if cam_pos is None:
            cam_pos = [(463.14450307610286, 271.1171723376318, 156.56895424388603),
                   (225.90027381807235, 164.9577775224395, 71.42188811921902),
                   (-0.27787912231751677, -0.1411181984824172, 0.950194110399093)]
        return self.show(cam_pos=cam_pos, interactive_update=interactive)

    def _add_trajectory(self, traj, energies=None, radius=0.7, step=1, scalar_name='scalars_t', button_name='1', color='', cmap='plasma'):
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
        if energies is None:
            energies = []
        obj = self._render_trajectories(traj=traj, energies=energies, radius=radius, step=step, name=scalar_name)
        self.__prepare_obj(obj, button_name, cmap, color)


    def _add_3Darray(self, arr, lower_t=None, upper_t=None, exclude_zeros=False, opacity=0.5, clim=None, below_color=None, above_color=None, show_edges=None, nan_opacity=None, scalar_name='scalars_s', button_name='NoName', color=None, show_scalar_bar=True, cmap=None, n_colors=256, log_scale=False, invert=False, texture=None):
        """
        Adds 3D structure from a Numpy array to the Pyvista plot

        :param arr: numpy array
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
        self.obj = self._render_3Darray(arr=arr, lower_t=lower_t, upper_t=upper_t, exclude_zeros=exclude_zeros, name=scalar_name, invert=invert)
        self.__prepare_obj(self.obj, button_name, cmap, color, show_scalar_bar, clim, below_color, above_color, n_colors, log_scale=log_scale, opacity=opacity, nan_opacity=nan_opacity, show_edges=show_edges, texture=texture)

    def __prepare_obj(self, obj, name, cmap, color, show_scalar_bar=True, clim=None, below_color=None, above_color=None, n_colors=256, log_scale=False, opacity=0.5, nan_opacity=0.5, show_edges=None, texture=None):
        while True:
            try:
                if not cmap:
                    obj_a = self.p.add_mesh(obj, style='surface', opacity=opacity, nan_opacity=nan_opacity, clim=clim, below_color=below_color, above_color=above_color, name=name, label='Structure', log_scale=log_scale, show_scalar_bar=show_scalar_bar, n_colors=n_colors, color=color, lighting=True, show_edges=show_edges, texture=texture, render=False) # adding data to the plot
                    break
                else:
                    obj_a = self.p.add_mesh(obj, style='surface', opacity=opacity, use_transparency=False, nan_opacity=nan_opacity, clim=clim, below_color=below_color, above_color=above_color, name=name, label='Structure', log_scale=log_scale, show_scalar_bar=show_scalar_bar, cmap=cmap, n_colors=n_colors, lighting=True, show_edges=show_edges, texture=texture, render=False)
                    break
            except Exception as e:
                print(f'Error:{e.args}')
                return
        self.p.add_text(name, font_size=self.font, position=(self.x_pos + 5, self.y_pos)) # captioning button
        obj_aa = self.SetVisibilityCallback(obj_a)
        self.p.add_checkbox_button_widget(obj_aa, value=True, position=(5, self.y_pos), size=self.size, color_on='blue') # adding button
        self.y_pos += self.size
        self.meshes_count += 1


    def _render_3Darray(self, arr, lower_t=None, upper_t=None, exclude_zeros=False, name='scalars_s', invert=False ):
        """
        Renders a 3D numpy array and trimms values

        :param arr: array
        :param lower_t: lower cutoff threshold
        :param upper_t: upper cutoff threshold
        :return: pyvista.PolyData object
        """
        # if upper_t is None: upper_t = arr.max()
        # if lower_t is None: lower_t = arr.min()
        grid = numpy_to_vtk(arr, self.cell_dim, data_name=name, grid=None)
        if exclude_zeros:
            grid.remove_cells((arr==0).flatten())
        if upper_t is not None or lower_t is not None:
            if upper_t is None: upper_t = arr.max()
            if lower_t is None: lower_t = arr.min()
            grid = grid.threshold([lower_t,upper_t], continuous=True, invert=invert) # trimming
        return grid


    def _render_trajectories(self, traj, energies, radius=0.7, step=1, name='scalars_t'):
        """
        Renders mapped trajectories as splines with the given thickness

        :param traj: collection of trajectories
        :param energies: collection of energies
        :param radius: line width
        :return: pyvista.PolyData object
        """

        mesh = pv.PolyData()
        # If energies are provided, they are gonna be used as scalars to color trajectories
        start = timeit.default_timer()
        if len(energies) != 0:
            print('Rendering PEs...', end='')
            for i in tqdm(range(0, len(traj), step)): #
            #     mesh = mesh + self.__render_trajectory(traj[i], energies[i], radius, name)
            #     mesh[name] = energies
                mesh_d = self.__render_trajectories(np.asarray(traj[i]), 'line')
                mesh_d[name] = np.asarray(energies[i])
                mesh += mesh_d
            print(f'took {timeit.default_timer()-start}')
        else:
            print('Rendering SEs...', end='')
            # for i in tqdm(range(0, len(traj), step)):
                # mesh = mesh + self.__render_trajectory(traj[i], 0, radius, name)
            traj = traj.reshape(traj.shape[0] * 2, 3)
            mesh = self.__render_trajectories(traj, 'seg')
            print(f'took {timeit.default_timer() - start}')
        return mesh.tube(radius=radius) # it is important for color mapping to create tubes after all trajectories are added

    def __render_trajectories(self, traj, kind='line'):
        """
        Turn a collection of points to line/lines.

        kind: 'line' to create a line connecting all the points
              'seg' to create separate segments (2 points each)

        :param traj: collection of points
        :param kind: type of connection
        :return:
        """

        if kind not in ['line', 'seg']:
            raise RuntimeWarning('Wrong \'type\' argument in Render.__render_trajectories. Method accepts \'line\' or \'seg\'')
        traj[:, 0], traj[:,2] = traj[:, 2], traj[:,0].copy() # swapping x and z coordinates
        if kind == 'line':
            mesh = pv.lines_from_points(np.asfortranarray(traj))
        if kind == 'seg':
            mesh = pv.line_segments_from_points(traj)
        return mesh

    def __render_trajectory(self, traj, energies=0, radius=0.7, name='scalars'):
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

    def update_mask(self, mask):
        index = np.zeros_like(mask, dtype=np.uint8)
        index[mask == 0] = vtkDataSetAttributes.HIDDENCELL
        last_scalars = self.p.mesh.array_names[0]
        self.p.mesh.cell_data[vtkDataSetAttributes.GhostArrayName()] = index.ravel()
        self.p.mesh.set_active_scalars(last_scalars)

    def save_3Darray(self, filename, arr, data_name='scalar'):
        """
        Dump a Numpy array to a vtk file with a specified name and creation date

        :param filename: distinct name of the file
        :param arr: array to save

        :param data_name: name of the data to include in the vtk dataset
        :return:
        """
        grid = numpy_to_vtk(arr, self.cell_dim, data_name)
        print("File is saved in the same directory with current python script. Current time is appended")
        grid.save(f'{sys.path[0]}{os.sep}{filename}{time.strftime("%H:%M:%S", time.localtime())}.vtk')

    def show(self, screenshot=False, show_grid=True, keep_plot=False, interactive_update=False, cam_pos=None):
        """
        Shows plotting scene

        :param screenshot: if True, a screenshot of the scene will be saved upon showing
        :param show_grid: indicates axes and scales
        :param keep_plot: if True, creates a copy of current Plotter before showing
        :param interactive_update: if True, code execution does not stop while scene window is opened
        :param cam_pos: camera view
        :return: current camera view
        """
        if show_grid:
            self.p.show_grid()
        if keep_plot:
            p1 = copy.deepcopy(self.p)
        camera_pos = self.p.show(screenshot=screenshot, interactive_update=interactive_update, cpos=cam_pos, return_cpos=True)
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
        self.y_pos -= self.size*self.meshes_count
        self.meshes_count = 0
        # self.p.clear()


def numpy_to_vtk(arr, cell_dim, data_name='scalar', grid=None, unstructured=False):
        if not grid:
            grid = pv.UniformGrid()
            grid.dimensions = np.asarray([arr.shape[2], arr.shape[1], arr.shape[0]]) + 1  # creating a grid with the size of the array
            grid.spacing = (cell_dim, cell_dim, cell_dim)  # assigning dimensions of a cell
            grid_given = False
        else:
            grid_given = True
        grid.cell_data[data_name] = arr.ravel()  # writing values
        if unstructured and not grid_given:
            grid = grid.cast_to_unstructured_grid()
        return grid

def save_deposited_structure(structure, filename=None):
    """
    Saves current deposition result to a vtk file
    if filename does not contain path, saves to the current directory

    :param structure: an instance of the current state of the process
    :return:
    """

    cell_dim = structure.cell_dimension
    vtk_obj = numpy_to_vtk(structure.deposit, cell_dim, 'deposit', unstructured=False)
    vtk_obj = numpy_to_vtk(structure.precursor, cell_dim, data_name='precursor_density', grid=vtk_obj)
    vtk_obj = numpy_to_vtk(structure.surface_bool, cell_dim, data_name='surface_bool', grid=vtk_obj)
    vtk_obj = numpy_to_vtk(structure.semi_surface_bool, cell_dim, data_name='semi_surface_bool', grid=vtk_obj)
    vtk_obj = numpy_to_vtk(structure.ghosts_bool, cell_dim, data_name='ghosts_bool', grid=vtk_obj)
    # vtk_obj.__setattr__('features', True) # Availability of this parameter will show if vtk file is either just a structure or a simulation result
    # vtk_obj.__setattr__('substrate_val', structure.substrate_val)
    # vtk_obj.__setattr__('substrate_height', structure.substrate_height)
    # vtk_obj.__setattr__('deposit_val', structure.deposit_val)
    # vtk_obj.__setattr__('volume_prefill', structure.vol_prefill)
    # a = vtk_obj.features
    # b = vtk_obj.cell_data['surface_bool']
    # c = vtk_obj.cell_data['deposit']
    if filename == None:
        filename = "Structure"
    # file = open(f'{filename}{time.strftime("%H:%M:%S", time.localtime())}.vtk', 'wb')
    # pickle.dump(vtk_obj, file,protocol=4)
    # Eventually, vtk does not save those new attributes
    vtk_obj.save(f'{filename}_{time.strftime("%H:%M:%S", time.localtime())}.vtk')

def open_deposited_structure(filename=None, return_structure=False):
    vtk_obj = pv.read(filename)
    structure = Structure()
    cell_dimension = vtk_obj.spacing[0]
    deposit = np.asarray(vtk_obj.cell_data['deposit'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
    substrate = np.asarray(vtk_obj.cell_data['precursor_density'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
    surface_bool = np.asarray(vtk_obj.cell_data['surface_bool'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
    semi_surface_bool = np.asarray(vtk_obj.cell_data['semi_surface_bool'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
    ghosts_bool = np.asarray(vtk_obj.cell_data['ghosts_bool'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))

    if return_structure:
        return structure
    else:
        return (cell_dimension, deposit, substrate, surface_bool, semi_surface_bool, ghosts_bool)

def export_obj(structure, filename=None):
    """
    Export deposited structure as an .obj file

    :param structure: Structure class instance, must have 'deposit' array and 'cell_dimension' value
    :param filename: full path with file name
    :return:
    """
    grid = numpy_to_vtk(structure.deposit, structure.cell_dimension, 'Deposit', None, True)
    grid = grid.threshold([-2,-0.001], continuous=True)
    p = pv.Plotter()
    p.add_mesh(grid)
    p.export_obj(filename)
    return 1

def show_animation(directory=''):
    """
    Show animated process from series of vtk files
    Files must have consequent creation dates to align correctly

    :param directory: folder with vtk files
    :return:
    """
    #TODO: this function could be a standalone app that allows playing series of snapshots as animation
    # Possible features: stop/play button, next/previous snapshot, selection of layers, animation export(?)
    # also it could import and show secondary electron flux distribution
    a=0
    if not directory:
        directory = fd.askdirectory()
    font_size = 12
    files, times = open_file(directory)
    cell_dim, deposit, substrate, surface_bool, semi_surface_bool, ghosts_bool = open_deposited_structure(os.path.join(directory, files[0]))
    substrate[np.isnan(substrate)] = 0  # setting all NAN values to 0
    render = Render(cell_dim)
    render._add_3Darray(substrate, 0.00000001, 1, opacity=0.5, show_edges=True, exclude_zeros=False, scalar_name='Precursor',button_name='Precursor', cmap='plasma')
    cam_pos = render.show(interactive_update=False, cam_pos=[(206.34055818793468, 197.6510638707941, 100.47106597548205),
                                                  (0.0, 0.0, 0.0),
                                                  (-0.23307751464125356, -0.236197909312718, 0.9433373838690787)])
    render = Render(cell_dim)
    render._add_3Darray(substrate, 0.00000001, 1, opacity=0.5, show_edges=True, exclude_zeros=False,
                        scalar_name='Precursor', button_name='Precursor', cmap='plasma')
    render.show(interactive_update=True, cam_pos=cam_pos)
    init_layer =np.count_nonzero(deposit==-2) # substrate layer
    total_dep_cells = [np.count_nonzero(deposit[deposit<0])-init_layer] # total number of fully deposited cells
    growth_rate=[] # growth rate on each step
    for i in range(1, len(files)):
        # filename = os.path.join(directory, filename)
        # os.renames(filename, filename.replace('.0',''))
        # with pv.read(os.path.join(directory, filename)) as vtk_obj:
        cell_dim, deposit, substrate, surface_bool, semi_surface_bool, ghosts_bool = open_deposited_structure((os.path.join(directory, files[i])))
        total_dep_cells.append(np.count_nonzero(deposit[deposit<0])-init_layer)
        growth_rate.append((total_dep_cells[i]-total_dep_cells[i-1])/((times[i]-times[i-1]).total_seconds())*60*60)
        # render.add_3Darray(deposit, cell_dim,-2, -0.001, 0.7, scalar_name='Solid Deposit', button_name='Deposit')
        # render.add_3Darray(deposit, cell_dim, 0.001, 1, 0.7, scalar_name='Surface Deposit', button_name='Deposit(S)')
        # substrate[substrate < 0] = 0
        # substrate[substrate > 1] = 0
        substrate[np.isnan(substrate)] = 0
        # if i == 0:
        render.p.clear()
        render._add_3Darray(substrate, 0.00000001, 1, opacity=1, show_edges=True, exclude_zeros=False, scalar_name='Precursor', button_name='Precursor', cmap='plasma') # adding structure
        render.p.add_text(str(times[i]-times[0])) # showing time passed
        render.p.add_text(f'Cells: {total_dep_cells[i]} \n' # showing total number of deposited cells
                          f'Height: {int(np.nonzero(deposit)[0].max()*cell_dim)} nm \n'# showing current height of the structure
                          f'Growth rate: {int(np.asarray(growth_rate).mean())} cell/h \n' # showing average growth rate
                          f'Frame {i}/{len(files)} \n', position='upper_right', font_size=font_size)
        # else:
        #     render.p.update_scalars(substrate.ravel(), render=True)
        render.update(force_redraw=True) # redrawing scene
    else:
        cell_dim, deposit, substrate, surface_bool, semi_surface_bool, ghosts_bool = open_deposited_structure(
            (os.path.join(directory, files[-1])))
        total_dep_cells.append(np.count_nonzero(deposit[deposit < 0]) - init_layer)
        growth_rate.append(
            (total_dep_cells[-1] - total_dep_cells[-2]) / ((times[-1] - times[-2]).total_seconds()) * 60 * 60)
        # render.add_3Darray(deposit, cell_dim,-2, -0.001, 0.7, scalar_name='Solid Deposit', button_name='Deposit')
        # render.add_3Darray(deposit, cell_dim, 0.001, 1, 0.7, scalar_name='Surface Deposit', button_name='Deposit(S)')
        # substrate[substrate < 0] = 0
        # substrate[substrate > 1] = 0
        substrate[np.isnan(substrate)] = 0
        # if i == 0:
        # render.p.clear()
        render._add_3Darray(substrate, 0.00000001, 1, opacity=1, show_edges=True, exclude_zeros=False,
                            scalar_name='Precursor', button_name='Precursor', cmap='plasma')  # adding structure
        render.p.add_text(str(times[-1] - times[0]))  # showing time passed
        render.p.add_text(f'Cells: {total_dep_cells[i-1]} \n'  # showing total number of deposited cells
                          f'Height: {int(np.nonzero(deposit)[0].max() * cell_dim)} nm \n'  # showing current height of the structure
                          f'Growth rate: {int(np.asarray(growth_rate).mean())} cell/h \n'  # showing average growth rate
                          f'Frame {i}/{len(files)} \n', position='upper_right', font_size=font_size)
        render.show(interactive_update=False)
    input()


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


if __name__ == '__main__':
    dir = fd.askdirectory()
    show_animation(dir)
