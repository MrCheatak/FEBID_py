"""
Core visualization module
"""
# Default packages
import os, sys, time
import timeit
import datetime
import copy

# Core packages
import numpy as np
import pyvista as pv
from vtk import vtkDataSetAttributes

# Axillary packeges
from tqdm import tqdm

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
        self.arrow = None # serves to indicate beam position
    class SetVisibilityCallback:
        """
        Helper callback to keep a reference to the actor being modified.
        This helps button show and hide plot elements
        """
        def __init__(self, actor):
            self.actor = actor

        def __call__(self, state):
            self.actor.SetVisibility(state)

    def show_full_structure(self, structure:Structure, struct=True, deposit=True, precursor=True, surface=True, semi_surface=True, temperature=True, ghosts=True, t=None, sim_time=None, beam=None, cam_pos=None):
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
        if temperature:
            self._add_3Darray(structure.temperature, 1, opacity=1, scalar_name='temperature', button_name='Max. temperature', cmap='inferno')
        init_layer = np.count_nonzero(structure.deposit == -2)  # substrate layer
        total_dep_cells = np.count_nonzero(structure.deposit[structure.deposit < 0]) - init_layer  # total number of fully deposited cells
        self.p.add_text(f'Cells: {total_dep_cells} \n'  # showing total number of deposited cells
                        f'Height: {int(np.nonzero(structure.deposit)[0].max() * structure.cell_dimension)} nm \n'                           # showing current height of the structure
                        f'Deposited volume: {int(total_dep_cells + structure.deposit[structure.deposit>0].sum()) * structure.cell_dimension**3} nm^3\n',
                        position='upper_right', font_size=self.font)
        text = ''
        if t:
            text += f'Time: {t} \n'
        if sim_time:
            text += f'Simulation time: {sim_time:.7f} s \n'
        self.p.add_text(text, position='upper_left', font_size=self.font)

        if beam is not None:
            x_pos, y_pos = beam
            x, y = int(x_pos/self.cell_dim), int(y_pos/self.cell_dim)
            max_z = structure.deposit[:, y, x].nonzero()[0].max()
            start = np.array([0, 0, 100]).reshape(1, 3)  # position of the center of the arrow
            end = np.array([0, 0, -100]).reshape(1, 3)  # direction and resulting size
            self.arrow = self.p.add_arrows(start, end, color='tomato')
            self.arrow.SetPosition(x_pos, y_pos, max_z * self.cell_dim + 30)  # relative to the initial position

        if cam_pos is None:
            cam_pos = [(463.14450307610286, 271.1171723376318, 156.56895424388603),
                       (225.90027381807235, 164.9577775224395, 71.42188811921902),
                       (-0.27787912231751677, -0.1411181984824172, 0.950194110399093)]
        return self.show(cam_pos=cam_pos)

    def show_mc_result(self, grid=None, pe_traj=None, surface_flux=None, se_traj=None, heat_t=None,
                       heat_pe=None, heat_se=None, t=None, sim_time=None, beam=None, cam_pos=None, interactive=True):
        pe_traj = copy.deepcopy(pe_traj)
        se_traj = copy.deepcopy(se_traj)
        if grid is not None:
            self._add_3Darray(grid, -2, -0.01, opacity=0.9, show_edges=True, scalar_name='Structure', button_name='Structure', color='white')
        if pe_traj is not None:
            self._add_trajectory(pe_traj[:,0], pe_traj[:,1], 0.2, step=1, scalar_name='PE Energy, keV', button_name='PEs', cmap='viridis')
        if surface_flux is not None:
            self._add_3Darray(surface_flux, 1, opacity=1, show_edges=False, scalar_name='SE Flux, 1/(nm^2*s)', button_name="SE surface flux", cmap='plasma', log_scale=True)
        if se_traj is not None:
            max_trajes = 4000
            step = int(se_traj.shape[0]/max_trajes)+1
            self._add_trajectory(se_traj, radius=0.1, step=step, button_name='SEs', cmap=None, color='red')
        if heat_t is not None:
            self._add_3Darray(heat_t, 1, opacity=0.7, scalar_name='Total energy transfered to heat, eV', button_name='Total heat', cmap='coolwarm', log_scale=True)
        if heat_pe is not None:
            self._add_3Darray(heat_pe, 1, opacity=0.7, scalar_name='Total PE energy transferred to heat, eV',
                              button_name='PE heat', cmap='coolwarm', log_scale=True)
        if heat_se is not None:
            self._add_3Darray(heat_se, 1, opacity=0.7, scalar_name='Total SE energy transferred to heat, eV',
                              button_name='SE heat', cmap='coolwarm', log_scale=True)
        text = ''
        if t:
            text += f'Time: {t} \n'
        if sim_time:
            text += f'Simulation time: {sim_time:.7f} s \n'
        if beam is not None:
            text += f'Beam position: {beam[0], beam[1]}'
        self.p.add_text(text, position='upper_left', font_size=self.font)
        if cam_pos is None:
            cam_pos = [(463.14450307610286, 271.1171723376318, 156.56895424388603),
                   (225.90027381807235, 164.9577775224395, 71.42188811921902),
                   (-0.27787912231751677, -0.1411181984824172, 0.950194110399093)]
        cam_pos = self.show(cam_pos=cam_pos, interactive_update=interactive)
        return cam_pos


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
        self.p.add_text(name, font_size=self.font, position=(self.x_pos + 5, self.y_pos), name=name+'_caption') # captioning button
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


def read_field_data(vtk_obj):
    """
    Read run time, simulation time and beam position from vtk-file.

    :param vtk_obj: VTK-object (UniformGrid)
    :return:
    """
    t = vtk_obj.field_data.get('time', None)
    sim_time = vtk_obj.field_data.get('simulation_time', None)
    beam_position = vtk_obj.field_data.get('beam_position', None)
    if t:
        t = t[0]
    if sim_time:
        sim_time = sim_time[0]
    if beam_position is not None:
        beam_position = beam_position[0]
    return t, sim_time, beam_position


def numpy_to_vtk(arr, cell_dim, data_name='scalar', grid=None, unstructured=False):
    """
    Convert numpy array to a VTK-datastructure (UniformGrid or UnstructuredGrid).
    If grid is provided, add new dataset to that grid.

    :param arr: numpy array
    :param cell_dim: array cell (cubic) edge length
    :param data_name: name of data
    :param grid: existing UniformGrid
    :param unstructured: if True, return an UnstructuredGrid
    :return:
    """
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


def save_deposited_structure(structure, sim_t=None, t=None, beam_position=None, filename=None):
    """
    Save current deposition result to a vtk file.
    If filename does not contain path, saves to the current directory.

    :param structure: an instance of the current state of the process
    :param sim_t: simulation time, s
    :param t: run time
    :param beam_position: (x,y) current position of the beam
    :param filename: full file name
    :return:
    """

    cell_dim = structure.cell_dimension
    # Accumulating data from the array in a VTK datastructure
    vtk_obj = numpy_to_vtk(structure.deposit, cell_dim, 'deposit', unstructured=False)
    vtk_obj = numpy_to_vtk(structure.precursor, cell_dim, data_name='precursor_density', grid=vtk_obj)
    vtk_obj = numpy_to_vtk(structure.surface_bool, cell_dim, data_name='surface_bool', grid=vtk_obj)
    vtk_obj = numpy_to_vtk(structure.semi_surface_bool, cell_dim, data_name='semi_surface_bool', grid=vtk_obj)
    vtk_obj = numpy_to_vtk(structure.ghosts_bool, cell_dim, data_name='ghosts_bool', grid=vtk_obj)
    vtk_obj = numpy_to_vtk(structure.temperature, cell_dim, data_name='temperature', grid=vtk_obj)
    # Attaching times and beam position
    vtk_obj.field_data['date'] = [datetime.datetime.now()]
    vtk_obj.field_data['time'] = [str(datetime.timedelta(seconds=int(t)))]
    vtk_obj.field_data['simulation_time'] = [sim_t]
    vtk_obj.field_data['beam_position'] = [beam_position]
    if filename == None:
        filename = "Structure"
    vtk_obj.save(f'{filename}_{time.strftime("%H.%M.%S", time.localtime())}.vtk')


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


if __name__ == '__main__':
    raise NotImplementedError
