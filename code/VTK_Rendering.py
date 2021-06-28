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


    def add_3Darray(self, arr, cell_dim, lower_t=0, upper_t=1, scalar_name='scalars_s', button_name='1', color='', cmap='viridis', log_scale=False, invert=False):
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
        obj = self.render_3Darray(arr=arr, cell_dim=cell_dim, lower_t=lower_t, upper_t=upper_t, name=scalar_name, invert=invert)
        self.prepare_obj(obj, button_name, cmap, color, log_scale=log_scale)

    def prepare_obj(self, obj, name, cmap, color, log_scale=False):
        while True:
            if color:
                obj_a = self.p.add_mesh(obj, style='surface', opacity=0.5, label='Structure', log_scale=log_scale, color=color) # adding data to the plot
                break
            if cmap:
                obj_a = self.p.add_mesh(obj, style='surface', opacity=0.5, label='Structure', log_scale=log_scale, cmap=cmap)
                break
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



    def show(self, screenshot=False, show_grid=True, keep_plot=False):
        if show_grid:
            self.p.show_grid()
        if keep_plot:
            p1 = copy.deepcopy(self.p)
        camera_pos = self.p.show(screenshot=screenshot)
        if keep_plot:
            self.p = copy.deepcopy(p1)
        self.y_pos = 5
        return camera_pos


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
    # file = open(f'{sys.path[0]}{os.sep}{filename}{time.strftime("%H:%M:%S", time.localtime())}.vtk', 'wb')
    # pickle.dump(vtk_obj, file,protocol=4)
    # Eventually, vtk does not save those new attributes
    vtk_obj.save((f'{sys.path[0]}{os.sep}{filename}{time.strftime("%H:%M:%S", time.localtime())}.vtk'))