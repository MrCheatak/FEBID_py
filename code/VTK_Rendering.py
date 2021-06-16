import copy

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import pylab as p
from matplotlib import cm
import pyvista as pv
from tqdm import tqdm
import os

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


    def add_3Darray(self, arr, cell_dim, lower_t=0, upper_t=1, scalar_name='scalars_s', button_name='1', color='', cmap='viridis'):
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
        obj = self.render_3Darray(arr=arr, cell_dim=cell_dim, lower_t=lower_t, upper_t=upper_t, name=scalar_name)
        self.prepare_obj(obj, button_name, cmap, color)

    def prepare_obj(self, obj, name, cmap, color):
        while True:
            if color:
                obj_a = self.p.add_mesh(obj, style='surface', opacity=0.5, label='Structure', color=color) # adding data to the plot
                break
            if cmap:
                obj_a = self.p.add_mesh(obj, style='surface', opacity=0.5, label='Structure', cmap=cmap)
                break
        self.p.add_text(name, font_size=self.font, position=(self.x_pos + 5, self.y_pos)) # captioning button
        obj_aa = self.SetVisibilityCallback(obj_a)
        self.p.add_checkbox_button_widget(obj_aa, value=True, position=(5, self.y_pos), size=self.size, color_on='blue') # adding button
        self.y_pos += self.size


    def render_3Darray(self, arr, cell_dim, lower_t=0, upper_t=1, name='scalars_s' ):
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
        grid = pv.UniformGrid()
        grid.dimensions = np.asarray([arr.shape[2], arr.shape[1], arr.shape[0]]) + 1 # creating grid with the size of the array
        grid.spacing = (cell_dim, cell_dim, cell_dim) # assigning dimensions of a cell
        grid.cell_arrays[name] = arr.flatten() # writing values
        grid = grid.threshold([lower_t,upper_t]) # trimming
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