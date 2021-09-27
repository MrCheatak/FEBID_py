###################################################################
#
#  FEBID Simulation
#
#  Version 0.9
#
####################################################################
import datetime
import math
import os
import sys
import timeit
from contextlib import suppress
from tkinter import filedialog as fd

import line_profiler
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
import pyvista as pv
import scipy.constants as scpc
import yaml
# import ipyvolume as ipv
from numexpr_mod import evaluate_cached, cache_expression
from numpy import zeros, copy, s_
from tqdm import tqdm

import VTK_Rendering as vr
import etraj3d
from modified_libraries.rolling import roll

# from timebudget import timebudget
# TODO: look into k-d trees
# TODO: add a benchmark to determine optimal threads number for current machine


# <editor-fold desc="Parameters">
# td = 1E-6  # dwell time of a beam, s
# Ie = 1E-10  # beam current, A
# beam_d = 10  # electron beam diameter, nm
# effective_diameter = beam_d * 3.3 # radius of an area which gets 99% of the electron beam
# f = Ie / scpc.elementary_charge / (math.pi * beam_d * beam_d / 4)  # electron flux at the surface, 1/(nm^2*s)
# F = 3000  # precursor flux at the surface, 1/(nm^2*s)   here assumed a constant, but may be dependent on time and position
# tau = 500E-6  # average residence time, s; may be dependent on temperature
#
# # Precursor properties
# sigma = 2.2E-2  # dissociation cross section, nm^2; is averaged from cross sections of all electron types (PE,BSE, SE1, SE2)
# n0 = 1.9  # inversed molecule size, Me3PtCpMe, 1/nm^2
# M_Me3PtCpMe = 305  # molar mass of the precursor Me3Pt(IV)CpMe, g/mole
# p_Me3PtCpMe = 1.5E-20  # density of the precursor Me3Pt(IV)CpMe, g/nm^3
# V = 4 / 3 * math.pi * math.pow(0.139, 3)  # atomic volume of the deposited atom (Pt), nm^3
# D = np.float32(1E5)  # diffusion coefficient, nm^2/s
#
#
# kd = F / n0 + 1 / tau + sigma * f  # depletion rate
# kr = F / n0 + 1 / tau  # replenishment rate
# nr = F / kr  # absolute density after long time
# nd = F / kd  # depleted absolute density
# t_out = 1 / (1 / tau + F / n0)  # effective residence time
# p_out = 2 * math.sqrt(D * t_out) / beam_d
# cell_dimension = 5  # side length of a square cell, nm
#
# effective_radius_relative = math.floor(effective_diameter / cell_dimension / 2)
#
# nn=1 # default number of threads for numexpr
# </editor-fold>

# <editor-fold desc="Timings">
# dt = np.float32(1E-6)  # time step, s
# t_flux = 1/(sigma*f)  # dissociation event time
# diffusion_dt = math.pow(cell_dimension * cell_dimension, 2) / (2 * D * (cell_dimension * cell_dimension + cell_dimension * cell_dimension))   # maximum stability lime of the diffusion solution
# tau = 500E-6  # average residence time, s; may be dependent on temperature
# </editor-fold>

# <editor-fold desc="Framework" >
# Main cell matrices
# system_size = 50
# height_multiplyer = 2
# substrate = zeros((system_size*height_multiplyer, system_size, system_size), dtype=np.float32) # substrate[z,x,y] holds precursor density
# deposit = zeros((system_size*height_multiplyer, system_size, system_size), dtype=np.float32) # deposit[z,y,x] holds deposit density
# substrate[0, :, :] = nr  # filling substrate surface with initial precursor density
# # deposit[0, 20:40, 20:40] = 0.95
# zmax, ymax, xmax = substrate.shape # dimensions of the grid

# It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus able produce deposit under irradiation.
# The idea is to avoid iterating through the whole 3D matrix and address only surface cells
# Thus the number of surface cells is fixed.

# Semi-surface cells are cells that have precursor density but do not have deposit right under them
# Thus they cannot produce deposit and their precursor density is calculated without disosisiation term.
# They are introduced to allow diffusion on the walls of the deposit.
# Basically these are all other surface cells
# </editor-fold>

# <editor-fold desc="Helpers">
# center = effective_radius_relative * cell_dimension  # beam center in array-coordinates
# index_y, index_x = mgrid[0:(effective_radius_relative*2+1), 0:(effective_radius_relative*2+1)] # for indexing purposes of flux matrix
# index_yy, index_xx = index_y*cell_dimension-center, index_x*cell_dimension-center

# A dictionary of expressions for numexpr.evaluate_cached
# Debug note: before introducing a new cached expression, that expression should be run with the default 'evaluate' function for fetching the signature list.
# This is required, because variables in it must be in the same order as Numexpr fetches them, otherwise Numexpr compiler will throw an error
# TODO: this has to go the Structure class
expressions = dict(pe_flux=cache_expression("f*exp(-r*r/(2*beam_d*beam_d))", [('beam_d', np.int32), ('f', np.float64), ('r', np.float64)]),
                   rk4=cache_expression("(k1+k4)/6 +(k2+k3)/3", [('k1', np.float64), ('k2', np.float64), ('k3', np.float64), ('k4', np.float64)]),
                   #precursor_density_=cache_expression("(F * (1 - (sub + addon) / n0) - (sub + addon) / tau - (sub + addon) * sigma * flux_matrix)*dt", [('F', np.int64), ('addon', np.float64), ('dt', np.float64), ('flux_matrix', np.int64), ('n0', np.float64), ('sigma',np.float64), ('sub', np.float64), ('tau', np.float64)]),
                   precursor_density=cache_expression("F_dt - (sub + addon) * (F_dt_n0_1_tau_dt + sigma_dt * flux_matrix)", [('F_dt', np.float64), ('F_dt_n0_1_tau_dt', np.float64), ('addon', np.float64), ('flux_matrix', np.int64), ('sigma_dt',np.float64), ('sub', np.float64)]),
                   laplace1=cache_expression("grid_out*dt_D", [('dt_D', np.float64), ('grid_out', np.float64)]),
                   laplace2=cache_expression("grid_out*dt_D_div", [('dt_D_div', np.float64), ('grid_out', np.float64)]),
                   flux_matrix=cache_expression("((index_xx-center)*(index_xx-center)+(index_yy-center)*(index_yy-center))**0.5", [('center', np.int32), ('index_xx', np.int32), ('index_yy', np.int32)]))
# </editor-fold>


class Structure():
    """
    Represents simulation chamber and holds grid parameters
    """
    def __init__(self, cell_dim=5, width=50, length=50, height=100, substrate_height=4, nr=1, vtk_obj: pv.UniformGrid = None, volume_prefill=0.0,):
        """
        Frame initializer. Either a vtk object should be specified or initial conditions given.

        vtk object can either represent only a solid structure or a result of a deposition process with several parameters and arrays.
        If parameters are specified despite being present in vtk file (i.e. cell dimension), newly specified values are taken.

        :param cell_dim: size of a cell in nm
        :param width: width of the simulation chamber (along X-axis)
        :param length: length of the simulation chamber (along Y-axis)
        :param height: height of the simulation chamber (along Z-axis)
        :param substrate_height: thickness of the substrate in a number of cells along Z-axis
        :param nr: initial precursor density
        :param vtk_obj: a vtk object from file
        :param volume_prefill: level of initial filling for every cell. This is used to artificially speed up the depositing process
        """
        if vtk_obj:
            self.cell_dimension = 1
            if vtk_obj.spacing[0] != vtk_obj.spacing[1] != vtk_obj.spacing[2]:
                choice = input(f'Cell\'s dimensions must be equal and represent a cube. \nType x, y or z to specify dimension value that will be used for all three. \nThis may lead to a change of structure\'s shape. Press any other key to exit.')
                if choice == 'x':
                    self.cell_dimension = vtk_obj.spacing[0]
                if choice == 'y':
                    self.cell_dimension = vtk_obj.spacing[1]
                if choice == 'z':
                    self.cell_dimension = vtk_obj.spacing[2]
                else:
                    sys.exit("Exiting.")
            else:
                self.cell_dimension = vtk_obj.spacing[0]
            self.zdim, self.ydim, self.xdim = vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1
            self.shape = (self.zdim, self.ydim, self.xdim)
            if 'surface_bool' in vtk_obj.array_names: # checking if it is a complete result of a deposition process
                self.deposit = np.asarray(vtk_obj.cell_arrays['deposit'].reshape((vtk_obj.dimensions[2]-1, vtk_obj.dimensions[1]-1, vtk_obj.dimensions[0]-1)))
                self.substrate = np.asarray(vtk_obj.cell_arrays['precursor_density'].reshape((vtk_obj.dimensions[2]-1, vtk_obj.dimensions[1]-1, vtk_obj.dimensions[0]-1)))
                self.surface_bool = np.asarray(vtk_obj.cell_arrays['surface_bool'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
                self.semi_surface_bool= np.asarray(vtk_obj.cell_arrays['semi_surface_bool'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
                self.ghosts_bool = np.asarray(vtk_obj.cell_arrays['ghosts_bool'].reshape((vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
                # An attempt to attach new attributes to vtk object failed:
                # self.substrate_height = vtk_obj['substrate_height']
                # self.substrate_val = vtk_obj['substrate_val']
                # self.deposit_val = vtk_obj['deposit_val']
                # self.vol_prefill = vtk_obj['volume_prefill']
                self.substrate_val = -2
                self.deposit_val = -1
                self.substrate_height = np.nonzero(self.deposit==self.substrate_val)[0].max()+1
                self.vol_prefill = self.deposit[-1,-1,-1]
            else:
                # TODO: if a sample structure would be provided, it will be necessary to create a substrate under it
                self.deposit = np.asarray(vtk_obj.cell_arrays.active_scalars.reshape((vtk_obj.dimensions[2]-1, vtk_obj.dimensions[1]-1, vtk_obj.dimensions[0]-1)))
                self.substrate = np.zeros((self.zdim, self.ydim, self.xdim), dtype=np.float64)
                self.substrate_height = substrate_height
                self.substrate_val = -2
                self.deposit_val = -1
                self.vol_prefill = self.deposit[-1, -1, -1] # checking if there is a prefill by probing top corner cell
                self.surface_bool = np.zeros((self.zdim, self.ydim, self.xdim), dtype=bool)
                self.semi_surface_bool = np.zeros((self.zdim, self.ydim, self.xdim), dtype=bool)
                self.ghosts_bool = np.zeros((self.zdim, self.ydim, self.xdim), dtype=bool)
                self.define_surface()
                self.define_ghosts()
        else:
            self.cell_dimension = cell_dim
            self.zdim, self.ydim, self.xdim = height, width, length
            self.shape = (self.zdim, self.ydim, self.xdim)
            self.deposit = np.zeros((self.zdim+substrate_height, self.ydim, self.xdim), dtype=np.float64)
            self.substrate = np.zeros((self.zdim+substrate_height, self.ydim, self.xdim), dtype=np.float64)
            self.substrate_val = -2
            self.deposit_val = -1
            self.substrate_height = substrate_height
            self.vol_prefill = volume_prefill
            self.nr = nr
            self.flush_structure()
            self.surface_bool = np.zeros((self.zdim+substrate_height, self.ydim, self.xdim),dtype=bool)
            self.semi_surface_bool = np.zeros((self.zdim+substrate_height, self.ydim, self.xdim),dtype=bool)
            self.ghosts_bool = np.zeros((self.zdim+substrate_height, self.ydim, self.xdim),dtype=bool)
            self.define_surface()
            self.define_ghosts()
        self.t = 0

    def flush_structure(self):
        """
        Resets and prepares initial state of the grid

        :param substrate: 3D precursor density array
        :param deposit: 3D deposit array
        :param init_density: initial precursor density on the surface
        :param init_deposit: initial deposit on the surface, can be a 2D array with the same size as deposit array along 0 and 1 dimensions
        :param volume_prefill: initial deposit in the volume, can be a predefined structure in an 3D array same size as deposit array (constant value is virtual and used for code development)
        :return:
        """
        self.substrate[...] = 0
        self.substrate[0:self.substrate_height, :, :] = 0  # substrate surface
        self.substrate[self.substrate_height, :, :] = self.nr  # filling substrate surface with initial precursor density
        if self.vol_prefill == 0:
            self.deposit[...] = 0
        else:
            self.deposit[...] = self.vol_prefill  # partially filling cells with deposit
            # if init_deposit != 0:
            #     self.deposit[1, :, :] = init_deposit  # partially fills surface cells with deposit
        self.deposit[0:self.substrate_height, :, :] = -2


    def define_surface(self):
        """
        Determining surface of the initial structure

        :return:
        """

        # The whole idea is to derive surface according to neighboring cells
        # 1. Firstly, a boolean array marking non-solid cells is created (positive)
        # 2. Then, an average of each cell+neighbors is calculated (convolution applied)
        #   after this only cells that are on the surfaces(solid side and gas side) are gonna be changed
        # 3. Surface cells now have changed values and have to be separated from surface on the solid side
        #   it achieved by the intersection of 'positive' and convoluted arrays, as surface is ultimately not a solid

        positive = np.full((self.deposit.shape), False, dtype=bool)
        positive[self.deposit >= 0] = True  # gas cells
        grid = np.copy(self.deposit)
        # Applying convolution;  simple np.roll() does not work well, as it connects the edges(i.E rolls top layer to the bottom)
        grid[:, :, :-1] += self.deposit[:, :, 1:]  # rolling forward (actually backwards)
        grid[:, :, -1] += self.deposit[:, :, -1]  # taking care of edge values
        grid[:, :, 1:] += self.deposit[:, :, :-1]  # rolling backwards
        grid[:, :, 0] += self.deposit[:, :, 0]
        grid[:, :-1, :] += self.deposit[:, 1:, :]
        grid[:, -1, :] += self.deposit[:, -1, :]
        grid[:, 1:, :] += self.deposit[:, :-1, :]
        grid[:, 0, :] += self.deposit[:, 0, :]
        grid[:-1, :, :] += self.deposit[1:, :, :]
        grid[-1, :, :] += self.deposit[-1, :, :]
        grid[1:, :, :] += self.deposit[:-1, :, :]
        grid[0, :, :] += self.deposit[0, :, :]
        grid /= 7  # six neighbors + cell itself
        # Trimming unchanged cells:     using tolerance in case of inaccuracy
        grid[abs(grid - self.deposit_val) < 0.0000001] = 0  # fully deposited cells
        grid[abs(grid - self.substrate_val) < 0.0000001] = 0  # substrate
        grid[abs(grid - self.vol_prefill) < 0.000001] = 0  # prefilled cells
        # Now making a boolean array of changed cells
        combined = np.full((self.deposit.shape), False, dtype=bool)
        combined[abs(grid) > 0] = True
        grid[...] = 0
        # Now, surface is intersection of these boolean arrays:
        grid += positive
        grid += combined
        self.surface_bool[grid == 2] = True


    def define_ghosts(self):
        """
        Determining ghost shell wrapping surface
        This is crucial for the diffusion to work

        :return:
        """

        # Rolling in all directions marks all the neighboring cells
        # Subtracting surface from that selection results in a "shell" around the surface
        self.ghosts_bool = np.copy(self.surface_bool)
        self.ghosts_bool[:, :, :-1] += self.surface_bool[:, :, 1:]  # rolling forward (actually backwards)
        self.ghosts_bool[:, :, -1] += self.surface_bool[:, :, -1]  # taking care of edge values
        self.ghosts_bool[:, :, 1:] += self.surface_bool[:, :, :-1]  # rolling backwards
        self.ghosts_bool[:, :, 0] += self.surface_bool[:, :, 0]
        self.ghosts_bool[:, :-1, :] += self.surface_bool[:, 1:, :]
        self.ghosts_bool[:, -1, :] += self.surface_bool[:, -1, :]
        self.ghosts_bool[:, 1:, :] += self.surface_bool[:, :-1, :]
        self.ghosts_bool[:, 0, :] += self.surface_bool[:, 0, :]
        self.ghosts_bool[:-1, :, :] += self.surface_bool[1:, :, :]
        self.ghosts_bool[-1, :, :] += self.surface_bool[-1, :, :]
        self.ghosts_bool[1:, :, :] += self.surface_bool[:-1, :, :]
        self.ghosts_bool[0, :, :] += self.surface_bool[0, :, :]
        self.ghosts_bool[self.surface_bool] = False

    def max_z(self):
        return self.deposit.nonzero()[0].max()

    def save_to_vtk(self):
        import time
        grid = pv.UniformGrid()
        grid.dimensions = np.asarray([self.deposit.shape[2], self.deposit.shape[1], self.deposit.shape[0]]) + 1  # creating grid with the size of the array
        grid.spacing = (self.cell_dimension, self.cell_dimension, self.cell_dimension)  # assigning dimensions of a cell
        grid.cell_arrays["deposit"] = self.deposit.flatten()
        grid.save('Deposit_'+time.strftime("%H:%M:%S", time.localtime()))

class Material:
    """
    Represents a material object, that stores some physical properties
    """
    def __init__(self, name='noname', Z=1, A=1, rho=1, e=1, lambda_escape=1):
        self.name = name
        self.rho = rho
        self.Z = Z
        self.A = A
        self.e = e
        self.lambda_escape = lambda_escape

def open_stream_file(offset=1.5):
    file = fd.askopenfilename()
    data = None

    :param substrate: 3D precursor density array
    :param deposit: 3D deposit array
    :param init_density: initial precursor density on the surface
    :param init_deposit: initial deposit on the surface, can be a 2D array with the same size as deposit array along 0 and 1 dimensions
    :param volume_prefill: initial deposit in the volume, can be a predefined structure in an 3D array same size as deposit array (constant value is virtual and used for code development)
    :return:
    """
    substrate[...] = 0
    substrate[0:4, :, :] = 0  # substrate surface
    substrate[4, :, :] = init_density  # filling substrate surface with initial precursor density
    if volume_prefill == 0:
        deposit[...] = 0
    else:
        deposit[...] = volume_prefill # partially filling cells with deposit
        if init_deposit != 0:
            deposit[1, :, :] = init_deposit # partially fills surface cells with deposit
    deposit[0:4, :, :] = -2


def define_surface(surf, deposit):
    """
    Determining surface of the initial structure

    :param surf: boolean array
    :param deposit: deposit array
    :return:
    """

    # The whole idea is to derive surface according to neighboring cells
    # 1. Firstly, a boolean array marking non-solid cells is created (positive)
    # 2. Then, an average of each cell+neighbors is calculated (convolution applied)
    #   after this only cells that are on the surfaces(solid side and gas side) are gonna be changed
    # 3. Surface cells now have changed values and have to be separated from surface on the solid side
    #   it achieved by the intersection of 'positive' and convoluted arrays, as surface is ultimately not a solid

    surf[...]=False
    positive = np.full((deposit.shape), False, dtype=bool)
    positive[deposit >= 0] = True # gas cells
    grid = np.copy(deposit)
    # grid +=np.roll(deposit, 1, 0)
    # grid += np.roll(deposit, -1,0)
    # grid += np.roll(deposit, 1, 1)
    # grid += np.roll(deposit, -1, 1)
    # grid += np.roll(deposit, 1, 2)
    # grid += np.roll(deposit, -1, 2)
    # Applying convolution
    grid[:, :, :-1] += deposit[:, :, 1:]  # rolling forward (actually backwards)
    grid[:, :, -1] += deposit[:, :, -1]  # taking care of edge values
    grid[:, :, 1:] += deposit[:, :, :-1]  # rolling backwards
    grid[:, :, 0] += deposit[:, :, 0]
    grid[:, :-1, :] += deposit[:, 1:, :]
    grid[:, -1, :] += deposit[:, -1, :]
    grid[:, 1:, :] += deposit[:, :-1, :]
    grid[:, 0, :] += deposit[:, 0, :]
    grid[:-1, :, :] += deposit[1:, :, :]
    grid[-1, :, :] += deposit[-1, :, :]
    grid[1:, :, :] += deposit[:-1, :, :]
    grid[0, :, :] += deposit[0, :, :]
    grid /= 7 # six neighbors + cell itself
    # Trimming unchanged cells:
    grid[abs(grid + 1)<0.0000001] = 0 # fully deposited cells
    grid[abs(grid + 2)<0.0000001] = 0 # substrate
    grid[abs(grid - deposit[deposit.shape[0] - 1,deposit.shape[1] - 1,deposit.shape[2] - 1])<0.000001] = 0 # prefilled cells
    # Now making a boolean array marking changed cells
    combined = np.full((deposit.shape), False, dtype=bool)
    combined[abs(grid)>0] = True
    grid[...] = 0
    # Now, surface is intersection of these boolean arrays:
    grid += positive
    grid += combined
    surf[grid == 2] = True
    # for i in range(deposit.shape[0]-1):
    #     for j in range(deposit.shape[1]-1):
    #         for k in range(deposit.shape[2]-1):
    #             if deposit[i, j, k] < 1:
    #                 kernel = [[i + 1, i - 1, i, i, i, i], [j, j, j - 1, j + 1, j, j], [k, k, k, k, k - 1, k + 1]]
    #                 # try:
    #                     # if (deposit[i+1,j,k] or deposit[i-1,j,k] or deposit[i,j-1,k] or deposit[i,j+1,k] or deposit[i,j,k-1] or deposit[i,j,k+1])>=1:
    #                 if np.any(deposit[kernel]>=1):
    #                     surf[i,j,k] = True
    #                 # except:
    #                 #     if i==deposit.shape[0]:
    #                 #         i


def define_ghosts_2(surface):
    grid = np.copy(surface)
    grid[:, :, :-1] += surface[:, :, 1:]  # rolling forward (actually backwards)
    grid[:, :, -1] += surface[:, :, -1]  # taking care of edge values
    grid[:, :, 1:] += surface[:, :, :-1]  # rolling backwards
    grid[:, :, 0] += surface[:, :, 0]
    grid[:, :-1, :] += surface[:, 1:, :]
    grid[:, -1, :] += surface[:, -1, :]
    grid[:, 1:, :] += surface[:, :-1, :]
    grid[:, 0, :] += surface[:, 0, :]
    grid[:-1, :, :] += surface[1:, :, :]
    grid[-1, :, :] += surface[-1, :, :]
    grid[1:, :, :] += surface[:-1, :, :]
    grid[0, :, :] += surface[0, :, :]
    grid[surface] = False
    return grid


# @jit(nopython=True, parallel=True)
def deposition(deposit, precursor, flux_matrix, surface_bool, sigma_V_dt, gr = 1):

    """
    Calculates deposition on the surface for a given time step dt (outer loop)

    :param deposit: 3D deposit array
    :param precursor: 3D precursor density array
    :param flux_matrix: matrix of electron flux distribution
    :param dt: time step
    :return: writes back to deposit array
    """
    # Instead of processing cell by cell and on the whole surface, it is implemented to process only (effectively) irradiated area and array-wise(thanks to Numpy)
    # Math here cannot be efficiently simplified, because multiplication of constant variables here produces a value below np.float32 accuracy
    # np.float32 — ~1E-7, produced value — ~1E-10
    deposit[surface_bool] += precursor[surface_bool] * flux_matrix[surface_bool] * sigma_V_dt * gr


# @jit(nopython=True, parallel=True)
def update_surface(deposit, precursor, surface_bool, semi_surf_bool, ghosts_bool):
    """
    Evolves surface upon a full deposition of a cell. This method holds has the vast majority of logic

    :param deposit: 3D deposit array
    :param precursor: 3D precursor density array
    :param surface_bool: array representing surface cells
    :param semi_surf_bool: array representing semi-surface cells
    :param ghosts_bool: array representing ghost cells
    :return: changes surface, semi-surface and ghosts arrays
    """

    new_deposits = np.argwhere(deposit>=1) # looking for new deposits
    if new_deposits.any():
        for cell in new_deposits:
            # deposit[cell[0]+1, cell[1], cell[2]] += deposit[cell[0], cell[1], cell[2]] - 1  # if the cell was filled above unity, transferring that surplus to the cell above
            deposit[cell[0], cell[1], cell[2]] = -1  # a fully deposited cell is always a minus unity
            precursor[cell[0], cell[1], cell[2]] = -1
            ghosts_bool[cell[0], cell[1], cell[2]] = True # deposited cell belongs to ghost shell
            surface_bool[cell[0], cell[1], cell[2]] = False  # rising the surface one cell up (new cell)

            # Instead of using classical conditions, boolean arrays are used to select elements
            # First, a condition array is created, that picks only elements that satisfy conditions
            # Then this array is used as index

            # Kernels for choosing cells
            neibs_sides = np.array([[[0, 0, 0],  # chooses side neighbors
                                     [0, 1, 0],
                                     [0, 0, 0]],
                                    [[0, 1, 0],
                                     [1, 0, 1],
                                     [0, 1, 0]],
                                    [[0, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]]])
            neibs_edges = np.array([[[0, 1, 0],  # chooses edge neighbors
                                     [1, 0, 1],
                                     [0, 1, 0]],
                                    [[1, 0, 1],
                                     [0, 0, 0],
                                     [1, 0, 1]],
                                    [[0, 1, 0],
                                     [1, 0, 1],
                                     [0, 1, 0]]])

            # Creating a view with the 1st nearest neighbors to the deposited cell
            z_min, z_max, y_min,y_max, x_min, x_max = 0,0,0,0,0,0
            if cell[0]-1 < 0:
                z_min = 0
                neibs_sides = neibs_sides[1:,:,:]
                neibs_edges = neibs_edges[1:,:,:]
            else:
                z_min = cell[0] - 1
            if cell[0]+2 > deposit.shape[0]:
                z_max = deposit.shape[0]
                neibs_sides = neibs_sides[:1,:,:]
                neibs_edges = neibs_edges[:1,:,:]
            else:
                z_max = cell[0] + 2
            if cell[1]-1<0:
                y_min = 0
                neibs_sides = neibs_sides[:,1:,:]
                neibs_edges = neibs_edges[:,1:,:]
            else:
                y_min = cell[1] - 1
            if cell[1]+2 > deposit.shape[1]:
                y_max = deposit.shape[1]
                neibs_sides = neibs_sides[:,:1,:]
                neibs_edges = neibs_edges[:,:1,:]
            else:
                y_max = cell[1] + 2
            if cell[2]-1 < 0:
                x_min = 0
                neibs_sides = neibs_sides[:,:,1:]
                neibs_edges = neibs_edges[:,:,1:]
            else:
                x_min = cell[2] - 1
            if cell[2]+2>deposit.shape[2]:
                x_max = deposit.shape[2]
                neibs_sides = neibs_sides[:, :, :1]
                neibs_edges = neibs_edges[:, :, :1]
            else:
                x_max = cell[2] + 2
            # neighbors_1st = s_[cell[0]-1:cell[0]+2, cell[1]-1:cell[1]+2, cell[2]-1:cell[2]+2]
            neighbors_1st = s_[z_min:z_max, y_min:y_max, x_min:x_max]
            # Creating a view with the 2nd nearest neighbors to the deposited cell
            if cell[0]-2 < 0:
                z_min = 0
            else:
                z_min = cell[0] - 2
            if cell[0]+3 > deposit.shape[0]:
                z_max = deposit.shape[0]
            else:
                z_max = cell[0] + 3
            if cell[1]-2<0:
                y_min = 0
            else:
                y_min = cell[1] - 2
            if cell[1]+3 > deposit.shape[1]:
                y_max = deposit.shape[1]
            else:
                y_max = cell[1] + 3
            if cell[2]-2 < 0:
                x_min = 0
            else:
                x_min = cell[2] - 2
            if cell[2]+3>deposit.shape[2]:
                x_max = deposit.shape[2]
            else:
                x_max = cell[2] + 3
            # neighbors_2nd = s_[cell[0]-2:cell[0]+3, cell[1]-2:cell[1]+3, cell[2]-2:cell[2]+3]
            neighbors_2nd = s_[z_min:z_max, y_min:y_max, x_min:x_max]


            deposit_kern = deposit[neighbors_1st]
            semi_s_kern = semi_surf_bool[neighbors_1st]
            surf_kern = surface_bool[neighbors_1st]
            # Creating condition array
            condition = np.logical_and(deposit_kern==0, neibs_sides) # True for elements that are not deposited and are side neighbors
            semi_s_kern[condition] = False
            surf_kern[condition] = True
            condition = np.logical_and(np.logical_and(deposit_kern==0, surf_kern==0), neibs_edges) # True for elements that are not deposited, not surface cells and are edge neighbors
            semi_s_kern[condition] = True
            ghosts_kern = ghosts_bool[neighbors_1st]
            ghosts_kern[...] = False

            surf_kern = surface_bool[neighbors_2nd]
            semi_s_kern = semi_surf_bool[neighbors_2nd]
            ghosts_kern = ghosts_bool[neighbors_2nd]
            condition = np.logical_and(surf_kern==0, semi_s_kern==0) # True for elements that are neither surface nor semi-surface cells
            ghosts_kern[condition] = True

            deposit[cell[0], cell[1], cell[2]] = -1  # a fully deposited cell is always a minus unity
            precursor[cell[0], cell[1], cell[2]] = -1
            ghosts_bool[cell[0], cell[1], cell[2]] = True # deposited cell belongs to ghost shell
            surface_bool[cell[0], cell[1], cell[2]] = False  # rising the surface one cell up (new cell)
            # refresh(precursor, surface_bool, semi_surf_bool, ghosts_bool, cell[0] + 1, cell[1], cell[2])
        else: # when loop finishes
        #     if cell[0] > h_max-3:
        #         h_max += 1
            return True

    return False


# @jit(nopython=True) # parallel=True)
def refresh(precursor, surface_bool, semi_s_bool, ghosts_bool, z,y,x):
    """
    Updates surface, semi-surface and ghost cells collections according to the provided coordinate of a newly deposited cell

    :param precursor: 3D precursor density array
    :param semi_s_bool: array representing semi-surface cells
    :param ghosts_bool: array representing ghost cells
    :param z: z-coordinate of the cell above the new deposit
    :param y: y-coordinate of the deposit
    :param x: x-coordinate of the deposit
    :return: changes surface array, semi-surface and ghosts collections
    """
    # this is needed, due to the precursor view being 2 cells wider in case of semi-surface or ghost cell falling out of the bounds of the view
    semi_s_bool[z, y, x] = False # removing the new cell from the semi_surface collection, because it belongs to surface now
    semi_s_bool[z-1, y-1, x] = False
    surface_bool[z-1, y-1, x] = True
    semi_s_bool[z-1, y+1, x] = False
    surface_bool[z-1, y+1, x] = True
    semi_s_bool[z-1, y, x-1] = False
    surface_bool[z-1, y, x-1] = True
    semi_s_bool[z-1, y, x+1] = False
    surface_bool[z-1, y, x+1] = True
    ghosts_bool[z, y, x] = False # removing the new cell from the ghost shell collection
    precursor[z, y, x] += precursor[z - 1, y, x] # if the deposited cell had precursor in it, transfer that surplus to the cell above
    # this may lead to an overfilling of a cell above unity, but it is not causing any anomalies due to diffusion process
    precursor[z - 1, y, x] = -1  # precursor density is NaN in the fully deposited cells (it was previously set to zero, but later some of the zero cells were added back to semi-surface)
    if precursor[z+1, y, x] == 0: # if the cell that is above the new cell is empty, then add it to the ghost shell collection
        ghosts_bool[z+1, y, x] = True
    # Adding neighbors(in x-y plane) of the new cell to the semi_surface collection
    # and updating ghost shell for every neighbor:
    with suppress(IndexError): # It basically skips operations that occur out of the array
        if precursor[z, y - 1, x] == 0:
            semi_s_bool[z, y - 1, x] = True
            precursor[z, y - 1, x] += 1E-7 # this "marks" cell as a surface one, because some of the checks refer to if the cell is empty. This assignment is essential. It corresponds to the smallest value that float32 can hold and should be changed corrspondingly to the variable type.
            refresh_ghosts(precursor, ghosts_bool, x, y-1, z) # update ghost shell around
        if precursor[z, y + 1, x] == 0:
            semi_s_bool[z, y + 1, x] = True
            precursor[z, y + 1, x] += 1E-7
            refresh_ghosts(precursor, ghosts_bool,  x, y+1, z)
        if precursor[z, y, x - 1] == 0:
            semi_s_bool[z, y, x - 1] = True
            precursor[z, y, x - 1] += 1E-7
            refresh_ghosts(precursor, ghosts_bool, x-1, y, z)
        if precursor[z, y, x + 1] == 0:
            semi_s_bool[z, y, x + 1] = True
            precursor[z, y, x + 1] += 1E-7
            refresh_ghosts(precursor, ghosts_bool, x+1, y, z)

# @jit(nopython=True) # parallel=True)
def refresh_ghosts(precursor, ghosts_bool, x, y, z):
    """
    Updates ghost cells collection around the specified cell

    :param precursor: 3D precursor density array
    :param ghosts_bool: array representing ghost cells
    :param x: x-coordinates of the cell
    :param y: y-coordinates of the cell
    :param z: z-coordinates of the cell
    :return: changes ghosts array
    """
    # It must be kept in mind, that z-coordinate here is absolute, but x and y are relative to the view
    # Firstly, deleting current cell from ghost shell and then adding all neighboring cells(along all axis) if they are zero
    ghosts_bool[z, y, x] = False
    if precursor[z - 1, y, x] == 0:
        ghosts_bool[z - 1, y, x] = True
    if precursor[z + 1, y, x] == 0:
        ghosts_bool[z + 1, y, x] = True
    if precursor[z, y - 1, x] == 0:
        ghosts_bool[z, y - 1, x] = True
    if precursor[z, y + 1, x] == 0:
        ghosts_bool[z, y + 1, x] = True
    if precursor[z, y, x - 1] == 0:
        ghosts_bool[z, y, x - 1] = True
    if precursor[z, y, x + 1] == 0:
        ghosts_bool[z, y, x + 1] = True


# @profile
def precursor_density(flux_matrix, precursor, surface_bool, ghosts_bool, F, n0, tau, sigma, D, dt):
    """
    Recalculates precursor density on the whole surface

    :param flux_matrix: matrix of electron flux distribution
    :param precursor: 3D precursor density array
    :param surface_bool: array representing surface cells
    :param semi_surf_bool: array representing semi-surface cells
    :param ghosts_bool: array representing ghost cells
    :param dt: time step
    :return: changes precursor array
    """
    diffusion_matrix = laplace_term_rolling(substrate, ghosts_bool, D, dt)  # Diffusion term is calculated separately and added in the end
    substrate[surface_bool] += rk4(dt, substrate[surface_bool], F, n0, tau, sigma, flux_matrix[surface_bool]) # An increment is calculated through Runge-Kutta method without the diffusion term
    # substrate[semi_surf_bool] += rk4(dt, substrate[semi_surf_bool])  # same process for semi-cells, but without dissociation term
    substrate+=diffusion_matrix # finally adding diffusion term
    # if substrate[substrate > 1] != 0:
    #     raise Exception("Normalized precursor density is more than 1")


# @jit(nopython=False, parallel=True)
# noinspection PyUnusedLocal
def rk4(dt, sub, F, n0, tau, sigma, flux_matrix=0):
    """
    Calculates increment of precursor density by Runge-Kutta method

    :param dt: time step
    :param sub: array of surface cells
    :param flux_matrix: matrix of electron flux distribution
    :return:
    """
    k1 = precursor_density_increment(dt, sub, flux_matrix, F, n0, tau, sigma,) # this is actually an array of k1 coefficients
    k2 = precursor_density_increment(dt/2, sub, flux_matrix, F, n0, tau, sigma, k1 / 2)
    k3 = precursor_density_increment(dt/2, sub, flux_matrix, F, n0, tau, sigma, k2 / 2)
    k4 = precursor_density_increment(dt, sub, flux_matrix, F, n0, tau, sigma, k3)
    # numexpr: 1 core performs better
    # numexpr.set_num_threads(nn)
    return evaluate_cached(expressions["rk4"], casting='same_kind')


# @jit(nopython=False, parallel=True)
# noinspection PyUnusedLocal
def precursor_density_increment(dt, sub, flux_matrix, F, n0, tau, sigma, addon=0):
    """
    Calculates increment of the precursor density without a diffusion term

    :param dt: time step
    :param sub: array of surface cells (2D for surface cells, 1D dor semi-surface cells)
    :param flux_matrix: matrix of electron flux distribution
    :param addon: Runge Kutta term
    :return: to sub array
    """
    # numexpr: 1 core performs better
    # numexpr.set_num_threads(nn)
    return evaluate_cached(expressions["precursor_density"], local_dict={'F_dt':F*dt, 'F_dt_n0_1_tau_dt': (F*dt*tau+n0*dt)/(tau*n0), 'addon':addon, 'flux_matrix':flux_matrix, 'sigma_dt':sigma*dt, 'sub':sub}, casting='same_kind')


def rk4_diffusion(grid, ghosts_bool, D, dt):
    '''
    Calculates increment of the diffusion term

    :param grid: 3D array
    :param ghosts_bool: array representing ghost cells
    :param D: diffusion coefficient
    :param dt: time step
    :return: 3D array with the increments
    '''
    k1=laplace_term_rolling(grid, ghosts_bool, D, dt)
    k2=laplace_term_rolling(grid, ghosts_bool, D, dt, add=k1/2)
    k3=laplace_term_rolling(grid, ghosts_bool, D, dt, add=k2/2)
    k4=laplace_term_rolling(grid, ghosts_bool, D, dt, add=k3)
    # numexpr.set_num_threads(nn)
    return evaluate_cached(expressions["rk4"], casting='same_kind')


# @jit(nopython=True, parallel=True, forceobj=False)
def laplace_term_rolling(grid, ghosts_bool, D, dt, add = 0, div: int = 0):
    """
    Calculates diffusion term for all surface cells using rolling


    :param grid: 3D precursor density array
    :param ghosts_bool: array representing ghost cells
    :param D: diffusion coefficient
    :param dt: time step
    :param add: Runge-Kutta intermediate member
    :param div:
    :return: to grid array
    """

    # Debugging note: it would be more elegant to just use numpy.roll() on the ghosts_bool to assign neighboring values
    # to ghost cells. But Numpy doesn't retain array structure when utilizing boolean index streaming. It rather extracts all the cells
    # (that correspond to True in our case) and processes them as a flat array. It caused the shifted values for ghost cells to
    # be assigned to the previous(first) layer, which was not processed by numpy.roll() when it rolled backwards.
    # Thus, borders(planes) that are not taking part in rolling(shifting) are cut off by using views to an array
    grid = grid + add
    grid_out = copy(grid)
    grid_out *= -6

    # X axis:
    # No need to have a separate array of values, when whe can conveniently call them from the original data
    shore = grid[:, :, 1:]
    wave = grid[:, :, :-1]
    shore[ghosts_bool[:, :, 1:]] = wave[ghosts_bool[:, :, 1:]] # assigning values to ghost cells forward along X-axis
    # grid_out[:,:, :-1]+=grid[:,:, 1:] #rolling forward (actually backwards)
    roll.rolling_3d(grid_out[:,:,:-1], grid[:,:,1:])
    index = ghosts_bool.reshape(-1).nonzero()
    # grid_out[:,:,-1] += grid[:,:,-1] #taking care of edge values
    roll.rolling_2d(grid_out[:,:,-1], grid[:,:,-1])
    # grid[ghosts_bool] = 0 # flushing ghost cells
    grid.reshape(-1)[index] = 0
    # Doing the same, but in reverse
    shore = grid[:, :, :-1]
    wave = grid[:, :, 1:]
    shore[ghosts_bool[:, :, :-1]] = wave[ghosts_bool[:, :, :-1]]
    # grid_out[:,:,1:] += grid[:,:,:-1] #rolling backwards
    roll.rolling_3d(grid_out[:,:,1:], grid[:,:,:-1])
    # grid_out[:, :, 0] += grid[:, :, 0]
    roll.rolling_2d(grid_out[:, :, 0], grid[:, :, 0])
    # grid[ghosts_bool] = 0
    grid.reshape(-1)[index] = 0

    # Y axis:
    shore = grid[:, 1:, :]
    wave = grid[:, :-1, :]
    shore[ghosts_bool[:, 1:, :]] = wave[ghosts_bool[:, 1:, :]]
    # grid_out[:, :-1, :] += grid[:, 1:, :]
    roll.rolling_3d(grid_out[:, :-1, :], grid[:, 1:, :])
    # grid_out[:, -1, :] += grid[:, -1, :]
    roll.rolling_2d(grid_out[:, -1, :], grid[:, -1, :])
    # grid[ghosts_bool] = 0
    grid.reshape(-1)[index] = 0
    shore = grid[:, :-1, :]
    wave = grid[:, 1:, :]
    shore[ghosts_bool[:, :-1, :]] = wave[ghosts_bool[:, :-1, :]]
    # grid_out[:, 1:, :] += grid[:, :-1, :]
    roll.rolling_3d(grid_out[:, 1:, :], grid[:, :-1, :])
    # grid_out[:, 0, :] += grid[:, 0, :]
    roll.rolling_2d(grid_out[:, 0, :], grid[:, 0, :])
    # grid[ghosts_bool] = 0
    grid.reshape(-1)[index] = 0

    # Z axis:
    shore = grid[1:, :, :]
    wave = grid[:-1, :, :]
    shore[ghosts_bool[1:, :, :]] = wave[ghosts_bool[1:, :, :]]
    # c
    roll.rolling_3d(grid_out[:-1, :, :], grid[1:, :, :])
    # grid_out[-1, :, :] += grid[-1, :, :]
    roll.rolling_2d(grid_out[-1, :, :], grid[-1, :, :])
    # grid[ghosts_bool] = 0
    grid.reshape(-1)[index] = 0
    shore = grid[:-1, :, :]
    wave = grid[1:, :, :]
    shore[ghosts_bool[:-1, :, :]] = wave[ghosts_bool[:-1, :, :]]
    # grid_out[1:, :, :] += grid[:-1, :, :]
    roll.rolling_3d(grid_out[1:, :, :], grid[:-1, :, :])
    # grid_out[0, :, :] += grid[0, :, :]
    roll.rolling_2d(grid_out[0, :, :], grid[0, :, :])
    # grid[ghosts_bool] = 0
    grid.reshape(-1)[index] = 0
    # grid_out[ghosts_bool]=0
    grid_out.reshape(-1)[index] = 0 # result also has to be cleaned as it contains redundant values in ghost cells
    # numexpr: 1 core performs better
    # numexpr.set_num_threads(nn)
    return evaluate_cached(expressions["laplace1"], local_dict={'dt_D': dt*D, 'grid_out':grid_out}, casting='same_kind')
    # else:
    #     return evaluate_cached(expressions["laplace2"], local_dict={'dt_D_div': dt*D/div, 'grid_out':grid_out}, casting='same_kind')


def define_irr_area(beam_matrix):
    indices = np.nonzero(beam_matrix)
    return indices[1].min(), indices[1].max(), indices[2].min(), indices[2].max()

# def path_generator(type, height=1, width=5, length=5)

# @jit(nopython=True, parallel=True, cache=True)
def show_yield(deposit, summ, summ1, res):
    summ1 = np.sum(deposit)
    res = summ1-summ
    return summ, summ1, res


def initialize_framework(from_file=False):
    """
    Prepare simulation framework and parameters from input configuration files
    :param from_file: load structure from vtk file
    :return:
    """
    precursor = yaml.load(open(f'{sys.path[0]}{os.sep}Me3PtCpMe.yml', 'r'),
                          Loader=yaml.Loader)  # Precursor and substrate properties(substrate here is the top layer)
    settings = yaml.load(open(f'{sys.path[0]}{os.sep}Parameters.yml', 'r'),
                         Loader=yaml.Loader)  # Parameters of the beam, dwell time and precursor flux
    sim_params = yaml.load(open(f'{sys.path[0]}{os.sep}Simulation.yml', 'r'),
                           Loader=yaml.Loader)  # Size of the chamber, cell size and time step
    mc_config, equation_values, timings, nr = buffer_constants(precursor, settings, sim_params)
    structure = None
    if from_file:
        try:
            vtk_file = fd.askopenfilename() # '/Users/sandrik1742/Documents/PycharmProjects/FEBID/code/New Folder With Items/35171.372k_of_1000000.0_loops_k_gr4 15/02/42.vtk'#
            vtk_obj = pv.read(vtk_file)
            structure = Structure(vtk_obj=vtk_obj)
        except FileNotFoundError:
            structure = Structure(sim_params['cell_dimension'], sim_params['width'], sim_params['length'],
                                  sim_params['height'], sim_params['substrate_height'], nr)
    else:
        structure = Structure(sim_params['cell_dimension'], sim_params['width'], sim_params['length'],
                              sim_params['height'], sim_params['substrate_height'], nr)
    return structure, mc_config, equation_values, timings

def buffer_constants(precursor: dict, settings: dict, sim_params: dict):
    """
    Calculate necessary constants and prepare parameters for modules

    :param precursor: precursor properties
    :param settings: simulation conditions
    :param sim_params: parameters of the simulation
    :return:
    """
    td = settings["dwell_time"]  # dwell time of a beam, s
    Ie = settings["beam_current"]  # beam current, A
    beam_FWHM = 2.36*settings["gauss_dev"]  # electron beam diameter, nm
    F = settings["precursor_flux"]  # precursor flux at the surface, 1/(nm^2*s)   here assumed a constant, but may be dependent on time and position
    effective_diameter = beam_FWHM * 3.3  # radius of an area which gets 99% of the electron beam
    f = Ie / scpc.elementary_charge / (math.pi * beam_FWHM * beam_FWHM / 4)  # electron flux at the surface, 1/(nm^2*s)
    e = precursor["SE_emission_activation_energy"]
    l = precursor["SE_mean_free_path"]

    # Precursor properties
    sigma = precursor["cross_section"]  # dissociation cross section, nm^2; is averaged from cross sections of all electron types (PE,BSE, SE1, SE2)
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
    dt = sim_params["time_step"]
    cell_dimension = sim_params["cell_dimension"]  # side length of a square cell, nm

    t_flux = 1 / (sigma + f)  # dissociation event time
    diffusion_dt = math.pow(cell_dimension * cell_dimension, 2) / (2 * D * (
            cell_dimension * cell_dimension + cell_dimension * cell_dimension))  # maximum stability

    # Parameters for Monte-Carlo simulation
    mc_config = {'name': precursor["deposit"], 'E0': settings["beam_energy"], 'Emin': settings["minimum_energy"],
              'Z': precursor["average_element_number"],
              'A': precursor["average_element_mol_mass"], 'rho': precursor["average_density"],
              'I0': settings["beam_current"], 'sigma': settings["gauss_dev"],
              'N': Ie * dt / scpc.elementary_charge, 'sub': settings["substrate_element"],
              'cell_dim': sim_params["cell_dimension"],
              'e': precursor["SE_emission_activation_energy"], 'l': precursor["SE_mean_free_path"]}
    # Parameters for reaction-equation solver
    equation_values = {'F': settings["precursor_flux"], 'n0': precursor["max_density"],
                       'sigma': precursor["cross_section"], 'tau': precursor["residence_time"] * 1E-6,
                       'V': precursor["dissociated_volume"], 'D': precursor["diffusion_coefficient"],
                       'dt': sim_params["time_step"]}
    # Stability time steps
    timings = {'t_diff': diffusion_dt, 't_flux': t_flux, 't_desorption': tau, 'dt': dt}

    # effective_radius_relative = math.floor(effective_diameter / cell_dimension / 2)
    return mc_config, equation_values, timings, nr
    
# /The printing loop.
# @jit(nopython=True)
# @profile
def printing(loops=1, dwell_time=1):
    """
    Performs FEBID printing process in a zig-zag manner for given number of times

    :param loops: number of repetitions of the route
    :return: changes deposit and precursor arrays
    """

    structure, mc_config, equation_values, timings = initialize_framework()

    F = equation_values['F']
    n0 = equation_values['n0']
    sigma = equation_values['sigma']
    tau = equation_values['tau']
    V = equation_values['V']
    D = equation_values['D']
    dt = equation_values['dt']
    sub_h = structure.substrate_height

    t = 2E-6 # absolute time, s
    refresh_dt = dt*2 # dt for precursor density recalculation

    beam_matrix = zeros(structure.shape, dtype=int)  # matrix for electron beam flux distribution

    surface_bool = structure.surface_bool
    semi_surface_bool = structure.semi_surface_bool
    ghosts_bool = structure.ghosts_bool
    deposit = structure.deposit
    precursor = structure.precursor

    sim = etraj3d.cache_params(mc_config, deposit, surface_bool)

    y, x = structure.ydim / 2, structure.xdim / 2

    max_z=int(sub_h + np.nonzero(surface_bool)[0].max()+3) # used to track the highest point
    y_start, y_end, x_start, x_end = 0, 0, 0, 0
    irradiated_area_3D = None

    flag = True
    render = vr.Render()
    render.add_3Darray(precursor, structure.cell_dimension, 0.00001, 1, opacity=0.9, nan_opacity=1, scalar_name='Precursor',
                       button_name='precursor', cmap='plasma')
    render.show(interactive_update=True, cam_pos=[(206.34055818793468, 197.6510638707941, 100.47106597548205),
                                                  (0.0, 0.0, 0.0),
                                                  (-0.23307751464125356, -0.236197909312718, 0.9433373838690787)])
    time_step = 15 # s
    time_stamp = 0 # s
    init_cells = np.count_nonzero(deposit[deposit < 0]) # substrate layer
    total_dep_cells = [np.count_nonzero(deposit[deposit < 0]) - init_cells]  # total number of fully deposited cells
    growth_rate = []  # growth rate on each step
    i = 0
    for l in tqdm(range(loops)):  # loop repeats, currently beam travels in a zig-zack manner (array indexing)
        if timeit.default_timer() >= time_stamp:
            i += 1
            # structure.cell_dimension = cell_dimension
            # structure.deposit = deposit
            # structure.precursor = precursor
            # structure.ghosts_bool = ghosts_bool
            # structure.surface_bool = surface_bool
            # vr.save_deposited_structure(structure, f'{l/1000}k_of_{loops/1000}_loops_k_gr16 ')
            time_stamp += time_step
            print(f'Time passed: {timeit.default_timer()}, Av.speed: {l/timeit.default_timer()}')
            total_dep_cells.append(np.count_nonzero(deposit[deposit < 0]) - init_cells)
            growth_rate.append((total_dep_cells[i] - total_dep_cells[i - 1]) / (time_stamp - time_step+0.001) * 60 * 60)
            render.add_3Darray(precursor, structure.cell_dimension, 0.0001, 1, 1, show_edges=True, scalar_name='Precursor',
                       button_name='precursor', cmap='plasma')
            # render.add_3Darray(deposit, structure.cell_dimension, -2, -0.5, 0.7, scalar_name='Deposit',
            #            button_name='Deposit', color='white', show_scalar_bar=False)
            render.p.add_text(f'Time: {str(datetime.timedelta(seconds=int(timeit.default_timer())))} \n'
                            f'Sim. time: {(t):.8f} s \n'
                            f'Speed: {(t/timeit.default_timer()):.8f} \n'  # showing time passed
                            f'Relative growth rate: {int(total_dep_cells[i]/timeit.default_timer()*60*60)} cell/h \n'  # showing average growth rate
                            f'Real growth rate: {int(total_dep_cells[i] / t * 60)} cell/min \n', position='upper_left', font_size=12)  # showing average growth rate
            render.p.add_text(f'Cells: {total_dep_cells[i]} \n' # showing total number of deposited cells
                              f'Height: {max_z*structure.cell_dimension} nm \n', position='upper_right', font_size=12)  # showing current height of the structure

            render.update(100000, force_redraw=True)
        if flag:
            start = timeit.default_timer()
            beam_matrix = etraj3d.rerun_simulation(y, x, deposit, surface_bool, sim, dt)
            print(f'Took total {timeit.default_timer() - start}')
            y_start, y_end, x_start, x_end = define_irr_area(beam_matrix[sub_h:max_z])
            max_z = int(sub_h + np.nonzero(surface_bool)[0].max() + 3)
            flag = False
        y_start, y_end, x_start, x_end = define_irr_area(beam_matrix[sub_h:max_z])
        irradiated_area_3D = s_[sub_h:max_z, y_start:y_end,x_start:x_end]  # a slice of the currently irradiated area
        # irradiated_area_3D_ext = s_[:max_z, y_start - ext:y_end + ext, x_start - ext:x_end + ext]
        # beam_exposure[irradiated_area_3D] += beam_matrix[irradiated_area_3D] # accumulates beam exposure for precursor density if it is called with an interval bigger that dt
        deposition(deposit[irradiated_area_3D],
                   precursor[irradiated_area_3D],
                   beam_matrix[irradiated_area_3D],
                   surface_bool[irradiated_area_3D], sigma*V*dt, 4)  # depositing on a selected area
        flag = update_surface(deposit[irradiated_area_3D],
                               precursor[irradiated_area_3D],
                               surface_bool[irradiated_area_3D],
                               semi_surface_bool[irradiated_area_3D],
                               ghosts_bool[irradiated_area_3D],)  # updating surface on a selected area
        # if t % refresh_dt < 1E-6:
        # TODO: look into DASK for processing arrays by chunks in parallel
        precursor_density(beam_matrix[sub_h:max_z, :, :],
                          precursor[sub_h:max_z, :, :],
                          surface_bool[sub_h:max_z, :, :],
                          ghosts_bool[sub_h:max_z, :, :], F, n0, tau, sigma, D, dt)
            # if l==3:
            #     profiler = line_profiler.LineProfiler()
            #     profiled_func = profiler(precursor_density)
            #     try:
            #         profiled_func(beam_matrix, precursor[:surface.max()+3,:,:], surface, ghosts_index, refresh_dt)
            #     finally:
            #         profiler.print_stats()
            # beam_exposure[:max_z, y_offset-effective_radius_relative:y_limit+effective_radius_relative,x_offset-effective_radius_relative:x_limit+effective_radius_relative] = 0 # flushing accumulated radiation
            # beam_exposure[...] =  0
        t += dt
        # beam_matrix[irradiated_area_3D] = 0
        if flag:
            beam_matrix[...] = 0
        #     except Exception as e:
        #         logging.exception('Caught an Error:')
        # except Exception as e:
        #     e = sys.exc_info()[0]
        #     print("<p>Error: %s</p>" % e)
    a=0
    b=0






def open_params():
    input("Press Enter and specify your Precursor&Substrate properties file:")
    precursor_cfg = fd.askopenfilename()
    input("Press Enter and specify your Technical parameters file:")
    tech_cfg = fd.askopenfilename()
    input("Press Enter and specify your Simulation parameters file:")
    sim_cfg = fd.askopenfilename()
    return precursor_cfg, tech_cfg, sim_cfg


if __name__ == '__main__':
    # precursor_cfg, tech_cfg, sim_cfg = open_params()
    printing(100000)
    # cProfile.runctx('printing(100)',globals(),locals())
    # <editor-fold desc="Plot">
    q=0
