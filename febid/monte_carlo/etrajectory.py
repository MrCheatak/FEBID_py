"""
Primary electron trajectory simulator
"""
# Default packages
import os, sys
import random as rnd
import warnings
from math import *
import multiprocessing

# Core packages
import numpy as np

# Axillary packeges
import pickle
from timeit import default_timer as dt
import traceback as tb

# Local packages
from febid.libraries.ray_traversal import traversal
from febid.monte_carlo.compiled import etrajectory_c
from febid.monte_carlo.mc_base import MC_Sim_Base


class Electron():
    """
    A class representing a single electron with its properties and methods to define its scattering vector.
    """

    def __init__(self, x, y, parent):
        # Python uses dictionaries to represent class attributes, which causes significant memory usage
        # __slots__ attribute forces Python to use a small array for attribute storage, which reduces amount
        # of memory required for every copy of the class.
        # As a result, it reduces multiprocessing overhead as every process obtains a full copy of all objects it works with.
        # However, if __slots__ attribute is declared, declaration of new attributes is only possible by adding a new entry
        # to the __slots__.
        __slots__ = ["E", "x", "y", "z", "point", "point_prev", "x_prev", "y_prev", "z_prev",
                     "cx", "cy", "cz", "direction_c", "ctheta", "stheta", "psi",
                     "cell_dim", "zdim", "ydim", "zdim_abs", "ydim_abs", "xdim_abs"]
        self.E = parent.E
        self.x = x
        self.y = y
        self.z = parent.zdim_abs - 1.0
        self.point = np.zeros(3)
        self.point_prev = np.zeros(3)
        self.x_prev = 0
        self.y_prev = 0
        self.z_prev = 0
        self.cx = 0
        self.cy = 0
        self.cz = 1
        self.direction_c = np.asarray([1.0, 0, 0])
        self.ctheta = 0
        self.stheta = 0
        self.psi = 0
        self.cell_dim = parent.cell_dim
        self.zdim, self.ydim, self.xdim = parent.zdim, parent.ydim, parent.xdim
        self.zdim_abs, self.ydim_abs, self.xdim_abs = parent.zdim_abs, parent.ydim_abs, parent.xdim_abs

    @property
    def coordinates(self):
        """
        Current coordinates (z, y, x)

        :return: tuple
        """
        return (self.z, self.y, self.x)

    @coordinates.setter
    def coordinates(self, value):
        self.x_prev = self.point_prev[2] = self.x
        self.y_prev = self.point_prev[1] = self.y
        self.z_prev = self.point_prev[0] = self.z
        self.z = self.point[0] = value[0]
        self.y = self.point[1] = value[1]
        self.x = self.point[2] = value[2]

    @property
    def coordinates_prev(self):
        """
        Previous coordinates (z, y, x)

        :return: tuple
        """
        return (self.z_prev, self.y_prev, self.x_prev)

    @property
    def direction(self):
        return self.point - self.point_prev

    @property
    def indices(self, z=0, y=0, x=0):
        """
        Gets indices of a cell in an array according to its position in the space

        :return: i(z), j(y), k(x)
        """
        if z == 0: z = self.z
        if y == 0: y = self.y
        if x == 0: x = self.x
        return int(z / self.cell_dim), int(y / self.cell_dim), int(x / self.cell_dim)

    def index_corr(self):
        """
        Corrects indices according to the direction if coordinates are on the cell wall

        :return:
        """
        sign = np.sign(self.point)
        sign[sign == 1] = 0
        delta = self.point % self.cell_dim
        return np.int64((delta == 0) * sign)

    def check_boundaries(self, z=0, y=0, x=0):
        """
        Check if the given (z,y,x) position is inside the simulation chamber.
        If bounds are crossed, return corrected position

        :param z:
        :param y:
        :param x:
        :return:
        """
        if x == 0: x = self.x
        if y == 0: y = self.y
        if z == 0: z = self.z
        flag = True
        if 0 <= x < self.xdim_abs:
            pass
        else:
            flag = False
            if x < 0:
                x = 0.000001
            else:
                x = self.xdim_abs - 0.0000001

        if 0 <= y < self.ydim_abs:
            pass
        else:
            flag = False
            if y < 0:
                y = 0.000001
            else:
                y = self.ydim_abs - 0.000001

        if 0 <= z < self.zdim_abs:
            pass
        else:
            flag = False
            if z < 0:
                z = 0.000001
            else:
                z = self.zdim_abs - 0.000001
        if flag:
            return flag
        else:
            return (z, y, x)

    def __generate_angles(self, a):
        """
        Generates cos and sin of lateral angle and the azimuthal angle

        :param a: alpha at the current step
        :return:
        """
        rnd2 = rnd.random()
        self.ctheta = 1.0 - 2.0 * a * rnd2 / (
                    1.0 + a - rnd2)  # scattering angle cosines , 0 <= angle <= 180˚, it produces an angular distribution that is obtained experimentally (more chance for low angles)
        self.stheta = sqrt(1.0 - self.ctheta * self.ctheta)  # scattering angle sinus
        self.psi = 2.0 * pi * rnd.random()  # azimuthal scattering angle

    def __get_direction(self, ctheta=None, stheta=None, psi=None):
        """
        Calculate cosines of the new direction according
        Special procedure from D.Joy 1995 on Monte Carlo modelling
        :param ctheta:
        :param stheta:
        :param psi:
        :return:
        """
        if ctheta != None:
            self.ctheta, self.stheta, self.psi = ctheta, stheta, psi
        if self.cz == 0.0: self.cz = 0.00001
        self.cz, self.cy, self.cx = traversal.get_direction(self.ctheta, self.stheta, self.psi, self.cz, self.cy,
                                                            self.cx)
        self.direction_c[:] = self.cz, self.cy, self.cx

    def __get_next_point(self, step):
        """
        Calculate coordinates of the next point from previous point and direction cosines.
        Does boundary check.
        :param step: current electron free path
        :return:
        """
        self.x_prev = self.point_prev[2] = self.x
        self.y_prev = self.point_prev[1] = self.y
        self.z_prev = self.point_prev[0] = self.z
        self.x += step * self.cx
        self.y += step * self.cy
        self.z -= step * self.cz
        check = self.check_boundaries()
        if check is True:
            pass
        else:
            self.z, self.y, self.x = check
        self.point[0] = self.z
        self.point[1] = self.y
        self.point[2] = self.x
        if check is True:
            return True
        else:
            return False

    def get_next_point(self, a, step):
        self.__generate_angles(a)
        self.__get_direction()
        return self.__get_next_point(step)

    def get_direction(self, ctheta=None, stheta=None, psi=None):
        return self.__get_direction(ctheta, stheta, psi)


class ETrajectory(MC_Sim_Base):
    """
    A class responsible for the generation and scattering of electron trajectories
    """
    def __init__(self):
        self.passes = [] # keeps the last result of the electron trajectory simulation

        rnd.seed()

        # Beam properties
        self.E0 = 0 # energy of the beam, keV
        self.Emin = 0 # cut-off energy for electrons, keV
        self.sigma = 0 # standard Gaussian deviation
        self.n = 0 # power of the super Gaussian distribution
        self.N = 0 # number of electron trajectories to simulate

        # Solid structure properties
        self.material = None # material in the current point

        self.z0, self.y0, self.x0 = 0, 0, 0
        self.zdim, self.ydim, self.xdim = 0, 0, 0
        self.zdim_abs, self.ydim_abs, self.xdim_abs = 0, 0, 0
        self.ztop = 0

        self.norm_factor = 0 # a ratio of the actual number of electrons emitted to the number of electrons simulated

    def setParameters(self, structure, params, stat=1000):
        """
        Initialise the instance and set all the necessary parameters

        :param structure: solid structure representation
        :param params: contains all input parameters for the simulation
        :param stat: number of simulated trajectories
        """
        #TODO: material tracking can be more universal
        # instead of using deponat and substrate variables, there can be a dictionary, where substrate is always last
        self.E0 = params['E0']
        self.Emin = params['Emin']
        self.I0 = params['I0']
        self.grid = structure.deposit
        self.surface = structure.surface_bool
        self.s_neghib = structure.surface_neighbors_bool
        self.cell_dim = params['cell_dim']
        self.sigma = params['sigma']
        self.n = params.get('n', 1)
        self.N = stat
        self.norm_factor = (self.I0 / self.elementary_charge) / self.N

        self.deponat = params['deponat']
        self.substrate = params['substrate']
        self.substrate.mark = -2

        self.__calculate_attributes()

    def __calculate_attributes(self):
        self.x0, self.y0, self.z0 = self.grid.shape[2] * self.cell_dim / 2, self.grid.shape[1] * self.cell_dim / 2, self.grid.shape[0] * self.cell_dim - 1
        self.zdim, self.ydim, self.xdim = self.grid.shape
        self.zdim_abs, self.ydim_abs, self.xdim_abs = self.zdim * self.cell_dim, self.ydim * self.cell_dim, self.xdim * self.cell_dim
        self.chamber_dim = np.asarray([self.zdim_abs, self.ydim_abs, self.xdim_abs])
        self.ztop = np.nonzero(self.surface)[0].max() + 1  # highest point of the structure

    def get_norm_factor(self, N=None):
        """
        Calculate norming factor with the given number of generated trajectories

        :param N: number of trajectories
        :return:
        """
        if N is None:
            N = self.N
        return self.I0 / N / self.elementary_charge

    def rnd_super_gauss(self, x0, y0, N):
        """
        Generate a specified number of points according to a Super Gaussian distribution.
        Standard deviation and order of the super gaussian are class properties.

        :param x0: mean along X-axis
        :param y0: mean along Y-axis
        :param N: number of points to generate
        :return: two arrays of N-length with x and y positions
        """

        # The function uses rejection method to generate a custom distribution (Super Gauss)
        # with a given probability density function
        # Firstly, probability density is calculated for a number of generated points within the given boundaries.
        # Then, points of the meshgrid are assigned random probability r values (0>r>p.max()).
        # If the r value in the given point is less than p value, the point is saved. Otherwise, it is disgarded.

        # Keeping in mind, that electrons might miss the target, a given number of points is generated first and then
        # checked to be inside the main grid boundaries.

        # Probability density function
        def super_gauss_2d_mod(x, y, st_dev, n):
            return 1 / sqrt(2 * pi) / st_dev * e ** (-0.5 * (((x-x0) ** 2 + (y-y0) ** 2) / (st_dev ** 2*(1+5*(n-1))**0.5+(n-1)**1.5)) ** n)
        def super_gauss_2d(x, y, st_dev, n):
            return 1 / sqrt(2 * pi) / st_dev * e ** (-0.5 * (((x - x0) ** 2 + (y - y0) ** 2) / (st_dev ** 2)) ** n)
        if N == 0: N = self.N
        rnd = np.random.default_rng()
        # Boundaries of the distribution
        bonds = self.sigma * 5
        x_all = np.array([0])
        y_all = np.array([0])
        N = N + 1 # because the first element [0] is discarded regardless
        i = 0
        while x_all.shape[0] <= N:
            # Uniform meshgrid
            x = rnd.uniform(-bonds+x0, bonds+x0, 100)
            y = rnd.uniform(-bonds+y0, bonds+y0, 100)
            # Probability density grid
            p = super_gauss_2d(x, y, self.sigma, self.n)
            # p = super_gauss_2d_mod(x, y, self.sigma, self.n)
            # Assigning random numbers
            rand = rnd.uniform(0, p.max(), x.shape[0])
            # Sieving out points
            choice = (rand < p)
            x = x[choice]
            y = y[choice]
            x_all = np.concatenate((x_all, x))
            y_all = np.concatenate((y_all, y))
            i += 1
            if i > 100000:
                raise OverflowError("Stuck in an endless loop in Gauss distribution creation. \n Terminating.")
        # Discarding excess points
        x_all = x_all[:N]
        y_all = y_all[:N]
        # Discarding points that are out of the main grid
        if x_all.max() > self.xdim_abs or y_all.max() > self.ydim_abs:
            condition = np.logical_and(x_all < self.xdim_abs, y_all < self.ydim_abs)
            x_all = x_all[condition]
            y_all = y_all[condition]
        if x_all.min() <= 0 or y_all.min() <= 0:
            condition = np.logical_and(x_all > 0, y_all > 0)
            x_all = x_all[condition]
            y_all = y_all[condition]
        return x_all, y_all

    def rnd_gauss_xy(self, x0, y0, N):
        """
        Generate a specified number of points according to a Gaussian distribution.
        Standard deviation and order of the super gaussian are class properties.

        :param x0: mean along X-axis
        :param y0: mean along Y-axis
        :param N: number of points to generate
        :return: two arrays of N-length with x and y positions
        """
        if N==0: N=self.N
        i=0
        rnd = np.random.default_rng()
        while True:
            x = rnd.normal(0, self.sigma, N) + x0
            y = rnd.normal(0, self.sigma, N) + y0
            try:
                if x.max()>self.xdim_abs or y.max()>self.ydim_abs:
                    condition = np.logical_and(x<self.xdim_abs, y<self.ydim_abs)
                    x = x[condition]
                    y = y[condition]
                if x.min() <= 0 or y.min() <= 0:
                    condition = np.logical_and(x > 0, y > 0)
                    x = x[condition]
                    y = y[condition]
            except:
                i += 1
                if i > 10000:
                    raise OverflowError("Stuck in an endless loop in Gauss distribution creation. \n Terminating.")
                continue
            if x.size > 0 and y.size > 0:
                if x.shape[0] != y.shape[0]:
                    print(f'x and y shape mismatch in \'rnd_gauss_xy\'')
                    raise ValueError
                return x, y
            i += 1
            if i > 10000:
                raise OverflowError("Stuck in an endless loop in Gauss distribution creation. \n Terminating.")

    def map_wrapper(self, y0, x0, N=0):
        """
        Create normally distributed electron positions and run trajectory mapping

        :param y0: y-position of the beam, nm
        :param x0: x-position of the beam, nm
        :param N: number of electrons to create
        :return:
        """
        if N == 0:
            N = self.N
        x0, y0 = self.rnd_super_gauss(x0, y0, N)  # generate gauss-distributed beam positions
        print('Running \'map trajectory\'...', end='')
        start = dt()
        self.passes = self.map_trajectory(x0, y0)
        print(f'finished. \t {dt() - start}')
        if not len(self.passes) > 0:
            raise ValueError('Zero trajectories generated!')
        return self.passes

    def map_wrapper_cy(self, y0, x0, N=0):
        """
        Create normally distributed electron positions and run trajectory mapping in Cython

        :param y0: y-position of the beam, nm
        :param x0: x-position of the beam, nm
        :param N: number of electrons to create
        :return:
        """
        if N == 0:
            N = self.N
        x0, y0 = self.rnd_super_gauss(x0, y0, N) # generate gauss-distributed beam positions
        print('Running \'map trajectory\'...', end='')
        start = dt()
        try:
            self.passes = etrajectory_c.start_sim(self.E0, self.Emin, y0, x0, self.cell_dim, self.grid, self.surface.view(dtype=np.uint8), [self.substrate, self.deponat])
        except Exception as e:
            raise RuntimeError(f'An error occurred while generating trajectories: {e.args}')
        print(f'finished. \t {dt() - start}')
        if not len(self.passes) > 0:
            raise ValueError('Zero trajectories generated!')
        return self.passes

    def map_trajectory(self, x0, y0):
        """
        Simulate trajectory of the electrons with a specified starting position.

        :param x0: x-positions of the electrons
        :param y0: y-positions of the electrons
        :return:
        """
        self.passes = []
        self.ztop = np.nonzero(self.surface)[0].max()
        count = -1
        print('\nStarting PE trajectories mapping...')
        for x,y in zip(x0,y0):
            count += 1
            # print(f'{count}', end=' ')
            flag = False
            self.E = self.E0  # getting initial beam energy
            trajectories = []  # trajectory will be a sequence of points
            energies = []
            mask = []
            # Due to memory race problem, all the variables that are changing(coordinates, energy) have been moved to a separate class,
            # that gets instanced for every trajectory and thus for every process
            coords = Electron(x, y, self)
            # Getting initial point. It is set to the top of the chamber
            trajectories.append(coords.coordinates)  # every time starting from the beam origin (a bit above the highest point of the structure)
            energies.append(self.E0)  # and with the beam energy
            coords.coordinates = coords.coordinates
            # To get started, we need to know what kind of cell(material) we're in
            i, j, k = coords.indices
            if self.grid[i, j, k] > -1:  # if current cell is not deposit or substrate, electron flies straight to the surface
                coords.point_prev[0] = coords.z_prev = coords.z
                # Finding the highest solid cell
                coords.point[0] = coords.z = np.amax((self.grid[:, j, k]<0).nonzero()[0], initial=0)*self.cell_dim + self.cell_dim # addition here is required to position at the top of the incident solid cell
                trajectories.append(coords.coordinates)  # saving current point
                energies.append(coords.E)
                mask.append(0)
                # Finish trajectory if electron does not meet any solid cell
                if coords.z == self.cell_dim: # if there are no solid cells along the trajectory, we same bottom point and move on to the next trajectory
                    self.passes.append((trajectories, energies, mask))
                    continue
                self.material = self.substrate

            # Generating trajectory
            try:
                while coords.E > self.Emin:  # going on with every electron until energy is depleted or escaping chamber
                    # Getting Alpha value and electron path length
                    a, step = self._get_alpha_and_step(coords) # actual distance an electron travels
                    # Next coordinates:
                    # First thing to do: boundary check
                    check = coords.get_next_point(a, step)
                    # Trimming new segment and heading for finishing the trajectory
                    if check is False:
                        flag = True
                        step = traversal.det_1d(coords.direction) # recalculating step length
                    # What cell are we in now?
                    i,j,k = coords.indices #
                    # If its a solid cell, get energy loss and continue
                    if self.grid[i,j,k] < 0:
                        coords.E += traversal.get_Eloss(coords.E, self.material.Z, self.material.rho, self.material.A, self.material.J, step)
                        trajectories.append(coords.coordinates)  # saving current point
                        energies.append(coords.E)
                        mask.append(1)
                        # Getting material in the new point
                        if self.grid[i,j,k] != self.material.mark:
                            if self.grid[i, j, k] == -2:  # current cell is substrate
                                self.material = self.substrate
                            if self.grid[i, j, k] == -1:  # current cell is deponat
                                self.material = self.deponat
                    # If next cell is empty(non solid),
                    # first find the intersection with the surface
                    # then either with walls or next solid cell
                    else:
                        flag, crossing, crossing1 = self.get_next_crossing(coords)
                        coords.coordinates = crossing
                        coords.E += self._getELoss(coords) * traversal.det_1d(coords.point-coords.point_prev)
                        trajectories.append(coords.coordinates)  # saving current point
                        energies.append(coords.E)  # saving electron energy at this point
                        if flag == 2: # if electron escapes chamber without crossing surface (should not happen ideally!)
                            mask.append(0)
                        if flag < 2: # if electron crossed the surface
                            mask.append(1)
                            coords.coordinates = crossing1
                            trajectories.append(coords.coordinates)  # saving current point
                            energies.append(coords.E)  # saving electron energy at this point
                            mask.append(0)
                    # <editor-fold desc="Accurate tracking of material">
                    ############ Enable this snippet instead of the code after getting index,
                    ############ if interfaces between solid materials have to be tracked precisely.
                    ############ It only misses the 'get_crossing_point' function implementation,
                    ############ that looks for the next cell with different material.
                    #                 if self.grid[i,j,k] == self.material.mark:
                    #                     coords.E += self._getELoss(coords) * step
                    #                 elif self.grid[i,j,k] != self.material.mark and self.grid[i,j,k] < 0:
                    #                     coords.coordinates = self.get_crossing_point(coords, self.material.mark)
                    #                     coords.E += self._getELoss(coords) * coords.corr_step()
                    #                     # Determining material of the current voxel
                    #                     if self.grid[i, j, k] == -2:  # current cell is substrate
                    #                         self.material = self.substrate
                    #                     if self.grid[i, j, k] == -1:  # current cell is deponat
                    #                         self.material = self.deponat
                    #                     flag = False
                    #                     trajectories.append(coords.coordinates)  # saving current point
                    #                     energies.append(coords.E)  # saving electron energy at this point
                    #                     mask.append(1)
                    #                 else:  # checking if the cell is void
                    #                     flag, crossing = self.get_next_crossing(coords)
                    #                     coords.coordinates = crossing
                    #                     trajectories.append(coords.coordinates)  # saving current point
                    #                     energies.append(coords.E)  # saving electron energy at this point
                    #                     mask.append(0)
                    # </editor-fold>

                    if flag > 0: # finishing trajectory mapping if electron is beyond chamber walls
                        flag = False
                        break
                self.passes.append((trajectories, energies, mask))  # collecting mapped trajectories and energies
            except Exception as e:
                print(e.args)
                tb.print_exc()
                print(f'Current electron properties: ', end='\n\t')
                print(*vars(coords).items(), sep='\n\t')
                print(f'Generated trajectory:', end='\n\t')
                print(*zip(trajectories, energies,mask), sep='\n\t')
                try:
                    print(f'Last indices: {i, j, k}')
                except:
                    pass
        print('\nFinished PE sim')
        return self.passes

    def map_trajectory_verbose(self, x0, y0):
        """
        Simulate trajectory of the electrons with a specified starting position.
        Version with step-by-step output to console.

        :param x0: x-positions of the electrons
        :param y0: y-positions of the electrons
        :return:
        """
        print('Starting PE simulation...')
        self.passes = []
        self.ztop = np.nonzero(self.surface)[0].max()
        i = -1
        for x, y in zip(x0, y0):
            i +=1
            print(f'Trajectory {i}, coordinates: {x, y}')
            flag = False
            self.E = self.E0  # getting initial beam energy
            trajectories = []  # trajectory will be a sequence of points
            energies = []
            mask = []
            # Due to memory race problem, all the variables that are changing(coordinates, energy) have been moved to a separate class,
            # that gets instanced for every trajectory and thus for every process
            coords = Electron(x, y, self)
            # Getting initial point. It is set to the top of the chamber
            print(f'Recording first coordinates: {coords.coordinates}')
            trajectories.append(coords.coordinates)  # every time starting from the beam origin (a bit above the highest point of the structure)
            print(f'Recording first energy: {self.E0}')
            energies.append(self.E0)  # and with the beam energy
            coords.coordinates = coords.coordinates
            # To get started, we need to know what kind of cell(material) we're in
            i, j, k = coords.indices
            print(f'Initial indices: {i, j, k}')
            if self.grid[i, j, k] > -1:  # if current cell is not deposit or substrate, electron flies straight to the surface
                print(f'Current cell is void: {self.grid[i,j,k]}. Flying through...')
                print(f'Setting z-coordinate as previous one: {coords.z} <— {coords.z_prev}')
                coords.point_prev[0] = coords.z_prev = coords.z
                # Finding the highest solid cell
                coords.point[0] = coords.z = np.amax((self.grid[:, j, k] < 0).nonzero()[0],
                                                     initial=0) * self.cell_dim + self.cell_dim  # addition here is required to position at the top of the incident solid cell
                print(f'Got new z coordinate at the surface: {coords.z}')
                print(f'Recording coordinates: {coords.coordinates}')
                trajectories.append(coords.coordinates)  # saving current point
                print(f'Recording energy: {coords.E}')
                energies.append(coords.E)
                print(f'Recording mask: 0')
                mask.append(0)
                # Finish trajectory if electron does not meet any solid cell
                if coords.z == self.cell_dim:  # if there are no solid cells along the trajectory, we same bottom point and move on to the next trajectory
                    print(f'Electron did not hit any surface, terminating trajectory.')
                    self.passes.append((trajectories, energies, mask))
                    continue
                print(f'Setting current material as substrate: {self.substrate.name}')
                self.material = self.substrate

            # Generating trajectory
            try:
                print(f'Starting scattering loop...')
                while coords.E > self.Emin:  # going on with every electron until energy is depleted or escaping chamber
                    print(f'Current coordinates and energy: {coords.coordinates, coords.E}')
                    # a = self._getAlpha(coords)
                    # step = -self._getLambda_el(coords, a) * log( rnd.uniform(0.00001, 1))  # actual distance an electron travels
                    # Getting Alpha value and electron path length
                    a, step = self._get_alpha_and_step(coords)
                    print(f'Got alpha and step: {a, step}')
                    # Next coordinates:
                    check = coords.get_next_point(a, step)
                    print(f'Got new direction and coordinates: {coords.direction, coords.coordinates} <— {coords.coordinates_prev}')
                    # First thing to do: boundary check
                    # check = coords.check_boundaries()
                    # Trimming new segment and heading for finishing the trajectory
                    if check is False:
                        flag = True
                        step = traversal.det_1d(coords.direction)  # recalculating step length
                        print(f'Recalculating step after correction: {step}')
                    # What cell are we in now?
                    i, j, k = coords.indices  # + coords.index_corr()
                    print(f'Getting current index and cell type: {i, j, k} cell value {self.grid[i, j, k]}')
                    # If its a solid cell, get energy loss and continue
                    if self.grid[i, j, k] < 0:
                        print(f'Cell is solid...')
                        # coords.E += self._getELoss(coords) * step
                        coords.E += traversal.get_Eloss(coords.E, self.material.Z, self.material.rho, self.material.A,
                                                        self.material.J, step)
                        print(f'Getting new energy with losses: {coords.E}')
                        trajectories.append(coords.coordinates)  # saving current point
                        energies.append(coords.E)
                        mask.append(1)
                        print(f'Recording coordinates, energy and mask: {coords.coordinates, coords.E} 1')
                        # Getting material in the new point
                        if self.grid[i, j, k] != self.material.mark:
                            print(f'Material in the current cell is different...', end='')
                            if self.grid[i, j, k] == -2:  # current cell is substrate
                                print(f'its substrate.')
                                self.material = self.substrate
                            if self.grid[i, j, k] == -1:  # current cell is deponat
                                print(f'its deponat.')
                                self.material = self.deponat
                    # If next cell is empty(non solid),
                    # first find the intersection with the surface
                    # then either with walls or next solid cell
                    else:
                        print(f'Cell is not solid...flying through....')
                        flag, crossing, crossing1 = self.get_next_crossing(coords)
                        if flag == 0:
                            print(f'Electron has crossed solid/surface in {crossing} and then hit solid in {crossing1}')
                        if flag == 1:
                            print(f'Electron has crossed solid/surface in {crossing} and then exited the volume in {crossing1}')
                        if flag == 2:
                            print(f'Warning! Electron has NOT crossed solid/surface and exited the volume in {crossing1}.')
                        print(f'Setting next scattering point: {crossing} <— {coords.coordinates} <— {coords.coordinates_prev}')
                        coords.coordinates = crossing
                        coords.E += self._getELoss(coords) * traversal.det_1d(coords.point - coords.point_prev)
                        print(f'Getting new energy with losses: {coords.E}')
                        trajectories.append(coords.coordinates)  # saving current point
                        energies.append(coords.E)  # saving electron energy at this point
                        print(f'Recording coordinates, energy and mask: {coords.coordinates, coords.E} ', end='')
                        if flag == 2:  # if electron escapes chamber without crossing surface (should not happen ideally!)
                            print(f'0')
                            mask.append(0)
                        if flag < 2:  # if electron crossed the surface
                            print(f'1')
                            mask.append(1)
                            print(f'Setting next scattering point: {crossing1} <— {coords.coordinates} <— {coords.coordinates_prev}')
                            coords.coordinates = crossing1
                            trajectories.append(coords.coordinates)  # saving current point
                            energies.append(coords.E)  # saving electron energy at this point
                            mask.append(0)
                            print(f'Recording coordinates, energy and mask: {coords.coordinates, coords.E, 0} ')
                            # if flag == 1: # if electron escapes after crossing surface and traversing void
                            #     mask.append(0)
                            # else: # if electron meets a solid cell
                            #     mask.append(1)

                    # <editor-fold desc="Accurate tracking of material">
                    ############ Enable this snippet instead of the code after getting index,
                    ############ if interfaces between solid materials have to be tracked precisely.
                    ############ It only misses the 'get_crossing_point' function implementation,
                    ############ that looks for the next cell with different material.
                    #                 if self.grid[i,j,k] == self.material.mark:
                    #                     coords.E += self._getELoss(coords) * step
                    #                 elif self.grid[i,j,k] != self.material.mark and self.grid[i,j,k] < 0:
                    #                     coords.coordinates = self.get_crossing_point(coords, self.material.mark)
                    #                     coords.E += self._getELoss(coords) * coords.corr_step()
                    #                     # Determining material of the current voxel
                    #                     if self.grid[i, j, k] == -2:  # current cell is substrate
                    #                         self.material = self.substrate
                    #                     if self.grid[i, j, k] == -1:  # current cell is deponat
                    #                         self.material = self.deponat
                    #                     flag = False
                    #                     trajectories.append(coords.coordinates)  # saving current point
                    #                     energies.append(coords.E)  # saving electron energy at this point
                    #                     mask.append(1)
                    #                 else:  # checking if the cell is void
                    #                     flag, crossing = self.get_next_crossing(coords)
                    #                     coords.coordinates = crossing
                    #                     trajectories.append(coords.coordinates)  # saving current point
                    #                     energies.append(coords.E)  # saving electron energy at this point
                    #                     mask.append(0)
                    # </editor-fold>

                    if flag > 0:  # finishing trajectory mapping if electron is beyond chamber walls
                        print(f'Finishing trajectory.')
                        flag = False
                        break
                self.passes.append((trajectories, energies, mask))  # collecting mapped trajectories and energies
            except Exception:
                tb.print_exc()
                print(f'Current electron properties: ', end='\n\t')
                print(*vars(coords).items(), sep='\n\t')
                print(f'Generated trajectory:', end='\n\t')
                print(*zip(trajectories, energies, mask), sep='\n\t')
                try:
                    print(f'Last indices: {i, j, k}')
                except:
                    pass
            i += 1
        return self.passes

    def get_crossing_point(self, coords, curr_material):
        # STUB: this function is a part of precise solid material tracking
        #  Misses corresponding Cython function!
        p0 = np.asarray(coords.coordinates_prev)
        pn = np.asarray(coords.coordinates)
        direction = pn - p0  #
        sign = np.int8(np.sign(direction))
        step = np.sign(direction) * self.cell_dim
        step_t = step / direction
        delta = -(p0 % self.cell_dim)
        t = np.abs((delta + (step == self.cell_dim) * self.cell_dim + (delta == 0) * step) / direction)
        sign[sign == 1] = 0
        crossing = np.empty(3)
        traversal.get_surface_crossing(self.surface.view(dtype=np.uint8), self.cell_dim, p0, direction, t, step_t, sign, curr_material, crossing)
        if not crossing.any():
            return pn
        return crossing

    def get_next_crossing(self, coords):
        """
        Get next two crossing points and a flag showing if volume boundaries are met

        :param coords:
        :return:
        """

        # Finding where electron would exit chamber along current direction
        p0 = coords.point_prev
        direction = coords.direction_c
        direction[0] *= -1 # main MC algorithm runs with inverse z-component, but here we need the direct one
        sign = np.int8(np.sign(direction))
        t = np.abs((-p0 + (sign == 1) * self.chamber_dim)/direction)
        pn = p0 + np.min(t) * direction
        pn[pn>=self.chamber_dim] = self.chamber_dim[pn>=self.chamber_dim]-0.000001 # reducing coordinate if it is beyond the boundary
        pn[pn<=0] = 0.000001
        direction[0] *= -1 # recovering original sign

        direction = pn - p0
        direction[direction==0] = rnd.choice((0.000001, -0.000001)) # sometimes next point is so close to the previous, that their subtraction evaluates to 0
        step = sign * self.cell_dim
        step_t = step / direction
        delta = -(p0 % self.cell_dim)
        t = np.abs((delta + (sign > 0) * self.cell_dim + (delta == 0) * step) / direction)
        sign[sign == 1] = 0
        crossing = np.ones(3)
        crossing1 = np.ones(3)
        flag = traversal.get_surface_solid_crossing(self.surface.view(dtype=np.uint8), self.grid, self.cell_dim, p0, pn, direction, t, step_t, sign, crossing, crossing1)
        if flag != 0:
            if flag == 2:
                crossing = pn # ideally, this line should not execute at all, because there is always a surface layer between solid and void
            if flag == 1:
                crossing1 = pn
        return flag, crossing, crossing1

    def save_passes(self, fname, type):
        """
        Save passes to a text file or by pickling

        :param fname: name of the file
        :param type: saving type: accepts 'pickle' or 'text'
        :return:
        """

        if type not in ['pickle', 'text']:
            raise Warning(f"No type option named {type}. Type argument accepts either \'text\' or \'pickle\'")
        try:
            if type == 'pickle':
                self.__save_passes_obj(fname)
            if type == 'text':
                self.__save_passes_text(fname)
        except Exception as e:
            print(f'Something went wrong, was unable to write the file. Error raised:{e.args}')
            return

    def __save_passes_text(self, fname):
        """
        Save electron trajectories to individual text files.
        Mask is an axillary value 0 or 1, that helps to filter segments, that are outside of the solid material
        Formatting:
        x/nm  y/nm   z/nm   E/keV   mask
        x1     y1     z1     E1      mask1
        x2     y2     z2     E2      mask2
        ...

        :param fname: name of the file
        :return:
        """
        i = 0
        print('Dumping trajectories as a text file ...', end='')
        cwd = os.getcwd()
        fname = os.path.join(cwd, fname+'.txt')
        with open(fname, mode='w') as f:
            f.write('# List of generated electron trajectories' + '\n')
            for p in self.passes:
                f.write(f'# Trajectory {i} \n')
                points = np.asarray(p[0])
                energies = np.asarray(p[1])
                mask = np.asarray(p[2])
                mask = np.insert(mask ,0, nan)

                f.write('# x/nm\t\t y/nm\t\t z/nm\t\t E/keV\t\t mask\n')
                for pt, e, m in zip(points, energies, mask):
                    f.write('{:f}\t'.format(pt[0]) + '{:f}\t'.format(pt[1]) +
                      '{:f}\t'.format(pt[2]) + '{:f}\t'.format(e) + str(m) + '\n')
                i += 1
        print('done!')

    def __save_passes_obj(self, fname):
        """
        Pickle electron trajectories.

        :param fname: name of the file
        :return:
        """
        cwd = os.getcwd()
        fname = os.path.join(cwd, fname)
        with open(fname, mode='w') as f:
            pickle.dump(self.passes, f)

    def __inspect_passes(self, short=True, exclude_first = False):
        """
        Inspect generated trajectories for duplicate points, points with identical values in at least on
        of the components, negative components, negative energies and unmarked segments that are significantly
        longer that the actual electron free path.

        :param short: exclude empty reports
        :param exclude_first: the first point is always at the top of the simulation volume and always projected down to the surface
        :return:
        """
        def print_results(message, index, color_code = 30, *args):
            if short:
                if index.shape[0] == 0: return
                print(f'\033[0;{color_code};48m')
            print(f' {message}')
            print(f'T    S  \t\t\t\t p0 \t\t\t\t\t\t\t\t\t pn') # trajectory and segment number
            str = ''
            for j in range(len(args[0])):
                str = f'{i}  {index[j]}\t'
                for arg in args:
                    str += f'{arg}\t'
                print(str+'\n')
            if str == '':
                print(f'None\n')
        for i, traj in enumerate(self.passes):
            points = np.asarray(traj[0])
            energies = np.asarray(traj[1])
            mask = np.asarray(traj[2])
            if points.shape[0] != energies.shape[0] or points.shape[0] - mask.shape[0] != 1:
                print(f'Incorrect number of points/energies/masks: '
                      f'Points: {points.shape[0]}'
                      f'Energies: {energies.shape[0]}'
                      f'Masks: {mask.shape[0]}')
            if exclude_first:
                points = points[1:]
                energies =energies[1:]
                mask = mask[1:]
            p0 = points[:-1]
            pn = points[1:]
            direction = pn-p0
            L = np.linalg.norm(direction, axis=1)
            zero_dir = np.nonzero(direction==0)[0] # segments with 0 in at least one component
            zero_dir = np.unique(zero_dir)
            zero_length = np.nonzero(L==0)[0] # segments with zero length
            negative_E = np.nonzero(energies<0)[0] # points with negative energies
            negative_positions = np.nonzero(points<0)[0] # points with at least on negative component
            _, max_step = traversal.get_alpha_and_lambda(self.E0, self.deponat.Z, self.deponat.rho, self.deponat.A)
            max_step *= -log(0.00001) # longest step possible in the deposited material
            long_segments = (L > max_step) # segments, that are longer than the maximum step length
            long_unmasked = np.logical_and(long_segments, mask!=0) # the latter, but also unmasked
            long_segments = long_segments.nonzero()[0]
            long_unmasked = long_unmasked.nonzero()[0]

            message = 'Points with zero difference in at least one component:'
            print_results(message, zero_dir, 36, p0[zero_dir], pn[zero_dir])
            message = 'Points with zero difference (zero length):'
            print_results(message, zero_length, 31, p0[zero_length], pn[zero_length])
            message = 'Negative coordinates:'
            print_results(message, negative_positions, 31, points[negative_positions])
            message = 'Segments longer than maximum step:'
            print_results(message, long_segments, 33, p0[long_segments], pn[long_segments], L[long_segments])
            message = 'The latter unmasked:'
            print_results(message, long_unmasked, 31, p0[long_unmasked], pn[long_unmasked], L[long_unmasked])
            message = 'Points with negative energies:'
            print_results(message, negative_E, 31, p0[negative_E], pn[negative_E], L[negative_E])

            print('\033[0;30;48m ', end='')
            a=0

    def plot_distribution(self, x, y, func=None):
        """
        Plot a scatter plot of the (x,y) points with 2D histograms depicting axial distribution

        :param x: array of x-coordinates
        :param y: array of y-coordinates
        :param func: 2D probability density function
        :return:
        """
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        # global fig, scatter, x_hist, y_hist, bins
        print('Preparing plot:')
        left, width = 0.1, 0.6
        bottom, height = 0.1, 0.6
        spacing = 0.005

        print('Creating canvas and axes...')
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        # start with a square Figure
        fig = plt.figure(figsize=(9, 9))

        ax: Axes = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)

        # the scatter plot:
        print('Drawing scatter plot...')
        scatter = ax.scatter(x, y, s=1, linewidths=0, alpha=0.7, label=f'{x.shape[0]} points \n'
                                                                       f'{self.sigma} st_dev \n'
                                                                       f'{self.n} order')

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # now determine nice limits by hand:
        n_bins = 200

        # making histograms
        print('Drawing histograms...')
        binwidth_x = (x.max() - x.min()) / 100
        xmax = np.max(np.abs(x))
        lim_x = (int(xmax / binwidth_x) + 1) * binwidth_x

        center_x = x.mean()
        margin = lim_x - center_x
        bins_x = np.arange(center_x - margin, center_x + margin + binwidth_x, binwidth_x)

        binwidth_y = (y.max() - y.min()) / 100
        ymax = np.max(np.abs(y))
        lim_y = (int(ymax / binwidth_y) + 1) * binwidth_y

        center_y = y.mean()
        margin = lim_y - center_y
        bins_y = np.arange(center_y - margin, center_y + margin + binwidth_y, binwidth_y)

        nx, _ = np.histogram(x, bins_x, density=True)
        ny, _ = np.histogram(y, bins_y, density=True)

        x_p = x[np.fabs(y - center_y) < (y.max() - center_y) * 0.1]
        y_p = y[np.fabs(x - center_x) < (x.max() - center_x) * 0.1]
        _, _, x_hist = ax_histx.hist(x_p, bins=bins_x, density=True)
        _, _, y_hist = ax_histy.hist(y_p, bins=bins_y, density=True, orientation='horizontal')
        if func is not None:
            x_p = np.arange(x.min(), x.max(), 0.01)
            p_x = func(x_p, center_y, self.sigma, self.n)
            ax_histx.plot(x_p, p_x, lw=3, label="Probability density distr.")
            y_p = np.arange(y.min(), y.max(), 0.01)
            p_y = func(center_x, y_p, self.sigma, self.n)
            ax_histy.plot(p_y, y_p, lw=3)
            ax.set_title(f"Super Gauss distribution, σ={self.sigma} n={self.n}", pad=170, fontdict={'fontsize': 18})
            ax_histx.legend(loc='upper left')

        print('Setting sliders...')
        ax_histx.set_title('x distribution')
        ax_histy.set_title('y distribution')
        ax.legend()
        ax.grid(True, 'both')

        # axstdev = plt.axes([0.25, 0.01, 0.65, 0.03])
        # axnorder = plt.axes([0.25, 0.05, 0.65, 0.03])

        # s_stdev = Slider(axstdev, 'Standard deviation', 0.1, 30.0, valinit=3, valstep=0.1)
        # s_order = Slider(axnorder, 'Function order', 0.1, 20.0, valinit=1, valstep=1)

        # s_stdev.on_changed(update_stdev)
        # s_order.on_changed(update_order)

        print('Plotting...')
        return plt.show()

    # Equations used in trajectory calculation
    def _get_alpha_and_step(self, coords):
        """
        Calculate alpha value and electron path from the electron energy and material properties

        :param coords:
        :return:
        """
        a = self._getAlpha(coords, self.material)
        l = self._getLambda_el(coords, a, self.material)
        # NOTE: excluding unity to prevent segments with zero length (log(1)=0).
        #  Not only they are practically useless, but also cause bugs due to division by zero
        step = l * -log(rnd.uniform(0.00001, 0.9999))
        return a, step

    def _getJ(self, Z=0):
        """
        Mean ionization potential, keV
        Represents the effective average energy loss per interaction between the incident electron and the solid.
        This parameter incorporates into its value all possible mechanisms for energy loss that an electron can encounter (X-ray, Auger, phonon, SE)
        Z – atomic number of the target material
        :return:
        """
        if Z==0: Z=self.Z
        return (9.76*Z + 58.5/Z**0.19)*1.0E-3 # +

    def _getAlpha(self, coords, material=nan, E=0):
        """
        Screening factor, that accounts for the fact that the incident electron
        does not see all of the charge on the nucleus because of the cloud of orbiting electrons
        E – electron energy, keV
        Z – atomic number of the target material
        :return:
        """
        if E == 0:
            E = coords.E
        return 3.4E-3*self.material.Z**0.67/E

    def _getSigma(self, coords, a, material=nan): # in nm^2/atom
        """
        Calculates Elastic cross section (by Rutherford)
        E – electron energy, keV
        Z – atomic number of the target material
        a – screening factor
        :return:
        """
        # a = self._getAlpha()
        return 5.21E-7*self.material.Z**2/coords.E**2*4.0*pi/(a*(1.0+a))*((coords.E+511.0)/(coords.E+1022.0))**2

    def _getLambda_el(self, coords,  a, material=nan): # in nm
        """
        Mean free path of an electron
        A – atomic weight, g/mole
        Na – Avogadro number
        rho – target material density, g/cm^3
        sigma – elastic cross section
        :return:
        """
        # sigma = self._getSigma()
        return self.material.A/(self.NA*self.material.rho*1.0E-21*self._getSigma(coords, a))

    def _getLambda_in(self, material=nan, Emin=0.1):
        mfp1, mfp2, mfp3 = 0,0,0
        FSE = False # enable tracking of secondary electrons
        if self.E > Emin:
            mfp1 = self._getLambda_el()# elastic MFP in Å
            mfp2 = self.material.A*self.E**2*2.55/(self.material.rho*self.material.Z) # inelastic MFP in Å
            mfp3 = (mfp1*mfp2)/(mfp1+mfp2) # total MFP

            if FSE:
                self.PEL = mfp3/mfp1
            else:
                self.PEL = 1 # eliminating probability of SE generation
                mfp3 = mfp1 # considering only elastic MFP

        return mfp3

    def _getELoss(self, coords, material=nan, E=0): # in keV/nm
        """
        Energy loss rate per distance traveled due to inelastic events dE/dS (stopping power)
        Applicable down to PE energies of 100 eV
        rho – target material density, g/cm^3
        Z – atomic number of the target material
        A – atomic weight, g/mole
        E – electron energy, keV
        J – mean ionisation potential, keV
        :return:
        """
        if E == 0:
            E = coords.E
        return -7.85E-3*self.material.rho*self.material.Z/(self.material.A*E)*log(1.166*(E/self.material.J + 0.85))


if __name__ == "__main__":
    print("Current script does not have an entry point.....")
    input('Press Enter to exit.')

