#######################################################
#                                                     #
#       Primary electron trajectory simulator         #
#                                                     #
#######################################################
import math
import multiprocessing
import os
import random as rnd
from math import *

import line_profiler
import matplotlib.pyplot as plt
import numpy as np

from modified_libraries.ray_traversal import traversal


class ETrajectory(object):
    def __init__(self, name='noname'):
        self.name = name
        self.passes = []
        self.NA = 6.022141E23 # Avogadro number
        rnd.seed()

    class Electron():
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
            self.z = parent.zdim_abs-1.0
            self.point = np.zeros(3)
            self.point_prev = np.zeros(3)
            self.x_prev = 0
            self.y_prev = 0
            self.z_prev = 0
            self.cx = 0
            self.cy = 0
            self.cz = 1
            self.direction_c = np.asarray([1.0,0,0])
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
            sign = np.sign(self.direction)
            sign[sign==1] = 0
            delta = self.point%self.cell_dim
            return np.int64((delta==0)*sign)

        def check_boundaries(self, z=0, y=0, x=0):
            """
            Checks is the given (z,y,x) position is inside the simulation chamber

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
                return (z,y,x)

        def __generate_angles(self, a):
            """
            Generates cos and sin of lateral angle and the azimuthal angle

            :param a: alpha at the current step
            :return:
            """
            rnd2 = rnd.random()
            self.ctheta = 1.0 - 2.0 * a * rnd2 / (1.0 + a - rnd2)  # scattering angle cosines , 0 <= angle <= 180˚, it produces an angular distribution that is obtained experimentally (more chance for low angles)
            self.stheta = sqrt(1.0 - self.ctheta * self.ctheta)  # scattering angle sinus
            self.psi = 2.0 * pi * rnd.random()  # azimuthal scattering angle

        def __get_direction(self, ctheta = None, stheta = None, psi = None):
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
            self.cz, self.cy, self.cx = traversal.get_direction(self.ctheta, self.stheta, self.psi, self.cz, self.cy, self.cx)
            self.direction_c[:] = self.cz, self.cy, self.cx

        def __get_next_point(self, step):
            """
            Calculate next point coordinates
            :param step:
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

    def setParameters(self, params, deposit, surface, stat=100000):
        #TODO: material tracking can be more universal
        # instead of using deponat and substrate variables, there can be a dictionary, where substrate is always last
        self.E0 = params['E0']
        self.Z = params['Z']
        self.A = params['A']
        self.rho = params['rho']
        self.e = params['e'] # fitting parameter related to energy required to initiate a SE cascade, material specific, eV
        self.lambda_escape = params['l'] # mean free escape path, material specific, nm
        self.grid = deposit
        self.surface = surface
        self.cell_dim = params['cell_dim']
        self.J = self._getJ()
        self.sigma = params['sigma']
        self.N = stat
        self.norm_factor = params['N'] / self.N

        self.PEL = 0
        self.SE_passes = []
        self.deponat = Element(params['name'], params['Z'], params['A'], params['rho'], params['e'], params['l'], -1)
        self.substrate = substrates[params['sub']]
        self.substrate.mark = -2
        self.material = None
        # self.dt = dt
        self.Emin = params['Emin'] # cut-off energy for PE, keV

        self.__calculate_attributes()

    def setParams_MC_test(self, structure, params):
        self.E0 = params['E0']
        self.N = params['N']
        self.deponat = substrates[params['material']]
        self.deponat.mark = -1
        self.substrate = substrates[params['material']]
        self.substrate.mark = -1
        self.Emin = params['Emin']  # cut-off energy for PE, keV
        self.sigma = params['sigma']

        self.grid = structure.deposit
        self.surface = np.logical_or(structure.surface_bool, structure.semi_surface_bool)
        self.cell_dim = structure.cell_dimension

        self.__calculate_attributes()

    def __calculate_attributes(self):
        self.x0, self.y0, self.z0 = self.grid.shape[2] * self.cell_dim / 2, self.grid.shape[1] * self.cell_dim / 2, self.grid.shape[0] * self.cell_dim - 1
        self.zdim, self.ydim, self.xdim = self.grid.shape
        self.zdim_abs, self.ydim_abs, self.xdim_abs = self.zdim * self.cell_dim, self.ydim * self.cell_dim, self.xdim * self.cell_dim
        self.chamber_dim = np.asarray([self.zdim_abs, self.ydim_abs, self.xdim_abs])
        self.ztop = np.nonzero(self.surface)[0].max() + 1  # highest point of the structure


    def rnd_gauss_xy(self, x0, y0, N):
        '''Gauss-distributed (x, y) positions.
           sigma: standard deviation
           n: number of positions
           Returns lists of n x and y coordinates which are gauss-distributed.
        '''
        # x, y = [], []
        if N==0: N=self.N
        r = np.random.normal(0, self.sigma, N)
        phi = np.random.uniform(0, 2 * pi - np.finfo(float).tiny, N)
        rnd.seed()
        # for i in range(self.N):
        #     ra = rnd.gauss(0.0, self.sigma)
        #     phis = rnd.uniform(0, 2 * pi - np.finfo(float).tiny)
        #     x.append(ra * cos(phis)+x0)
        #     y.append(ra * sin(phis)+y0)
        x = r*np.cos(phi) + x0
        y = r*np.sin(phi) + y0
        max_x = x[x>self.xdim_abs]
        reduct_coeff = max_x/self.xdim_abs +0.1
        x[x>self.xdim_abs] /= reduct_coeff
        max_y = y[y>self.ydim_abs]
        reduct_coeff = max_y/self.ydim_abs +0.1
        y[y > self.ydim_abs] /= reduct_coeff
        x = np.where(x<0, np.abs(x), x)
        y = np.where(y<0, np.abs(y), y)
        return (x, y)

    def run(self, i0, j0, N=0):
        """
        Runs Monte-Carlo electron scattering simulation for the given structure

        :param Emin: cut-off energy
        :param passes:
        :return:
        """
        """
        This is the first step of the simulation.
        In the previous version, PE trajectories were first simulated until electron looses all its energy or exits the chamber. 
        Then, in the next step, trajectories were shifted according to the Gaussian distribution and lowered to origin at the specimen surface.
        After that trajectories were adjusted to correspond scattering in the real structure.
        Two of those steps are now done in the process of initial trajectory simulation:
        1. With the given beam position(x,y) a Gauss distribution is mapped. 
            Beam origin is always at the top of the chamber and thus the first trajectory piece is a straight line down to the surface(solid)
        2. Then trajectories are mapped and their energy losses per trajectory step are calculated
            Trajectories are no longer mapped fully, but only until electron exits chamber or energy

        """

        y0, x0 = self.rnd_gauss_xy(i0*self.cell_dim, j0*self.cell_dim, N)  # generate gauss-distributed beam positions
        self.passes[:] = [] # prepare a list for energies and trajectories
        print("\nGenerating trajectories")
        # for x,y in tqdm(zip(x0,y0)):
        #     self.map_trajectory(x, y)
        # for i in tqdm(range(passes)):
        # Splitting collections to correspond to the number of processes:
        x = list(np.array_split(x0, 8))
        y = list(np.array_split(y0, 8))
        # Launching a processes manager:
        with multiprocessing.Pool(8) as pool:
            results = pool.starmap(self.map_wrapper, zip(x, y))
        # COllectings results:
        for p in results:
            self.passes += p
        # print(f'Done')
        # points = [(x, y) for (x,y) in zip(x0, y0)]
        # with concurrent.futures.ProcessPoolExecutor() as ex:
        #     results = [ex.submit(self.map_trajectory, point) for point in points]
        # for f in concurrent.futures.as_completed(results):
        #     self.passes.append(f.result())
        # self.prep_plot_traj() # view trajectories

    def map_wrapper(self, x0, y0, N=0):
        passes = []
        if N == 0:
            N = self.N
        if type(x0) in [float, int]:
            x0, y0 = self.rnd_gauss_xy(x0 * self.cell_dim, y0 * self.cell_dim, N)  # generate gauss-distributed beam positions
        # profiler = line_profiler.LineProfiler()
        # profiled_func = profiler(self.map_trajectory)
        # try:
        #     profiled_func(x0, y0)
        # finally:
        #     profiler.print_stats()
        passes = self.map_trajectory(x0, y0)
        # for x,y in zip(x0,y0):
        #     passes.append(self.map_trajectory(x,y))
        return passes

    def map_trajectory(self, x0, y0):

        # TODO: Vectorize !
        # TODO: if electron escapes
        self.passes = []
        self.ztop = np.nonzero(self.surface)[0].max()

        for x,y in zip(x0,y0):
            flag = False
            self.E = self.E0  # getting initial beam energy
            trajectories = []  # trajectory will be a sequence of points
            energies = []
            mask = []
            # Due to memory race problem, all the variables that are changing(coordinates, energy) have been moved to a separate class,
            # that gets instanced for every trajectory and thus for every process
            coords = self.Electron(x, y, self)
            # Getting initial point. It is set to the top of the chamber
            trajectories.append(coords.coordinates)  # every time starting from the beam origin (a bit above the highest point of the structure)
            energies.append(self.E0)  # and with the beam energy
            coords.coordinates = coords.coordinates
            # To get started, we need to know what kind of cell we're in
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
            while (coords.E > self.Emin):  # going on with every electron until energy is depleted or escaping chamber
                # a = self._getAlpha(coords)
                # step = -self._getLambda_el(coords, a) * log( rnd.uniform(0.00001, 1))  # actual distance an electron travels
                # Getting Alpha value and electron path length
                a, step = self._get_alpha_and_step(coords)
                # Next coordinates:
                check = coords.get_next_point(a, step)
                # First thing to do: boundary check
                # check = coords.check_boundaries()
                # Trimming new segment and heading for finishing the trajectory
                if check is False:
                    flag = True
                    step = traversal.det_1d(coords.direction) # recalculating step length
                # What cell are we in now?
                i,j,k = coords.indices# + coords.index_corr()
                # If its a solid cell, get energy loss and continue
                if self.grid[i,j,k] < 0 :
                    # coords.E += self._getELoss(coords) * step
                    coords.E += traversal.get_Eloss(coords.E, self.material.Z, self. material.rho, self.material.A, self.material.J, step)
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
                        # mask.append(1)
                        coords.coordinates = crossing1
                        trajectories.append(coords.coordinates)  # saving current point
                        energies.append(coords.E)  # saving electron energy at this point
                        if flag == 1: # if electron escapes after crossing surface and traversing void
                            mask.append(0)
                        else: # if electron meets a solid cell
                            mask.append(1)

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

        return self.passes

    def get_crossing_point(self, coords, curr_material):
        # Misses corresponding Cython function!
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
        Get next two crossing points and a flag showing if boundaries are met

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
        direction[0] *= -1 # recovering original sign

        direction = pn - p0
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

    def savePasses(self, fname):
        i = 0
        for p in self.passes:
            points = p[0]
            energies = p[1]
            f = open(fname + '_{:05d}'.format(i), 'w')
            f.write('# ' + self.name + '\n')
            f.write('# x/nm y/nm z/nm E/keV\n')
            for pt, e in zip(points, energies):
                f.write('{:f}\t'.format(pt[0]) + '{:f}\t'.format(pt[1]) +
                  '{:f}\t'.format(pt[2]) + '{:f}\t'.format(e) + '\n')
            f.close()
            i += 1

    def _get_alpha_and_step(self, coords):
        """
        Calculate alpha value and electron path from the electron energy and material properties

        :param coords:
        :return:
        """
        a, step = traversal.get_alpha_and_lambda(coords.E, self.material.Z, self.material.rho, self.material.A)
        step *= -log( rnd.uniform(0.00001, 1))
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


    def _getCxCyCz(self):
        r = sqrt(self.x**2 + self.y**2 + self.z**2)
        return (self.x/r,self.y/r,self.z/r)

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

    def show(self, dir_name):
        fig = plt.figure(figsize=(16,8))
        ax_t = fig.add_subplot(121)
        ax_e = fig.add_subplot(122)
        files = os.listdir(dir_name + '.')
        for f in files:
            if f.find('.dat') == -1: continue
            x, y, z, E = np.loadtxt(dir_name + f, unpack=True)
            ax_t.plot(x, -z, '-')
            ax_e.plot(E, '*')
        ax_t.set_xlabel('x (nm)')
        ax_t.set_ylabel('z (nm)')
        ax_e.set_xlabel('# elastic event')
        ax_e.set_ylabel('residual energy (eV)')
        plt.show()


class Element:
    """
    Represents a material
    """
    def __init__(self, name='noname', Z=1, A=1.0, rho=1.0, e=50, lambda_escape=1.0, mark=1):
        self.name = name # name of the material
        self.rho = rho # density, g/cm^3
        self.Z = Z # atomic number (or average if compound)
        self.A = A # molar mass, g/mol
        self.J = ETrajectory._getJ(self) # ionisation potential
        self.e = e # effective energy required to produce an SE, eV [lin]
        self.lambda_escape = lambda_escape # effective SE escape path, nm [lin]
        self.mark = mark

        # [lin] Lin Y., Joy D.C., Surf. Interface Anal. 2005; 37: 895–900

    def __add__(self, other):
        if other == 0:
            return self

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

substrates = {}
substrates["Au"] = Element(name='Au', Z=79, A=196.967, rho=19.32, e=35, lambda_escape=0.5)
substrates["Si"] = Element(name='Si', Z=14, A=29.09, rho=2.33, e=90, lambda_escape=2.7)



if __name__ == "__main__":
    params = {}

