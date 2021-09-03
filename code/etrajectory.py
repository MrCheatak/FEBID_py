#######################################################
#                                                     #
#       Primary electron trajectory simulator         #
#                                                     #
#######################################################
import math
from math import *
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from matplotlib import cm
import pyvista as pv
import os
from tqdm import tqdm
import line_profiler
import multiprocessing
import concurrent.futures
import itertools
from timebudget import timebudget

NA = 6.022141E23 # Avogadro number


class Element:
    """
    Represents a material
    """
    def __init__(self, name='noname', Z=1, A=1, rho=1):
        self.name = name
        self.rho = rho
        self.Z = Z
        self.A = A
        self.J = ETrajectory._getJ(self)

class ETrajectory(object):
    def __init__(self, name='noname'):
        self.name = name
        self.passes = []
        rnd.seed()

    class Coordinates():
        def __init__(self, x, y, parent):
            # Python uses dictionaries to represent class attributes, which causes significant memory usage
            # __slots__ attribute forces Python to use a small array for attribute storage, which reduces amount
            # of memory required for every copy of the class.
            # As a result, it reduces multiprocessing overhead as every process obtains a full copy of all objects it works with.
            # However, if __slots__ attribute is declared, declaration of new attributes is only possible by adding a new entry
            # to the __slots__.
            __slots__ = ["E", "x", "y", "z", "cx", "cy", "cz", "ctheta", "stheta", "psi",
                         "cell_dim", "zdim", "ydim", "zdim_abs", "ydim_abs", "xdim_abs"]
            self.E = parent.E
            self.x = x
            self.y = y
            self.z = (parent.zdim - parent.ztop) * parent.cell_dim / 3
            self.cx = 0
            self.cy = 0
            self.cz = 1
            self.ctheta = 0
            self.stheta = 0
            self.psi = 0
            self.cell_dim = parent.cell_dim
            self.zdim, self.ydim, self.xdim = parent.zdim, parent.ydim, parent.xdim
            self.zdim_abs, self.ydim_abs, self.xdim_abs = self.zdim * self.cell_dim, self.ydim * self.cell_dim, self.xdim * self.cell_dim

        def get_indices(self, z=0, y=0, x=0, cell_dim=1):
            """
            Gets indices of a cell in an array according to its position in the space

            :param x:
            :param y:
            :param z:
            :param cell_dim:
            :return: i(z), j(y), k(x)
            """
            if z == 0: z = self.z
            if y == 0: y = self.y
            if x == 0: x = self.x
            if cell_dim == 1: cell_dim = self.cell_dim
            return int(z / cell_dim), int(y / cell_dim), int(x / cell_dim)

        def __generate_angles(self, a):
            """
            Generates cos and sin of lateral angle and the azimuthal angle

            :param a: alpha at the current step
            :return:
            """
            rnd2 = rnd.random()
            self.ctheta = 1.0 - 2.0 * a * rnd2 / (
                        1.0 + a - rnd2)  # scattering angle cosines , 0 <= angle <= 180˚, it produces an angular distribution that is obtained experimentally (more chance for low angles)
            self.stheta = sqrt(1.0 - self.ctheta ** 2)  # scattering angle sinus
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
            # Coefficients for calculating direction cosines
            AM = -self.cx / self.cz
            AN = 1.0 / sqrt(1.0 + AM ** 2)
            V1 = AN * self.stheta
            V2 = AN * AM * self.stheta
            V3 = cos(self.psi)
            V4 = sin(self.psi)
            # New direction cosines
            # On every step a sum of squares of the direction cosines is always a unity
            ca = self.cx * self.ctheta + V1 * V3 + self.cy * V2 * V4
            cb = self.cy * self.ctheta + V4 * (self.cz * V1 - self.cx * V2)
            cc = self.cz * self.ctheta + V2 * V3 - self.cy * V1 * V4
            self.cz, self.cy, self.cx = cc, cb, ca
            return  self.cz, self.cy, self.cx

        def __get_next_point(self, step):
            """
            Calculate next point coordinates
            :param step:
            :return:
            """
            self.x += step * self.cx
            self.y += step * self.cy
            self.z -= step * self.cz

        def get_next_point(self, a, step):
            self.__generate_angles(a)
            self.__get_direction()
            self.__get_next_point(step)



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
            if 0 <= x < self.xdim_abs:
                if 0 <= y < self.ydim_abs:
                    if 0 <= z < self.zdim_abs:
                        return True
            return False

        def get_direction(self, ctheta=None, stheta=None, psi=None):
            return self.__get_direction(ctheta, stheta, psi)

    def setParameters(self, params, deposit, surface, cell_dim, beam_ef_rad, dt, stat=1000):
        self.E0 = params['E0']
        self.Z = params['Z']
        self.A = params['A']
        self.rho = params['rho']
        self.e = 72 # fitting parameter related to energy required to initiate a SE cascade, material specific, eV
        self.lambda_escape = 3.5 # mean free escape path, material specific, nm
        self.grid = deposit
        self.surface = surface
        self.cell_dim = cell_dim
        self.x0, self.y0, self.z0 = deposit.shape[2]*cell_dim/2, deposit.shape[1]*cell_dim/2, self.grid.shape[0]*self.cell_dim-1
        self.zdim, self.ydim, self.xdim = self.grid.shape
        self.zdim_abs, self.ydim_abs, self.xdim_abs = self.zdim * self.cell_dim, self.ydim * self.cell_dim, self.xdim * self.cell_dim
        self.ztop = np.nonzero(self.surface)[0].max()+1 # highest point of the structure
        self.J = self._getJ()
        self.sigma = params['sigma']
        self.N = stat
        self.norm_factor = params['N'] / self.N
        self.beam_ef_rad = beam_ef_rad
        self.PEL = 0
        self.SE_passes = []
        self.deponat = Element(params["name"], params["Z"], params["A"], params["rho"])
        self.substrate = Element(params["sub"], params["Z_s"], params["A_s"], params["rho_s"])
        self.material = nan
        self.dt = dt
        self.Emin = 0.1 # cut-off energy for PE, keV



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
        return (x, y)


    def __get_indices(self, z=0, y=0, x=0, cell_dim=1):
        """
        Gets indices of a cell in an array according to its position in the space

        :param x:
        :param y:
        :param z:
        :param cell_dim:
        :return: i(z), j(y), k(x)
        """
        # if z == 0: z = self.z
        # if y == 0: y = self.y
        # if x == 0: x = self.x
        if cell_dim == 1: cell_dim = self.cell_dim
        return int(z/cell_dim), int(y/cell_dim), int(x/cell_dim)

    def __generate_angles(self, coords, material=nan):
        """
        Generates cos and sin of lateral angle and azimuthal angle

        :param material:
        :return:
        """
        a = self._getAlpha(coords)
        rnd2 = rnd.random()
        ctheta = 1.0 - 2.0 * a * rnd2 / (1.0 + a - rnd2)  # scattering angle cosines , 0 <= angle <= 180˚, it produces an angular distribution that is obtained experimentally (more chance for low angles)
        stheta = sqrt(1.0 - ctheta ** 2)  # scattering angle sinus
        psi = 2.0 * pi * rnd.random()  # azimuthal scattering angle
        return ctheta, stheta, psi, a

    def __check_boundaries(self, z=0, y=0, x=0):
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
        if 0<=x<self.xdim_abs:
            if 0<=y<self.ydim_abs:
                if 0<=z<self.zdim_abs:
                    return True
        return False


    def test_run(self, i0, j0):
        profiler = line_profiler.LineProfiler()
        profiled_func = profiler(self.run)
        try:
            profiled_func(i0, j0)
        finally:
            profiler.print_stats()
        a=0


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
        if type(x0) == float:
            y0, x0 = self.rnd_gauss_xy(x0 * self.cell_dim, y0 * self.cell_dim, N)  # generate gauss-distributed beam positions
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
        for x,y in zip(x0,y0):
            flag = False
            trajectories = []  # trajectory will be a sequence of points
            energies = []
            mask = []
            # x, y = points
            trajectories.append(((self.zdim - self.ztop) * self.cell_dim / 3, y, x))  # every time starting from the beam origin (a bit above the highest point of the structure)
            energies.append(self.E0)  # and with the beam energy
            self.E = self.E0  # getting initial beam energy
            # Due to memory race problem, all the variables that are changing(coordinates, energy) have been moved to a separate class,
            # that gets instanced for every trajectory and thus for every process
            coords = self.Coordinates(x, y, self)#(self.E, (self.zdim - self.ztop) * self.cell_dim / 3, y, x, self.cell_dim)
            # self.x, self.y, self.z = x, y, (self.zdim - self.ztop) * self.cell_dim / 3
            # self.cx, self.cy, self.cz = 0, 0, 1  # direction cosines
            ctheta, stheta = 0, 0
            i, j, k = coords.get_indices()
            if self.grid[i, j, k] > -1:  # if current cell is not deposit or substrate, electron flies straight to the surface
                coords.z = ((self.grid[:, j, k]<0).nonzero()[0].max()+1) * self.cell_dim
                trajectories.append((coords.z, coords.y, coords.x))  # saving current point
                energies.append(coords.E)
                mask.append(0)
            while (coords.E > self.Emin):  # and (self.z >= 0.0):  # going on with every electron until energy is depleted or escaping chamber
                i, j, k = coords.get_indices()
                if coords.cz > 0:
                    if i > 0:
                        i -= 1
                if coords.cy < 0:
                    if j > 0:
                        j -= 1
                if coords.cx < 0:
                    if k > 0:
                        k -= 1
                if self.grid[i, j, k] > -1:  # checking if the cell is void
                    # If yes continuing previous trajectory until impact
                    x1, y1, z1 = coords.x, coords.y, coords.z
                    while True:
                        x1 += self.cell_dim * coords.cx
                        y1 += self.cell_dim * coords.cy
                        z1 -= self.cell_dim * coords.cz
                        if coords.check_boundaries(z1, y1, x1):
                            if self.grid[coords.get_indices(z1, y1, x1)] <= -1:
                                trajectories.append((z1, y1, x1))  # saving current point
                                energies.append(coords.E)  # saving electron energy at this point
                                mask.append(0)
                                coords.x, coords.y, coords.z = x1, y1, z1
                                break
                        else:
                            if z1>=self.zdim_abs:
                                z1 = self.zdim_abs-0.000001
                            if z1 <= 0:
                                z1 = 0.000001
                            if y1>=self.ydim_abs:
                                y1 = self.ydim_abs-0.000001
                            if y1 <= 0:
                                y1 = 0.000001
                            if x1>=self.xdim_abs:
                                x1 = self.xdim_abs-0.000001
                            if x1 <= 0:
                                x1 = 0.000001
                            trajectories.append((z1, y1, x1))  # saving current point
                            m=0
                            if self.grid[coords.get_indices(z1, y1, x1)] <= -1:
                                step = np.asarray([z1-coords.z, y1-coords.y, x1-coords.x])
                                step = math.sqrt(step.dot(step))
                                coords.E += self._getELoss(coords) * step
                                m = 1
                            energies.append(coords.E)  # saving electron energy at this point
                            mask.append(m)
                            coords.x, coords.y, coords.z = x1, y1, z1
                            flag = True
                            break
                    i, j, k = coords.get_indices()
                    if flag:  # finishing trajectory mapping if electron is beyond chamber walls
                        flag = False
                        break

                # Determining material of the current voxel
                if self.grid[i, j, k] == -2:  # curent cell is substrate
                    self.material = self.substrate
                if self.grid[i, j, k] == -1:  # current cell is deponat
                    self.material = self.deponat
                # ctheta, stheta, psi, a = self.__generate_angles(coords)
                a = self._getAlpha(coords)
                step = -self._getLambda_el(coords, a) * log(rnd.uniform(0.00001, 1))  # actual distance an electron travels
                # cc, cb, ca = coords.get_direction(ctheta, stheta, psi)  # direction cosines
                # Next step coordinates:
                # x1 = coords.x + step * ca
                # y1 = coords.y + step * cb
                # z1 = coords.z - step * cc
                coords.get_next_point(a, step)
                if not coords.check_boundaries():
                    break
                coords.E += self._getELoss(coords) * step
                # trajectories.append((z1, y1, x1))  # saving current point
                trajectories.append((coords.z, coords.y, coords.x))  # saving current point
                energies.append(coords.E)  # saving electron energy at this point
                mask.append(1)
                # Making the new point the current point for the next iteration
                # coords.x, coords.y, coords.z = x1, y1, z1
                # coords.cx, coords.cy, coords.cz = ca, cb, cc
            self.passes.append((trajectories, energies, mask))  # collecting mapped trajectories and energies
        # return (trajectories, energies, mask)
        return self.passes

    # def __get_direction(self, ctheta, stheta, psi):
    #     """
    #     Calculate cosines of the new direction according
    #     Special procedure from D.Joy 1995 on Monte Carlo modelling
    #     :param ctheta:
    #     :param stheta:
    #     :param psi:
    #     :return:
    #     """
    #     if self.cz == 0.0: self.cz = 0.00001
    #     # Coefficients for calculating direction cosines
    #     AM = -self.cx / self.cz
    #     AN = 1.0 / sqrt(1.0 + AM ** 2)
    #     V1 = AN * stheta
    #     V2 = AN * AM * stheta
    #     V3 = cos(psi)
    #     V4 = sin(psi)
    #     # New direction cosines
    #     # On every step a sum of squares of the direction cosines is always a unity
    #     ca = self.cx * ctheta + V1 * V3 + self.cy * V2 * V4
    #     cb = self.cy * ctheta + V4 * (self.cz * V1 - self.cx * V2)
    #     cc = self.cz * ctheta + V2 * V3 - self.cy * V1 * V4
    #     return ca, cb, cc

    def _FSEmfp(self, E): # in
        """
        Calculates mean free path for an SE
        :param E:
        :return:
        """
        a = self._getAlpha(E)
        sigma = self.Z**2*9842.7/(E**2*a*(1+a))
        return self.A*1E8/(self.rho*sigma)


    def _track_FSE(self, Emin = 0.001):
        """
        Simulates SE trajectory

        :param Emin:
        :return:
        """
        # FSE_count +=1
        traj = []
        energies = []
        a=rnd.random()
        E_frac = 1/(1000-998*a)
        E = self.E * E_frac # SE energy is a fraction of PE energy
        dE = self.E-E # incident energy loss is SE energy
        if E < Emin:
            escape = 750*E**1.66/self.rho # escape range
        x, y, z = self.x, self.y, self.z
        cx, cy, cz = self.cx, self.cy, self.cz
        while E > Emin:
            traj.append((x, y, z))
            energies.append(E)
            # Get the initial scattering angles
            stheta = 2 * (1 - E_frac) / (2 + E_frac * E / 511)
            ctheta = sqrt(1 - stheta)  # cos
            stheta = sqrt(stheta)  # sin
            step = -self._FSEmfp(E)*log(rnd.random())
            psi = 2.0 * pi * rnd.random()
            if cz == 0: cz = 0.00001 # avoid division by zero
            AM = -cx / cz
            AN = 1.0 / sqrt(1.0 + AM ** 2)
            V1 = AN * stheta
            V2 = AN * AM * stheta
            V3 = cos(psi)
            V4 = sin(psi)
            ca = cx * ctheta + V1 * V3 + cy * V2 * V4
            cb = cy * ctheta + V4 * (cz * V1 - cx * V2)
            cc = cz * ctheta + V2 * V3 - cy * V1 * V4
            x1 = x + step * ca
            y1 = y + step * cb
            z1 = z + step * cc
            deltaE = step*self._getELoss(E)
            E = E + deltaE
            x, y, z = x1, y1, z1
            cx, cy, cz = ca, cb, cc
            if int(z1/self.cell_dim) == 0:
                # terminate tracking
                ddd = 0
        self.SE_passes.append((traj, energies))
        stheta = E_frac*2/(2+(self.E/511)-(self.E*E_frac/511))
        ctheta = sqrt(1-stheta)
        stheta = sqrt(stheta)
        return ctheta, stheta


    def getPasses(self):
        return self.passes

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
        return self.A/(NA*self.material.rho*1.0E-21*self._getSigma(coords, a))


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


    def prep_plot_traj(self):
        self.p = pv.Plotter()
        grid = render_3Darray(self.grid, self.cell_dim, -2,0)
        self.p.add_mesh(grid, color='white', opacity=0.7)
        try:
            for trajectories, energies in self.passes:
                line = render_trajectory(trajectories, energies, 2, name="Energies, keV")
                self.p.add_mesh(line.tube(radius=1), opacity=0.5,smooth_shading=True)
        except:
            for trajectories in self.passes:
                line = render_trajectory(trajectories, radius=1)
                self.p.add_mesh(line, opacity=0.5,smooth_shading=True)
        # try:
        for trajectories, energies in tqdm(self.SE_passes):
            # lines = np.asarray(trajectories)
            # m = pv.PolyData()
            # m.points = lines
            # if len(lines) <5:
            #     if len(lines)<5:
            #         continue
            #     else:
            #         cells = np.full((len(lines), 3), 2, dtype=np.int_)
            #         cells[:, 1] = np.arange(0, len(lines), dtype=np.int_)
            #         cells[:, 2] = np.arange(1, len(lines), dtype=np.int_)
            # else:
            #     cells = np.full((len(lines)-1, 3), 2, dtype=np.int_)
            #     cells[:, 1] = np.arange(0, len(lines) - 1, dtype=np.int_)
            #     cells[:, 2] = np.arange(1, len(lines), dtype=np.int_)
            # m.lines = cells
            # m["scalars"] = np.asarray(energies)
            # line = m.tube(radius=2)
            line = render_trajectory(trajectories, energies, 0.7)
            self.p.add_mesh(line, color='red')
        # except:
        #     pass
        bcpos = [(774.8344494642572, 503.47871482113965, 375.82390286760705),
                (215.36398275262687, 162.21964754293566, 104.19286428573474),
                (-0.32454558774426506, -0.20322828037708204, 0.9237794257996352)]
        self.p.camera_position = bcpos
        cam = self.p.show() #(screenshot='PE_trajes.png')


def render_3Darray(arr, cell_dim, lower_t=0, upper_t=1, name='scalars_s', invert=False ):
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
    grid = grid.threshold([lower_t,upper_t], invert=invert) # trimming
    return grid


def render_trajectories(traj, energies=[], radius=0.7, step=1, name='scalars_t'):
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
            mesh = mesh + render_trajectory(traj[i], energies[i], radius, name)
            # mesh.plot()
    else:
        for i in tqdm(range(0, len(traj), step)):
            mesh = mesh + render_trajectory(traj[i], 0, radius, name)
    return mesh.tube(radius=radius) # it is important for color mapping to creaate tubes after all trajectories are added

def render_trajectory(traj, energies=0, radius=0.7, name='scalars'):
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

if __name__ == "__main__":
    params = {}
    name = 'Si_5kV'
    dir_name = './trajektorien/SiFul_5kV/'
    C = np.array([1.0, 16.0, 32.0, 45.0]) # composition N-C-Si-Cl
    Z = np.array([7.0, 6.0, 14.0, 17.0]) # Z number of constituent elements
    A = np.array([14.0, 12.0, 28.1, 35.5]) # A number of constituent elements
    params['E0'] = 5.0
    params['Z'] = np.dot(Z, C)/np.sum(C)
    params['A'] = np.dot(A, C)/np.sum(C)
    params['rho'] = 1.9
    params['x0'] = 0.0
    params['y0'] = 0.0
    params['z0'] = 0.0
    sim = ETrajectory(name=name)
    sim.setParameters(params)
    sim.run(passes=1000)
    sim.savePasses(dir_name + name + '.dat')
    sim.show(dir_name)
