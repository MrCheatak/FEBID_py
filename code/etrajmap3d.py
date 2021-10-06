import logging
import multiprocessing
import random as rnd
import sys
import timeit
# import sandbox.HelloWorld as hw
from functools import total_ordering
from math import *

import numpy as np
import pyvista as pv
from numpy.random import default_rng
import os, sys
from tqdm import tqdm
import multiprocessing
import logging
import line_profiler
# import sandbox.HelloWorld as hw
from functools import total_ordering
import numba as nb
from modified_libraries.ray_traversal import traversal



@total_ordering # realises all comparison operations without having to define them explicitly
class ETrajMap3d(object):
    def __init__(self, deposit, surface, sim, segment_min_length = 1):
        #TODO: ETrajMap3d class should probably be merged with Etrajectory
        # These classes
        self.grid = deposit
        self.state = deposit
        self.surface = surface
        self.cell_dim = sim.cell_dim # absolute dimension of a cell, nm
        self.nz, self.ny, self.nx = np.asarray(self.grid.shape)- 1 # simulation chamber dimensions
        self.zdim_abs, self.ydim_abs, self.xdim_abs = [x*self.cell_dim for x in [self.nz, self.ny, self.nx]]
        self.DE = np.zeros((self.nz+1, self.ny+1, self.nx+1)) # array for storing of deposited energies
        self.flux = np.zeros((self.nz+1, self.ny+1, self.nx+1)) # array for storing SE fluxes
        self.amplifying_factor = 10000 # artificially increases SE yield to preserve accuracy
        # self.e = e # fitting parameter related to energy required to initiate a SE cascade, material specific, eV
        self.deponat= sim.deponat
        self.substrate = sim.substrate
        # self.lambda_escape = lambda_escape # mean free escape path, material specific, nm
        # self.dn = floor(self.lambda_escape * 2 / self.cell_dim) # number of cells an SE can intersect
        self.trajectories = [] # holds all trajectories mapped to 3d structure
        self.se_traj = []
        self.x0, self.y0, self.z0 = 0, 0, 0  # origin of 3d grid
        rnd.seed()
        self.segment_min_length = segment_min_length

    def __lt__(self, x):
        return x < self.xdim_abs

    def __eq__(self, x):
        return x == self.xdim_abs

    # @nb.jit(nopython=True)
    def __arr_min(self, x):
        if x[0] >= x[1]:
            if x[1] >= x[2]:
                return x[2], 2
            else:
                return x[1], 1
        else:
            if x[0] >= x[2]:
                return x[2], 2
            else:
                return x[0], 0

    # @nb.jit(nopython=True)
    def traverse_cells(self, p0, pn, direction, t, step_t):
        """
            AABB Ray-Voxel traversal algorithm.
            Gets coordinates, where ray crosses voxel walls

        :param p0: ray origin
        :param pn: ray endpoint
        :param direction: direction of the ray
        :param t: first t-value
        :param step_t: step of the t-value
        :return:
        """

        crossings = [p0]  # first point is always ray origin
        while True:  # iterating until the end of the ray
            next_t, ind = self.__arr_min(t)  # minimal t-value corresponds to the box wall crossed; 2x faster than t.min() !!
            if next_t > 1:  # finish if trajectory ends inside a cell (t>1)
                crossings.append(pn)
                break
            crossing_point = p0 + next_t * direction  # actual crossing point
            # if crossing_point[0] >= self.zdim_abs or crossing_point[1] >= self.ydim_abs or crossing_point[2] >= self.xdim_abs:  # 5x faster than ().any() !!!!!
            #     break
            t[ind] += step_t[ind]  # going to the next wall; 7x faster than with (t==next_t)
            crossings.append(crossing_point)  # saving crossing point
        return crossings

    def follow_segment(self, points_all, dEs_all):  #(self, points: np.ndarray, dEs):
        """
        Calculates distances traversed by a trajectory segment inside cells
        and gets energy losses corresponding to the distances.

        :param points: array of (z, y, x) points representing a trajectory from MC simulation
        :param dEs:  list of energies losses between consecutive points. dEs[0] corresponds to a loss between p[0] and p[1]
        :return:
        """

        # Cell traversal algorithm taken from https://www.shadertoy.com/view/XddcWn#
        # Electron trajectory segments are treated as rays

        # Algorithm is vectorized, thus following calculations are done for all segments in a trajectory
        # cells = []
        # profiler = line_profiler.LineProfiler()
        # profiled_func = profiler(self.generate_se)
        for points, dEs in zip(points_all, dEs_all):
            # points, dEs = self.__setup_trajectory(one_pass[0][1:], one_pass[1][1:], one_pass[2][1:])  # Adding distance from the beam origin to surface to all the points (shifting trajectories down) and getting energy losses
            if not dEs.any():
                continue
            direction = points[:, 1, :] - points[:, 0, :] # vector of a ray
            # L = np.linalg.norm(direction, axis=1) # length of a rays
            L = np.empty_like(dEs)
            traversal.det_2d(direction, L)
            des = dEs/L
            step = np.sign(direction) * self.cell_dim # distance traveled by a ray along each axis in the ray direction, when crossing a cell
            step_t = step / direction # iteration step of the t-values
            delta = -(points[:, 0] % self.cell_dim) # positions of the ray origin relative to its enclosing cell position
            t = np.abs((delta + (step == self.cell_dim) * self.cell_dim + (delta == 0) * step) / direction) # initial t-value
            p0, pn = points[:, 0], points[:, 1]  # ray origin and end
            max_traversed_cells = int(L.max()/self.cell_dim*2)+10 # maximum number of cells traversed by a segment in the a trajectory;
            # this is essential to allocate enough memory for the traversal algorithm
            # TODO: produced NaN at gr=4
            # max_traversed_cells = ceil(np.sum(1/step_t, axis=1).max())+5
            traversal.traverse_segment(self.DE, self.grid, self.cell_dim, p0, pn, direction, t, step_t, des, max_traversed_cells)


    def prep_se_emission(self, points_all, dEs_all):
        """
        The idea behind this method is to divide trajectory segments into pieces
        and emit SEs from every piece based on the energy lost on it.

        :param passes:
        :return:
        """
        energies_all = []
        coords_all = []
        for points, dEs in zip(points_all, dEs_all):  # go through trajectories
            # points, dEs = self.__setup_trajectory(passes[i][0][1:], passes[i][1][1:], passes[i][2][1:])  # Adding distance from the beam origin to surface to all the points (shifting trajectories down) and getting energy losses
            # direction = points[:, 1, :] - points[:, 0, :] # vector of a ray
            # L = np.linalg.norm(direction, axis=1)
            L = np.empty(dEs.shape, dtype=np.float64)
            traversal.det_2d(points[:, 1, :] - points[:, 0, :], L)
            # Segments are divided into even parts, that become SE emission centers.
            # It has been observed, that segments are often smaller than the cell (~0.5nm average)
            # Thus there is no point in dividing all segments and going through every piece in a loop.
            # Instead of that, short segments are separated from longer ones
            # and considered as emitting pieces.
            # Emission proceeds from the segment starting point!
            # All SE segments or vectors are collected in a single array

            # Collecting short segments that have energy loss
            short = np.logical_and(L <= self.segment_min_length, dEs != 0).nonzero() # If the energy loss is 0, it means that electron escaped solid should be discarded
            # short = np.nonzero(L <= 1.5 and dEs>0)
            # coords_all.append(np.asarray((points[short], points[np.asarray(short[0])+1])))
            if short[0].shape[0]>0:
                coords_all.append(points[short,0].reshape(short[0].shape[0], 3))
                energies_all.append(dEs[short])

            # Collecting long segments
            long = np.logical_and(L > self.segment_min_length, dEs != 0).nonzero()
            if long[0].any():
                coords = []
                # [coords.append([points[index,0], points[index,1]]) for index in long[0]]
                coords_long = np.take(points, long[0], axis=0)
                # coords_long = np.asarray(coords)
                # coords_long = np.asarray([points[long], points[np.asarray(long[0]) + 1]]) # creating coordinates pairs for all long segments
                shorts = []
                energies_l = []
                vector = coords_long[:, 1, :] - coords_long[:, 0, :]
                # BUG: np.ceil refuses to cast to integer even with 'casting=unsafe'
                num = np.intp(np.ceil(L[long[0]] / self.segment_min_length)) # np.ceil r
                delta = vector / np.broadcast_to(num, (3, num.shape[0])).T
                pieces = np.zeros((np.sum(num, dtype=int), 3))
                energies = np.zeros(np.sum(num, dtype=int))
                count = 0
                for i in range(len(long[0])):
                    # delta = np.fabs(coords_long[i, 1] - coords_long[i,0])
                    # num = int(delta.max()/1.5)
                    # num = ceil(L[long[0][i]] / self.segment_min_length) # number of pieces
                    # delta = vector[i]/num

                    # Evenly dividing segment into pieces
                    energies[count:count+num[i]] = np.repeat(dEs[long[0][i]] / (num[i]), num[i])
                    for j in range(num[i]):
                        pieces[count] = coords_long[i,0]+delta[i]*j
                        count += 1
                    # shorts.append(pieces)

                    # shorts.append(np.asarray((np.linspace(coords_long[i, 0, 0], coords_long[i, 1, 0], num, False), np.linspace(coords_long[i, 0, 1], coords_long[i, 1, 1], num, False),np.linspace(coords_long[i, 0, 2], coords_long[i, 1, 2], num, False))).T)
                    # energies_l.append(np.repeat(dEs[long[0][i]] / (num), num))
                # longs = np.concatenate((shorts), axis=0)
                # coords_all.append(np.asarray((longs[:-1], longs[1:])))
                coords_all.append(pieces)
                # e_l = np.concatenate((energies_l), axis=0)
                energies_all.append(energies)
            coords_all.append(points[points.shape[0]-1,1].reshape(1,3))
            l= len(energies_all)-1
            energies_all.append(energies_all[l][energies_all[l].shape[0]-1].reshape(1))

        # Combining all the collected segments into one array
        coords_all = np.concatenate((coords_all), axis=0)
        energies_all = np.concatenate((energies_all), axis=0)
        self.dES_all = energies_all
        self.coords_all = coords_all

    def generate_se(self):
        rng = default_rng()
        alpha = rng.uniform(0, 1, self.dES_all.shape) * 2 * pi
        # z = rng.uniform(0, 1, self.dES_all.shape)
        # y = np.sin(alpha) * np.sqrt(1-z*z) * length
        # x = np.cos(alpha) * np.sqrt(1-z*z) * length
        direction = np.empty((self.dES_all.shape[0], 3))
        direction[:, 0] = rng.uniform(0, 1, self.dES_all.shape)
        z_sqrt = np.sqrt(1 - direction[:, 0] * direction[:, 0])
        direction[:, 1] = np.sin(alpha) * z_sqrt
        direction[:, 2] = np.cos(alpha) * z_sqrt
        sign = np.int8(np.sign(direction))
        sign[sign==-1] = 0
        sign[sign==1] = -1
        delta = -(self.coords_all % self.cell_dim)
        sign = (delta==0) * sign


        # coords = (np.int64(self.coords_all/self.cell_dim + (delta == 0) * sign)).T
        coords = (np.int64(self.coords_all / self.cell_dim)).T
        cell_material = self.grid[coords[0], coords[1], coords[2]]
        # e = np.where(cell_material==-1, self.deponat.e, 0) + np.where(cell_material==-2, self.substrate.e, 0)
        e = np.empty_like(cell_material)
        e[cell_material==-1] = self.deponat.e
        e[cell_material==-2] = self.substrate.e
        e[cell_material>=0] = 1000000
        lambda_escape = np.where(cell_material == -1, self.deponat.lambda_escape * 2, 0.00001) + np.where(cell_material == -2, self.substrate.lambda_escape * 2, 0.00001)
        n_se = self.dES_all / e * self.amplifying_factor  # number of generated SEs, usually ~0.1

        length = lambda_escape
        # direction = np.column_stack((z*length, y, x))
        # direction *=  np.broadcast_to(length, (length.shape, 3))
        direction[:,0] *= length
        direction[:,1] *= length
        direction[:,2] *= length
        pn = direction + self.coords_all
        step = np.sign(direction) * self.cell_dim
        step_t = step / direction

        t = np.abs((delta + np.maximum(step, 0) + (delta == 0) * step) / direction)
        max_traversed_cells = int(np.amax(length, initial=0)/self.cell_dim*2+5)
        traversal.generate_flux(self.flux, self.surface.view(dtype=np.uint8), self.cell_dim, self.coords_all, pn, direction, sign, t, step_t, n_se, max_traversed_cells)

        self.coords_all = np.hstack((self.coords_all.reshape((pn.shape[0],1,3)), pn.reshape((pn.shape[0],1,3))))

    def __setup_trajectory(self, points, energies, mask):
        '''Setup trajectory from MC simulation data for further computation.
           points: list of (x, y, z) points of trajectory from MC simulation
           energies: list of residual energies of electron at points of trajectory in keV
           Returns arrays of points and energy losses (in eV)
        '''
        """
        Gauss distribution is now handled before PE trajectories simulation, where they are mapped according to the real structure
        Z-positions are also taken into account in that step
        """
        # Trajectories are divided into segments represented by a pair of points
        # Then mask is applied, selecting only segments that traverse solid
        # This reduces the unnecessary analysis of trajectory segments that lie in void(geometry features or backscattered electrons)
        mask = np.asarray(mask)
        pnp = np.array(points[0:len(points)]) # to get easy access to x, y, z coordinates of points
        p0, pn = pnp[:-1], pnp[1:]
        pairs = np.stack((p0,pn))
        # TODO: Thrown 'axis don't match array' exception :
        pairs = np.transpose(pairs, axes=(1,0,2))[mask.nonzero()]
        # np.delete(pairs, (mask==0), axis=0)
        # result = pairs[mask.nonzero()]
        dE = np.asarray(energies)
        dE -= np.roll(dE, -1)
        # dE.resize(len(dE)-1, refcheck=False)
        dE = dE[:-1] # last element is discarded
        dE = dE[mask.nonzero()]*1000
        # dE *=1000
        return pairs, dE

    def map_trajectory_multiprocessing(self, passes, n=8):
        '''
        Wrapper, that enables multicore processing. Check called function for the description.
        '''
        print("\nDepositing energy and generating SEs")
        pas = list(np.array_split(np.asarray(passes), n))
        with multiprocessing.Pool(n) as pool:
            results = pool.map(self.map_follow, pas)
        for p in results:
            self.flux += p[0]
            self.DE += p[1]
            self.se_traj += p[2]
        # print("Done")
        a=0

    def map_follow(self, passes, switch=1):
        """
        Get energy losses in the structure per cell

        :param passes: a collection of trajectories
        :param switch:
        :return:
        """
        if switch:
            # for one_pass in tqdm(passes):
            # for one_pass in passes:
            #     pts, dEs = self.__setup_trajectory(one_pass[0][1:], one_pass[1][1:])  # Adding distance from the beam origin to surface to all the points (shifting trajectories down) and getting energy losses
            #     self.follow_segment(pts, dEs)


            # profiled_func = profiler(hw.follow_trajectory_vec)
            # try:
            #     profiled_func(passes, self.grid, self.cell_dim)
            # finally:
            #     profiler.print_stats()
            start = timeit.default_timer()
            points = []
            dEs = []
            for one_pass in passes:
                if len(one_pass[1][:])<3:
                    continue
                pairs, energies = self.__setup_trajectory(one_pass[0][1:], one_pass[1][1:], one_pass[2][1:])
                if pairs.shape[0]:
                    points.append(pairs)
                    dEs.append((energies))
            # profiler = line_profiler.LineProfiler()
            # profiled_func = profiler(self.follow_segment)
            # try:
            #     profiled_func(points, dEs)
            # finally:
            #     profiler.print_stats()
            self.follow_segment(points, dEs)
            print(f'{timeit.default_timer()-start}', end='\t\t')
            # cell = self.DE.nonzero()
            # for i in range(len(cell[0])):
            #     self.generate_se(self.DE[cell[0][i], cell[1][i], cell[2][i]], cell[0][i], cell[1][i], cell[2][i], np.asarray([cell[0][i]*self.cell_dim+self.cell_dim/2, cell[1][i]*self.cell_dim, cell[2][i]*self.cell_dim]))
            start = timeit.default_timer()
            self.prep_se_emission(points, dEs)
            print(f'{timeit.default_timer()-start}', end='\t\t')
            start = timeit.default_timer()
            # profiler = line_profiler.LineProfiler()
            # profiled_func = profiler(self.generate_se)
            # try:
            #     for i in range(100):
            #         profiled_func()
            # finally:
            #     profiler.print_stats()
            self.generate_se()
            print(f'{timeit.default_timer()-start}', end='\t')
            a=0


        # else:
        #     for one_pass in passes:
        #         pts, dEs = self.__setup_trajectory(one_pass[0][1:], one_pass[1][1:])  # Adding distance from the beam origin to surface to all the points (shifting trajectories down) and getting energy losses
        #         p1 = pts[0]
        #         traj = []
        #         for i in range(len(pts) - 1):
        #             p2 = p1 + (pts[i + 1] - pts[i])  # set p1 and p2 as endpoints of current segment
        #             p1, cont = self.__follow_segment(traj, p1, p2, dEs[i])
        #             if not cont:  # if endpoint leaves simulation volume break
        #                 break
                # self.trajectories.append(traj)
        return self.flux, self.DE # has to be returned, as every process (when using multiprocessing) gets its own copy of the whole class and thus does not write to the original


############ Old code ##############
    def read_vtk(self, fname):
        '''Read vtk file with 3d voxel data.
           fname: name of vtk file.
           Creates uniform grid and sets empty(=0), surface(=1) and volume(=2) state data.
        '''
        self.grid = pv.read(fname)
        nx, ny, nz = self.grid.dimensions # is 1 larger than number of cells in each direction
        self.nx, self.ny, self.nz = nx - 1, ny - 1, nz - 1
        self.cell_dim, self.cell_dim, self.cell_dim = self.grid.spacing
        self.DE = np.zeros((self.nx, self.ny, self.nz))
        self.state = np.reshape(self.grid.cell_arrays['state'], (self.nx, self.ny, self.nz), order='F')
        self.x0, self.y0, self.z0 = self.grid.origin

    def __find_zshift(self, x, y):
        '''Finds and returns z-position where beam at (x, y) hits 3d structure.'''
        i, j = int((x - self.x0)/self.cell_dim), int((y - self.y0)/self.cell_dim) # converting absolute coordinates to array-coordinates
        for k in range(self.nz - 1, 0, -1): # proceeding from up to down
            if self.state[k,j,i] > self.state[self.nz-1, self.ny-1, self.nx-1]:# == 2 or self.state[i,j,k] == 1:
                return self.z0 + k*self.cell_dim # finishing on the first incident

    def __triple(self, p): # need always 'left-most' indices
        return  int(floor((p[0] - self.z0)/self.cell_dim)), int(floor((p[1] - self.y0)/self.cell_dim)), int(floor((p[2] - self.x0)/self.cell_dim))

    def __crossings(self, i, j, k, istp, jstp, kstp, p0, vd):
        if vd[0] == 0: # segment is perpendicular to z
            t0 = sys.float_info.max
        else:
            d = self.z0 + i*self.cell_dim # position of bottom wall of voxel
            if istp == 1: # top wall of voxel -> add cell_dim to d
                d += self.cell_dim
            t0 = (d - p0[0])/vd[0]
        if vd[1] == 0: # segment is perpendicular to y
            t1 = sys.float_info.max
        else:
            d = self.y0 + j*self.cell_dim # position of front wall of voxel
            if jstp == 1: # back wall of voxel -> add cell_dim to d
                d += self.cell_dim
            t1 = (d - p0[1])/vd[1]
        if vd[2] == 0: # segment is perpendicular to x
            t2 = sys.float_info.max
        else:
            d = self.x0 + k*self.cell_dim # position of left wall of voxel
            if kstp == 1: # right wall of voxel -> add cell_dim to d
                d += self.cell_dim
            t2 = (d - p0[2])/vd[2]
        return (t0, t1, t2)

    def __sign(self, x):
        if x < 0.0:
            return -1
        elif x > 0.0:
            return 1
        else:
            return 0

    def det(self, x):
        return sqrt(x.dot(x))

    def generate_se_old(self, de, i, j, k, pr):
        """
        Generates SEs depending on the deposited energy

        :param de: deposited energy
        :param i: z array position
        :param j: y array position
        :param k: x array position
        :param pr: initial scattering point
        :return:
        """
        # Current SE generation algorithm works per trajectory segment.
        # This works for small average segment lengths, i.e. as it is now(~0.5 nm).
        # Though with lower material density and Z, segments will get longer and
        # SE yield on the surface will be reduced
        # TODO: SE generation has to be implemented per segment piece eventually

        if self.surface[i,j,k]:
            self.flux[i, j, k] += de / self.e  # number of generated SEs
        else:
            if self.surface[self.__make_box(i,j,k)].any(): # check if SE can reach surface
                p0, p1 = pr - self.lambda_escape, pr + self.lambda_escape
                p0[p0<0] = 0
                p1[p1<0] = 0
                box = np.intp((np.array([p0, p1]) / self.cell_dim))
                slice = np.s_[box[0, 0]:box[1, 0]+1, box[0, 1]:box[1, 1]+1, box[0, 2]:box[1, 2]+1]
                view = self.surface[slice]
                pick = (rnd.randint(0, view.shape[0]-1), rnd.randint(0, view.shape[1]-1), rnd.randint(0, view.shape[2]-1))
                if view[pick]:
                    view  = self.flux[slice]
                    view[pick] += de / self.e  # number of generated SEs
                    return
                else:
                    return
                # for g in range(int(n_se)):
                alpha = rnd.uniform(-1, 1) * 2 * pi
                gamma = rnd.uniform(-1, 1) * 2 * pi
                length = self.lambda_escape * 2
                # s1, s2, s3 = int((pr[0]+length*cos(gamma))/self.cell_dim), int((pr[1]+length*sin(alpha))/self.cell_dim), int((pr[2]+length*cos(alpha))/self.cell_dim)
                s1, s2, s3 = (pr[0] + length * cos(gamma)), (pr[1] + length * sin(alpha)), (pr[2] + length * cos(alpha))
                # self.se_traj.append([pr, np.array([s1, s2, s3])]) # save SE trajectory for plotting
                try: # using try-except clause instead of checking boundaries, because in case of escape electron is just abandoned
                    if self.grid[self.__get_indices(s1, s2, s3)] > -1:  # if electron escapes solid
                        dz, dy, dx = (pr - (s1, s2, s3))/3
                        for n in range(3):  # # track which surface cell catches it
                            i,j,k = self.__get_indices(pr[0] - dz*n, pr[1] - dy*n, pr[2] - dx*n)
                            if self.surface[i,j,k] == True:
                                self.flux[i,j,k] += de / self.e  # number of generated SEs

                                break
                            # else:
                            #     s1 += dz
                            #     s2 += dy
                            #     s3 += dx
                except Exception as e: # printing out the actual exception just in case
                    logging.exception('Caught an Error:')
                    print("Skipping an electron")


    def __get_indices(self, z=0, y=0, x=0, cell_dim=0.0001):
        """
        Gets indices of a cell in an array according to its position in the space

        :param x: X-coordinate
        :param y: Y-coordinate
        :param z: Z-coordinate
        :param cell_dim: dimension of a cell
        :return: i(z), j(y), k(x)
        """
        if cell_dim == 0.0001: cell_dim = self.cell_dim
        return int(z/cell_dim), int(y/cell_dim), int(x/cell_dim)


    def __check_boundaries(self, z=0, y=0, x=0):
        """
        Checks is the given (z,y,x) position is inside the simulation chamber

        :param z:
        :param y:
        :param x:
        :return:
        """
        if 0 <= x < self.xdim_abs:
            if 0 <= y < self.ydim_abs:
                if 0 <= z < self.zdim_abs:
                    return True
        return False


    def __make_box(self, i, j, k):
        i_0 = i - self.dn
        i_n = i + self.dn + 1
        j_0 = j - self.dn
        j_n = j + self.dn + 1
        k_0 = k - self.dn
        k_n = k + self.dn + 1
        if i_0 < 0:
            i_0 = 0
        if j_0 < 0:
            j_0 = 0
        if k_0 < 0:
            k_0 = 0
        if i_n > self.nz:
            i_n = self.nz
        if j_n > self.ny:
            j_n = self.ny
        if k_n > self.nx:
            k_n = self.nx
        return np.s_[i_0:i_n, j_0:j_n,k_0:k_n]

    # TODO: maybe __follow_segment can be written in cpython?
    def __follow_segment(self, traj, p1, p2, dE):
        """
           Map line segment between points p1 and p2 onto real 3D structure and calculate energy deposited
           in each volume or surface voxel along the mapped segment.

           traj: list of segments as result of mapping of original segment
           dE: list of energy deposited in each element of traj

        :param traj: resulting trajectory
        :param p1: first point
        :param p2: next point
        :param dE:
        :return:
        """
        """
        Algorithm:
            1. Take the endpoints of the line segment and the energy loss associated with this line segment as input
            2. Calculate where the first point of the line segments is positioned within the simulation volume -> voxel
            3. Calculate where the line segment spanned by the fist and second point crosses the voxel surface of the voxel
            4. This point is stored for later use as it will become the first point for the next loop
            5. Check whether the type of voxel for which two segment endpoints are now known is of volume or surface type (i.e. filled)
            6. If it is filled the energy deposited in the voxel is calculated using the length of the segment
            7. The original length of the starting segment is reduced by the length of the segment inside the voxel
            8. If the remaining length runs below 0, the function returns
            9. If in the loop the mapped segments leave the simulation volume, the function returns with cont = False
            10. With the voxel intersection point 2 now becoming the first point the loop is repeated
        """
        vd = p2 - p1 # vector between two points
        L0 = sqrt(vd.dot(vd)) # length of segment
        L = L0 # L will be reduced from iteration to iteration until length <= 0, then leave
        # indicators for orientation of segment in space
        istp, jstp, kstp = self.__sign(vd[0]), self.__sign(vd[1]), self.__sign(vd[2]) # steps in the array
        p0 = p1 # p0 stores first endpoint of segment
        pr = np.copy(p2)  # will be changed in program and returned as point where next segment has to attach to
        i, j, k = self.__triple(p0)  # index triple of voxel where p0 lies within 3D structure
        # auxiliary real numbers to find where segment crosses surface of voxel
        t0, t1, t2 = self.__crossings(i, j, k, istp, jstp, kstp, p0, vd)
        t = 0.0
        cont = True # monitors when mapped segment leaves simulation volume -> cont = False
        while True:
            di = dj = dk = 0
            traj.append(p0) # appending the first point
            # following if-sequence and __crossing from fast method to find intersection of line segment
            # with box surface; taken from book about graphics programming in C and translated to python
            if t2 < t1:
                if t2 < t0:
                    t += t2
                    k += kstp
                    dk = kstp
                else:
                    t += t0
                    i += istp
                    di = istp
            else:
                if t1 < t0:
                    t += t1
                    j += jstp
                    dj = jstp
                else:
                    t += t0
                    i += istp
                    di = istp
            # t here is basically the smallest ratio between coordinate components and the vector
            ps = p1 + t*vd # actual point where box surface is crossed
            dp = ps - p0
            dL = sqrt(dp.dot(dp)) # length of segment from start point to crossing point
            if i < 0 or i >= self.nz or j < 0 or j >= self.ny or k < 0 or k >= self.nx: # checking if we are out of the array
                cont = False # segment runs out of simulation box
                traj.append(ps)
                break
            # calculate energy deposited in mapped segment if segment goes through surface or volume voxel
            state_old = self.state[i-di,j-dj,k-dk]
            if state_old == -2 or state_old == -1:
                # Here is the end of a segment is handled
                # Every step over a cell is subtracted from the total length of a segment
                # if the following step goes further than the end of the segment (reduced length is negative)
                # endpoint of a segment is chosen over the next step
                L -= dL
                if L < 0.0:
                    if i - di >= 0:
                        dp = pr - traj[-1]
                        de = sqrt(dp.dot(dp))/L0*dE
                        self.DE[i-di,j-dj,k-dk] += de # depositing energy in the corresponding cell
                        # if de != 0:
                        #     # TODO: when cell size is reduced, deposited energy per cell may go lower than activation energy
                        #     #  eliminating any SE emission
                        #     #  SE emission has to be evaluated per step of a trajectory segment
                        #     self.generate_se(de, i-di,j-dj,k-dk, pr)

                    traj.append(pr)
                    break
                traj.append(ps)
                if i - di >= 0:
                    self.DE[i-di,j-dj,k-dk] += dE*dL/L0
                    # if dE*dL/L0 != 0:
                    #     self.generate_se(self.DE[i-di,j-dj,k-dk], i-di,j-dj,k-dk, ps)
            else:
                pr += ps - traj[-1]
                traj.append(ps)
            p0 = ps  # box crossing point becomes new start point
            # calculate new auxillary numbers for next crossing point calculation
            t0, t1, t2 = self.__crossings(i, j, k, istp, jstp, kstp, p0, vd)
        return pr, cont