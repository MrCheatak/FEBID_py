"""
Electron-matter interaction simulator
"""

import copy
import inspect
import random as rnd
import timeit

import numexpr_mod as ne
import numpy as np
from numpy.random import default_rng

from febid.libraries.ray_traversal import traversal
from febid.monte_carlo.mc_base import MC_Sim_Base


def process_trajectories(points, energies, mask):
    """
    Convert raw trajectories into a collection of segments

    :param points: consequent scattering points
    :param energies: remaining energy at each point
    :param mask: marks segments that lie outside of solid

    :return: an array of start- and end-points of segments, energy loss at segment
    """
    # Trajectories are divided into segments represented by a pair of points
    # Then mask is applied, selecting only segments that traverse solid
    # This reduces the unnecessary analysis of trajectory segments that lie in void(geometry features or backscattered electrons)
    try:
        mask = mask.astype(bool)
        pnp = points
        dE = energies
    except AttributeError:
        mask = np.asarray(mask, dtype=bool)
        pnp = np.array(points[0:len(points)]) # to get easy access to x, y, z coordinates of points
        dE = np.asarray(energies)
    pairs = np.empty((pnp.shape[0]-1, 2, 3))
    pairs[:,0,:] = pnp[:-1]
    pairs[:,1,:] = pnp[1:]
    pairs = pairs[mask]
    # Workaround against duplicate points
    p0 = pairs[:,0,:]
    pn = pairs[:,1,:]
    pn[pn==p0] += rnd.choice((0.000001, -0.000001)) # protection against duplicate coordinates
    # p0, pn = pnp[:-1], pnp[1:]
    # pairs = np.stack((p0,pn))
    # TODO: Thrown 'axis don't match array' exception :
    # pairs = np.transpose(pairs, axes=(1,0,2))[mask.nonzero()]
    # np.delete(pairs, (mask==0), axis=0)
    # result = pairs[mask.nonzero()]
    dE[:-1] -= dE[1:]
    dE = dE[:-1] # last element is discarded
    dE = dE[mask]*1000
    return pairs, dE


class ETrajMap3d(MC_Sim_Base):
    """
    Implements energy deposition and surface secondary electron flux calculation.
    """
    def __init__(self):
        """
        Create an empty ETrajMap3d instance
        """
        self.DE = None # array for storing of deposited energies
        self.flux = None # array for storing SE fluxes

        self.amplifying_factor = 1 # artificially increases SE yield to preserve accuracy
        # self.e = e # fitting parameter related to energy required to initiate a SE cascade, material specific, eV
        # self.lambda_escape = lambda_escape # mean free escape path, material specific, nm
        self.trajectories = [] # holds all trajectories mapped to 3d structure
        self.se_traj = [] # holds all trajectories mapped to 3d structure
        self.se_coords_all = None # array coordinates of all the SE sources
        self.se_coords_included = None # array of coordinates of SEs that are close to the surface
        self.wasted_se = None # SEs that did not escape the surface, used for Joule heating
        self.heat_pe = None # Energy deposiuted by the PEs that is converted to heat
        self.heat = None # total heating from the inelastic energy
        self.segment_min_length = 1

    def setParametrs(self, structure, params, segment_min_length=0.3):
        """
        Initialise the instance and set all the necessary parameters

        :param structure: solid structure representation
        :param params: contains all input parameters for the simulation
        :param segment_min_length: segment subdivision length
        """
        self.grid = structure.deposit
        self.surface = structure.surface_bool
        self.s_neighb = structure.surface_neighbors_bool  # 3D array representing surface n-nearest neigbors
        self.cell_dim = structure.cell_dimension  # absolute dimension of a cell, nm
        self.DE = np.zeros_like(self.grid)  # array for storing of deposited energies
        self.flux = np.zeros_like(self.grid)  # array for storing SE fluxes

        self.amplifying_factor = 10000  # artificially increases SE yield to preserve accuracy
        self.se_E = 19  # eV, average SE energy
        self.emission_fraction = params['emission_fraction']  # fraction of total lost energy spent on secondary electron emission
        self.deponat = params['deponat'] # deposit material properties
        self.substrate = params['substrate'] # substrate material properties
        self.se_traj = []  # holds all trajectories mapped to 3d structure
        self.segment_min_length = segment_min_length

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

    def follow_segment(self, points, dEs):
        """
        Calculate total energy deposited by primary electrons per cell.

        :param points: array of (z, y, x) points representing a trajectory from MC simulation
        :param dEs: list of energies losses between consecutive points. dEs[0] corresponds to a loss between p[0] and p[1]
        :return:
        """

        # Cell traversal algorithm taken from https://www.shadertoy.com/view/XddcWn#
        # Electron trajectory segments are treated as rays

        # Algorithm is vectorized, thus following calculations are done for all segments in a trajectory
        p0, pn = points[:, 0], points[:, 1]  # ray origin and end
        direction = points[:, 1, :] - points[:, 0, :] # vector of a ray

        # L = np.linalg.norm(direction, axis=1) # length of a rays
        L = np.empty_like(dEs)
        traversal.det_2d(direction, L)
        des = dEs/L
        sign = np.int8(np.sign(direction))
        step = sign * self.cell_dim # distance traveled by a ray along each axis in the ray direction, when crossing a cell
        # step_t = step / direction  # iteration step of the t-values
        step_t = None
        # Catching 'Division by zero' error
        try:
            with np.errstate(divide='raise', invalid='raise'):
                step_t = step / direction # iteration step of the t-values
        except Exception as e:
            print(e.args)
            zeros = np.nonzero(direction==0)[0]
            for i in range(len(zeros)):
                print(f'p0: {p0[zeros[i]]}, pn: {pn[zeros[i]]}, direction: {direction[zeros[i]]}, L: {L[zeros[i]]}')
            step_t = step / direction

        delta = -(points[:, 0] % self.cell_dim) # positions of the ray origin relative to its enclosing cell position
        t = np.abs((delta + (step == self.cell_dim) * self.cell_dim + (delta == 0) * step) / direction) # initial t-value
        max_traversed_cells = int(L.max()/self.cell_dim*2)+10 # maximum number of cells traversed by a segment in the trajectory;
        # this is essential to allocate enough memory for the traversal algorithm
        self.DE = np.zeros_like(self.grid)
        traversal.traverse_segment(self.DE, self.grid, self.cell_dim, p0, pn, direction, t, step_t, des, max_traversed_cells)

    def prep_se_emission(self, points, dEs, ends):
        """
        Subdivide trajectory segments and energy losses

        :param points: segment start- and end-points
        :param dEs: energy loss
        :param ends: trajectory end positions, check comments
        :return:
        """

        # Segments are divided into even parts, that become SE emission centers.
        # It has been observed, that segments are often smaller than the cell (~0.5nm average)
        # Thus, firstly, short segments are separated from longer ones and are left untouched.
        # Long segments are divided into even parts based on the 'segment_min_length' attribute of the ETrajMap3d class
        # All SE segments or 'vectors' are collected in a single array
        # NOTE: because SE 'vectors' are 'emitted' from the first point of the segment, the last point
        #  at the exit point has no emitted SE. This may significantly reduce SE surface yield at the exit points
        #  in grids with bigger cell size.
        #  A one additional emitted SE vector is added at the end of each trajectory in order to mitigate the problem.
        #  It is assigned the vector and energy of the last vector in the list.
        #  While major PEs do not re-enter solid on a simple structure, they might re-enter in more complex structures,
        #  that are not effectively taken into account.
        energies_all = []
        coords_all = []
        L = np.empty(dEs.shape, dtype=np.float64)
        traversal.det_2d(points[:, 1, :] - points[:, 0, :], L)
        # Collecting short segments that have energy loss
        short = (L <= self.segment_min_length).nonzero()[0]
        if short.shape[0]>0:
            coords_all.append(points[short,0].reshape(short.shape[0], 3))
            energies_all.append(dEs[short])

        # Collecting long segments
        long = (L > self.segment_min_length).nonzero()[0]
        if long.shape[0]>0:
            coords_long = np.take(points, long, axis=0)
            vector = coords_long[:, 1, :] - coords_long[:, 0, :]
            # BUG: np.ceil refuses to cast to integer even with 'casting=unsafe'
            num = np.intc(np.ceil(L[long] / self.segment_min_length)) # np.ceil r
            # delta = vector / np.broadcast_to(num, (3, num.shape[0])).T
            delta = vector/num.reshape(num.shape[0],1)
            N = num.sum(dtype=int)
            pieces = np.empty((N, 3))
            energies = np.empty(N)
            traversal.divide_segments(dEs[long], coords_long[:,0], num, delta, pieces, energies) # Cython script
            if pieces.min() < 0:
                print(f'Encountered negative values in coordinates in prep_se_emission')
                print(f'Num: {num}, delta: {delta}, ')
                err_index = (pieces<0).nonzero()[0]
                print(f'Pieces:')
                print(*pieces[err_index], sep='\n\t')
                print('Energies:')
                print(*energies[err_index], sep='\n\t')
                print(f'Segments\' energies: {dEs[long]}')
                print(f'Segments\' coordinates: {coords_long[:,0]}')
                pieces[err_index] = np.fabs(pieces[err_index])
                frame = inspect.currentframe().f_back.f_back
                sim = frame.f_locals['sim']
                import os
                directory = os.path.dirname(os.path.realpath(__file__))
                filename = os.path.join(directory, '../passes_last.txt')
                sim.save_passes(filename, 'text')
            coords_all.append(pieces)
            energies_all.append(energies)
        coords_all.append(points[points.shape[0]-1,1].reshape(1,3))
        l = len(energies_all)-1
        energies_all.append(energies_all[l][energies_all[l].shape[0]-1].reshape(1))
        traj_ends = points[ends, 1,:]
        energies_ens = self.segment_min_length / L[ends] * dEs[ends]
        coords_all.append(traj_ends)
        energies_all.append(energies_ens)
        # Combining all the collected segments into one array
        coords_all = np.concatenate((coords_all), axis=0)
        if coords_all.min() < 0:
            print(f'Encountered negative values in coordinates in prep_se_emission')
            print(f'')
        energies_all = np.concatenate((energies_all), axis=0)
        self.dEs_all = energies_all
        self.coords_all = coords_all

    def generate_se(self):
        """
        Estimate surface secondary electron flux.

        :return:
        """
        # Generate a random vector for every coordinate, calculate SE source power (n) per each vector
        # and collect them when they cross surface

        # Getting cell index for each emission point
        self.se_coords_all = coords_all = np.int32(ne.evaluate('a/b', global_dict={'a': self.coords_all.T, 'b': self.cell_dim}))
        # Selecting only cells in surface proximity
        neighbors = self.s_neighb
        self.se_coords_included = include = neighbors[coords_all[0], coords_all[1], coords_all[2]]
        in_index = include.nonzero()[0]
        coords = self.coords_all[in_index]
        dEs = self.dEs_all[in_index]

        # Generating random direction for each emission point
        rng = np.random.default_rng()
        direction = rng.normal(0, 10, (dEs.shape[0], 3)) # creates spherically randomly distributed vectors
        L = np.empty_like(dEs)
        traversal.det_2d(direction, L)
        direction /= L.reshape(L.shape[0],1) # normalizing
        # Preparing from ray tracing algorithm
        sign = np.int8(np.sign(direction))
        sign[sign==-1] = 0
        sign[sign==1] = -1
        delta = ne.evaluate('-(a%b)', global_dict={'a':coords, 'b':5})
        sign = (delta==0) * sign

        coords_ind = coords_all[:, in_index]
        cell_material = self.grid[coords_ind[0], coords_ind[1], coords_ind[2]]
        e = np.empty_like(cell_material)
        e[cell_material==-1] = self.deponat.e
        e[cell_material==-2] = self.substrate.e
        e[cell_material>=0] = 1000000
        lambda_escape = np.where(cell_material == -1, self.deponat.lambda_escape * 2, 0.00001) + np.where(cell_material == -2, self.substrate.lambda_escape * 2, 0.00001)
        n_se = dEs / e * self.amplifying_factor  # number of generated SEs, usually ~0.1

        length = lambda_escape # explicitly says that every vector has same length that equals SE escape path
        direction[:,0] *= length
        direction[:,1] *= length
        direction[:,2] *= length
        pn = direction + coords
        step = np.sign(direction) * self.cell_dim
        step_t = step / direction

        t = ne.evaluate('abs((d + m0s + d0 * s)/dir)', global_dict={'d':delta, 'm0s':np.maximum(step,0), 'd0':delta==0, 's':step, 'dir':direction})
        max_traversed_cells = int(np.amax(length, initial=0)/self.cell_dim*2+5)
        # Here each vector is processed with the same ray traversal algorithm as in 'follow_segment' method.
        # It checks if vectors cross surface cells and if they do, the number of emitted SEs (n) associated
        # with the vector is collected in the cell crossed.
        traversal.generate_flux(self.flux, self.surface.view(dtype=np.uint8), self.cell_dim, coords, pn, direction, sign, t, step_t, n_se, max_traversed_cells) # Cython script

        self.coords = np.empty((coords.shape[0], 2, 3))
        self.coords[:,0] = coords[...]
        self.coords[:,1] = pn[...]

    def extract_se_heat(self):
        """
        Calculate energy loss by scattered secondary electrons per cell.

        :return:
        """

        # Collecting SEs that did not escape the surface
        exclude = self.se_coords_included
        exclude_index = exclude.nonzero()[0]
        coords_ind = self.se_coords_all[:, exclude_index]
        index = (coords_ind[0], coords_ind[1], coords_ind[2])
        # Defining material properties at the emission point
        cell_material = self.grid[coords_ind[0], coords_ind[1], coords_ind[2]]
        e = np.empty_like(cell_material)
        e[cell_material == -1] = self.deponat.e
        e[cell_material == -2] = self.substrate.e
        e[cell_material >= 0] = 1e6
        # Number of emitted SEs per point (source power)
        n_se = self.dEs_all[exclude_index] / e
        self.wasted_se = np.zeros_like(self.DE)
        # Deposited energy per emission point
        wasted_se_flat = self.se_E * n_se
        # Accumulating deposited energy in cells
        np.add.at(self.wasted_se, index, wasted_se_flat)
        self.wasted_se[self.grid >= 0] = 0

    def joule_heating(self):
        """
        Get total energy loss from primary and secondary electrons peel
        """
        self.extract_se_heat()
        self.heat_pe = self.DE * (1 - self.emission_fraction)
        self.heat = self.heat_pe + self.wasted_se

    def map_follow(self, passes, heating=False):
        """
        Get surface secondary electron flux and volumetric heat source distribution
         from primary electron trajectories.

        :param passes: a collection of trajectories
        :param heating: True will calculate collective heat effect from PEs and SEs
        :return:
        """
        # Segments in all the trajectories deposit energy and emit SEs independently.
        # Thus, their interconnection and relation to a certain trajectory does not have to be preserved.
        # This fact is used to collect all segments and energy losses in single arrays
        # and to vectorize operations on them.

        # Masks that are passed along with trajectories play an important role by
        # letting efficiently sieve out segments that did not have energy loss (i.e those traversing void)
        # Zeros in energy array cause errors as energy is divisor in numerous parts of the algorithm.

        points = [] # coordinate pairs (segments) are first collected here
        dEs = [] # same for energies
        traj_lengths = []
        traj_len = 0
        print(f'*Preparing trajectories...', end='')
        start = timeit.default_timer()
        passes = copy.deepcopy(passes)
        for one_pass in passes:
            if len(one_pass[1][:]) < 3:
                continue
            pairs, energies = process_trajectories(one_pass[0][1:], one_pass[1][1:], one_pass[2][1:])
            if pairs.shape[0]:
                points.append(pairs)
                dEs.append(energies)
                traj_len += pairs.shape[0]
                traj_lengths.append(traj_len)
        traj_lengths = np.asarray(traj_lengths)
        segments_all = np.concatenate(points, axis=0)
        dEs_all = np.concatenate(dEs, axis=0)
        print(f'finished. \t {timeit.default_timer() - start}')

        print(f'**Running \'divide_segments\'....', end='')
        start = timeit.default_timer()
        self.prep_se_emission(segments_all, dEs_all, traj_lengths-1)
        print(f'finished. \t {timeit.default_timer() - start}')

        self.flux = np.zeros_like(self.grid)

        print(f'***Running \'generate_flux\'...', end='')
        start = timeit.default_timer()
        self.generate_se()
        print(f'finished. \t {timeit.default_timer() - start}')

        if heating:
            print(f'****Running \'traverse_segment\'...', end='')
            start = timeit.default_timer()
            self.follow_segment(segments_all, dEs_all)
            print(f'finished. \t {timeit.default_timer() - start}')

            print(f'*****Running \'joule_heating\'...', end='')
            start = timeit.default_timer()
            self.joule_heating()
            print(f'finished. \t {timeit.default_timer() - start}')
        a=0

        return self.flux, self.heat, self.dEs_all # has to be returned, as every process (when using multiprocessing) gets its own copy of the whole class and thus does not write to the original
