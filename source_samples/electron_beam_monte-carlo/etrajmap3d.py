import pyvista as pv
import numpy as np
from math import *
import os

class ETrajMap3d(object):

    def __init__(self):
        self.trajectories = [] # holds all trajectories mapped to 3d structure
        self.DE = None # will hold accumulated deposited energy for each voxel
        self.state = None # wild hold states of voxels after read_vtk()
        self.grid = None # will hold uniform grid after read_vtk()
        self.dx, self.dy, self.dz = 0.0, 0.0, 0.0 # distance between grid points; will be set after read_vtk()
        self.nx, self.ny, self.nz = 0, 0, 0 # number of cells; will be set after read_vtk()
        self.x0, self.y0, self.z0 = 0, 0, 0 # origin of 3d grid; will be set after read_vtk()

    def read_vtk(self, fname):
        '''Read vtk file with 3d voxel data.
           fname: name of vtk file.
           Creates uniform grid and sets empty(=0), surface(=1) and volume(=2) state data.
        '''
        self.grid = pv.read(fname)
        nx, ny, nz = self.grid.dimensions # is 1 larger than number of cells in each direction
        self.nx, self.ny, self.nz = nx - 1, ny - 1, nz - 1
        self.dx, self.dy, self.dz = self.grid.spacing
        self.DE = np.zeros((self.nx, self.ny, self.nz))
        self.state = np.reshape(self.grid.cell_arrays['state'], (self.nx, self.ny, self.nz), order='F')
        self.x0, self.y0, self.z0 = self.grid.origin

    def __find_zshift(self, x, y):
        '''Finds and returns z-position where beam at (x, y) hits 3d structure.'''
        i, j = int((x - self.x0)/self.dx), int((y - self.y0)/self.dy)
        for k in range(self.nz - 1, 0, -1):
            if self.state[i,j,k] == 2 or self.state[i,j,k] == 1:
                return self.z0 + k*self.dz

    def __triple(self, p): # need always 'left-most' indices
        return int(floor((p[0] - self.x0)/self.dx)), int(floor((p[1] - self.y0)/self.dy)), int(floor((p[2] - self.z0)/self.dz))

    def __crossings(self, i, j, k, istp, jstp, kstp, p0, vd):
        if vd[0] == 0: # segment is perpendicular to x
            t0 = sys.float_info.max
        else:
            d = self.x0 + i*self.dx # position of left wall of voxel
            if istp == 1: # right wall of voxel -> add dx to d
                d += self.dx
            t0 = (d - p0[0])/vd[0]
        if vd[1] == 0: # segment is perpendicular to y
            t1 = sys.float_info.max
        else:
            d = self.y0 + j*self.dy # position of front wall of voxel
            if jstp == 1: # back wall of voxel -> add dy to d
                d += self.dy
            t1 = (d - p0[1])/vd[1]
        if vd[2] == 0: # segment is perpendicular to z
            t2 = sys.float_info.max
        else:
            d = self.z0 + k*self.dz # position of bottom wall of voxel
            if kstp == 1: # top wall of voxel -> add dz to d
                d += self.dz
            t2 = (d - p0[2])/vd[2]
        return (t0, t1, t2)

    def __sign(self, x):
        if x < 0.0:
            return -1
        elif x > 0.0:
            return 1
        else:
            return 0

    def __follow_segment(self, traj, p1, p2, dE): # set traj points inside geometry for given p1, p2
        vd = p2 - p1
        L0 = sqrt(vd.dot(vd))
        L = L0
        istp, jstp, kstp = self.__sign(vd[0]), self.__sign(vd[1]), self.__sign(vd[2])
        p0 = p1
        pr = np.copy(p2) # will be changed in program and returned
        i, j, k = self.__triple(p0)
        t0, t1, t2 = self.__crossings(i, j, k, istp, jstp, kstp, p0, vd)
        t = 0.0
        cont = True
        while True:
            di = dj = dk = 0
            traj.append(p0)
            if t0 < t1:
                if t0 < t2:
                    t += t0
                    i += istp
                    di = istp
                else:
                    t += t2
                    k += kstp
                    dk = kstp
            else:
                if t1 < t2:
                    t += t1
                    j += jstp
                    dj = jstp
                else:
                    t += t2
                    k += kstp
                    dk = kstp
            ps = p1 + t*vd
            dp = ps - p0
            dL = sqrt(dp.dot(dp))
            if i < 0 or i >= self.nx or j < 0 or j >= self.ny or k < 0 or k >= self.nz:
                cont = False # segment runs out of simulation box
                traj.append(ps)
                break
            state_new = self.state[i,j,k]
            state_old = self.state[i-di,j-dj,k-dk]
            if state_old == 2 or state_old == 1:
                L -= dL
                if L < 0.0:
                    if k - dk >= 0:
                        dp = pr - traj[-1]
                        de = sqrt(dp.dot(dp))/L0*dE
                        self.DE[i-di,j-dj,k-dk] += de
                    traj.append(pr)
                    break
                traj.append(ps)
                if k - dk >= 0:
                    self.DE[i-di,j-dj,k-dk] += dE*dL/L0
            else:
                pr += ps - traj[-1]
                traj.append(ps)
            p0 = ps
            t0, t1, t2 = self.__crossings(i, j, k, istp, jstp, kstp, p0, vd)
        return pr, cont

    def __setup_trajectory(self, points, energies, xb, yb):
        '''Setup trajectory from MC simulation data for further computation.
           points: list of (x, y, z) points of trajectory from MC simulation
           energies: list of residual energies of electron at points of trajectory in keV
           xb, yb: lateral position of beam to hit the 3d structure.
           Returns lists of points and deposited energies (in eV) with corrected coordinates for 3d structure mapping.
        '''
        pnp = np.array(points) # to get easy access to x, y, z coordinates of points
        x, y, z = pnp[:,0], pnp[:,1], pnp[:,2]
        z *= -1 # invert direction of z-axis, so that it points up
        # move x and y to be at position xb, yb
        x = x - x[0] + xb
        y = y - y[0] + yb
        zshift = self.__find_zshift(x[0], y[0])
        z = z - z[0] + zshift
        pts, dEs = [], []
        for i in range(len(x) - 1):
            pts.append(np.array([x[i], y[i], z[i]]))
            dEs.append((energies[i] - energies[i+1])*1000) # convert energies to eV
        pts.append(np.array([x[-1], y[-1], z[-1]]))
        return pts, dEs


    def map_trajectory(self, points, energies, xb, yb):
        '''Do actual mapping of trajectory and energy loss onto 3d structure.
           points: list of (x, y, z) points of trajectory from MC simulation
           energies: list of deposited energies at the (x, y, z) points from MC simulation
           xb, yb: lateral position of beam to hit the 3d structure.
           Adds calculated trajectory in 3d structure to self.trajectories and updates
           entries in self.DE regarding accumulated deposited energy in each voxel.
        '''
        pts, dEs = self.__setup_trajectory(points, energies, xb, yb)
        p1 = pts[0]
        traj = []
        for i in range(len(pts) - 1):
            p2 = p1 + (pts[i+1] - pts[i])
            p1, cont = self.__follow_segment(traj, p1, p2, dEs[i])
            if not cont:
                break
        self.trajectories.append(traj)
