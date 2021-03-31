from math import *
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

NA = 6.022141E23 # Avogadro number

class ETrajectory(object):

    def __init__(self, name='noname'):
        self.name = name
        self.passes = []
        rnd.seed()

    def setParameters(self, params):
        self.E0 = params['E0']
        self.Z = params['Z']
        self.A = params['A']
        self.rho = params['rho']
        self.x0, self.y0, self.z0 = params['x0'], params['y0'], params['z0']
        self.J = self._getJ()

    def run(self, Emin=0.1, passes=100):
        self.passes[:] = []
        for i in tqdm(range(passes)):
            trajectories = []
            energies = []
            trajectories.append((self.x0,self.y0,self.z0))
            energies.append(self.E0)
            self.E = self.E0
            self.x, self.y, self.z = self.x0, self.y0, self.z0
            cx, cy, cz = 0.0, 0.0, 1.0
            while (self.E > Emin) and (self.z >= 0.0):
                rnd1 = rnd.random()
                while rnd1 == 0.0: # exclude 0.0
                    rnd1 = rnd.random()
                rnd2 = rnd.random()
                rnd3 = rnd.random()
                step = -self._getLambda()*log(rnd1)
                a = self._getAlpha()
                ctheta = 1.0 - 2.0*a*rnd2/(1.0 + a - rnd2)
                stheta = sqrt(1.0 - ctheta**2)
                psi = 2.0*pi*rnd3
                if cz == 0.0: cz = 0.00001
                AM = -cx/cz
                AN = 1.0/sqrt(1.0 + AM**2)
                V1 = AN*stheta
                V2 = AN*AM*stheta
                V3 = cos(psi)
                V4 = sin(psi)
                ca = cx*ctheta + V1*V3 + cy*V2*V4
                cb = cy*ctheta + V4*(cz*V1 - cx*V2)
                cc = cz*ctheta + V2*V3 - cy*V1*V4
                x1 = self.x + step*ca
                y1 = self.y + step*cb
                z1 = self.z + step*cc
                E1 = self.E + self._getELoss()*step
                trajectories.append((x1,y1,z1))
                energies.append(E1)
                self.E = E1
                self.x, self.y, self.z = x1, y1, z1
                cx, cy, cz = ca, cb, cc
            self.passes.append((trajectories,energies))

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

    def _getJ(self):
        return (9.76*self.Z + 58.5/self.Z**0.19)*1.0E-3

    def _getAlpha(self):
        return 3.4E-3*self.Z**0.67/self.E

    def _getSigma(self): # in nm^2
        a = self._getAlpha()
        return 5.21E-7*self.Z**2/self.E**2*4.0*pi/(a*(1.0+a))*((self.E+511.0)/(self.E+1024.0))**2

    def _getLambda(self): # in nm
        sigma = self._getSigma()
        return self.A/(NA*self.rho*1.0E-21*sigma)

    def _getCxCyCz(self):
        r = sqrt(self.x**2 + self.y**2 + self.z**2)
        return (self.x/r,self.y/r,self.z/r)

    def _getELoss(self): # in keV/nm
        return -7.85E-3*self.rho*self.Z/(self.A*self.E)*log(1.166*(self.E/self.J + 0.85))

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
