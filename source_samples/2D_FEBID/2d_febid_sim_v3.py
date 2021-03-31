#!/usr/bin/env python3
'''Numerical solution of reaction diffusion problem for 2D FEBID with
   forward Euler method and speed-optimized code for laplacian.
   (C) Michael Huth, 2019-11-23.
'''
import numpy as np
from numexpr import evaluate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from math import *
import time
import yaml
import sys

#
# constants
#

kB = 1.38064852E-23 # in J/K
NA = 6.02214086E23
Echarge = 1.602E-19 # in As

#
# function definitions
#

def make_beam_path_rectangle(x0, y0, width, height, pitch):
    '''Returns beam path for rectangle in serpentine writing order.
       x0: lower left corner x-coordinate in nm
       y0: lower left corner y-coordinate in nm
       width: width in x-direction in nm
       height: height in y-directin in nm
       pitch: step width in x and y in nm.
       Returns list of [x, y] coordinates.
    '''
    x_steps = int(width//pitch + 1)
    y_steps = int(height//pitch + 1)
    dir = 1
    path = []
    for i in range(y_steps):
        y = y0 + i*pitch
        for j in range(x_steps):
            if dir == 1:
                x = x0 + j*pitch
            else:
                x = x0 + (x_steps - 1 - j)*pitch
            path.append([x, y])
        dir *= -1
    return path

def calc_D_tau(T, ED, Ed, D0, kd):
    '''Returns diffusion coefficient in nm^2/s and residence time in s.
       T: temperature in K
       ED: activation energy for diffusion in eV.
       Ed: activation energy for thermally induced desorption in eV.
           Typically Ed = 3 ... 10 ED (Utke review 2008).
       kD: prefactor for diffusion coefficient in nm^2/s.
       kd: prefactor for desorption rate in s^-1.
    '''
    kT = kB/Echarge*T
    return (D0*exp(-ED/kT), 1/kd*exp(Ed/kT))

def calc_Phig(p, m, T):
    '''Returns gas flux for given pressure, molar mass and temperature.
       p: pressure in Pa
       m: molar mass in g/mol
       T: temperature in K
       Returns gas flux in 1/nm^2s
    '''
    m = m/NA
    return p/sqrt(2*pi*m*kB*T)*1.0E-18

def calc_beam_flux(I, A, x0, y0, grid):
    '''Returns beam flux for given current, beam diameter, center position,
      and grid size.
       I: beam current in A
       A: Gaussian width parameter
       x0: x-position of beam in nm
       y0: y-position of beam in nm
       grid: (n, m) tuple for number of y and x positions
       Returns beam flux in 1/nm^2s
    '''
    n, m = grid
    a2 = A**2
    pre = I/Echarge/(2.0*pi*a2)
    f = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            r2 = (j - x0)**2 + (i - y0)**2
            f[i,j] += pre*exp(-r2/(2.0*a2))
    return f

def calc_molecule_diameter(M, rho):
    '''Returns molecule diameter in nm for given molar mass and density.
       M: molar mass i g/mol
       rho: density in g/cm^3
    '''
    return 1.122(M/(rho*NA))**(1.0/3.0)*1E7

def calc_adsorbate_density(d):
    '''Returns maximum adsorbate density n0 in 1/nm^2 for given molecule diameter.
       d: molecule diameter in nm
    '''
    return 1.154/d**2

def plot3d(z):
    '''Plot 3d perspective view of values z on grid.'''
    n = len(z[:,0])
    m = len(z[0,:])
    print(n, m)
    X, Y = np.meshgrid(np.linspace(0, m, m), np.linspace(0, n, n))
    mz = np.max(z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, z, rstride=4, cstride=4, alpha=1.0)
    cset = ax.contourf(X, Y, z, zdir='z', offset=-mz/5, cmap=cm.coolwarm)
    ax.set_zlim([-mz/5, mz])
    plt.show()

def save_data(fname, z, header):
    '''Save data as numpy array.
       Creates four files: fname_x.npy, fname_y.npy, fname_z.npy in
       binary numpy format and fname_info.txt containing info text in header.
    '''
    n = len(z[:,0])
    m = len(z[0,:])
    x, y = np.meshgrid(np.linspace(0, m, m), np.linspace(0, n, n))
    np.save(fname + '_x.npy', x)
    np.save(fname + '_y.npy', y)
    np.save(fname + '_z.npy', z)
    f = open(fname + '_info.txt', 'w')
    f.write(header)
    f.close()

def refresh(theta, n0, fg, s, tau):
    '''Set theta to equilibrium value for given parameters.
       n0: precusor density in 1/nm^2
       fg: gas flux in 1/nm^2s
       s: sticking coefficient
       tau: residence time in s
    '''
    a = s*fg/n0
    theta.fill(a*1.0/(1.0/tau + a))

def roll_add(rollee, shift, axis, out):
    '''Taken from book 'high performance python by Gorelick and Ozsvald.
       Compare section of numerical solution of diffusion problem in 2D.
    '''
    if shift == 1 and axis == 0:
        out[1:,:] += rollee[:-1,:]
        out[0,:] += rollee[-1,:]
    elif shift == -1 and axis == 0:
        out[:-1,:] += rollee[1:,:]
        out[-1,:] += rollee[0,:]
    elif shift == 1 and axis == 1:
        out[:,1:] += rollee[:,:-1]
        out[:,0] += rollee[:,-1]
    elif shift == -1 and axis == 1:
        out[:,:-1] += rollee[:,1:]
        out[:,-1] += rollee[:,0]

def laplace(theta_in, theta_out):
    '''Taken from book 'high performance python by Gorelick and Ozsvald.
       Compare section of numerical solution of diffusion problem in 2D.
    '''
    np.copyto(theta_out, theta_in)
    theta_out *= -4
    roll_add(theta_in, 1, 0, theta_out)
    roll_add(theta_in, -1, 0, theta_out)
    roll_add(theta_in, 1, 1, theta_out)
    roll_add(theta_in, -1, 1, theta_out)

#
# setup strings for using with numexpr generating fast code for vectorized
# expression with minimal cache misses and usage of temporary space.
#
eval_theta = '(theta_out*D'
eval_theta += ' + fg*s/n0*(1.0 - theta_in)'
eval_theta += ' - theta_in/tau'
eval_theta += ' - fe*sig*theta_in)*tD*dt + theta_in'
eval_height = 'V*sig*fe*n0*theta_in*dt*tD'
def iterate(theta_in, theta_out, dt, tD, D, n0, fg, s, tau, fe, sig):
    '''Iterate one time step.
       theta_in: array of normalized precusor density start values
       theta_out: array of normalized precusor density after one iteration
       dt: time step normalized to dwell time tD
       tD: dwell time in s
       D: diffusion coefficient in nm^2/s
       n0: precusor density in 1/nm^2
       fg: gas flux in 1/nm^2s
       s: sticking coefficient
       tau: residence time in s
       fe: electron flux array in 1/nm^2s
       sig: averaged dissociation cross section in nm^2
       Sets array theta_out to normalized precursor density after one iteration.
    '''
    laplace(theta_in, theta_out)
    evaluate(eval_theta, out=theta_out)

def sim_dwell_period(theta_in, theta_out, height, dt, tD, D, n0, fg, s, tau, fe, sig, V):
    '''Simulate for one complete dwell time.
       theta_in: array of normalized precusor density start values
       theta_out: array of normalized precusor density after one iteration
       height: array of height in nm
       dt: time step normalized to dwell time tD
       tD: dwell time in s
       D: diffusion coefficient in nm^2/s
       n0: precusor density in 1/nm^2
       fg: gas flux in 1/nm^2s
       s: sticking coefficient
       tau: residence time in s
       fe: electron flux array in 1/nm^2s
       sig: averaged dissociation cross section in nm^2
       V: volume of remaining deposit for each dissociated precursor molecule in nm^3
       Sets array theta_out to normalized precursor density and array height
       to new values after complete dwell event for duration tD.
    '''
    t = 0.0
    while t <= 1.0:
        iterate(theta_in, theta_out, dt, tD, D, n0, fg, s, tau, fe, sig)
        theta_in, theta_out = theta_out, theta_in
        height += evaluate(eval_height)
        t += dt

#
# main program
#

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('\n\n##########')
        print('usage: python3 2d_febid_sim_v3.py <sim parameters file (yaml)>', end='')
        print(' <precursor file (yaml)>', end='')
        print(' <pattern file (yaml)>')
        print('Terminate program.\n##########\n\n')
        exit(1)
    sim = yaml.load(open(sys.argv[1], 'r'), Loader=yaml.Loader)
    precursor = yaml.load(open(sys.argv[2], 'r'), Loader=yaml.Loader)
    pattern = yaml.load(open(sys.argv[3], 'r'), Loader=yaml.Loader)

    # set size of simulation grid
    #grid_shape = (sim['sim_width'], sim['sim_height'])
    grid_shape = (sim['sim_height'], sim['sim_width'])

    # set up zero beam array and Gaussian beam width parameter
    Phi_e_zero = np.zeros(grid_shape)
    A = sim['beam_fwhm']/(2.0*sqrt(2.0*log(2)))/sim['length_scale'] # Gaussian width parameter

    # set up gas flux
    Phi_g = calc_Phig(precursor['precursor_pressure'], precursor['molar_mass'],
                      precursor['precursor_temperature'])*sim['gis_factor']

    # create beam path
    if pattern['pattern_type'] == 'rectangle_filled':
        beam_path_rect_serpentine = make_beam_path_rectangle(pattern['x0']/sim['length_scale'],
          pattern['y0']/sim['length_scale'], pattern['width']/sim['length_scale'],
          pattern['height']/sim['length_scale'], pattern['pitch']/sim['length_scale'])
    else:
        print('\n\n##########Pattern type unknown. Terminate program.\n##########\n\n')

    # set up simulation
    D, tau = calc_D_tau(precursor['substrate_temperature'], precursor['diffusion_energy'],
                        precursor['desorption_energy'], precursor['diffusion_prefactor'],
                        precursor['desorption_frequency'])
    D /= sim['length_scale']**2 # adjust diffusion coefficient for length scale in x and y steps
    dt = 0.1/(D*pattern['dwell_time']) # this critertion will make sure that forward Euler converges
    theta_in = np.zeros(grid_shape)
    theta_out = np.zeros(grid_shape)
    height = np.zeros(grid_shape)
    print('\n\n################ Running simulation #################\n')
    print('time step (tD units) = {:4.3g}, time step (s) = {:4.3g}'.format(dt, dt*pattern['dwell_time']))
    start = time.time()

    # start with precursor density in equilibrium state
    refresh(theta_in, precursor['max_density'], Phi_g, precursor['sticking_coefficient'], tau)

    # run simulation proper
    for i in range(pattern['loops']):
        for x0, y0 in beam_path_rect_serpentine: #beam_path_dot:
            Phi_e = calc_beam_flux(sim['beam_current'], A, x0, y0, grid_shape)
            print('loop: {:0d}, x0 = {:04.3g} nm, y0 = {:04.3g} nm'.format(i+1, x0, y0), end='\r', flush=True)
            sim_dwell_period(theta_in, theta_out, height, dt, pattern['dwell_time'],
                             D, precursor['max_density'], Phi_g, precursor['sticking_coefficient'],
                             tau, Phi_e, precursor['cross_section'], precursor['dissociated_volume'])
        if pattern['refresh']:
            refresh(theta_in, precursor['max_density'], Phi_g, precursor['sticking_coefficient'], tau)
    print('\nElapsed time = {:4.3g} s'.format(time.time() - start))
    print('Maximum height of deposit = {:5.2g} nm'.format(np.max(height)))
    print('\n################ End of simulation #################\n')

    # save simulation result
    header = 'precursor: ' + precursor['name'] + '\n'
    header += 'grid shape = ({:d} nm, {:d} nm) \n'.format(grid_shape[0], grid_shape[1])
    header += 'diffusion coefficient = {:5.2g} nm^2/s\n'.format(D)
    header += 'residence time = {:5.2g} s\n'.format(tau)
    header += 'cross section = {:5.2g} nm^2\n'.format(precursor['cross_section'])
    header += 'max adsorbate density n0 = {:5.2g} 1/nm^2\n'.format(precursor['max_density'])
    header += 'single molecule deposit volume = {:5.2g} nm^3\n'.format(precursor['dissociated_volume'])
    header += 'pressure = {:5.2g} Pa\n'.format(precursor['precursor_pressure'])
    header += 'molar mass = {:5.2f} g/mol\n'.format(precursor['molar_mass'])
    header += 'sticking coefficient = {:5.2f}\n'.format(precursor['sticking_coefficient'])
    header += 'precursor temperature = {:5.2f} K\n'.format(precursor['precursor_temperature'])
    header += 'gas flux density = {:5.2g} 1/nm^2s\n'.format(Phi_g)
    header += 'peak electron flux density = {:5.2g} 1/nm^2s\n'.format(np.max(Phi_e))
    header += 'current I = {:5.2g} A\n'.format(sim['beam_current'])
    header += 'beam width FWHM = {:5.2f} nm\n'.format(sim['beam_fwhm'])
    header += 'pitch (x and y) = {:5.2f} nm\n'.format(pattern['pitch'])
    header += 'dwell time = {:5.2g} s\n'.format(pattern['dwell_time'])
    header += 'refresh = {:0}\n'.format(pattern['refresh'])
    header += 'loop number = {:d}\n'.format(pattern['loops'])
    header += 'time step size dt = {:5.2g} s\n'.format(dt*pattern['dwell_time'])
    header += 'length scale unit dl = {:5.2f} nm\n'.format(sim['length_scale'])

    save_data(precursor['name'] + '_precursor_density', theta_in, header)
    save_data(precursor['name'] + '_deposit_height', height, header)

    # plot final result
    plot3d(theta_in)
    plot3d(height)
