# Default packages
import warnings

# Core packages
import numpy as np
import pyvista as pv

# Axillary packeges
from tkinter import filedialog as fd
import timeit

# Local packages
from febid.Structure import Structure
from febid.libraries.vtk_rendering import VTK_Rendering as vr
from febid.monte_carlo import etrajectory as et
from febid.monte_carlo import etrajmap3d as map3d

# TODO: implement a global flag for collecting data(Se trajes) for plotting

class MC_Simulation_instance():
    def __init__(self, structure, **mc_params):
        mc_params['substrate'] = substrates[mc_params['substarte_element']]
        mc_params['deponat'] = Element(mc_params['name'], mc_params['Z'], mc_params['A'], mc_params['rho'], mc_params['e'], mc_params['l'], -1)
        self.pe_sim = et.ETrajectory()
        self.pe_sim.setParameters(structure, surface_neighbors)
        self.se_sim = map3d.ETrajMap3d()
        self.se_sim.setParametrs(structure, 0.3, mc_params)

        self.se_surface_flux = np.zeros(structure.shape, dtype=np.int32)
        self.deposited_energy = np.zeros(structure.shape)

    def update_structure(self, structure):
        self.pe_sim.grid = self.se_sim.grid = structure.deposit
        self.pe_sim.surface = self.se_sim.surface = structure.surface_bool
        self.se_sim.s_neighb = structure.surface_neighbors_bool
        self.se_surface_flux = np.zeros(structure.shape, dtype=np.int32)
        self.deposited_energy = np.zeros(structure.shape)

    def run_simulation(self, x0, y0, dt=1):
        start = timeit.default_timer()
        self.pe_sim.map_wrapper_cy(y0, x0)
        self.se_sim.map_follow(self.pe_sim.passes)
        if self.se_sim.flux.max() > 10000 * self.se_sim.amplifying_factor:
            print(f' Encountered infinity in the beam matrix: {np.nonzero(m3d.flux > 10000 * m3d.amplifying_factor)}')
            self.se_sim.flux[self.se_sim.flux > 10000 * self.se_sim.amplifying_factor] = 0
        if self.se_sim.flux.min() < 0:
            print(f'Encountered negative in beam matrix: {np.nonzero(m3d.flux < 0)}')
            self.se_sim.flux[self.se_sim.flux < 0] = 0
        const = self.pe_sim.norm_factor / (dt * self.pe_sim.cell_dim * self.pe_sim.cell_dim) / self.se_sim.amplifying_factor
        self.se_surface_flux[...] = np.int32(self.se_sim.flux * const)
        self.deposited_energy[...] = self.se_sim.DE * self.pe_sim.norm_factor
        return self.se_surface_flux, self.deposited_energy

    def plot(self, primary_e=True, deposited_E=True, secondary_flux=True, secondary_e=False):
        render = vr.Render(self.pe_sim.cell_dim)
        kwargs = {}
        if primary_e:
            pe_trajectories = np.asarray(self.pe_sim.passes)
            kwargs['pe_traj'] = pe_trajectories
        if deposited_E:
            kwargs['deposited_E'] = self.se_sim.DE
        if secondary_flux:
            kwargs['surface_flux'] = self.se_sim.flux
        if secondary_e:
            kwargs['se_traj'] = self.se_sim.coords_all
        render.show_mc_result(self.pe_sim.grid, **kwargs, interactive=False)


class Element:
    """
    Represents a material
    """
    def __init__(self, name='noname', Z=1, A=1.0, rho=1.0, e=50, lambda_escape=1.0, mark=1):
        self.name = name # name of the material
        self.rho = rho # density, g/cm^3
        self.Z = Z # atomic number (or average if compound)
        self.A = A # molar mass, g/mol
        self.J = (9.76*Z + 58.5/Z**0.19)*1.0E-3 # ionisation potential
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
substrates['Au'] = Element(name='Au', Z=79, A=196.967, rho=19.32, e=35, lambda_escape=0.5)
substrates['Si'] = Element(name='Si', Z=14, A=29.09, rho=2.33, e=90, lambda_escape=2.7)

def run_mc_simulation(vtk_obj, E0=20, sigma=5, N=100, pos='center', material='Au', Emin=0.1):
    """
    Create necessary objects and run the MC simulation

    :param vtk_obj:
    :param E0:
    :param sigma:
    :param N:
    :param pos:
    :param material:
    :param Emin:
    :return:
    """
    structure = Structure()
    structure.load_from_vtk(vtk_obj)
    params={'E0': E0, 'sigma':sigma, 'N':N, 'material':material, 'Emin':Emin}
    sim = et.ETrajectory()
    sim.setParams_MC_test(structure, params)
    x, y = 0, 0
    if pos == 'center':
        x = structure.shape[2]/2
        y = structure.shape[1]/2
    else:
        x, y = pos
    cam_pos = None
    while True:
        print(f'{N} PE trajectories took:   \t Energy deposition took:   \t SE preparation took:   \t Flux counting took:')
        start = timeit.default_timer()
        sim.map_wrapper(x,y)
        print(f'{timeit.default_timer() - start}', end='\t\t')
        m3d = map3d.ETrajMap3d(structure.deposit, structure.surface_bool, sim)
        m3d.map_follow(sim.passes, 1)
        pe_trajectories = np.asarray(sim.passes)
        render = vr.Render(structure.cell_dimension)
        cam_pos = render.show_mc_result(sim.grid, pe_trajectories, m3d.DE, m3d.flux, cam_pos=cam_pos)


def mc_simulation():
    """
    Fetch necessary data and start the simulation

    :return:
    """
    #TODO: This standalone Monte Carlo module should provide more data than in FEBID simulation

    print(f'Monte-Carlo electron beam - matter interaction module.\n'
          f'First load desired structure from vtk file, then enter parameters.')
    print(f'Select .vtk file....')
    while True:
        try:
            file = fd.askopenfilename()
            # file = '/Users/sandrik1742/Documents/PycharmProjects/FEBID/_source_samples/electron_beam_monte-carlo/hockeystick.vti'
            vtk_obj = pv.read(file)
        except Exception as e:
            print(f'Unable to read vtk file. {e.args} \n'
                  f'Try again:')
        else:
            print(f'Got file!\n')
            break

    print(f'Input parameters: Beam energy(keV), gauss st. deviation, number of electrons to emit, (beam x position, beam y position) , structure material(i.e. Au)\n'
          f'Note: \'center\' can be used instead of a coordinates pair (x,y) to set the beam to the center')
    E0=read_param('Beam energy', [int, float])
    sigma = read_param('Gauss standard deviation', [int, float])
    N = read_param('Number of electrons', [int])
    pos = read_param('Beam position', [tuple, str], check_string=['center'])
    material = read_param('Structure material', [str], check_string=['Au'])
    print(f'Got paramers!\n')
    run_mc_simulation(vtk_obj, E0, sigma, N, pos, material)
    # run_mc_simulation(vtk_obj, 20, 15, 1000, (12,17), 'Au')


def read_param(name, expected_type, message="Inappropriate input for ", check_string=None):
    """
    Read and parse a parameter from the input

    :param name: name of teh parameter
    :param expected_type: data type
    :param message: error text
    :param check_string: look for a specific string
    :return:
    """
    while True:
        item = input(f'Enter {name}:')
        try:
            result = eval(item)
        except:
            result = item
        if type(result) in expected_type:
            if type(result) is str:
                if check_string is not None:
                    if result in check_string:
                        return result
                    else:
                        warnings.warn("Input does not match any text choice")
                        print(f'Try again')
                        continue
            return result
        else:
            # unlike 'raise Warning()', this does not interrupt code execution
            warnings.warn(message+name)
            print(f'Try again')
            continue


def plot(m3d:map3d.ETrajMap3d, sim:et.ETrajectory, primary_e=True, deposited_E=True, secondary_flux=True,
         secondary_e=False, heat_total=False, heat_pe=False, heat_se=False): # plot energy loss and all trajectories
    """
    Show the structure with surface electron flux and electron trajectories

    :param m3d:
    :param sim:
    :return:
    """
    render = vr.Render(sim.cell_dim)
    kwargs = {}
    if primary_e:
        pe_trajectories = np.asarray(sim.passes)
        kwargs['pe_traj'] = pe_trajectories
    if deposited_E:
        kwargs['deposited_E'] = m3d.DE
    if secondary_flux:
        kwargs['surface_flux'] = m3d.flux
    if secondary_e:
        kwargs['se_traj'] = m3d.coords
    if heat_total:
        kwargs['heat_t'] = m3d.heat
    if heat_pe:
        kwargs['heat_pe'] = m3d.heat_pe
    if heat_se:
        kwargs['heat_se'] = m3d.wasted_se
    render.show_mc_result(sim.grid, **kwargs, interactive=False)


def cache_params(params, deposit, surface, surface_neighbors):
    """
    Creates an instance of simulation class and fetches necessary parameters

    :param fn_cfg: dictionary with simulation parameters
    :param deposit: initial structure
    :param surface: array pointing to surface cells
    :param cell_dim: dimensions of a cell
    :param dt: time step of the simulation
    :return:
    """

    sim = et.ETrajectory(name=params['name']) # creating an instance of Monte-Carlo simulation class
    sim.setParameters(params, deposit, surface, surface_neighbors) # setting parameters
    return sim


def rerun_simulation(y0, x0, sim:et.ETrajectory, heat):
    """
    Rerun simulation using existing MC simulation instance

    :param y0: beam y-position, absolute
    :param x0: beam x-position, absolute
    :param deposit: array representing solid structure
    :param surface: array representing surface shape
    :param sim: MC simulation instance
    :param heat: True will enable calculation of the beam heating
    :return:
    """

    if heat:
        N = 20000
        norm_factor = sim.get_norm_factor(N)
    else:
        N = sim.N
        norm_factor = sim.norm_factor
    sim.map_wrapper_cy(y0, x0, N)
    m3d = sim.m3d
    m3d.map_follow(sim.passes, heat)
    # plot(m3d, sim, True, True, True, True, True, True, True)
    if m3d.flux.max() > 10000*m3d.amplifying_factor:
        print(f' Encountered infinity in the beam matrix: {np.nonzero(m3d.flux>10000*m3d.amplifying_factor)}')
        m3d.flux[m3d.flux>10000*m3d.amplifying_factor] = 0
    if m3d.flux.min() < 0:
        print(f'Encountered negative in beam matrix: {np.nonzero(m3d.flux<0)}')
        m3d.flux[m3d.flux<0] = 0
    const = norm_factor/m3d.amplifying_factor/sim.cell_dim**2/sim.m3d.segment_min_length
    if heat:
        m3d.heat *= norm_factor/sim.cell_dim**3
    return np.int32(m3d.flux*const)


if __name__ == '__main__':
    print("Current script does not have an entry point.....")
    input('Press Enter to exit.')