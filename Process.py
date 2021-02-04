import numpy as np
#import scipy
import scipy.constants as scpc
import math
#import array
import matplotlib.pyplot as plt
import numexpr



class Position:
    def __init__(self, z, y, x, dtype):
        if dtype == int:
            self.x = int(x)
            self.y = int(y)
            self.z = int(z)
        else:
            self.x = x
            self.y = y
            self.z = z


td = 1E-6  # dwell time of a beam, s
Ie = 1E-10  # beam current, A
beam_d = 10  # electron beam diameter, nm
effective_radius = beam_d * 3.3
f = Ie / scpc.elementary_charge / (math.pi * beam_d * beam_d / 4)  # electron flux at the surface, 1/(nm^2*s)
F = 3000  # precursor flux at the surface, 1/(nm^2*s)   here assumed a constant, but may be dependent on time and position
tau = 500E-6  # average residence time, s may be dependent on temperature

# Precursor properties
sigma = 2.2E-2  # dissociation cross section, nm^2 is averaged from cross sections of all electron types (PE,BSE, SE1, SE2)
n0 = 1.9  # inversed molecule size, Me3PtCpMe, 1/nm^2
M_Me3PtCpMe = 305  # molar mass of the precursor Me3Pt(IV)CpMe, g/mole
p_Me3PtCpMe = 1.5E-20  # density of the precursor Me3Pt(IV)CpMe, g/nm^3
V = 4 / 3 * math.pi * math.pow(0.139, 3)  # atomic volume of the deposited atom (Pt), nm^3
D = 1E5  # diffusion coefficient, nm^2/s

###
dt = 1E-6  # time step, s
t = 2E-6  # absolute time, s

kd = F / n0 + 1 / tau + sigma * f  # depletion rate
kr = F / n0 + 1 / tau  # replenishment rate
nr = F / kr  # absolute density after long time
nd = F / kd  # depleted absolute density
t_out = 1 / (1 / tau + F / n0)  # effective residence time
p_out = 2 * math.sqrt(D * t_out) / beam_d
cell_dimension = 5  # side length of a square cell, nm
diffusion_dt = math.pow(cell_dimension * cell_dimension, 2) / (2 * D * (cell_dimension * cell_dimension + cell_dimension * cell_dimension))  # maximum stability lime of the diffusion solution

# Main cell matrix.
# Substrate[z,x,y,0] holds precursor density,
# substrate[z,y,x,1] holds deposit density.
# </summary>
system_size = 50
substrate = np.zeros((system_size, system_size, system_size, 2), dtype=np.float)
for z in range(substrate.shape[0]):
    for y in range(substrate.shape[1]):
        for x in range(substrate.shape[2]):
            substrate[z, y, x, 1] = 2
for y in range(substrate.shape[1]):
    for x in range(substrate.shape[2]):
        substrate[0, y, x, 0] = nr
        substrate[0, y, x, 1] = 0
# <summary>
# Holds z coordinates of the surface cells that are allowed to produce deposit.
# The idea is to avoid iterating through the whole 3D matrix
# and checking if it is a surface cell. It is assumed, that surface cell is a cell with a fully deposited cell(or substrate) under it and thus can produce deposit.
# Thus the number of surface cells is fixed.
# </summary>
surface = np.zeros((system_size, system_size), dtype=np.int)
# <summary>
# Semi-surface cells are cells that have precursor density, but cannot produce deposit. They are introduced to allow diffusion on the walls of the deposit.
# </summary>
semi_surface = set()

# <summary>
# Calculates PE flux at the given radius according to Gaussian distribution
# </summary>
# <param name="r">Radius from the center of the beam</param>
# <returns></returns>
pe_flux_expression="f*exp(-r*r/(2*beam_d*beam_d))"
def pe_flux(r): return numexpr.evaluate(pe_flux_expression)


# <summary>
# Calculates distance between two points
# </summary>
# <param name="beam">Coordinates of the first point, absolute, nm</param>
# <param name="cell">Coordinates of the second point, relative</param>
# <returns></returns>
distance_expression = "((b_y - c_y * cell_dimension)**2 + (b_x - c_x * cell_dimension)**2)**0.5"
def distance(beam, cell_distance):
    return float(numexpr.evaluate(distance_expression, local_dict={'b_y': beam.y, 'b_x': beam.x, 'c_y': cell_distance.y, 'c_x': cell_distance.x}))
# <summary>
# Calculates deposition amount at the given cell and distance from the beam
# </summary>
# <param name="r">Radius from the center of the beam</param>
# <param name="cell">Coordinates of the current cell in the matrix, relative</param>
# <param name="dt">Time step of the simulation</param>
def deposition(r, cell):
    substrate[cell.z, cell.y, cell.x, 1] += substrate[cell.z, cell.y, cell.x, 0] * sigma * pe_flux(r) * V * dt
    if substrate[cell.z, cell.y, cell.x, 1] >= 1:  # if the cell is fully deposited
        surface[cell.y, cell.x] += 1
        cell.z += 1  # rising the surface one cell up (new cell)
        semi_surface.discard(cell)  # removing the new cell from the semi_surface list, if it is present there
        substrate[cell.z, cell.y, cell.x, 1] = substrate[cell.z - 1, cell.y, cell.x, 1] - 1  # if the cell was fulled above unity, transfering that surplus to the cell above
        substrate[cell.z - 1, cell.y, cell.x, 1] = 1  # a fully deposited cell is always a unity
        substrate[cell.z - 1, cell.y, cell.x, 0] = 1  # precursor density is also unity in the fully deposited cells
        # Adding neighbors(in x-y plane) of the new cell to semi_surface list
        # Checking if the cell falls out of the array and then if it is already has precursor in it
        # Simple check on precursor density also covers if the cell is already in the semi_surface list
        if cell.y - 1 >= 0:
            if substrate[cell.z, cell.y - 1, cell.x, 0] == 0:
                semi_surface.add(Position(cell.z, cell.y - 1, cell.x, int))  # adding cell to the list
        if cell.y + 1 < substrate.shape[1]:
            if substrate[cell.z, cell.y + 1, cell.x, 0] == 0:
                semi_surface.add(Position(cell.z, cell.y + 1, cell.x, int))
        if cell.x - 1 >= 0:
            if substrate[cell.z, cell.y, cell.x - 1, 0] == 0:
                semi_surface.add(Position(cell.z, cell.y, cell.x - 1, int))
        if cell.x + 1 < substrate.shape[2]:
            if substrate[cell.z, cell.y, cell.x + 1, 0] == 0:
                semi_surface.add(Position(cell.z, cell.y, cell.x + 1, int))


# <summary>
# Runs adsorption/desorption processes with the beam on
# </summary>
# <param name="beam">Coordinates of the beam center on the matrix, absolute, nm</param>
def precursor_density(beam):
    sub = np.zeros([substrate.shape[1], substrate.shape[2]])  # array for surface sells
    semi_sub = dict([])  # dictionary for semi-surface cells. It holds a precursor density increment(float) value for every position in the semi_surface List.
    cell = Position(0, 0, 0, int)
    # An increment is calculated for every cell first and
    # added only in the next loop not to influence diffusion
    for cell.y in range(substrate.shape[1]):  # calculating increment for surface cells
        for cell.x in range(substrate.shape[2]):
            cell.z = surface[cell.y, cell.x]
            sub[cell.y, cell.x] += rk4(cell, beam)

    for voxel in semi_surface:  # calculating increment for semi_surface cells (without electron induced desorption)
        semi_sub[voxel] = rk4_nb(voxel)

    for cell.y in range(substrate.shape[1]):  # adding increment to surface cells
        for cell.x in range(substrate.shape[1]):
            cell.z = surface[cell.y, cell.x]
            substrate[cell.z, cell.y, cell.x, 0] += sub[cell.y, cell.x]
    for entry in semi_sub:  # adding increment to semi_surface cells
        substrate[entry.z, entry.y, entry.x, 0] += semi_sub[entry]


# <summary>
# Runs adsorption/desorption in absence of the electron beam
# </summary>
def precursor_density_nb():
    sub = np.array([substrate.shape[1], substrate.shape[2]], dtype=float)
    cell = Position(0, 0, 0, int)
    for cell.y in range(substrate.shape[1]):
        for cell.x in range(substrate.shape[2]):
            cell.z = surface[cell.y, cell.x]
            sub[cell.y, cell.x] += rk4_nb(cell)
    for cell.y in range(substrate.shape[1]):
        for cell.x in range(substrate.shape[2]):
            cell.z = surface[cell.y, cell.x]
            substrate[cell.z, cell.y, cell.x, 0] += sub[cell.y, cell.x]


# <summary>
# Calculates increment of a function by Runge-Kutta method
# </summary>
# <param name="cell">Coordinates of the current cell in the matrix, relative</param>
# <param name="beam">Coordinates of the beam center on the matrix, absolute, nm</param>
# <returns></returns>
def rk4(cell, beam):
    k1 = precursor_density_increment_c_b(cell, beam)
    k2 = precursor_density_increment_c_b_a(cell, beam, dt / 2 * k1)
    k3 = precursor_density_increment_c_b_a(cell, beam, dt / 2 * k2)
    k4 = precursor_density_increment_c_b_a(cell, beam, dt * k3)
    return dt / 6 * (k1 + k2 * 2 + k3 * 2 + k4)


# <summary>
# Calculates increment of a function by Runge-Kutta method (in absence of the beam)
# </summary>
# <param name="cell">Coordinates of the current cell in the matrix, relative</param>
# <returns></returns>
def rk4_nb(cell):
    k1 = precursor_density_increment_c(cell)
    k2 = precursor_density_increment_c_a(cell, dt / 2 * k1)
    k3 = precursor_density_increment_c_a(cell, dt / 2 * k2)
    k4 = precursor_density_increment_c_a(cell, dt * k3)
    return dt / 6 * (k1 + k2 * 2 + k3 * 2 + k4)


# <summary>
# Calculates increment of the precursor density with all the terms
# </summary>
# <param name="cell">Coordinates of the current cell in the matrix, relative</param>
# <param name="beam">Coordinates of the beam center on the matrix, absolute, nm</param>
# <returns></returns>
adsorption_term = "F*(1-n/n0)+"
desorption_term = "n/tau+"
dissociation_term ="n*sigma*flux+"
diffusion_term = "D*Laplace"
def precursor_density_increment_c_b(cell, beam):
    radius = distance(beam, cell)
    return numexpr.evaluate(adsorption_term + desorption_term + dissociation_term + diffusion_term,
                            local_dict={'n': substrate[cell.z, cell.y, cell.x, 0], 'flux': pe_flux(radius), 'Laplace': laplace_term_c(cell)})
    # return F * (1 - substrate[cell.z, cell.y, cell.x, 0] / n0) - substrate[cell.z, cell.y, cell.x, 0] / tau - substrate[cell.z, cell.y, cell.x, 0] * sigma * pe_flux(radius) + D * laplace_term_c(cell)


# <summary>
# Calculates increment of the precursor density with all the terms
# </summary>
# <param name="cell">Coordinates of the current cell in the matrix, relative</param>
# <param name="beam">Coordinates of the beam center on the matrix, absolute, nm</param>
# <param name="addon">Coefficient for Runge-Kutta method</param>
# <returns></returns>
def precursor_density_increment_c_b_a(cell, beam, addon):
    radius = distance(beam, cell)
    value = substrate[cell.z, cell.y, cell.x, 0] + addon
    return numexpr.evaluate(adsorption_term + desorption_term + dissociation_term + diffusion_term,
                            local_dict={'n': value, 'flux': pe_flux(radius), 'Laplace': laplace_term_c_a(cell, addon)})
    # return F * (1 - value / n0) - value / tau - value * sigma * pe_flux(radius) + D * laplace_term_c_a(cell, addon)


# <summary>
# Calculates increment of the precursor density without electron induced desorption
# </summary>
# <param name="cell">Coordinates of the current cell in the matrix, relative</param>
# <param name="addon">Coefficient for Runge-Kutta method</param>
# <returns></returns>
def precursor_density_increment_c_a(cell, addon):
    value = substrate[cell.z, cell.y, cell.x, 0] + addon
    return numexpr.evaluate(adsorption_term + desorption_term + diffusion_term,
                            local_dict={'n': value, 'Laplace': laplace_term_c_a(cell, addon)})
    # return F * (1 - value / n0) - value / tau + D * laplace_term_c_a(cell, addon)


# <summary>
# Calculates increment of the precursor density without the electron induced desorption term
# </summary>
# <param name="cell">Coordinates of the current cell in the matrix, relative</param>
# <returns></returns>
def precursor_density_increment_c(cell):
    return numexpr.evaluate(adsorption_term + desorption_term + diffusion_term,
                            local_dict={'n': substrate[cell.z, cell.y, cell.x, 0], 'Laplace': laplace_term_c(cell)})
    # return F * (1 - substrate[cell.z, cell.y, cell.x, 0] / n0) - substrate[cell.z, cell.y, cell.x, 0] / tau + D * laplace_term_c(cell)


# <summary>
# Calculates the Laplace operator for the given position
# </summary>
# <param name="cell">Coordinates of the current cell in the matrix, relative</param>
# <returns></returns>
laplace_terms = "(xn-2*n+xp)/cell_dimension*cell_dimension+"
laplace_terms += "(yn-2*n+yp)/cell_dimension*cell_dimension+"
laplace_terms += "(zn-2*n+zp)/cell_dimension*cell_dimension"
def laplace_term_c(cell):
    # Making sure the neighboring cell indexes are not falling out of the array
    # This check has to be done separately, to avoid IndexOutOfRange exception
    x_previous = cell.x - 1
    x_next = cell.x + 1
    y_previous = cell.y - 1
    y_next = cell.y + 1
    z_previous = cell.z - 1
    z_next = cell.z + 1
    if cell.x == 0:                             x_previous = cell.x
    if cell.x >= substrate.shape[2] - 1:        x_next = cell.x
    if cell.y == 0:                             y_previous = cell.y
    if cell.y >= substrate.shape[1] - 1:        y_next = cell.y
    if cell.z == 0:                             z_previous = cell.z
    if cell.z >= substrate.shape[0] - 1:        z_next = cell.z

    # To prevent diffusion into fully deposited cell or into void
    # Neighboring full or empty cells are given the same value as the current cell
    _x = substrate[cell.z, cell.y, x_previous, 0]  # previous x
    x_ = substrate[cell.z, cell.y, x_next, 0]  # next x
    _y = substrate[cell.z, y_previous, cell.x, 0]
    y_ = substrate[cell.z, y_next, cell.x, 0]
    _z = substrate[z_previous, cell.y, cell.x, 0]
    z_ = substrate[z_next, cell.y, cell.x, 0]

    # 1 is a full deposit,  2 means void, thus a simple check is enough
    if substrate[cell.z, cell.y, x_previous, 1] >= 1:      _x = substrate[cell.z, cell.y, cell.x, 0]
    if substrate[cell.z, cell.y, x_next, 1] >= 1:          x_ = substrate[cell.z, cell.y, cell.x, 0]
    if substrate[cell.z, y_previous, cell.x, 1] >= 1:      _y = substrate[cell.z, cell.y, cell.x, 0]
    if substrate[cell.z, y_next, cell.x, 1] >= 1:          y_ = substrate[cell.z, cell.y, cell.x, 0]
    if substrate[z_previous, cell.y, cell.x, 1] >= 1:      _z = substrate[cell.z, cell.y, cell.x, 0]
    if substrate[z_next, cell.y, cell.x, 1] >= 1:          z_ = substrate[cell.z, cell.y, cell.x, 0]

    # ddx = (x_ - 2 * (substrate[cell.z, cell.y, cell.x, 0]) + _x) / cell_dimension * cell_dimension
    # ddy = (y_ - 2 * (substrate[cell.z, cell.y, cell.x, 0]) + _y) / cell_dimension * cell_dimension
    # ddz = (z_ - 2 * (substrate[cell.z, cell.y, cell.x, 0]) + _z) / cell_dimension * cell_dimension

    return np.float(numexpr.evaluate(laplace_terms, local_dict={'n': substrate[cell.z, cell.y, cell.x, 0], 'xn': x_, 'xp': _x, 'yn': y_, 'yp': _y, 'zn': z_, 'zp': _z}))
    # return ddz + ddy + ddx


# <summary>
# Calculates the Laplace operator for the given position
# </summary>
# <param name="cell">Coordinates of the current cell in the matrix, relative</param>
# <param name="addon">Coefficient for Runge-Kutta method</param>
# <returns></returns>
def laplace_term_c_a(cell, addon):
    # Making sure the neighboring cell indexes are not falling out of the array
    x_previous = cell.x - 1
    x_next = cell.x + 1
    y_previous = cell.y - 1
    y_next = cell.y + 1
    z_previous = cell.z - 1
    z_next = cell.z + 1
    if cell.x == 0:                             x_previous = cell.x
    if cell.x >= substrate.shape[2] - 1:        x_next = cell.x
    if cell.y == 0:                             y_previous = cell.y
    if cell.y >= substrate.shape[1] - 1:        y_next = cell.y
    if cell.z == 0:                             z_previous = cell.z
    if cell.z >= substrate.shape[0] - 1:        z_next = cell.z

    # To prevent diffusion into fully deposited cell or into void
    # neighboring fully deposited or empty cells are given the same value as the current cell
    _x = substrate[cell.z, cell.y, x_previous, 0]
    x_ = substrate[cell.z, cell.y, x_next, 0]
    _y = substrate[cell.z, y_previous, cell.x, 0]
    y_ = substrate[cell.z, y_next, cell.x, 0]
    _z = substrate[z_previous, cell.y, cell.x, 0]
    z_ = substrate[z_next, cell.y, cell.x, 0]

    # 1 is a full deposit,  2 means void, thus a simple check is enough
    if substrate[cell.z, cell.y, x_previous, 1] >= 1:      _x = substrate[cell.z, cell.y, cell.x, 0]
    if substrate[cell.z, cell.y, x_next, 1] >= 1:          x_ = substrate[cell.z, cell.y, cell.x, 0]
    if substrate[cell.z, y_previous, cell.x, 1] >= 1:      _y = substrate[cell.z, cell.y, cell.x, 0]
    if substrate[cell.z, y_next, cell.x, 1] >= 1:          y_ = substrate[cell.z, cell.y, cell.x, 0]
    if substrate[z_previous, cell.y, cell.x, 1] >= 1:      _z = substrate[cell.z, cell.y, cell.x, 0]
    if substrate[z_next, cell.y, cell.x, 1] >= 1:          z_ = substrate[cell.z, cell.y, cell.x, 0]

    # ddx = (x_ - 2 * (substrate[cell.z, cell.y, cell.x, 0] + addon) + _x) / cell_dimension * cell_dimension
    # ddy = (y_ - 2 * (substrate[cell.z, cell.y, cell.x, 0] + addon) + _y) / cell_dimension * cell_dimension
    # ddz = (z_ - 2 * (substrate[cell.z, cell.y, cell.x, 0] + addon) + _z) / cell_dimension * cell_dimension

    return np.float(numexpr.evaluate(laplace_terms, local_dict={'n': (substrate[cell.z, cell.y, cell.x, 0] + addon), 'xn': x_, 'xp': _x, 'yn': y_, 'yp': _y, 'zn': z_, 'zp': _z}))
    # return ddz + ddy + ddx


# /The printing loop.
dwell_step = beam_d / 2
x_offset = 80  # offset on the X-axis on both sides
y_offset = 80  # offset on the Y-axis on both sides
x_limit = substrate.shape[2] * cell_dimension - x_offset
y_limit = substrate.shape[1] * cell_dimension - y_offset
beam = Position(0, y_offset, x_offset, float)  # coordinates of the beam

for l in range(100):  # loop repeats

    for beam.y in np.arange(y_offset, y_limit, dwell_step):  # beam travel along Y-axis

        for beam.x in np.arange(x_offset, x_limit, dwell_step):  # beam travel alon X-axis

            # Determining the area around the beam that is effectively irradiated
            norm_y_start = 0
            norm_y_end = math.ceil((beam.y + effective_radius) / cell_dimension)
            norm_x_start = 0
            norm_x_end = math.ceil((beam.x + effective_radius) / cell_dimension)

            temp = math.floor((beam.y - effective_radius) / cell_dimension)
            if temp > 0:
                norm_y_start = temp
            if math.ceil((beam.y + effective_radius) / cell_dimension) > substrate.shape[1]:
                norm_y_end = substrate.shape[1]
            temp = math.floor((beam.x - effective_radius) / cell_dimension)
            if temp > 0:
                norm_x_start = temp
            if math.ceil((beam.x + effective_radius) / cell_dimension) > substrate.shape[2]:
                norm_x_end = substrate.shape[2]
            while True:  # iterating through every cell position in the selected area and depositing
                voxel = Position(0, norm_y_start, norm_x_start, int)
                for voxel.y in range(norm_y_start, norm_y_end):
                    for voxel.x in range(norm_x_start, norm_x_end):
                        r = distance(beam, voxel)
                        voxel.z = surface[voxel.y, voxel.x]
                        if r < effective_radius:
                            deposition(r, voxel)
                if t % diffusion_dt < 1E-6:  # adsorption and desorption is run with a max timestep that allows stable diffusion
                    precursor_density(beam)
                t += dt
                if not t % td > 1E-6:
                    break

t = t
section = np.zeros((substrate.shape[0], substrate.shape[1]), dtype=float)
fig, (ax0) = plt.subplots(1)
pos = np.int(substrate.shape[2]/2)
for i in range(0, substrate.shape[0]):
    for j in range(0, substrate.shape[1]):
        section[i,j] = substrate[i, j,pos,1]
ax0.pcolor(section)
        # if substrate[i,j,pos,0] ==0:
        #     ax0.pcolor([i,j,] substrate[i,j,pos,0], color="white")
        # else:
        #     if substrate[i,j,pos,1] == 1:
        #         ax0.pcolor([i,j,], substrate[i,j,pos,1], color="blue")
        #     else:
        #         ax0.pcolor([i,j,], substrate[i,j,pos,0], color="orange")
fig.tight_layout()
plt.show()
q=0

