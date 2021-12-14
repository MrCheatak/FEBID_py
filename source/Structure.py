import sys,os

import numpy as np
import pyvista as pv

### Structure class contains all necessary main (deposit, precursor, surface) and axillary (ghost_cells, semi_surface) arrays
### as well as methods, that are needed, if necessary, to generate the latter.
### The minimum information that has to be provided in order to construct the class is a solid structure (deposit) data.
### Class initialization method automatically dectects if provided dataset already includes all necessary arrays.
### Alternatively, class can be created from geometry parameters.
###

class Structure():
    """
    Represents simulation chamber and the current state of the process
    """

    def __init__(self, precursor_empty = 0., precursor_full = 1., deposit_empty = 0., deposit_full_substrate = -2., deposit_full_deponat = -1.):
        """
        Set values to mark empty and full cells for precursor and deposit arrays.

        STUB: These values are used in MC module as well to read the structure and has to be conveniently pipelined there

        :param precursor_empty:
        :param precursor_full:
        :param deposit_empty:
        :param deposit_full_substrate:
        :param deposit_full_deponat:
        """
        self.p_empty = precursor_empty
        self.p_full = precursor_full
        self.d_empty = deposit_empty
        self.d_full_s = deposit_full_substrate
        self.d_full_d = deposit_full_deponat

    def load_from_vtk(self, vtk_obj: pv.UniformGrid):
        """
        Frame initializer. Either a vtk object should be specified or initial conditions given.

        vtk object can either represent only a solid structure or a result of a deposition process with several parameters and arrays.
        If parameters are specified despite being present in vtk file (i.e. cell dimension), newly specified values are taken.

        :param cell_dim: size of a cell in nm
        :param width: width of the simulation chamber (along X-axis)
        :param length: length of the simulation chamber (along Y-axis)
        :param height: height of the simulation chamber (along Z-axis)
        :param substrate_height: thickness of the substrate in a number of cells along Z-axis
        :param nr: initial precursor density
        :param vtk_obj: a vtk object from file
        :param volume_prefill: level of initial filling for every cell. This is used to artificially speed up the depositing process
        """
        self.cell_dimension = 1
        if vtk_obj.spacing[0] != vtk_obj.spacing[1] or vtk_obj.spacing[0] != vtk_obj.spacing[2] or vtk_obj.spacing[1] != \
                vtk_obj.spacing[2]:
            choice = input(f'Cell\'s dimensions must be equal and represent a cube. '
            f'\nType x, y or z to specify dimension value that will be used for all three. '
            f'\nx={vtk_obj.spacing[0]} \ny={vtk_obj.spacing[1]} \nz={vtk_obj.spacing[2]} '
            f'\nThis may lead to a change of the structure\'s shape. Press any other key to exit.')
            if choice == 'x':
                self.cell_dimension = vtk_obj.spacing[0]
            if choice == 'y':
                self.cell_dimension = vtk_obj.spacing[1]
            if choice == 'z':
                self.cell_dimension = 5  # vtk_obj.spacing[2]
            if choice not in ['x', 'y', 'z']:
                sys.exit("Exiting.")
        else:
            self.cell_dimension = vtk_obj.spacing[0]
        self.cell_dimension = (self.cell_dimension)
        self.zdim, self.ydim, self.xdim = vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[
            0] - 1
        self.shape = (self.zdim, self.ydim, self.xdim)
        if 'surface_bool' in vtk_obj.array_names:  # checking if it is a complete result of a deposition process
            print(f'VTK file is FEBID file, reading arrays...')
            self.deposit = np.asarray(vtk_obj.cell_arrays['deposit'].reshape(
                (vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
            self.precursor = np.asarray(vtk_obj.cell_arrays['precursor_density'].reshape(
                (vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
            self.surface_bool = np.asarray(vtk_obj.cell_arrays['surface_bool'].reshape(
                (vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)), dtype=bool)
            self.semi_surface_bool = np.asarray(vtk_obj.cell_arrays['semi_surface_bool'].reshape(
                (vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)), dtype=bool)
            self.ghosts_bool = np.asarray(vtk_obj.cell_arrays['ghosts_bool'].reshape(
                (vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)), dtype=bool)
            # An attempt to attach new attributes to vtk object failed:
            # self.substrate_height = vtk_obj['substrate_height']
            # self.substrate_val = vtk_obj['substrate_val']
            # self.deposit_val = vtk_obj['deposit_val']
            # self.vol_prefill = vtk_obj['volume_prefill']
            self.substrate_val = -2
            self.deposit_val = -1
            self.substrate_height = np.nonzero(self.deposit == self.substrate_val)[0].max() + 1
            # self.vol_prefill = self.deposit[-1,-1,-1]
        else:
            # TODO: if a sample structure would be provided, it will be necessary to create a substrate under it
            self.deposit = np.asarray(vtk_obj.cell_arrays.active_scalars.reshape(
                (vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)))
            self.deposit[self.deposit != 0] = -1
            self.precursor = np.zeros(self.shape, dtype=np.float64)
            # self.substrate_height = substrate_height/self.cell_dimension
            self.substrate_val = -2
            self.deposit_val = -1
            # self.vol_prefill = self.deposit[-1, -1, -1] # checking if there is a prefill by probing top corner cell
            self.surface_bool = np.zeros(self.shape, dtype=bool)
            self.semi_surface_bool = np.zeros(self.shape, dtype=bool)
            self.ghosts_bool = np.zeros(self.shape, dtype=bool)
            self.define_surface()
            self.define_semi_surface()
            self.define_ghosts()
            self.substrate_height = 0
        self.precursor[self.precursor<0] = 0

    def create_from_parameters(self, cell_dim=5, width=50, length=50, height=100, substrate_height=4, nr=0):
        self.cell_dimension = cell_dim
        self.zdim, self.ydim, self.xdim = height, width, length
        self.shape = (self.zdim, self.ydim, self.xdim)
        self.deposit = np.zeros((self.zdim + substrate_height, self.ydim, self.xdim), dtype=np.float64)
        self.precursor = np.zeros((self.zdim + substrate_height, self.ydim, self.xdim), dtype=np.float64)
        self.substrate_val = -2
        self.deposit_val = -1
        self.substrate_height = substrate_height
        # self.vol_prefill = volume_prefill
        self.nr = nr
        self.flush_structure()
        self.surface_bool = np.zeros((self.zdim + substrate_height, self.ydim, self.xdim), dtype=bool)
        self.semi_surface_bool = np.zeros((self.zdim + substrate_height, self.ydim, self.xdim), dtype=bool)
        self.ghosts_bool = np.zeros((self.zdim + substrate_height, self.ydim, self.xdim), dtype=bool)
        self.define_surface()
        self.define_ghosts()
        self.t = 0

    def flush_structure(self):
        """
        Resets and prepares initial state of the grid. Use with caution as it will wipe

        :param substrate: 3D precursor density array
        :param deposit: 3D deposit array
        :param init_density: initial precursor density on the surface
        :param init_deposit: initial deposit on the surface, can be a 2D array with the same size as deposit array along 0 and 1 dimensions
        :param volume_prefill: initial deposit in the volume, can be a predefined structure in an 3D array same size as deposit array (constant value is virtual and used for code development)
        :return:
        """
        self.precursor[...] = 0
        self.precursor[0:self.substrate_height, :, :] = 0  # substrate surface
        if self.nr == 0:
            self.precursor[self.substrate_height, :, :] = 0.000001  # filling substrate surface with initial precursor density
        else:
            self.precursor[self.substrate_height, :, :] = self.nr  # filling substrate surface with initial precursor density
        # if self.vol_prefill == 0:
        #     self.deposit[...] = 0
        # else:
        #     self.deposit[...] = self.vol_prefill  # partially filling cells with deposit
        # if init_deposit != 0:
        #     self.deposit[1, :, :] = init_deposit  # partially fills surface cells with deposit
        self.deposit[...] = 0
        self.deposit[0:self.substrate_height, :, :] = -2

    def fill_surface(self, nr: float):
        """
        Covers surface of the deposit with initial precursor density

        :param nr: initial precursor density
        :return:
        """
        self.precursor[self.surface_bool] = nr

    def define_surface(self):
        """
        Determining surface of the initial structure

        :return:
        """

        # The whole idea is to derive surface according to neighboring cells
        # 1. Firstly, a boolean array marking non-solid cells is created (positive)
        # 2. Then, an average of each cell+neighbors is calculated (convolution applied)
        #   after this only cells that are on the surfaces(solid side and gas side) are gonna be changed
        # 3. Surface cells now have changed values and have to be separated from surface on the solid side
        #   it achieved by the intersection of 'positive' and convoluted arrays, as surface is ultimately not a solid

        positive = np.full((self.deposit.shape), False, dtype=bool)
        positive[self.deposit >= 0] = True  # gas cells
        grid = np.copy(self.deposit)
        # Applying convolution;  simple np.roll() does not work well, as it connects the edges(i.E rolls top layer to the bottom)
        grid[:, :, :-1] += self.deposit[:, :, 1:]  # rolling forward (actually backwards)
        grid[:, :, -1] += self.deposit[:, :, -1]  # taking care of edge values
        grid[:, :, 1:] += self.deposit[:, :, :-1]  # rolling backwards
        grid[:, :, 0] += self.deposit[:, :, 0]
        grid[:, :-1, :] += self.deposit[:, 1:, :]
        grid[:, -1, :] += self.deposit[:, -1, :]
        grid[:, 1:, :] += self.deposit[:, :-1, :]
        grid[:, 0, :] += self.deposit[:, 0, :]
        grid[:-1, :, :] += self.deposit[1:, :, :]
        grid[-1, :, :] += self.deposit[-1, :, :]
        grid[1:, :, :] += self.deposit[:-1, :, :]
        grid[0, :, :] += self.deposit[0, :, :]
        grid /= 7  # six neighbors + cell itself
        # Trimming unchanged cells:     using tolerance in case of inaccuracy
        grid[abs(grid - self.deposit_val) < 0.0000001] = 0  # fully deposited cells
        grid[abs(grid - self.substrate_val) < 0.0000001] = 0  # substrate
        # grid[abs(grid - self.vol_prefill) < 0.000001] = 0  # prefilled cells
        # Now making a boolean array of changed cells
        combined = np.full((self.deposit.shape), False, dtype=bool)
        combined[abs(grid) > 0] = True
        grid[...] = 0
        # Now, surface is intersection of these boolean arrays:
        grid += positive
        grid += combined
        self.surface_bool[grid == 2] = True

    def define_semi_surface(self):
        grid = np.zeros_like(self.deposit)
        grid[:, :, :-1] += self.surface_bool[:, :, 1:]  # rolling forward (actually backwards)
        grid[:, :, -1] += self.surface_bool[:, :, -1]  # taking care of edge values
        grid[:, :, 1:] += self.surface_bool[:, :, :-1]  # rolling backwards
        grid[:, :, 0] += self.surface_bool[:, :, 0]
        grid[:, :-1, :] += self.surface_bool[:, 1:, :]
        grid[:, -1, :] += self.surface_bool[:, -1, :]
        grid[:, 1:, :] += self.surface_bool[:, :-1, :]
        grid[:, 0, :] += self.surface_bool[:, 0, :]
        grid[:-1, :, :] += self.surface_bool[1:, :, :]
        grid[-1, :, :] += self.surface_bool[-1, :, :]
        grid[1:, :, :] += self.surface_bool[:-1, :, :]
        grid[0, :, :] += self.surface_bool[0, :, :]
        grid[self.deposit != 0] = 0
        grid[self.surface_bool] = 0
        grid[grid < 2] = 0
        self.semi_surface_bool[grid != 0] = True

    def define_ghosts(self):
        """
        Determining ghost shell wrapping surface
        This is crucial for the diffusion to work

        :return:
        """

        # Rolling in all directions marks all the neighboring cells
        # Subtracting surface from that selection results in a "shell" around the surface
        roller = np.logical_or(self.surface_bool, self.semi_surface_bool)
        self.ghosts_bool = np.copy(roller)
        self.ghosts_bool[:, :, :-1] += roller[:, :, 1:]  # rolling forward (actually backwards)
        self.ghosts_bool[:, :, -1] += roller[:, :, -1]  # taking care of edge values
        self.ghosts_bool[:, :, 1:] += roller[:, :, :-1]  # rolling backwards
        self.ghosts_bool[:, :, 0] += roller[:, :, 0]
        self.ghosts_bool[:, :-1, :] += roller[:, 1:, :]
        self.ghosts_bool[:, -1, :] += roller[:, -1, :]
        self.ghosts_bool[:, 1:, :] += roller[:, :-1, :]
        self.ghosts_bool[:, 0, :] += roller[:, 0, :]
        self.ghosts_bool[:-1, :, :] += roller[1:, :, :]
        self.ghosts_bool[-1, :, :] += roller[-1, :, :]
        self.ghosts_bool[1:, :, :] += roller[:-1, :, :]
        self.ghosts_bool[0, :, :] += roller[0, :, :]
        self.ghosts_bool[roller] = False

    def max_z(self):
        return self.deposit.nonzero()[0].max()

    def save_to_vtk(self):
        import time
        grid = pv.UniformGrid()
        grid.dimensions = np.asarray([self.deposit.shape[2], self.deposit.shape[1],
                                      self.deposit.shape[0]]) + 1  # creating grid with the size of the array
        grid.spacing = (self.cell_dimension, self.cell_dimension, self.cell_dimension)  # assigning dimensions of a cell
        grid.cell_arrays["deposit"] = self.deposit.flatten()
        grid.save('Deposit_' + time.strftime("%H:%M:%S", time.localtime()))
