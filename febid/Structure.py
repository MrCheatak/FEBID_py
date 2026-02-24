import sys

import numpy as np
import pyvista as pv

from febid.logging_config import setup_logger
from febid.mlcca import MultiLayerdCellCellularAutomata as MLCCA, initialize_structure_topology
# Setup logger
logger = setup_logger(__name__)


### Structure class contains all necessary main (deposit, precursor, surface) and axillary (ghost_cells, semi_surface) arrays
### as well as methods, that are needed, if necessary, to generate the latter.
### The minimum information that has to be provided in order to construct the class is a solid structure (deposit) data.
### Class initialization method automatically dectects if provided dataset already includes all necessary arrays.
### Alternatively, class can be created from geometry parameters.
###

# 1. Cell dimension is always in absolute units (nm)
# 2. Volume dimensions are always in relative units (array cells along each dimension)

class BaseSolidStructure:
    """
    This class represents a base solid structure.
    """

    def __init__(self):
        """
        Initializes the BaseSolidStructure object.
        """
        self.deposit = np.array([])
        self.cell_size = 1

    @property
    def shape(self):
        """
        Returns the shape of the deposit array.
        """
        return self.deposit.shape

    @property
    def shape_abs(self):
        """
        Returns the absolute shape of the deposit array.
        """
        return tuple(np.asarray(self.shape) * self.cell_size)

    @property
    def ydim_abs(self):
        """
        Returns the absolute y dimension of the deposit array.
        """
        return self.shape_abs[1]

    @property
    def xdim_abs(self):
        """
        Returns the absolute x dimension of the deposit array.
        """
        return self.shape_abs[2]

    @property
    def zdim(self):
        """
        Returns the z dimension of the deposit array.
        """
        return self.shape[0]

    @property
    def ydim(self):
        """
        Returns the y dimension of the deposit array.
        """
        return self.shape[1]

    @property
    def xdim(self):
        """
        Returns the x dimension of the deposit array.
        """
        return self.shape[2]

    @property
    def zdim_abs(self):
        """
        Returns the absolute z dimension of the deposit array.
        """
        return self.shape_abs[0]

    def size(self):
        """
        Returns the size of the deposit array.
        """
        return self.deposit.size


class Structure(BaseSolidStructure):
    """
    Represents the discretized space of the simulation volume and keeps the current state of the structure.
    """

    def __init__(self, precursor_empty=0., precursor_full=1., deposit_empty=0., deposit_full_substrate=-2.,
                 deposit_full_deponat=-1.):
        """
        Set values to mark empty and full cells for precursor and deposit arrays.

        STUB: These values are used in MC module as well to read the structure and has to be conveniently pipelined there

        :param precursor_empty:
        :param precursor_full:
        :param deposit_empty:
        :param deposit_full_substrate:
        :param deposit_full_deponat:
        """
        super().__init__()
        self.p_empty = precursor_empty
        self.p_full = precursor_full
        self.d_empty = deposit_empty
        self.d_full_s = deposit_full_substrate
        self.d_full_d = deposit_full_deponat

        self.room_temp = 294  # K, room temperature

        self.precursor = None
        self.surface_bool = None
        self.semi_surface_bool = None
        self.surface_neighbors_bool = None
        self.ghosts_bool = None
        self.temperature = None

        self.substrate_height = 0
        self.nr = 0.000001

        self.initialized = False
        self._mlcca = MLCCA()

    def _ensure_topology_arrays(self):
        """Ensure topology arrays exist with correct shape/dtype."""
        shape = self.deposit.shape
        if self.surface_bool is None or self.surface_bool.shape != shape:
            self.surface_bool = np.zeros(shape, dtype=bool)
        if self.semi_surface_bool is None or self.semi_surface_bool.shape != shape:
            self.semi_surface_bool = np.zeros(shape, dtype=bool)
        if self.surface_neighbors_bool is None or self.surface_neighbors_bool.shape != shape:
            self.surface_neighbors_bool = np.zeros(shape, dtype=bool)
        if self.ghosts_bool is None or self.ghosts_bool.shape != shape:
            self.ghosts_bool = np.zeros(shape, dtype=bool)

    def _rebuild_topology(self, n_surface_neighbors=0):
        """Canonical topology rebuild delegated to MLCCA."""
        initialize_structure_topology(self, n_surface_neighbors=n_surface_neighbors, mlcca=self._mlcca)

    @property
    def data_dict(self):
        return {'deposit': self.deposit, 'precursor': self.precursor, 'surface_bool': self.surface_bool,
                          'semi_surface_bool': self.semi_surface_bool, 'surface_neighbors_bool': self.surface_neighbors_bool,
                          'ghosts_bool': self.ghosts_bool, 'temperature': self.temperature}
    def load_from_vtk(self, vtk_obj: pv.DataSet, add_substrate=4):
        """
        Frame initializer. Load structure from a .vtk file.


        A vtk object can either represent only a single solid structure array or a result of a deposition process
        with the full set of arrays.

            Important requirement: vtk data must be a ImageData with 'spacing' attribute.

        :param vtk_obj: a vtk object from file
        :param add_substrate: if a value is specified, a substrate with such height will be created for simple vtk files. 0 or False otherwise. If the value is not a multiple of the 'spacing' attribute, it will be rounded down.
        """
        self.cell_size = 1
        if vtk_obj.spacing[0] != vtk_obj.spacing[1] or vtk_obj.spacing[0] != vtk_obj.spacing[2] or vtk_obj.spacing[1] != \
                vtk_obj.spacing[2]:
            raise ValueError('VTK file is not a uniform grid')
        else:
            self.cell_size = vtk_obj.spacing[0]
        self.cell_size = int(self.cell_size)
        shape = (vtk_obj.dimensions[2] - 1, vtk_obj.dimensions[1] - 1, vtk_obj.dimensions[0] - 1)
        if 'deposit' in vtk_obj.array_names:  # checking if it is a complete result of a deposition process
            logger.info('VTK file is a FEBID file, reading arrays...')

            try:
                self.deposit = np.asarray(vtk_obj.cell_data['deposit'].reshape(shape))
                cell_data_keys = vtk_obj.cell_data.keys()
                if 'precursor' in cell_data_keys:
                    self.precursor = np.asarray(vtk_obj.cell_data['precursor'].reshape(shape))
                else:
                    self.precursor = np.asarray(vtk_obj.cell_data['precursor_density'].reshape(shape)) # legacy support
                logger.debug('retrieved deposit and precursor data...')
            except Exception as e:
                logger.exception('Failed to read data from the .vtk FEBID file')
                raise
            try:
                self.surface_bool = np.asarray(vtk_obj.cell_data['surface_bool'].reshape(shape), dtype=bool)
                logger.debug('retrieved surface index data...')
            except:
                logger.debug('failed to retrieve surface index data...')
                self.surface_bool = np.zeros_like(self.deposit, dtype=bool)
                self.surface_bool = self._mlcca.compute_surface_topology(
                    self.deposit, d_full_d=self.d_full_d, d_full_s=self.d_full_s, out=self.surface_bool
                )
            try:
                self.semi_surface_bool = np.asarray(vtk_obj.cell_data['semi_surface_bool'].reshape(shape), dtype=bool)
                logger.debug('retrieved semi-surface index data...')
            except:
                logger.debug('failed to retrieve semi-surface index data...')
                self.semi_surface_bool = np.zeros_like(self.deposit, dtype=bool)
                self.semi_surface_bool = self._mlcca.compute_semi_surface_topology(
                    self.deposit, self.surface_bool, out=self.semi_surface_bool
                )
            try:
                self.surface_neighbors_bool = np.asarray(vtk_obj.cell_data['surface_neighbors_bool'].reshape(shape),
                                                         dtype=bool)
                logger.debug('retrieved surface nearest neighbors index data...')
            except:
                logger.debug('failed to retrieve surface nearest neighbors index data...')
                self.surface_neighbors_bool = np.zeros_like(self.deposit, dtype=bool)
                self._mlcca.compute_surface_neighbors(
                    self.deposit, self.surface_bool, n=3, out=self.surface_neighbors_bool
                )
            try:
                self.ghosts_bool = np.asarray(vtk_obj.cell_data['ghosts_bool'].reshape(shape), dtype=bool)
                logger.debug('retrieved ghost cells index data...')
            except:
                logger.debug('failed to retrieve ghost cells index data...')
                self.ghosts_bool = np.zeros_like(self.deposit, dtype=bool)
                self.ghosts_bool = self._mlcca.compute_ghost_shell(
                    self.surface_bool, self.semi_surface_bool, out=self.ghosts_bool
                )
            try:
                self.temperature = np.asarray(vtk_obj.cell_data['temperature'].reshape(shape))
                logger.debug('retrieved temperature data...')
            except:
                logger.debug('failed to retrieve temperature data...')
                self.temperature = np.zeros_like(self.deposit)
                self.temperature[self.deposit < 0] = self.room_temp
        else:
            logger.info('VTK file is a regular file, generating auxiliary arrays...')
            self.deposit = np.asarray(vtk_obj.cell_data.active_scalars.reshape(shape))
            # self.deposit[self.deposit != 0] = -1
            # Electrons need at least one empty cell layer for creation
            empty_layer = np.zeros((5, shape[1], shape[2]), dtype=np.float64)
            self.deposit = np.concatenate((self.deposit, empty_layer), axis=0)
            if add_substrate:
                self.substrate_height = add_substrate
                substrate = np.full((self.substrate_height, shape[1], shape[2]), -2)
                self.deposit = np.concatenate((substrate,self.deposit), axis=0)
                shape = self.shape
            self.precursor = np.zeros_like(self.deposit)
            self.surface_bool = np.zeros(shape, dtype=bool)
            self.semi_surface_bool = np.zeros(shape, dtype=bool)
            self.ghosts_bool = np.zeros(shape, dtype=bool)
            self.surface_neighbors_bool = np.zeros(shape, dtype=bool)
            self.temperature = np.zeros(shape, dtype=np.float64)
            self._rebuild_topology()
        logger.info('Loaded the VTK file!')
        self.precursor[self.precursor < 0] = 0
        if self.substrate_height == 0:
            self.substrate_height = (self.deposit == -2).nonzero()[0].max()
        self.temperature[self.deposit < 0] = self.room_temp
        self.initialized = True

    def create_from_parameters(self, cell_size=5, width=50, length=50, height=100, substrate_height=4, nr=0):
        """
        Frame initializer. Create a discretized simulation volume framework from parameters.

        :param cell_size: size of a cell, nm
        :param width: width of the simulation chamber (along X-axis), number of cells
        :param length: length of the simulation chamber (along Y-axis), number of cells
        :param height: height of the simulation chamber (along Z-axis), number of cells
        :param substrate_height: thickness of the substrate along Z-axis, number of cells
        :param nr: initial precursor density, normalized
        :return:
        """
        self.cell_size = cell_size
        self.deposit = np.zeros((height + substrate_height, length, width), dtype=np.float64)
        self.precursor = np.zeros_like(self.deposit)  # precursor array
        self.substrate_height = substrate_height
        self.nr = nr
        self.flush_structure()
        self.surface_bool = np.zeros_like(self.deposit, dtype=bool)
        self.semi_surface_bool = np.zeros_like(self.deposit, dtype=bool)
        self.surface_neighbors_bool = np.zeros_like(self.deposit, dtype=bool)
        self.ghosts_bool = np.zeros_like(self.deposit, dtype=bool)
        self.temperature = np.zeros_like(self.deposit)
        self.temperature[self.deposit < 0] = self.room_temp
        self._rebuild_topology(n_surface_neighbors=1)
        self.t = 0

        self.initialized = True

    def flush_structure(self):
        """
        Resets and prepares initial state of the grid.

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
            self.precursor[self.substrate_height, :,
            :] = 0.000001  # filling substrate surface with initial precursor density
        else:
            self.precursor[self.substrate_height, :,
            :] = self.nr  # filling substrate surface with initial precursor density
        self.deposit[...] = 0
        self.deposit[0:self.substrate_height, :, :] = -2

    def resize_structure(self, delta_z=0, delta_y=0, delta_x=0):
        """
        Resize the data framework. The specified lengths are attached along the axes.

        If any of the data is referenced, only a warning is shown and data is resized anyway.

        Changing dimensions along y and x axes should be done mindful, because if these require extension
        in the negative direction, that data has to be centered after the resizing.

        :param delta_z: increment for the z-axis, nm
        :param delta_y: increment for the y-axis, nm
        :param delta_x: increment for the x-axis, nm
        :return:
        """
        d_i, d_j, d_k = np.asarray([delta_z, delta_y, delta_x], dtype=int) // self.cell_size
        shape_new = (self.zdim + d_i, self.ydim + d_j, self.xdim + d_k)
        shape_old = self.shape
        slice_old = np.s_[0:shape_old[0], 0:shape_old[1], 0:shape_old[2]]
        # slice_new = np.s_[0:shape_new[0], 0:shape_new[1], 0:shape_new[2]]

        def resize_all(ref_check=True):
            if ref_check and sys.getrefcount(self.deposit) - 1 > 0:
                raise ValueError
            temp = np.copy(self.deposit)
            self.deposit = np.zeros(shape_new)
            self.deposit[slice_old] = temp[:]
            temp = np.copy(self.precursor)
            self.precursor = np.zeros(shape_new)
            self.precursor[slice_old] = temp[:]
            temp = np.copy(self.surface_bool)
            self.surface_bool = np.zeros(shape_new, dtype=bool)
            self.surface_bool[slice_old] = temp[:]
            temp = np.copy(self.semi_surface_bool)
            self.semi_surface_bool = np.zeros(shape_new, dtype=bool)
            self.semi_surface_bool[slice_old] = temp[:]
            temp = np.copy(self.surface_neighbors_bool)
            self.surface_neighbors_bool = np.zeros(shape_new, dtype=bool)
            self.surface_neighbors_bool[slice_old] = temp[:]
            temp = np.copy(self.ghosts_bool)
            self.ghosts_bool = np.zeros(shape_new, dtype=bool)
            self.ghosts_bool[slice_old] = temp[:]
            temp = np.copy(self.temperature)
            self.temperature = np.zeros(shape_new)
            self.temperature[slice_old] = temp[:]
            if d_j > 0 or d_k > 0:
                self.deposit[:self.substrate_height] = self.d_full_s
                # Recompute topology after lateral growth and then refill fresh surface cells.
                self._rebuild_topology()
                self.precursor[np.logical_and(self.precursor == 0, self.surface_bool)] = self.precursor.max()
        try:
            resize_all(True)
        except ValueError:
            logger.exception(f'Resized Structure arrays are referenced. \n'
                  f'Resizing is forced, references has to be updated manually.')
            try:
                resize_all(False)
            except Exception as e:
                logger.exception(f'An error occurred while resizing Structure arrays.')
                raise

    def fill_surface(self, nr: float):
        """
        Covers surface of the deposit with initial precursor density

        :param nr: initial precursor density
        :return:
        """
        self.precursor[self.surface_bool] = nr

    def max_z(self):
        """
        Get the height of the structure.
        :return: 0-axis index of the highest cell
        """
        return self.deposit.nonzero()[0].max()

    def save_to_vtk(self):
        import time
        grid = pv.ImageData()
        grid.dimensions = np.asarray([self.deposit.shape[2], self.deposit.shape[1],
                                      self.deposit.shape[0]]) + 1  # creating grid with the size of the array
        grid.spacing = (self.cell_size, self.cell_size, self.cell_size)  # assigning dimensions of a cell
        grid.cell_data["deposit"] = self.deposit.flatten()
        grid.save('Deposit_' + time.strftime("%H:%M:%S", time.localtime()))

    @property
    def substrate_height_abs(self):
        return self.substrate_height * self.cell_size

