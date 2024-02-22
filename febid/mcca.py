"""
Mixed Cell Cellular Automata
"""
import numpy as np

from .slice_trics import get_3d_slice, get_boundary_indices


class MixedCellCellularAutomata:
    """
    Mixed Cell Cellular Automata class.

    A type of Cellular Automata where cells have several attributes.

    This class implements the mixed cell cellular automata algorithm.
    The rules are currently embedded in th logic of the get_converged_configuration() method.
    """
    def __init__(self):
        self.__get_utils()
        pass

    def get_converged_configuration(self, cell, deposit, surface, semi_surface, ghosts):
        """
        Define a static converged cell configuration based on the change in the provided cell.


        The center cell contains the change that is the single constant attribute driving all the changes.
        All cells around it and their attributes are subject to change if they violate the rules.

        :param cell: center cell
        :param deposit: deposit array, should contain the change
        :param surface: surface array
        :param semi_surface: semi-surface array
        :param ghosts: ghost cells array
        :return: a slice that was processed, converged surface, semi-surface and ghost cells arrays
        """

        # What here actually done is marking the filled cell as a solid and a ghost cell and then updating surface,
        # semi-surface, ghosts and precursor to describe the surface geometry around the newly filled cell.
        # The approach is cell-centric, which means all the surroundings are processed
        def get_init_slice(cell, shape, size=1):
            neibs_sides, neibs_edges = self.__neibs_sides, self.__neibs_edges
            neighbs_1st, cell_new = get_3d_slice(cell, shape, size)
            dummy = deposit[neighbs_1st]
            neighbs_mask, _ = get_3d_slice(cell_new, dummy.shape, size)
            neibs_edges = neibs_edges[neighbs_mask]
            neibs_sides = neibs_sides[neighbs_mask]
            return neibs_sides, neibs_edges

        # Instead of using classical conditions, boolean arrays are used to select elements
        # First, a condition array is created, that picks only elements that satisfy conditions
        # Then this array is used as index

        # Creating a view with the 1st nearest neighbors to the deposited cell
        # Taking into account cases when the cell is located at the edge and at sides:
        neibs_sides, neibs_edges = get_init_slice(cell, deposit.shape, 1)

        ### Preparing appropriate views of the processed arrays
        # Creating a view with the 2nd nearest neighbors to the deposited cell or a 5x5x5 view
        neighbors_2nd, cell_new2 = get_3d_slice(cell, deposit.shape, 2)
        deposit_view = deposit[neighbors_2nd]
        surface_view = surface[neighbors_2nd].copy()
        semi_surface_view = semi_surface[neighbors_2nd].copy()
        ghosts_view = ghosts[neighbors_2nd].copy()
        # Creating a view with the 1st nearest neighbors to the deposited cell or a 3x3x3 view
        slice_1 = get_boundary_indices(cell_new2, deposit_view.shape, 1)
        deposit_kern = deposit_view[slice_1]
        surf_kern = surface_view[slice_1]
        semi_s_kern = semi_surface_view[slice_1]
        ghosts_kern = ghosts_view[slice_1]

        ### Processing cells according to the cell evolution rules
        # Creating condition array for surface cells
        condition1 = np.logical_and(deposit_kern == 0,
                                    neibs_sides)  # True for elements that are not deposited and are side neighbors
        # Updating main arrays
        semi_s_kern[condition1] = False
        surf_kern[condition1] = True
        # Creating condition array for semi-surface cells
        condition2 = np.logical_and(np.logical_and(deposit_kern == 0, surf_kern == 0),
                                    neibs_edges)  # True for elements that are not deposited, not surface cells and are edge neighbors
        semi_s_kern[condition2] = True
        ghosts_kern[...] = False
        # Creating condition array for ghost cells
        condition4 = np.logical_and(surface_view == 0,
                                    semi_surface_view == 0)  # True for elements that are neither surface nor semi-surface cells
        ghosts_view[condition4] = True

        # ghosts[cell] = True  # deposited cell belongs to ghost shell
        # surface[cell] = False  # deposited cell is no longer a surface cell

        return neighbors_2nd, surface_view, semi_surface_view, ghosts_view

    def __get_utils(self):
        # Kernels for choosing cells
        self.__neibs_sides = np.array([[[0, 0, 0],  # chooses side neighbors
                                        [0, 1, 0],
                                        [0, 0, 0]],
                                       [[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]],
                                       [[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]]])
        self.__neibs_edges = np.array([[[0, 1, 0],  # chooses edge neighbors
                                        [1, 0, 1],
                                        [0, 1, 0]],
                                       [[1, 0, 1],
                                        [0, 0, 0],
                                        [1, 0, 1]],
                                       [[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]]])
