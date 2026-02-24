"""
Multi-Layerd Continuous Cellular Automata
"""
import numpy as np

from febid.slice_trics import get_3d_slice, get_boundary_indices


class MultiLayerdCellCellularAutomata:
    """
    Multi-Layerd Continuous Cellular Automata class.

    A type of Cellular Automata where cells have several attributes and may take continuous values (0..1).

    This class implements the multi-Layerd Continuous cellular automata algorithm.
    The rules are currently embedded in th logic of the get_converged_configuration() method.
    """
    def __init__(self):
        self.__get_utils()

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
            # Prepare some of the views considering cell position and boundaries
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

    @staticmethod
    def stencil_3d(grid_out, grid_in):
        """Accumulate 6-neighborhood + edge-safe boundaries into ``grid_out``."""
        grid_out[:, :, :-1] += grid_in[:, :, 1:]
        grid_out[:, :, -1] += grid_in[:, :, -1]
        grid_out[:, :, 1:] += grid_in[:, :, :-1]
        grid_out[:, :, 0] += grid_in[:, :, 0]
        grid_out[:, :-1, :] += grid_in[:, 1:, :]
        grid_out[:, -1, :] += grid_in[:, -1, :]
        grid_out[:, 1:, :] += grid_in[:, :-1, :]
        grid_out[:, 0, :] += grid_in[:, 0, :]
        grid_out[:-1, :, :] += grid_in[1:, :, :]
        grid_out[-1, :, :] += grid_in[-1, :, :]
        grid_out[1:, :, :] += grid_in[:-1, :, :]
        grid_out[0, :, :] += grid_in[0, :, :]

    def compute_surface_topology(self, deposit, d_full_d=-1.0, d_full_s=-2.0, out=None):
        """
        Compute surface mask from deposit grid.

        Returns a boolean array where gas-side surface cells are True.
        """
        if out is None:
            out = np.zeros_like(deposit, dtype=bool)
        else:
            out[...] = False

        positive = deposit >= 0
        grid = np.copy(deposit)
        grid[grid > 0] = 0
        grid1 = np.copy(deposit)
        grid1[grid1 > 0] = 0
        self.stencil_3d(grid, grid1)
        grid /= 7
        grid[np.abs(grid - d_full_d) < 1e-7] = 0
        grid[np.abs(grid - d_full_s) < 1e-7] = 0
        combined = np.abs(grid) > 0
        out[positive & combined] = True
        return out

    def compute_semi_surface_topology(self, deposit, surface_bool, out=None):
        """
        Compute semi-surface mask from deposit and surface masks.
        """
        if out is None:
            out = np.zeros_like(deposit, dtype=bool)

        grid = np.zeros_like(deposit)
        self.stencil_3d(grid, surface_bool)
        grid[deposit != 0] = 0
        grid[surface_bool] = 0
        # Preserve legacy behavior from Structure.define_semi_surface():
        # only mark True where condition matches, without explicit full reset.
        out[grid >= 2] = True
        return out

    def compute_surface_neighbors(self, deposit, surface_bool, n=0, out=None):
        """
        Compute nearest-neighbor shell of solid cells around surface cells.

        If ``out`` is provided, the computed values are written to its inner volume
        (excluding 1-cell border) to preserve original in-place update behavior.
        """
        if out is None:
            out = np.zeros_like(deposit, dtype=bool)

        grid = np.zeros_like(deposit)
        self.stencil_3d(grid, surface_bool)
        grid[grid > 1] = 1

        grid1 = np.zeros_like(grid)
        self.stencil_3d(grid1, grid)
        grid1[grid > 0] = 0
        grid1[grid1 < 2] = 0
        grid1[grid1 >= 2] = 1
        grid[grid1 > 0] = 1

        i = 1
        loop = 0 if n == 0 else 1
        while True:
            i += 1
            if loop:
                if i > n:
                    break
            elif grid[deposit < 0].min() > 0:
                break
            grid1 = np.zeros_like(grid)
            self.stencil_3d(grid1, grid)
            grid1[grid != 0] = 0
            grid1[grid1 > 0] = 1
            grid[grid1 > 0] = i
            grid1 = np.zeros_like(grid)
            self.stencil_3d(grid1, grid)
            grid1[grid > 0] = 0
            grid1[grid1 < i * 2] = 0
            grid1[grid1 >= i * 2] = 1
            grid[grid1 > 0] = i

        grid[deposit > -1] = 0
        out_view = out[1:-1, 1:-1, 1:-1]
        grid_view = grid[1:-1, 1:-1, 1:-1]
        out_view[...] = False
        out_view[grid_view > 0] = True
        return out

    def compute_ghost_shell(self, surface_bool, semi_surface_bool, out=None):
        """
        Compute ghost shell wrapping surface and semi-surface.
        """
        roller = np.logical_or(surface_bool, semi_surface_bool)
        if out is None:
            out = np.copy(roller)
        else:
            out[...] = roller

        self.stencil_3d(out, roller)
        out[roller] = False
        return out

    def check_neighbors(self, arr1, arr2):
        from scipy.ndimage import binary_dilation

        # Define a connectivity structure (3x3x3 cube around each cell)
        struct = np.array([[[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                           [[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]],
                           [[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]]], dtype=bool)

        # Dilate array arr1 to find its neighborhood
        arr1_dilated = binary_dilation(arr1, structure=struct)

        # Identify the cells in b that do not have a neighboring True cell in a
        no_neighbors = arr2 & ~arr1_dilated

        if no_neighbors.any():
            print("Indices in array b with no True neighbor in array a:")
            print(no_neighbors)
            return True
        return False

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


def visualize_kernel(*arrays):
    import pyvista as pv
    from febid.libraries.vtk_rendering.VTK_Rendering import SetVisibilityCallback

    # Step 1: Create a ImageData
    grids = []
    for arr in arrays:
        grid = pv.ImageData()
        grid.dimensions = np.array(arr.shape) + 1
        grid.cell_data["values"] = arr.flatten(order="F")  # Flatten the array in Fortran order
        grid1 = grid.threshold(0.000001, method='upper')
        grid2 = grid.threshold(-0.00001, method='lower')
        grid = grid1 + grid2
        grids.append(grid)

    # Step 2: Add the grid to the Plotter
    plotter = pv.Plotter()
    meshes = []
    # colors = list(pv.hexcolors.keys())
    colors_dict = {
    'Red': '#D32F2F',
    'Blue': '#1976D2',
    'Green': '#388E3C',
    'Yellow': '#FBC02D',
    'Purple': '#8E24AA',
    'Orange': '#F57C00',
    'Light Blue': '#0288D1',
    'Dark Purple': '#7B1FA2',
    'Pink': '#C2185B',
    'Teal': '#00796B'
    }
    colors = list(colors_dict.values())
    i = 0
    for grid in grids:
        try:
            color = colors[i]
            colors.remove(color)
            mesh = plotter.add_mesh(grid, show_edges=True, opacity=1, color=color, label=f'Array {i}')
            i += 1
            meshes.append(mesh)
        except ValueError:
            print(f"Array {i} could not be added due to being empty or invalid.")

    # Step 3: Add a checkbox widget to toggle visibility
    toggles = []
    i = 0
    for mesh in meshes:
        toggle = SetVisibilityCallback(plotter, mesh)
        toggles.append(toggle)
        plotter.add_checkbox_button_widget(toggle, value=True, position=(10, 10+i*30), size=25)
        plotter.add_text(f"Array {i}", position=(40, 10+i*30), font_size=18)
        i += 1

    # Step 4: Show the plot
    plotter.show_grid()
    plotter.show_axes()
    plotter.show_axes_all()
    plotter.show()


if __name__ == '__main__':
    bool_array = np.zeros((5, 5, 5), dtype=bool)
    bool_array[1:2, 1:4, :] = True
    bool_array1 = np.zeros((5, 5, 5), dtype=bool)
    bool_array1[1:4, 1:2, :] = True
    visualize_kernel(bool_array, bool_array1)
