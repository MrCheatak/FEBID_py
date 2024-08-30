import datetime
import sys
import time
import timeit
import warnings
from threading import Thread

import numpy as np
import pyvista as pv
from vtk import vtkDataSetAttributes
from PyQt5.QtWidgets import QMainWindow

from febid.Process import Process
from febid.libraries.vtk_rendering.VTK_Rendering import Render


class RenderWindow(QMainWindow):
    def __init__(self, process_obj, app=None):
        super().__init__()
        self.setWindowTitle("FEBID process")
        self.app = app

        # Create the main widget
        # main_widget = QWidget()
        # self.setCentralWidget(main_widget)

        # Set up the layout
        # self.layout = QVBoxLayout(main_widget)
        self.process_obj = process_obj
        # Create a PyVista BackgroundPlotter and add it to the layout
        self.render = Render(app=app, show=True, cell_size=process_obj.structure.cell_size)
        # self.plotter = self.render.p
        # self.arrow = self.render.arrow
        # self.plotter = BackgroundPlotter(app=app, show=True)
        # self.arrow = None
        # self.render.add_toolbars()
        # layout.addWidget(self.render.interactor)
        # Create the 3D pyramid from a numpy array
        # self.create_pyramid()
        # self.render.show()

    def create_voxelized_pyramid(self):
        # Define the size of the 3D array (e.g., 11x11x11 for a 10-layer pyramid)
        size = 100
        pyramid_array = np.zeros((size, size, size), dtype=int)

        # Create the voxelized pyramid by assigning values to the array
        for layer in range(size // 2 + 1):
            value = layer + 1  # Assign a value from 1 to 10 based on the layer
            pyramid_array[layer:size - layer, layer:size - layer, layer] = value
        mask = np.zeros_like(pyramid_array, dtype=np.uint8)
        mask[pyramid_array == 0] = vtkDataSetAttributes.HIDDENCELL

        # Convert the numpy array to a PyVista UnstructuredGrid
        self.grid = grid = pv.UniformGrid()
        grid.dimensions = np.array(pyramid_array.shape) + 1
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)
        grid.cell_data["values"] = pyramid_array.flatten(order="F")
        grid.cell_data[vtkDataSetAttributes.GhostArrayName()] = mask.flatten(order="F")

        # Extract the non-zero voxels
        # self.grid = thresholded = grid.threshold([1, 10], scalars="values")

        def disco():
            while self.plotter.isVisible():
                grid = self.grid.cell_data["values"]
                rand_pos = np.random.randint(0, grid.size)
                rand_val = np.random.randint(0, grid.max())
                grid[rand_pos] = rand_val
                mask = self.grid.cell_data[vtkDataSetAttributes.GhostArrayName()]
                mask[rand_pos] = vtkDataSetAttributes.HIDDENCELL if rand_val == 0 else 0
                rand_pos_z = rand_pos // (pyramid_array.shape[1] * pyramid_array.shape[2])
                rand_pos_y = rand_pos % (pyramid_array.shape[1] * pyramid_array.shape[2]) // pyramid_array.shape[2]
                rand_pos_x = rand_pos % pyramid_array.shape[2]
                if self.arrow is None:
                    self.arrow = self.plotter.add_arrows(np.array([0, 0, 10]), np.array([0, 0, -10]), mag=1,
                                                         color="red")
                self.arrow.SetPosition(rand_pos_x, rand_pos_y, rand_pos_z)

                time.sleep(0.02)

        def shrink():
            for i in range(50):
                self.grid.shrink(1 - i * 0.01)
                # self.grid.spacing = tuple([i * 0.9 for i in self.grid.spacing])
                time.sleep(0.1)

        # Plot the voxelized pyramid with a colormap
        self.plotter.add_mesh(self.grid, cmap="hsv", opacity=0.7, name="Pyramid")
        self.grid = self.plotter.mesh
        self.plotter.add_axes()
        # self.plotter.add_bounding_box()
        self.plotter.show_grid()
        thread = Thread(target=disco)
        thread.start()
        return thread

    def start(self):
        # thread = Thread(target=self.visualize_process_test, args=(self.render, self.process_obj))
        # thread = Thread(target=self.create_voxelized_pyramid)
        # thread.start()
        # thread = self.create_voxelized_pyramid()
        # return thread
        # self.visualize_process_test(self.render, self.process_obj)
        self.visualize_process(self.process_obj, self.render, frame_rate=1, displayed_data='precursor')

    def visualize_process(self, pr: Process, rn: Render, frame_rate=1, displayed_data='precursor', **kwargs):
        """
        A daemon process function to manage statistics gathering and graphics update.

        :param pr: object of the core deposition process
        :param run_flag: thread synchronization object, allows to stop visualization when simulation concludes
        :param frame_rate: redrawing delay
        :param displayed_data: name of the displayed data. Options: 'precursor', 'deposit', 'temperature', 'surface_temperature'
        :return:
        """
        start_time = timeit.default_timer()
        # Initializing graphical monitoring

        # rn = Render(pr.structure.cell_size, app=app)
        # rn.p.clear()
        pr.redraw = True
        now = timeit.default_timer()
        data = pr.structure.precursor
        mask = pr.structure.surface_bool
        cmap = 'inferno'
        displayed_data = 'precursor'
        self.create_scene(pr, rn, cmap, data, displayed_data, mask)
        pr.redraw = False
        rn.p.show_axes()
        rn.p.show_grid()


        # self.update_graphical(rn, pr, now, displayed_data)
        # Event loop
        def update():
            while self.render.p.isVisible():
                now = timeit.default_timer()
                return_val = self.update_graphical(rn, pr, now - start_time, displayed_data)
                # if return_val or not rn.p.render_window.IsCurrent():
                #     print('Visualization window closed, stopping rendering.')
                #     return
                time.sleep(frame_rate)
                if pr.redraw:
                    mask = pr.structure.surface_bool
                    data = pr.structure.precursor
                    cam_pos = rn.p.camera_position
                    self.create_scene(pr, rn, cmap, data, displayed_data, mask)
                    rn.p.camera_position = cam_pos
                    rn.p.show_axes()
                    rn.p.show_grid()
                    pr.redraw = False
        thread = Thread(target=update)
        thread.start()
        # rn.p.close()
        # print('Closing rendering.')
        # visualize_result(pr, now - start_time, displayed_data)
        # return now - start_time

    def create_scene(self, pr: Process, rn: Render, cmap, data, displayed_data, mask):
        try:
            # Clearing scene
            rn.y_pos = 5
            try:
                rn.p.button_widgets.clear()
            except Exception as e:
                print('Something went wrong while clearing widgets from the scene...')
            rn.p.clear()
            # Putting an arrow to indicate beam position
            start = np.array([0, 0, 100]).reshape(1, 3)  # position of the center of the arrow
            end = np.array([0, 0, -100]).reshape(1, 3)  # direction and resulting size
            rn.arrow = rn.p.add_arrows(start, end, color='tomato')
            rn.arrow.SetPosition(pr.x0, pr.y0,
                                 pr.max_z * pr.cell_size + 10)  # relative to the initial position
            # Plotting data
            rn.add_3Darray(data, opacity=1, scalar_name=displayed_data,
                           button_name=displayed_data, show_edges=True, cmap=cmap)
            scalar = rn.p.mesh.active_scalars_name
            rn.p.mesh[scalar] = data.reshape(-1)
            rn.update_mask(mask)
            rn.p.add_text('.', position='upper_left', font_size=12, name='time')
            rn.p.add_text('.', position='upper_right', font_size=12, name='stats')
            # rn.show(interactive_update=True, cam_pos=[(206.34055818793468, 197.6510638707941, 100.47106597548205),
            #                                           (0.0, 0.0, 0.0),
            #                                           (-0.23307751464125356, -0.236197909312718,
            #                                            0.9433373838690787)])
        except Exception as e:
            print('An error occurred while redrawing the scene.')
            print(e.args)

    def update_graphical(self, rn: Render, pr: Process, time_spent, displayed_data='precursor', update=True):
        """
        Update the visual representation of the current process state

        :param rn: visual scene object
        :param pr: process object
        :param time_spent:
        :param displayed_data:
        :param update:
        :return:
        """
        try:
            if displayed_data == 'precursor':
                data = pr.structure.precursor
                mask = pr.structure.surface_bool
                cmap = 'plasma'
            if displayed_data == 'deposit':
                data = pr.structure.deposit
                mask = pr.structure.surface_bool
                cmap = 'viridis'
            if displayed_data == 'temperature':
                data = pr.structure.temperature
                mask = pr.structure.deposit < 0
                cmap = 'inferno'
            if displayed_data == 'surface_temperature':
                data = pr.surface_temp
                mask = pr.structure.surface_bool
                cmap = 'inferno'
            if displayed_data not in ['precursor', 'deposit', 'temperature', 'surface_temperature']:
                raise AttributeError(f'Dataset {displayed_data} is not available for rendering.')
            # if pr.redraw:
            #     try:
            #         # Clearing scene
            #         rn.y_pos = 5
            #         # try:
            #         #     rn.p.button_widgets.clear()
            #         # except Exception as e:
            #         #     print('Something went wrong while clearing widgets from the scene...')
            #         # rn.p.clear()
            #         # Putting an arrow to indicate beam position
            #         # start = np.array([0, 0, 100]).reshape(1, 3)  # position of the center of the arrow
            #         # end = np.array([0, 0, -100]).reshape(1, 3)  # direction and resulting size
            #         # rn.arrow = rn.p.add_arrows(start, end, color='tomato')
            #         # rn.arrow.SetPosition(pr.x0, pr.y0,
            #         #                      pr.max_z * pr.cell_size + 10)  # relative to the initial position
            #         # Plotting data
            #         rn.add_3Darray(data, opacity=1, scalar_name=displayed_data,
            #                        button_name=displayed_data, show_edges=True, cmap=cmap)
            #         scalar = rn.p.mesh.active_scalars_name
            #         rn.p.mesh[scalar] = data.reshape(-1)
            #         rn.update_mask(mask)
            #         # rn.p.add_text('.', position='upper_left', font_size=12, name='time')
            #         # rn.p.add_text('.', position='upper_right', font_size=12, name='stats')
            #         # rn.show(interactive_update=True, cam_pos=[(206.34055818793468, 197.6510638707941, 100.47106597548205),
            #         #                                           (0.0, 0.0, 0.0),
            #         #                                           (-0.23307751464125356, -0.236197909312718,
            #         #                                            0.9433373838690787)])
            #     except Exception as e:
            #         print('An error occurred while redrawing the scene.')
            #         print(e.args)
            #     rn.meshes_count += 1
            #     pr.redraw = False

            # Changing arrow position
            x, y, z = rn.arrow.GetPosition()
            z_pos = pr.structure.deposit[:, int(pr.y0 / pr.cell_size), int(pr.x0 / pr.cell_size)].nonzero()[
                        0].max() * pr.cell_size
            if z_pos != z or pr.y0 != y or pr.x0 != x:
                rn.arrow.SetPosition(pr.x0, pr.y0, z_pos + 30)  # relative to the initial position
            # Calculating values to indicate
            pr.n_filled_cells.append(pr.filled_cells)
            i = len(pr.n_filled_cells) - 1
            time_real = str(datetime.timedelta(seconds=int(time_spent)))
            speed = pr.t / time_spent
            height = (pr.max_z - pr.substrate_height) * pr.structure.cell_size
            total_V = int(pr.dep_vol)
            delta_t = pr.t - pr.t_prev
            delta_V = total_V - pr.vol_prev
            if delta_t == 0 or delta_V == 0:
                growth_rate = pr.growth_rate
            else:
                growth_rate = delta_V / delta_t
                growth_rate = int(growth_rate)
                pr.growth_rate = growth_rate
            pr.t_prev += delta_t
            pr.vol_prev = total_V
            max_T = pr.structure.temperature.max()
            # Updating displayed text
            rn.p.actors['time'].SetText(2,
                                        f'Time: {time_real} \n'  # showing real time passed
                                        f'Sim. time: {(pr.t):.8f} s \n'  # showing simulation time passed
                                        f'Speed: {speed:.8f} \n'
                                        f'Av. growth rate: {growth_rate} nm^3/s \n'
                                        f'Max. temperature: {max_T:.1f} K')
            rn.p.actors['stats'].SetText(3,
                                         f'Cells: {pr.n_filled_cells[i]} \n'  # showing total number of deposited cells
                                         f'Height: {height} nm \n'
                                         f'Volume: {total_V:.0f} nm^3')
            # Updating scene
            rn.update_mask(mask)
            try:
                _min = data[data > 0.00001].min()
            except ValueError:
                _min = 1e-8
            rn.p.update_scalar_bar_range(clim=[_min, data.max()])

            # if update:
            #     return_val = rn.update()
            #     return return_val
        except Exception as e:
            warnings.warn(f"Failed to redraw the scene.\n"
                          f"{e.args}")
            pr.redraw = True
        return 0

    def visualize_process_test(self, rn: Render, pr: Process):
        data = pr.structure.precursor
        displayed_data = 'precursor'
        mask = pr.structure.surface_bool
        cmap = 'inferno'
        rn.add_3Darray(data, opacity=1, scalar_name=displayed_data,
                       button_name=displayed_data, show_edges=True, cmap=cmap)
        scalar = rn.p.mesh.active_scalars_name
        rn.p.mesh[scalar] = data.reshape(-1)
        rn.update_mask(mask)
        rn.p.show_axes()
        rn.p.show_grid()

        def update():
            while self.render.p.isVisible():
                if pr.redraw:
                    mask = pr.structure.surface_bool
                    self.create_scene(pr, rn, cmap, data, displayed_data, mask)
                    pr.redraw = False
                try:
                    grid = self.render.p.mesh.cell_data["precursor"]
                    rand_pos = np.random.randint(0, grid.size)
                    rand_val = np.random.random() * 0.16
                    grid[rand_pos] = rand_val
                    mask = self.render.p.mesh.cell_data[vtkDataSetAttributes.GhostArrayName()]
                    mask[rand_pos] = vtkDataSetAttributes.HIDDENCELL if rand_val == 0 else 0
                    rand_pos_z = rand_pos // (data.shape[1] * data.shape[2])
                    rand_pos_y = rand_pos % (data.shape[1] * data.shape[2]) // data.shape[2]
                    rand_pos_x = rand_pos % data.shape[2]
                except Exception as e:
                    if not self.render.p.isVisible():
                        break
                    else:
                        print(f"Failed to update the scene.\n{e.args}")

        time.sleep(0.02)

        ### For some reason creating a thread with the current function (visualize_process)
        # causes a wglMakeCurrent failed in MakeCurrent() error in VTK.
        # Thus starting a thread from within the function instead
        thread = Thread(target=update)
        thread.start()
        return thread
