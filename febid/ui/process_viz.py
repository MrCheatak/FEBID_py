import time
import timeit
import warnings
import datetime
from threading import Thread

import numpy as np
from PyQt5.QtWidgets import QMainWindow

from febid.libraries.vtk_rendering.VTK_Rendering import Render
from febid.febid_core import flag


class RenderWindow(QMainWindow):
    """
    Class for the visualization of the FEBID process
    """
    def __init__(self, process_obj, app=None):
        super().__init__()
        self.setWindowTitle("FEBID process")
        self.app = app
        # The object of the simulation process state
        self.process_obj = process_obj
        # Create a PyVista BackgroundPlotter and add it to the layout
        self.render = Render(app=app, show=True, cell_size=process_obj.structure.cell_size)

    def start(self, frame_rate=1):
        """
        Start the visualization of the process

        :param frame_rate: rendering speed in frames per second
        """
        self._visualize_process(frame_rate)

    def _visualize_process(self, frame_rate=1):
        """
        Visualize the process of the core deposition

        :param frame_rate: rendering speed in frames per second
        :return:
        """
        start_time = timeit.default_timer()
        pr = self.process_obj
        rn = self.render
        cmap = 'inferno'
        data = pr.structure.precursor
        mask = pr.structure.surface_bool
        displayed_data = 'precursor'
        # Initializing graphical monitoring
        pr.redraw = True
        ### For some reason adding a mesh to the scene in a child thread
        # causes a wglMakeCurrent failed in MakeCurrent() error in VTK.
        # Thus preparing the scene before starting the update loop.
        self._create_scene(data, mask, displayed_data, cmap)
        pr.redraw = False
        time.sleep(1)
        rn.p.show_axes()
        rn.p.show_grid()

        # Event loop
        def update(data, mask):
            while self.render.p.isVisible() and not flag:
                now = timeit.default_timer()
                if pr.redraw:
                    mask = pr.structure.surface_bool
                    data = pr.structure.precursor
                    cam_pos = rn.p.camera_position
                    self._create_scene(data, mask, displayed_data, cmap)
                    rn.p.camera_position = cam_pos
                    rn.p.show_axes()
                    rn.p.show_grid()
                    pr.redraw = False
                self.__update_graphical(data, mask, now - start_time)
                time.sleep(1 / frame_rate)

        thread = Thread(target=update, args=(data, mask))
        thread.start()
        return thread

    def _create_scene(self, data, mask, displayed_data='precursor', cmap='inferno'):
        """
        Create the initial visual representation of the process state

        :param data: 3D array of the data to visualize
        :param mask: 3D array of the mask for the data
        :param displayed_data: name of the displayed data
        :param cmap: colormap for the data
        """
        pr = self.process_obj
        rn = self.render
        try:
            # Clearing scene
            rn.y_pos = 5
            try:
                rn.p.button_widgets.clear()
            except Exception as e:
                print('Something went wrong while clearing widgets from the scene...')
                print(e.args)
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
        except Exception as e:
            print('An error occurred while creating the scene.')
            print(e.args)

    def __update_graphical(self, data, mask, time_spent):
        """
        Update the visual representation of the current process state

        :param rn: visual scene object
        :param pr: process object
        :param time_spent:
        :param displayed_data:
        :return:
        """
        pr = self.process_obj
        rn = self.render
        try:
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
            time_text = (f'Time: {time_real} \n'  # showing real time passed
                         f'Sim. time: {pr.t :.8f} s \n'  # showing simulation time passed
                         f'Speed: {speed:.8f} \n'
                         f'Av. growth rate: {growth_rate} nm^3/s \n'
                         f'Max. temperature: {max_T:.1f} K')
            stats_text = (f'Cells: {pr.n_filled_cells[i]} \n'  # showing total number of deposited cells
                          f'Height: {height} nm \n'
                          f'Volume: {total_V:.0f} nm^3')
            rn.p.actors['time'].SetText(2, time_text)
            rn.p.actors['stats'].SetText(3, stats_text)
            # Updating scene
            rn.update_mask(mask)
            try:
                _min = data[data > 0.00001].min()
            except ValueError:
                _min = 1e-8
            rn.p.update_scalar_bar_range(clim=[_min, data.max()])
        except Exception as e:
            if not rn.p.isVisible():
                print('The scene window was closed.')
            else:
                warnings.warn(f"Failed to redraw the scene.\n"
                              f"{e.args}")
                pr.redraw = True

    def isVisible(self):
        """
        Check if the plotting window is visible
        """
        return self.render.p.isVisible()

    def close(self):
        """
        Close the plotting window
        """
        self.render.p.close()
