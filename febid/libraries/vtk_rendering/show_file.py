"""
View the 3D-structure files produced by the simulation.
"""
import sys, os
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog as fd

from febid.libraries.vtk_rendering.VTK_Rendering import Render, read_field_data
from febid.Structure import Structure


def check_allowed_extensions(filenames, allowed_extensions=['vtk']):
    for filename in filenames:
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            return False
    return True


def ask_filenames(allowed_extensions=['vtk']):
    os.chdir('../../..')
    init_dir = os.getcwd()
    app = QApplication(sys.argv)
    extensions = ', '.join([f'*.{ext}' for ext in allowed_extensions])
    filenames = fd.getOpenFileName(None, 'Select the file to view', init_dir, f'VTK files ({extensions});;All files (*)')[0]
    app.quit()
    return filenames


def show_structure(filenames=None, **kwargs):
    if not filenames:
        filenames = ask_filenames()
    result = render_structure(filenames, **kwargs)
    return result


def render_structure(filenames, solid=True, deposit=True, precursor=True, surface=True, semi_surface=True, ghost=True):
    font_size = 12
    cam_pos = None
    if type(filenames) not in [list, tuple]:
        filenames = [filenames]
    for filename in filenames:
        print(f'Opening file {filename}')
        vtk_obj = pv.read(filename)
        print(f'Data arrays: {vtk_obj.array_names}')
        structure = Structure()
        structure.load_from_vtk(vtk_obj)
        d = structure.deposit
        d_er = 0
        p = structure.precursor
        p_er = 0
        if d.max() > 1:
            d_er += 1
        if d.min() < -2:
            d_er += 1
        if p.max() > 1:
            p_er += 1
        if p.min() < 0:
            p_er += 1
        print(f'Checking data.....', end='')
        if d_er + p_er:
            print(f'bad cells encountered in data arrays!')
            if d_er:
                print(f'\t Solid-cell data contains bad cells:')
                if d.max() > 1:
                    print(f'\t  Found {np.count_nonzero(d > 1)} cells above unity, maximum value is {d.max()}')
                if d.min() < -2:
                    print(f'\t  Found {np.count_nonzero(d < -2)} cells below zero, minimum value is {d.min()}')
            if p_er:
                print(f'\t Surface precursor density data contains bad cells:')
                if p.max() > 1:
                    print(f'\t  Found {np.count_nonzero(p > 1)} cells above unity, maximum value is {p.max()}')
                if p.min() < 0:
                    print(f'\t  Found {np.count_nonzero(p < 0)} cells below zero, minimum value is {p.min()}')
        else:
            print('ok!')
        t, sim_time, beam_position = read_field_data(vtk_obj)
        render = Render(structure.cell_size)
        cam_pos = render.show_full_structure(structure, True, solid, deposit, precursor, surface, semi_surface, ghost,
                                             t, sim_time, beam_position, cam_pos=cam_pos)
    return cam_pos


if __name__ == '__main__':
    filenames = None
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
        if not check_allowed_extensions(filenames):
            raise ValueError('Only .vtk files are allowed!')
    show_structure(filenames, solid=True, deposit=True, precursor=True, surface=True, semi_surface=True, ghost=True)
