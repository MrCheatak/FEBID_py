"""
View the 3D-structure files produced by the simulation.
"""
import sys, os
import numpy as np
import pyvista as pv
from tkinter import filedialog as fd
from febid.libraries.vtk_rendering import VTK_Rendering as vr
from febid.Structure import Structure


def show_structure(filenames, solid=True, deposit=True, precursor=True, surface=True, semi_surface=True, ghost=True):

    font_size = 12
    cam_pos = None
    if type(filenames) not in [list, tuple]:
        filenames = [filenames]
    for filename in filenames:
        print(f'Opening file {filename}')
        vtk_obj = pv.read(filename)
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
        if d_er+p_er:
            print(f'bad cells encountered in data arrays!')
            if d_er:
                print(f'\t Solid-cell data contains bad cells:')
                if d.max() > 1:
                    print(f'\t  Found {np.count_nonzero(d>1)} cells above unity, maximum value is {d.max()}')
                if d.min() < -2:
                    print(f'\t  Found {np.count_nonzero(d<-2)} cells below zero, minimum value is {d.min()}')
            if p_er:
                print(f'\t Surface precursor density data contains bad cells:')
                if p.max() > 1:
                    print(f'\t  Found {np.count_nonzero(p>1)} cells above unity, maximum value is {p.max()}')
                if p.min() < 0:
                    print(f'\t  Found {np.count_nonzero(p<0)} cells below zero, minimum value is {p.min()}')
        else:
            print('ok!')
        t, sim_time, beam_position = vr.read_field_data(vtk_obj)
        render = vr.Render(structure.cell_dimension)
        cam_pos = render.show_full_structure(structure, True, solid, deposit, precursor, surface, semi_surface, ghost, t, sim_time, beam_position, cam_pos=cam_pos)


if __name__ == '__main__':
    # show = Thread(target=show_structure)
    # show.start()
    filenames = None
    try:
        filename = sys.argv[1]
    except Exception as e:
        print(e.args)
        if not filenames:
            os.chdir('../../..')
            init_dir = os.getcwd()
            filenames = fd.askopenfilename(multiple=True)

    show_structure(filenames, True, True, True, True, True, True)
