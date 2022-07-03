import sys
import numpy as np
import pyvista as pv
from tkinter import filedialog as fd
from source.libraries.vtk_rendering import VTK_Rendering as vr
from source.Structure import Structure


def show_structure(solid=True, deposit=True, precursor=True, surface=True, semi_surface=True, ghost=True):

    filenames = fd.askopenfilename(multiple=True)
    font_size = 12
    for filename in filenames:
        print(f'Opening file {filename}')
        structure = Structure()
        structure.load_from_vtk(pv.read(filename))
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
        render = vr.Render(structure.cell_dimension)
        render.show_full_structure(structure, solid, deposit, precursor, surface, semi_surface, ghost)


if __name__ == '__main__':
    # show = Thread(target=show_structure)
    # show.start()
    filename = None
    try:
        filename = sys.argv[1]
    except Exception as e:
        print(e.args)
        pass
    show_structure(True, True, True, True, True, True)
