#!/usr/bin/env python3
import pyvista as pv
import numpy as np
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    print('Usage: python3 <fname (without_x, _y, _z and extension)> <z-axis scale factor> <projection (y/n)>')
    print('Exit.')
    exit(0)

fname = sys.argv[1]
scale_factor = float(sys.argv[2])
if sys.argv[3] == 'y' or sys.argv[3] == 'Y':
    proj = True
else:
    proj = False

x = np.load(fname + '_x.npy')
y = np.load(fname + '_y.npy')
z = np.load(fname + '_z.npy')*scale_factor

# make a mesh from point cloud
points = np.c_[x.reshape(-1), y.reshape(-1), z.reshape(-1)]
cloud = pv.PolyData(points)
surf = cloud.delaunay_2d()

# add scalar value to point positions so that plot shows z values in color-coded form
surf[fname] = z.reshape(-1)

# make scalebar look nice and being moveable in plot window
sargs = dict(
    #title_font_size=14,
    label_font_size=16,
    n_labels=5,
    fmt='%.1f',
    font_family='times',
    interactive=True
)

# instantiate plotter and add mesh
p = pv.Plotter()
pv.set_plot_theme("ParaView") # others are: document, default, night
p.add_mesh(surf, stitle='', scalar_bar_args=sargs) # without title over scalar bar

# if projection desired, create it and it to plotter
if proj == True:
    origin = surf.center
    origin[-1] -= surf.length/3.0 # shift projection 30% below max z values to negative
    projected = surf.project_points_to_plane(origin=origin)
    p.add_mesh(projected)

# finally show
p.show()
