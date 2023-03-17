Interface
===============

Control panel:
""""""""""""""""""
Here is the list of all settings available on the control panel.

**Load last session** – initially unchecked. Checking it will create a session file at the location, from where the
Python command was executed. Lunching from the same location again will load the saved settings.
This file can be as well be manually edited to change the settings preset,
i.e for a `series of simulation runs <series.html>`_.

In `File` menu, `Open session` option allows loading a session file from a different directory.

**Simulation volume:**

* VTK file – allows specifying a VTK-type file (.vtk) that contains a predefined 3D structure to be used in the simulation.
* Parameters – create a fresh simulation volume with specified dimensions and voxel(cell) size with a substrate at the bottom.
* Auto – to be used only when using a stream-file. The dimensions of the simulation volume will be defined automatically to encapsulate the printing path with a sufficient margin.
* Width, length, height – simulation volume dimensions, nm.
* Cell size – edge length of a cubic cell or voxel that is the simulation volume is divided into. The smallest volume fraction of the simulation volume
* Substrate height – the thickness of a substrate layer at the bottom of the simulation volume. By default, it has properties of gold. It should be a multiple of `Cell size`.

    Volume dimensions have to be set only if `Parameters` is chosen. When `VTK file` is chosen, they are set automatically from the file, as well as `Substrate height` and `Cell size`.
    For the `Auto` option, only `Cell size` and `Substrate height` have to be specified.


**Pattern:**

* Simple pattern – allows generation of a path with one of the available simple shapes:
  Available shapes: `point, line, square, rectangle and circle.`

* x, y – parameters of the selected shape. Position for a point, `length` for a line, `edge length` for a square and rectangle, `radius` for a circle. Except for the point, all shapes are placed in the center. Keep in mind, that the printing path should be inside the borders of the simulation volume.
* Pitch – the shape contour is divided into discrete points, which a beam visits in a sequence. This parameter defines the distance between two consequent positions of the beam along it's path.
* Dwell time – the amount of time the beam sits or dwells at a single position.
* Repeats – the number of times the pattern defined by shape, dwell time and pitch has to be repeated.
* Stream file – allows specifying a special stream-file, that defines a more complex printing path. It represents a sequence of beam positions with dwell times. This option requires `Auto` to be chosen in the `Simulation volume` section.
* HFW – Half Field Width sets the scale of the structure. Because pattern files are resolved in pixels, they have to be related to the actual distance units. This relation is provided by the magnification or HFW.

**Beam and precursor:**

* `Settings <settings_file.html>`_ – a YAML (.yml) file with beam parameters and precursor flux to be specified here.
* `Precursor parameters <precursor_file.html>`_ – a YAML (.yml) file with precursor(printing material) and deponat(printed material) properties.
* Temperature tracking – check to enable calculation of the temperature profile and temperature dependency of the precursor coverage.

.. warning:: Corresponding precursor parameters have to be included in the parameter file in order for the temperature tracking to work.


.. note:: If a loaded 3D structure does not have temperature profile data, it will be added automatically.

**Save file:**

* Save simulation data – check to regularly save statistical data of the simulation including time passed, deposition time passed and volume filled. The save interval is specified in the next field.
* Save structure snapshots – check to regularly save the state of the deposition process. The save interval is specified in the next field.


VTK file option:
    Read the volume with a structure from a .vtk file. The file can be a regular .vtk file with a structure in it
    or it can be a file produced by the simulation (by checking Save structure snapshots). If an arbitrary .vtk file is
    specified, it has to be a UniformGrid, have a cubic cell (equal spacings) and have a single cell array.

Graphical:
    When 'Show the process' is checked to view the simulation process in real-time, a window with a 3D scene will open.
    Refresh rate is set to 0.5 s, thus it may be slow to interact with.
    The scene is interactive, meaning it can be zoomed by scrolling, rotated  with a mouse, moved around (with Shift
    pressed) and focused at the cursor by pressing 'F'.
    The coloring and the corresponding scale represents the concentration of the precursor at the surface.
    Thus, the 3D object displayed is not the solid structure itself, but it's whole surface, that follows the shape of
    the solid 3D object.


Saving simulation results:
""""""""""""""""""""""""""""
When any of the ‚Save…‘ options are checked a new folder for the current simulation is created.
The intervals of statistics records and snapshots saving refer to the deposition time.

`Save simulation data` creates an .xlsx Excel file and records simulation setup information and statistical data.
Simulation setup is recorded before the simulation start and includes Precursor/deposit properties,
Beam/precursor flux settings and Simulation volume attributes, which are saved on separate sheets.
Statistical data is then recorded repeatedly during the simulation and includes the following default columns:

•	Precise time of record (real)
•	Time passed (real), s
•	Time passed (deposition/experiment), s
•   Current lowest precursor coverage 1/nm :sup:`2`
•   Temperature, K
•	Deposited volume, nm :sup:`3`
•   Growth rate

.. note::
    The data collected can be extended via Statistics class by adding columns at the simulation initialization and then
    providing data for timely records in the monitoring function.

.. hint::
    While real time refers to the real-world time, simulation/experiment refers to the time defined by the beam pattern.



`Save structure snapshots` enables regular dumping of the current state of structure. The data is saved in .vtk format,
and includes 3D arrays that define:

•	Grown structure
•   Surface deposit
•	Surface precursor coverage
•	Temperature
•	Surface cells
•	Semi-surface cells
•	Ghost cells

    Additionally, current time, time passed, deposition time passed and beam position are saved.

The files saved via this option can be then viewed as 3D models by the included show_file.py and show_animation.py
scripts or in ParaView®.

.. warning::
    3D structure file (.vtk) may reach 500 Mb for finer grids and, coupled with regular saving with short intervals,
    may occupy significant disc space. If only the end-result is needed, input an interval that is larger than the
    total deposition time.

.. important::
    Currently, patterning information is not included in the saved simulation setup info and has to be managed manually.


Viewing simulation results:
"""""""""""""""""""""""""""""
There are three options to inspect a 3D structure deposited by FEBID simulation.

The first one is viewing a specific
snapshot with all the corresponding data layers (precursor coverage, temperature etc.).

    ``python -m febid show_file``

The second option is to view the process based on a series of structure snapshots. Unlike viewing a single file, only
one data layer can be 'animated'.

    ``python -m febid show_animation``

Surface deposit, precursor coverage and temperature profile data are currently supported, it can be set up inside
the script.

The third option is to use `Paraview® <https://www.paraview.org/download/>`_.
`Examples <https://github.com/MrCheatak/FEBID_py/tree/master/Examples>`_ folder contains a process file, that has
all presets for each dataset included in the 3D structure file to render the same result as the `show_file` script.


