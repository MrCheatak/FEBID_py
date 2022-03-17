# FEBID
Simulation of the FEBID process written in Python.

## Installation

1. Clone the repository and choose virtual environment.
2. Install following packages using pip: *numpy, cython, pyvista, pyaml, ruamel, pandas, openpyxl, tqdm, pypt5, line_profiler and numexpr_mod* (https://github.com/MrCheatak/numexpr_mod).
	 You can use these two commands to install everything at once:
	
	`pip install numpy cython pyvista pyaml ruamel.yaml pandas openpyxl tqdm pyqt5 line_profiler`
   
    `pip install git+https://github.com/MrCheatak/numexpr_mod.git#numexpr_mod`

3. Compile Cython modules by navigating to the project folder and executing `python setup.py build_ext --inplace`

**Note**: Cython modules utilize OpenMP for parallel computations. You may need to install it in order to run the simulation.

On Mac OSX it can be installed using *brew* and does compile with clang/clang++ compiler shipped with macs:
	
In Terminal: `brew install libomp`

On Linux/Ubuntu its available through *apt-get* and is compatible with the standard gcc compiler:

In Terminal: `sudo apt-get install libomp-dev`

On Windows it should be shipped with standard gcc compiler, so no actions are required from the user.

## Usage:

1. The simulation setup is done via the control pannel by running the start.py script in /source folder.
2. Define simulation volume by specifying a prepared .vtk file or create an empty volume from parameters
 or automatically, if you intend to print from a stream-file.
3. Create a printing path from a number of simple shapes or load a more complex path from a stream-file.
4. Choose Settings file, that includes surface flux, beam parameters and substrate material.
5. Choose Precursor file, that contains properties of the precursor and its expected deponat.
6. Statistics from the simulation as well as current states of the simulation volume (snapshots) can be saved 
with the desired intervals. The program will create a folder with the specified name and put all the files 
inside
7. The deposition process can be observed visually when Show process is checked.


'Parameters' option and specifying the dimensions. The volume is created with a substrate. Another option 
   is to read the volume with a structure from a .vtk file. The file can be a regular .vtk file with a structure in it
   or it can be a file produced by the simulation (see Save file). If a regular .vtk file is specified, it
   has to be a UniformGrid and have a cubic cell (equal spacings). The last option 'auto' creates volume automatically
   based on the stream-file.
   
