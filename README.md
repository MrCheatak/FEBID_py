# FEBID
Simulation of the FEBID process witten in Python.

## Installation
1. Clone the repository and choose virtual environment.
2. Install following packages using pip: numpy, scipy, cython, pyvista, pyaml, tqdm, line_profiler and numexpr_mod(https://github.com/MrCheatak/numexpr_mod). 

	You can use this command to install everything at once:
	
	*pip install numpy scipy cython pyvista pyaml tqdm line_profiler git+https://github.com/MrCheatak/numexpr_mod.git#numexpr_mod*

3. Compile two necessary Cython modules in code/modified_libraries/ray_traversal and code/modified_libraries/rolling by navigating to those directories in Terminal and executing *python setup.py build_ext --inplace*

