import os, sys
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

# Using compiler of clang with llvm installed on mac OSX
platform = sys.platform
libraries = []
if 'darwin' in platform:  # darwin == OSX
    os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
    os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"
    openMP_arg = '-lomp'
if 'linux' in platform:
    openMP_arg = '-fopenmp'
    libraries.append('gomp')
if 'win32' in platform:
    openMP_arg = '-fopenmp'
    libraries.append('gomp')
if not openMP_arg:
    raise RuntimeError('Could not define System OS!.')
print(f'Platform: {platform, openMP_arg}')

# Add include/link dirs, and modify the stdlib to libc++
ext_modules = [
             Extension("source.monte_carlo.compiled.etrajectory_c", ['source/monte_carlo/compiled/etrajectory_c.pyx'],
                         include_dirs=["/usr/local/opt/llvm/include"],
                         library_dirs=["/usr/local/opt/llvm/lib"],
                         # extra_compile_args=["-w", '-fopenmp'],
                         # libraries=libraries,
                         # extra_link_args=[openMP_arg]
                         ),
               Extension("source.libraries.ray_traversal.traversal", ['source/libraries/ray_traversal/traversal.pyx'],
                         include_dirs=["/usr/local/opt/llvm/include"],
                         library_dirs=["/usr/local/opt/llvm/lib"],
                         # extra_compile_args=["-w", '-fopenmp'],
                         # libraries=libraries,
                         # extra_link_args=[openMP_arg]
                         ),
               Extension("source.libraries.rolling.roll", ['source/libraries/rolling/roll.pyx'],
                         include_dirs=["/usr/local/opt/llvm/include"],
                         library_dirs=["/usr/local/opt/llvm/lib"],
                         extra_compile_args=["-w", '-fopenmp'],
                         libraries=libraries,
                         extra_link_args=[openMP_arg]
                         ),
               ]

setup(
    author='Alexander Kuprava',
    ext_modules = cythonize(ext_modules),
    packages=['monte_carlo.compiled', 'traversal', 'rolling'],
    include_dirs=[np.get_include()])