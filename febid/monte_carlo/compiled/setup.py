import os, sys
from setuptools import  setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

# Using compiler of clang with llvm installed on mac OSX
if sys.platform is 'darwin':  # darwin == OSX
    os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
    os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"

# Add include/link dirs, and modify the stdlib to libc++
ext_module = [Extension("etrajectory_c", ['etrajectory_c.pyx'],
                  include_dirs=["/usr/local/opt/llvm/include"],
                  library_dirs=["/usr/local/opt/llvm/lib"],
                  extra_compile_args=["-w", "-fopenmp"],
                  extra_link_args=["-lomp"]
                  )]

# setup(ext_modules = cythonize("cytest.pyx", annotate=True, compiler_directives={'linetrace': True}), include_dirs=[np.get_include()])
setup(ext_modules = cythonize("etrajectory_c.pyx"), include_dirs=[np.get_include()])
# setup(ext_modules = cythonize(extensions, annotate=True), include_dirs='.')