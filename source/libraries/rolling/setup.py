import os
from setuptools import  setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension
from Cython.Compiler.Options import get_directive_defaults

# Using compiler of clang with llvm installed
os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"

# Add include/link dirs, and modify the stdlib to libc++
ext_module = [Extension("roll", ['roll.pyx'],
                  include_dirs=["/usr/local/opt/llvm/include"],
                  library_dirs=["/usr/local/opt/llvm/lib"],
                  language="c",
                  extra_compile_args=["-w", "-fopenmp"],
                  extra_link_args=["-lomp"]
                  )]

# setup(ext_modules = cythonize("roll.pyx", annotate=True))
setup(ext_modules = cythonize(ext_module, annotate=True), include_dirs='.')