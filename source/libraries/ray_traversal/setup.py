import cython
from setuptools import  setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

extensions = [Extension("traversal", ["traversal.pyx"], define_macros=[('CYTHON_TRACE', '1')])]

# setup(ext_modules = cythonize("cytest.pyx", annotate=True, compiler_directives={'linetrace': True}), include_dirs=[np.get_include()])
setup(ext_modules = cythonize("traversal.pyx", annotate=True), include_dirs=[np.get_include()])
# setup(ext_modules = cythonize(extensions, annotate=True), include_dirs='.')