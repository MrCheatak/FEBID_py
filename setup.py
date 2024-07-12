import os, sys, setuptools

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
    # os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
    # os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"
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
    Extension("febid.monte_carlo.compiled.etrajectory_c", ['febid/monte_carlo/compiled/etrajectory_c.pyx'],
              # include_dirs=["/usr/local/opt/llvm/include"],
              # library_dirs=["/usr/local/opt/llvm/lib"],
              # extra_compile_args=["-w", '-fopenmp'],
              # libraries=libraries,
              # extra_link_args=[openMP_arg]
              ),
    Extension("febid.libraries.ray_traversal.traversal", ['febid/libraries/ray_traversal/traversal.pyx'],
              # include_dirs=["/usr/local/opt/llvm/include"],
              # library_dirs=["/usr/local/opt/llvm/lib"],
              # extra_compile_args=["-w", '-fopenmp'],
              # libraries=libraries,
              # extra_link_args=[openMP_arg]
              ),
    Extension("febid.libraries.rolling.roll", ['febid/libraries/rolling/roll.pyx'],
              # include_dirs=["/usr/local/opt/llvm/include"],
              # library_dirs=["/usr/local/opt/llvm/lib"],
              # extra_compile_args=["-w", '-fopenmp'],
              # libraries=libraries,
              # extra_link_args=[openMP_arg]
              ),
    Extension("febid.libraries.pde.tridiag", ['febid/libraries/pde/tridiag.pyx'],
              # include_dirs=["/usr/local/opt/llvm/include"],
              # library_dirs=["/usr/local/opt/llvm/lib"],
              # extra_compile_args=["-w", "-fopenmp"],
              # libraries=libraries,
              # extra_link_args=[openMP_arg]
              ),
]
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setuptools.setup(
    name='febid',
    version='0.9.3',
    author='Alexander Kuprava, Michael Huth',
    author_email='sandro1742@gmail.com',
    description='FEBID process simulator',
    long_description='Direct-write nano- and microscale chemical vapor deposition method using a beam of accelerated electrons.',
    long_description_content_type="text/markdown",
    url='https://github.com/MrCheatak/FEBID_py',
    project_urls = {},
    license='MIT',
    packages=['febid', 'febid.monte_carlo', 'febid.monte_carlo.compiled', 'febid.ui', 'febid.libraries.vtk_rendering',
              'febid.libraries.rolling', 'febid.libraries.ray_traversal', 'febid.libraries.pde'],
    package_data = {'': ['*.pyx', 'ui/last_session_stub.yml']},
    include_package_data=True,
    install_requires=requirements,
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()],
    pyhton_requires='>=3.9',
)
