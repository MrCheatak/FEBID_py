from distutils.core import setup
from distutils.extension import Extension


# Compile an extension from the modified .c file
setup(name='roll',
      ext_modules = [Extension('roll', sources = ['roll.c'])])