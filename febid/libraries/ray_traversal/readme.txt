If any changes are brought to the source Cython file 'traversal.pyx' it has to be recompiled by executing:

    python setup.py build_ext --inplace

Note that you have to navigate to the containing folder first.

If Python fails to import the module, delete .cpp, .so files and build folder and recompile.
