===================================
Setting up a series of simulations
===================================

Optimisation of pattern files, simulation input parameters or simulation of several structures may require
running a significant number of simulations. The package offers some simple automation features for such tasks.
Setting up a simulation series requires composing a Python script.

The first feature allows executing a sequence of simulations arising from consequently changing a single parameter.
A series of such simulations is regarded as a `scan`. Such scan can be carried out on any parameter from
the `Precursor <precursor_file.html>`_ or `Settings <settings_file.html>`_ file.

.. code-block:: python

    # Initially, a session configuration file has to be specified.
    # This file, along settings and precursor parameters files specified in it, is to be modified
    # and then used to run a simulation. This routine is repeated until the desired parameter
    # has taken a given number of values.
    # The routine only changes a single parameter. All other parameters have to be preset forehand.
    session_file = '/home/kuprava/simulations/last_session.yml'

    # The first parameter change or scan modifies the Gaussian deviation parameter of the beam.
    # The file that will be modified in this case is the settings file.
    # Set up a folder (it will be created automatically) for simulation save files
    directory = '/home/kuprava/simulations/gauss_dev_scan/'
    write_param(session_file, 'save_directory', directory)
    # Specify parameter name
    param = 'gauss_dev'
    # Specify values that the parameter will take during consequent simulations
    vals = [2, 3, 4, 5, 6, 7, 8]
    # Launch the scan
    scan_settings(session_file, param, vals, 'hs')
    # Files that are saved during the simulation are named after the specified common name (here i.e. 'hs')
    # and the parameter name.``

It is also possible to run a 2D scan, meaning another parameter is scanned for each value of the first parameter.

The second option is to run simulations by using a collection of pattern files. This mode requires that all the
desired pattern files are collected in a single folder, that has to be provided to the script.

.. code-block:: python

    # Again, specify a desired location for simulation save files
    directory = '/home/kuprava/simulations/longs/'
    # Optionally, an initial structure can be specified. This will 'continue' deposition
    # onto a structure obtained in one of the earlier simulations.
    # It can be used i.e. when all planned structures share a same initial feature such as a pillar.
    # Keep in mind that it can be used only for patterning files with the same patterning area.
    # To that, the patterning area must correspond to one that is defined by the simulation for the current
    # pattern including margins.
    initial_structure = '/home/kuprava/simulations/hockey_stick_therm_050_5_01_15:12:31.vtk'
    write_param(session_file, 'structure_source', 'vtk')
    write_param(session_file, 'vtk_filename', initial_structure)
    write_param(session_file, 'save_directory', directory)
    # Specifying a folder with patterning files
    stream_files = '/home/kuprava/simulations/steam_files_long_s'
    # Launching the series
    scan_stream_files(session_file, stream_files)

.. note::
    Scanning only modifies the selected parameter(s). Thus, all other parameters as well as saving options and output
    directory have to be preset.


