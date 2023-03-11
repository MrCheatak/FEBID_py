.. FEBID Simulation documentation master file, created by
   sphinx-quickstart on Wed Jul  6 10:28:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:

   Getting started <intro>
   Manual <manual>
   How it works <how_it_works>
   API <_autosummary/febid>
   Github Repo <https://github.com/MrCheatak/FEBID_py>

Welcome to FEBID Simulation documentation!
============================================

The package is virtual representation of the direct-write nanofabrication technique called
`FEBID <https://www.beilstein-journals.org/bjnano/articles/3/70>`_ driven by an electron beam that typically takes place
in a `SEM <https://en.wikipedia.org/wiki/Scanning_electron_microscope>`_. The simulation takes in a handful of
parameters and allows prediction of the deposit shape expected from an experiment. It features a live visual process
representation, periodical save of the current state of the 3D deposited structure and
recording of the process parameters like time and growth rate. Additionally, the package features an electron beam - matter
simulator, that can be run separately using a previously saved 3D structure to reveal beam related details of the process.

The saved 3D structure files can then be interactively viewed or compiled into a animated series depicting the process.

The `Getting started <intro.html>`_ section will let you quickly install the package and run an example simulation.
A more detailed `interface manual <interface.html>`_, input parameter files explanation and features list will give a full understanding
on how to use the simulation.

For more in-deep understanding of the simulation design and code details check the `API <_autosummary/febid.html>`_ section.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
