Experimental settings
======================

An example of a settings file can be found in the
`Examples <https://github.com/MrCheatak/FEBID_py/tree/master/Examples>`_ folder of the repository.

Beam:
"""""""""""""""
Experiment beam settings:

- **beam_energy** – energy of the electron beam, keV
- **beam_current** – current of the electron beam,  A

Modulation of the beam profile:

- **gauss_dev** – standard deviation of a Gaussian beam shape function in nm
- **n** – order of the Gaussian function (see super or higher order gaussian distribution)

Electron trajectory settings:

- **minimum_energy** – energy at which electron trajectory following concludes, keV

Other:
""""""""""""

- **precursor_flux** – precursor flux at the surface,  1/(nm^2*s)
- **substrate_element** – material of the substrate, i.e. 'Au'
- **deposition_scaling** – multiplier for deposited volume for artificial speed up of the simulation
- **emission_fraction** – fraction of the total energy lost by primary electrons that is converted to secondary electron emission


