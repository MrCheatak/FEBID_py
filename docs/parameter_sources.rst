=========================
Parameter approximations
=========================

Deposited volume
"""""""""""""""""
The result of dissociation process is the added deposited volume, that is proportional to to number of dissociated
precursor molecules. In the simulation each molecule is assumed to produce a certain volume of deposit.
A volume of the smallest deposit volume then can be derived from density and composition of the bulk deposit:

:math:`\Delta V=\frac M{N_A\cdot\rho}`

where:
    :math:`M` is the molecular mass of the model molecule reflecting bulk composition :math:`\left[\frac g mol\right]`

    :math:`N_A` is the Avogadro number

    :math:`\rho` is density of the deposit


Atomic number
"""""""""""""""
Effective or average atomic number of a multi-component material can be estimated based on two criteria:

#. Atom number density:
    :math:`n_a=\frac{n_m\cdot N_A\cdot\rho}M`

    where:
            :math:`n_m` is the number of atoms in the model molecule

            :math:`M` is the molecular mass of the model molecule reflecting bulk composition :math:`\left[\frac g mol\right]`

#. Atom number averaging:
    :math:`\overline Z = \sqrt{\sum_{i=1}^{n_m} a_i\cdot Z_i^2}`

Both values have to be checked against those of elements in the periodic table to find the best matching element.


Residence time
"""""""""""""""

The heating and temperature tracking serves for the definition of temperature dependent time at runtime.
The dependence is *Arrhenius*-like and is described by the following relation:

:math:`\tau=\frac{1}{k_0}exp\left( \frac{E_a}{k_B T} \right)`

where:
    :math:`k_0` is the exponential prefactor representing desorption attempt frequency [Hz]

    :math:`E_a` is the adsorption energy [meV]

    :math:`k_b` is the Bolzman constant

    :math:`T` is temperature [K]



Diffusion
""""""""""""
