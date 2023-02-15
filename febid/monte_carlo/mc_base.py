"""
Monte Carlo simulator utility module
"""

from abc import ABC

import numpy as np

class Element:
    """
    Represents a solid material.
    Contains properties necessary for electron beam-matter interaction.
    """
    def __init__(self, name='noname', Z=1, A=1.0, rho=1.0, e=50, lambda_escape=1.0, mark=1):
        self.name = name # name of the material
        self.rho = rho # density, g/cm^3
        self.Z = Z # atomic number (or average if compound)
        self.A = A # molar mass, g/mol
        self.J = (9.76*Z + 58.5/Z**0.19)*1.0E-3 # ionisation potential
        self.e = e # effective energy required to produce an SE, eV [lin]
        self.lambda_escape = lambda_escape # effective SE escape path, nm [lin]
        self.mark = mark

        # [lin] Lin Y., Joy D.C., Surf. Interface Anal. 2005; 37: 895–900

    def __add__(self, other):
        if other == 0:
            return self

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


class MC_Sim_Base(ABC):
    cell_dim:int
    grid: np.ndarray
    surface: np.ndarray
    s_neighb: np.ndarray
    deponat: Element
    substrate:Element
    materials: list

    NA = 6.022141E23  # Avogadro number
    elementary_charge = 1.60217662e-19  # Coulon

    @property
    def shape(self):
        return self.grid.shape

    @property
    def shape_abs(self):
        return tuple([dim * self.cell_dim for dim in self.grid.shape])


substrates = {} # available substrates
substrates['Au'] = Element(name='Au', Z=79, A=196.967, rho=19.32, e=35, lambda_escape=0.5)
substrates['Si'] = Element(name='Si', Z=14, A=29.09, rho=2.33, e=90, lambda_escape=2.7)