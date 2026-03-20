from febid.Structure import Structure

def prepare_equation_values(precursor: dict, settings: dict):
    """
    Prepare equation values for the reaction-equation solver.

    :param precursor: dictionary containing precursor properties
    :param settings: dictionary containing beam and precursor flux settings
    :return: dictionary containing equation values for the solver
    """
    equation_values = {}
    try:
        equation_values['F'] = settings["precursor_flux"]
        equation_values['n0'] = precursor["max_density"]
        equation_values['sigma'] = precursor["cross_section"]
        equation_values['tau'] = precursor["residence_time"] * 1E-6
        equation_values['Ea'] = precursor['desorption_activation_energy']
        equation_values['k0'] = precursor['desorption_attempt_frequency']
        equation_values['V'] = precursor["dissociated_volume"]
        equation_values['D'] = precursor["diffusion_coefficient"]
        equation_values['Ed'] = precursor['diffusion_activation_energy']
        equation_values['D0'] = precursor['diffusion_prefactor']
        equation_values['rho'] = precursor['average_density']
        equation_values['heat_cond'] = precursor['thermal_conductivity']
        equation_values['cp'] = precursor['heat_capacity']
        equation_values['deposition_scaling'] = settings['deposition_scaling']
    except KeyError as e:
        raise KeyError(f"Missing key in precursor or settings dictionary: {str(e)}")
    return equation_values


def prepare_ms_config(precursor: dict, settings: dict, structure: Structure):
    """
    Prepare the configuration for Monte-Carlo simulation.

    :param precursor: dictionary containing precursor information
    :param settings: dictionary containing simulation settings
    :param structure: Structure object representing the simulation volume
    :return: dictionary containing the Monte-Carlo simulation configuration
    :raises TypeError: if the 'structure' parameter is not an instance of the 'Structure' class
    :raises KeyError: if any key is missing in the precursor or settings dictionaries
    """
    # Parameters for Monte-Carlo simulation
    try:
        mc_config = {'name': precursor["deposit"],
                     'E0': settings["beam_energy"],
                     'Emin': settings["minimum_energy"],
                     'Z': precursor["average_element_number"],
                     'A': precursor["average_element_mol_mass"],
                     'rho': precursor["average_density"],
                     'I0': settings["beam_current"],
                     'sigma': settings["gauss_dev"],
                     'n': settings['n'],
                     'substrate_element': settings["substrate_element"],
                     'cell_size': structure.cell_size,
                     'e': precursor["SE_emission_activation_energy"],
                     'l': precursor["SE_mean_free_path"],
                     'emission_fraction': settings['emission_fraction']}
    except KeyError as e:
        raise KeyError(f"Missing key in precursor or settings dictionary: {str(e)}")
    return mc_config
