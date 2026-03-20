"""
PhysicsEngine - CPU-based physics calculations

This module contains all CPU-based physics calculations for the FEBID simulation.
It is acceleration-agnostic and uses the uniform expression pattern via DataViewManager.
"""

import numpy as np
import numexpr_mod as ne

from febid.process.simulation_state import SimulationState
from febid.process.data_view_manager import DataViewManager, DepositionView, PrecursorDensityView, DiffusionView
from febid.thermal.temperature_manager import TemperatureManager
import febid.diffusion as diffusion
from febid.slice_trics import any_where, index_where
from ..expressions import cache_numexpr_expressions
from timeit import default_timer as df
from febid.logging_config import setup_logger

logger = setup_logger(__name__)


class PhysicsEngine:
    """
    CPU-based physics engine for FEBID simulation.

    Handles:
    - Deposition calculations
    - Precursor density evolution (RDE solver)
    - Diffusion calculations (FTCS)
    - RK4 integration
    - Cell filling checks

    Design principles:
    - Acceleration-agnostic: doesn't know about optimization mode
    - Uses uniform expressions: view.deposit[view.index] += ...
    - Stateless: all state changes go through SimulationState
    - View-based: gets data via DataViewManager
    """

    def __init__(self, state: SimulationState, view_manager: DataViewManager,
                 temp_manager: TemperatureManager):
        """
        Initialize PhysicsEngine.
        
        :param state: Read-only access to simulation data
        :type state: SimulationState
        :param view_manager: Provides optimized views and indices
        :type view_manager: DataViewManager
        :param temp_manager: Provides temperature-dependent coefficients
        :type temp_manager: TemperatureManager
        """
        self.state = state
        self.view_manager = view_manager
        self.temp_manager = temp_manager

        self.__expressions()  # Prepare numexpr expressions for faster calculations

    def compute_deposition(self, dt: float) -> None:
        """
        Calculate deposition increment for all irradiated cells over time step.
        
        Uses uniform expression approach that works for both acceleration modes:
        - Acceleration ON: view.index is fancy tuple, beam_matrix is 1D
        - Acceleration OFF: view.index is np.s_[:], beam_matrix is 3D
        
        :param dt: Time step in seconds
        :type dt: float
        """
        view: DepositionView = self.view_manager.get_deposition_view()

        # Calculate constant (multiplying by 1e6 to preserve accuracy)
        const = (self.state.precursor.sigma * self.state.precursor.V * dt * 1e6 *
                 self.state.deposition_scaling / self.state.cell_V * self.state.cell_size ** 2)

        # UNIFORM EXPRESSION: Works for both acceleration ON and OFF
        # When acceleration ON: view.index is fancy tuple (z,y,x), beam_matrix is 1D
        # When acceleration OFF: view.index is np.s_[:], beam_matrix is 3D
        view.deposit[view.index] += view.precursor[view.index] * view.beam_matrix * const / 1e6

    def compute_precursor_density(self, dt: float) -> None:
        """
        Calculate precursor density increment for all surface cells.
        
        Solves the reaction-diffusion equation using RK4 integration with FTCS diffusion.
        
        :param dt: Time step in seconds
        :type dt: float
        """
        view: PrecursorDensityView = self.view_manager.get_precursor_density_view(self.temp_manager)

        # surface_all represents surface + semi_surface cells
        # Boolean indexing: precursor[surface_all] extracts values at surface cells (1D flat array)
        view.precursor[view.surface] += self._rk4_with_ftcs(view, dt)

    def check_cells_filled(self) -> bool:
        """
        Check if any deposit cells are fully filled (deposit >= 1.0).
        
        Searches in reverse order (faster for bottom-up growth).
        
        :return: (bool) True if any cells are filled
        """
        view = self.view_manager.get_deposition_view()
        surface_cells = view.deposit[view.index]
        cells_filled = any_where(surface_cells, '>=', 1, reverse=True)
        return cells_filled

    def _rk4_with_ftcs(self, view: PrecursorDensityView, dt: float) -> np.ndarray:
        """
        RK4 integration with FTCS diffusion for precursor density evolution.
        
        Calculates k1, k2, k3, k4 coefficients for Runge-Kutta 4th order integration,
        with diffusion calculated via FTCS at each stage.
        
        :param view: View containing precursor, beam_matrix, surface, tau, D
        :type view: PrecursorDensityView
        :param dt: Time step in seconds
        :type dt: float
        
        :return: (np.ndarray) Precursor density increment (1D flat array)
        """
        beam_matrix = view.beam_matrix
        surface = view.surface
        precursor = view.precursor
        tau = view.tau  # Extract tau from view
        prec_flat = precursor[surface]

        # Extract tau at surface locations if it's an array (temperature tracking)
        if isinstance(tau, np.ndarray) and tau.shape != prec_flat.shape:
            # tau is 2D/3D array, extract surface values
            tau_flat = tau[surface] if tau.ndim > 1 else tau
        else:
            # tau is already flat or scalar
            tau_flat = tau

        # If no diffusion, use simpler RK4 without FTCS
        if np.any(view.D) == 0:
            return self._rk4(prec_flat, beam_matrix, dt, tau_flat)



        # k1
        diff_flat = self._diffusion(precursor, dt, flat=True)
        semi_surface_position = diff_flat.size - view.semi_surface_index[0].size
        k1 = self._precursor_density_increment(prec_flat, beam_matrix, dt, diff_flat[:semi_surface_position], tau=tau_flat)
        k1_semi = diff_flat[semi_surface_position:]

        # k2
        k1_div = k1 / 2
        k1_semi_div = k1_semi / 2
        k1_full_div = np.concatenate([k1_div, k1_semi_div])
        diff_flat = self._diffusion(precursor, dt / 2, add=k1_full_div, flat=True)
        k2 = self._precursor_density_increment(prec_flat, beam_matrix, dt / 2, diff_flat[:semi_surface_position], k1_div, tau=tau_flat)
        k2_semi = diff_flat[semi_surface_position:]

        # k3
        k2_div = k2 / 2
        k2_semi_div = k2_semi / 2
        k2_full_div = np.concatenate([k2_div, k2_semi_div])
        diff_flat = self._diffusion(precursor, dt / 2 , add=k2_full_div, flat=True)
        k3 = self._precursor_density_increment(prec_flat, beam_matrix, dt / 2, diff_flat[:semi_surface_position], k2_div, tau=tau_flat)
        k3_semi = diff_flat[semi_surface_position:]

        # k4
        k3_full = np.concatenate([k3, k3_semi])
        diff_flat = self._diffusion(precursor, dt , add=k3_full, flat=True)
        k4 = self._precursor_density_increment(prec_flat, beam_matrix, dt, diff_flat[:semi_surface_position], k3, tau=tau_flat)
        k4_semi = diff_flat[semi_surface_position:]

        precursor[view.semi_surface_index] += ne.re_evaluate("rk4", casting='same_kind',
                                                             local_dict={'k1': k1_semi, 'k2': k2_semi, 'k3': k3_semi, 'k4': k4_semi})

        # Combine RK4 coefficients
        return ne.re_evaluate("rk4", casting='same_kind',
                             local_dict={'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4})

    def _rk4(self, precursor: np.ndarray, beam_matrix: np.ndarray, dt: float,
             tau: np.ndarray = None) -> np.ndarray:
        """
        RK4 integration without diffusion (simplified version).
        
        Used when diffusion coefficient D = 0.
        
        :param precursor: Flat precursor array
        :type precursor: np.ndarray
        :param beam_matrix: Flat surface electron flux array
        :type beam_matrix: np.ndarray
        :param dt: Time step in seconds
        :type dt: float
        :param tau: Residence time (if None, fetches from view_manager)
        :type tau: np.ndarray or float, optional
        
        :return: (np.ndarray) Precursor density increment
        """
        k1 = self._precursor_density_increment(precursor, beam_matrix, dt, tau=tau)
        k2 = self._precursor_density_increment(precursor, beam_matrix, dt / 2, addon=k1 / 2, tau=tau)
        k3 = self._precursor_density_increment(precursor, beam_matrix, dt / 2, addon=k2 / 2, tau=tau)
        k4 = self._precursor_density_increment(precursor, beam_matrix, dt, addon=k3, tau=tau)
        return ne.re_evaluate("rk4", casting='same_kind',
                             local_dict={'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4})

    def _precursor_density_increment(self, precursor: np.ndarray, beam_matrix: np.ndarray,
                                     dt: float, diffusion_matrix: float = 0,
                                     addon: float = 0.0, tau: np.ndarray = None) -> np.ndarray:
        """
        Calculate precursor density increment for RDE.
        
        Includes adsorption, desorption, dissociation, and diffusion terms.
        
        :param precursor: Flat precursor array
        :type precursor: np.ndarray
        :param beam_matrix: Flat surface electron flux array
        :type beam_matrix: np.ndarray
        :param dt: Time step in seconds
        :type dt: float
        :param diffusion_matrix: Diffusion term from FTCS
        :type diffusion_matrix: float or np.ndarray, optional
        :param addon: Runge-Kutta intermediate term
        :type addon: float or np.ndarray, optional
        :param tau: Residence time (if None, fetches from view_manager)
        :type tau: np.ndarray or float, optional
        
        :return: (np.ndarray) Precursor density increment
        """
        # Get tau from parameter or from view if not provided
        if tau is None:
            view: PrecursorDensityView = self.view_manager.get_precursor_density_view(self.temp_manager)
            tau = view.tau

        n_d = diffusion_matrix

        try:
            return ne.re_evaluate('rde_temp',
                                 local_dict={
                                     'F': self.state.precursor.F,
                                     'dt': dt,
                                     'n0': self.state.precursor.n0,
                                     'sigma': self.state.precursor.sigma,
                                     'n': precursor + addon,
                                     'tau': tau,
                                     'se_flux': beam_matrix,
                                     'n_d': n_d
                                 },
                                 casting='same_kind')
        except ValueError as e:
            logger.error(
                f"Failed numexpr.re_evaluate() in PhysicsEngine._precursor_density_increment "
                f"due to array size mismatch. \n"
                f"Precursor array size: {precursor.size} \n"
                f"Beam matrix size: {beam_matrix.size} \n"
                f"Diffusion matrix size: {n_d.size if hasattr(n_d, 'size') else 'scalar'} \n"
                f"Tau size: {tau.size if hasattr(tau, 'size') else 'scalar'} \n"
            )
            raise e

    def _diffusion(self, grid: np.ndarray, dt: float,
                   add: float = 0, flat: bool = False) -> np.ndarray:
        """
        Calculate diffusion term via FTCS scheme.
        
        :param grid: Precursor coverage array
        :type grid: np.ndarray
        :param surface: Boolean surface array
        :type surface: np.ndarray
        :param dt: Time step in seconds
        :type dt: float
        :param add: Runge-Kutta intermediate term
        :type add: float or np.ndarray, optional
        :param flat: If True, return flattened array
        :type flat: bool, optional
        
        :return: (np.ndarray) Diffusion term
        """
        # Get diffusion view with D coefficient from TemperatureManager
        view: DiffusionView = self.view_manager.get_diffusion_view(self.temp_manager)
        surface = view.surface_all
        D = view.D

        return diffusion.diffusion_ftcs(
            grid, surface, D, dt, self.state.cell_size,
            view.surface_all_index, flat=flat, add=add
        )

    def equilibrate(self, dt, max_it=10000, eps=1e-8):
        """
        Bring precursor coverage to a steady state with a given accuracy

        It is advised to run this method after updating the surface in order to determine a more accurate precursor
        density value for newly acquired cells

        :param max_it: number of iterations
        :param eps: desired accuracy
        """
        start = df()
        for i in range(max_it):
            # Get current precursor state
            view = self.view_manager.get_precursor_density_view(self.temp_manager)
            p_prev = view.precursor.copy()

            # Update precursor density
            self.compute_precursor_density(dt)

            # Check convergence
            view_new = self.view_manager.get_precursor_density_view(self.temp_manager)
            norm = np.linalg.norm(view_new.precursor - p_prev) / np.linalg.norm(view_new.precursor)
            if norm < eps:
                print(f'Took {i+1} iteration(s) to equilibrate, took {df() - start}')
                return 1
        else:
            acc = str(norm)[:int(3-np.log10(eps))]
            logger.warning(f'Failed to reach {eps} accuracy in {max_it} iterations in Process.equilibrate. Achieved accuracy: {acc} \n'
                          f'Terminating loop.', RuntimeWarning)
            print(f'Took {i + 1} iteration(s) to equilibrate, took {df() - start}')
            return 0


    def __expressions(self):
        """
        Prepare math expressions for faster calculations. Expression are stored in the package.
        This method should be called only once.

        :return:
        """
        cache_numexpr_expressions()
