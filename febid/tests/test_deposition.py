import unittest
import pytest
from timeit import default_timer as timer
import numpy as np
from ruamel.yaml import YAML
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List
from abc import ABC, abstractmethod
from febid import simple_patterns as sp
from febid.Process import Process
from febid.Structure import Structure
from febid.tests.sample_beam_matrix import test_beam_matrix as beam_matrix_test
from febid.monte_carlo.etraj3d import MC_Simulation
from febid.febid_core import prepare_equation_values, prepare_ms_config
from febid.tests.lib_1d_febid import Params, run_1d_simulation_metrics
from febid.mlcca import initialize_structure_topology







# from line_profiler import LineProfiler

# def profile(func):
#     def wrapper(*args, **kwargs):
#         profiler = LineProfiler()
#         profiler.add_function(func)
#         result = profiler(func)(*args, **kwargs)
#         profiler.print_stats()
#         return result
#     return wrapper


# Custom YAML parser
def load_yaml(path):
    yaml = YAML(typ='safe')
    with open(path, 'r') as f:
        return yaml.load(f)

def create_pattern(params):
    """
    Create a pattern based on the provided parameters and structure.
    This function uses the Starter class to generate a pattern.
    """
    pattern = 'Point'
    p1 = float(params['param1'])
    p2 = float(params['param2'])
    dwell_time = float(params['dwell_time']) * 1E-6
    pitch = float(params['pitch'])
    repeats = int(float(params['repeats']))
    x, y = p1, p2
    printing_path = sp.generate_pattern(pattern, repeats, dwell_time, x, y, (p1, p2), pitch)
    return printing_path


@dataclass
class SimulationResults(ABC):
    """Base interface for simulation results."""
    # Common properties
    D: float
    case_name: str
    params: Dict

    @property
    @abstractmethod
    def V(self) -> float:
        """Volume of deposited structure."""
        pass

    @property
    @abstractmethod
    def theta_center(self) -> float:
        """Surface coverage at center."""
        pass

    @property
    @abstractmethod
    def theta_edge(self) -> float:
        """Surface coverage at edge."""
        pass

    @property
    @abstractmethod
    def theta_replenished(self) -> float:
        """Replenished surface coverage."""
        pass

    @property
    @abstractmethod
    def theta_depleted(self) -> float:
        """Depleted surface coverage."""
        pass


@dataclass
class SimulationResults1D(SimulationResults):
    """Stores results from 1D analytical simulation."""
    # Raw simulation data
    p1d: 'Params'  # 1D simulation parameters object with calculation methods
    volume_calculated: float  # Calculated deposited volume (nm³)
    r: np.ndarray  # Radial positions (nm)
    profile: np.ndarray  # Precursor coverage profile at final time (molecules/nm²)

    @property
    def V(self) -> float:
        """Total deposited volume (from calculated result)."""
        return self.volume_calculated

    @property
    def theta_center(self) -> float:
        """Surface coverage at center (r=0) - extracted from profile."""
        # The center is at index 0 in the profile
        return self.profile[0]

    @property
    def theta_edge(self) -> float:
        """Surface coverage at edge (far from beam) - computed from p1d."""
        return self.p1d.n_replenished()

    @property
    def theta_replenished(self) -> float:
        """Replenished surface coverage (far from beam) - computed from p1d."""
        return self.p1d.n_replenished()

    @property
    def theta_depleted(self) -> float:
        """Depleted surface coverage (at beam center) - computed from p1d."""
        return self.p1d.n_depleted()


@dataclass
class SimulationResults3D(SimulationResults):
    """Stores results from 3D numerical simulation."""
    # Raw simulation objects
    process: 'Process'  # Process object containing deposition state
    structure: 'Structure'  # Structure object containing geometry and precursor field

    @property
    def V(self) -> float:
        """Total deposited volume - extracted from process."""
        return self.process._deposited_vol

    @property
    def theta_center(self) -> float:
        """Surface coverage at center - extracted from structure precursor field."""
        s = self.structure.surface_bool.argmax(axis=0).max()
        l = self.structure.shape[1]
        center_idx = l // 2
        value = self.structure.precursor[s, center_idx, center_idx]
        return value

    @property
    def theta_edge(self) -> float:
        """Surface coverage at edge - extracted from structure precursor field."""
        s = self.structure.substrate_height
        l = self.structure.shape[1]
        edge_idx = 0
        value = self.structure.precursor[s, edge_idx, edge_idx]
        return value

    @property
    def theta_replenished(self) -> float:
        """Replenished surface coverage (approximated as edge coverage)."""
        return self.theta_edge

    @property
    def theta_depleted(self) -> float:
        """Depleted surface coverage - computed as minimum in precursor profile."""
        s = self.structure.substrate_height
        l = self.structure.shape[1]
        profile = self.structure.precursor[s, l // 2, :]
        return profile.min()

    @property
    def x(self) -> np.ndarray:
        """Spatial positions array for plotting - computed from structure geometry."""
        l = self.structure.shape[1]
        c = self.structure.cell_size
        return np.linspace(-l // 2 * c, l // 2 * c, l)

    @property
    def profile(self) -> np.ndarray:
        """Precursor coverage profile along center line - extracted from structure."""
        s = self.structure.substrate_height
        l = self.structure.shape[1]
        return self.structure.precursor[s, l // 2, :]

    @property
    def filled_cells(self):
        return self.process.filled_cells


@dataclass
class MetricResult:
    """Stores result of a single metric assessment."""
    metric_name: str
    value_1d: float
    value_3d: float
    relative_error: float
    absolute_error: float
    tolerance: float
    passed: bool


class TestSimulationVolume:
    """
    Test suite comparing 1D analytical and 3D numerical FEBID simulations.
    Since the assessed metrics are coupled through the same simulation runs, a cache based approach is used to avoid running identical simulations.
    """
    # Class-level tolerance constant
    TOLERANCE = 0.05

    # Simulation setups: (case_name, D_override)
    SIMULATION_SETUPS = [
        ("D=0", 0.0),
        ("D=1", 1.0),
        ("D=from_file", None),
    ]

    ACCELERATION_GRID_SETUPS = [
        ("ON", True),
        ("OFF", False),
    ]

    # Cache for simulation results
    _simulation_cache = {}

    # Track which cases have been reported (for printing/plotting only once)
    _reported_cases = set()

    params = {
            'structure_source': 'geom',
            'cell_size': 2,
            'width': 200,
            'length': 200,
            'height': 50,
            'substrate_height': 20,
            'pattern_source': 'simple',
            'pattern': 'Point',
            'param1': 101,  # x
            'param2': 101,  # y
            'dwell_time': 2000,  # us
            'pitch': 0,
            'repeats': 1,
            'settings_filename': r"febid/tests/Parameters.yml",
            'precursor_filename': r"febid/tests/Me3PtCpMe.yml",
            'save_simulation_data': False,
            'save_structure_snapshot': False,
            'simulation_data_interval': 1000,
            'structure_snapshot_interval': 1000,
            'unique_name': '',
            'save_directory': '',
            'temperature_tracking': False,
            'gpu': False
        }

    # ==================== UNIVERSAL TEST METHODS (Parameterized for 9 results) ====================

    @pytest.mark.parametrize("case_name,D_override", SIMULATION_SETUPS)
    def test_volume(self, case_name, D_override):
        """Test deposited volume metric for all setups (generates 3 test results)"""
        # Execute simulation (with caching)
        results_1d, results_3d = self.get_or_run_simulation(case_name, D_override)

        # Extract metrics
        metrics = self.extract_metrics(results_1d, results_3d)

        # Report and plot only once per case (on first metric test)
        if case_name not in self._reported_cases:
            self._reported_cases.add(case_name)
            self.print_report([(results_1d, results_3d)], [metrics])
            self.plot_comparison([(results_1d, results_3d)])

        # Assess volume metric (index 0)
        self._assert_metric(metrics[0])

    @pytest.mark.parametrize("case_name,D_override", SIMULATION_SETUPS)
    def test_center_coverage(self, case_name, D_override):
        """Test center coverage metric for all setups (generates 3 test results)"""
        # Execute simulation (with caching)
        results_1d, results_3d = self.get_or_run_simulation(case_name, D_override)

        # Extract metrics
        metrics = self.extract_metrics(results_1d, results_3d)

        # Assess center coverage metric (index 1)
        self._assert_metric(metrics[1])

    @pytest.mark.parametrize("case_name,D_override", SIMULATION_SETUPS)
    def test_edge_coverage(self, case_name, D_override):
        """Test edge coverage metric for all setups (generates 3 test results)"""
        # Execute simulation (with caching)
        results_1d, results_3d = self.get_or_run_simulation(case_name, D_override)

        # Extract metrics
        metrics = self.extract_metrics(results_1d, results_3d)

        # Assess edge coverage metric (index 2)
        self._assert_metric(metrics[2])

    def test_semi_surface_cells(self):
        """Test equivalence of deposited volume on a flat surface vs a surface with single layer square patch.
        This test is designed to test consistency of the simulation's handling of semi-surface cells.
        Running test with and without acceleration grid."""

        ca = False
        case_name = "Semi-surface cells test"
        acceleration_enabled = True
        params = self.params
        # print(f"Reduced exposure time: default is 0.03, current 0.0 s")
        # params["dwell_time"] = 10

        # Load settings and precursor using ruamel.yaml
        settings = load_yaml(params['settings_filename'])
        precursor = load_yaml(params['precursor_filename'])

        # Generate point exposure pattern
        printing_path = create_pattern(params)

        # Run simulation with flat surface and cellular automata disabled to get baseline results
        start = timer()
        results_3d_flat = self.run_3d_sim(ca, params, precursor, printing_path, settings, case_name, acceleration_enabled=acceleration_enabled)
        end = timer()
        surface_deposit_cells_flat = (results_3d_flat.structure.deposit > 0).sum()
        # Print simulation results
        print(f"\nFlat surface without CA:")
        print(f"    Total run time: {(end - start):.2f} s")

        # Define a structure with a single layer patch to test the acceleration grid's handling of semi-surface cells
        structure = self.setup_domain(params)
        surface_cells_count_flat = np.count_nonzero(structure.surface_bool)
        sub_height = structure.substrate_height
        dep = structure.deposit
        patch_size_abs = 10  # nm
        patch_size_cells = int(patch_size_abs / structure.cell_size)
        center = (structure.shape[1] // 2, structure.shape[2] // 2)
        x1 = center[1] - patch_size_cells // 2
        x2 = center[1] + patch_size_cells // 2
        y1 = center[0] - patch_size_cells // 2
        y2 = center[0] + patch_size_cells // 2
        dep[sub_height, y1:y2, x1:x2] = -1
        initialize_structure_topology(structure)
        prec = structure.precursor
        surface_mask = structure.surface_bool
        semi_surface_mask = structure.semi_surface_bool
        prec[:] = 0
        prec[surface_mask] = structure.nr
        prec[semi_surface_mask] = structure.nr
        surface_cells_count = np.count_nonzero(structure.surface_bool)
        assert surface_cells_count_flat == surface_cells_count
        print(f"Surface cells count with flat surface: {surface_cells_count_flat}")
        print(f"Surface cells count with single layer patch: {surface_cells_count}")

        # Run simulation with cellular automata enabled
        start = timer()
        results_3d = self.run_3d_sim(ca, params, precursor, printing_path, settings, case_name,
                                     acceleration_enabled=acceleration_enabled, structure=structure)
        end = timer()
        surface_deposit_cells = (results_3d.structure.deposit > 0).sum()

        # Print simulation results
        print(f"\nSingle deposit layer without CA:")
        print(f"    Total run time: {(end - start):.2f} s")

        # Assert with tolerance (consistent with other tests)
        # Check number of cells with deposit
        rel_error_cells = abs(surface_deposit_cells_flat - surface_deposit_cells) / surface_deposit_cells_flat if surface_deposit_cells_flat != 0 else 0
        assert rel_error_cells <= self.TOLERANCE, (
            f"Cell count mismatch: "
            f"flat={surface_deposit_cells_flat}, patch={surface_deposit_cells}, "
            f"rel_error={rel_error_cells*100:.3f}%, tolerance={self.TOLERANCE*100:.3f}%"
        )

        # Check deposited volume
        rel_error_vol = abs(results_3d_flat.V - results_3d.V) / results_3d_flat.V if results_3d_flat.V != 0 else 0
        assert rel_error_vol <= self.TOLERANCE, (
            f"Volume mismatch: "
            f"flat={results_3d_flat.V:.3f} nm³, patch={results_3d.V:.3f} nm³, "
            f"rel_error={rel_error_vol*100:.3f}%, tolerance={self.TOLERANCE*100:.3f}%"
        )


    @pytest.mark.parametrize("case_name,acceleration_enabled", ACCELERATION_GRID_SETUPS)
    def test_acceleration_grid(self, case_name, acceleration_enabled):
        """Test acceleration grid consistency by running the simulation with and without acceleration"""
        ca = True
        params = self.params
        # print(f"Reduced exposure time: default is 0.03, current 0.0 s")
        # params["dwell_time"] = 300

        # Load settings and precursor using ruamel.yaml
        settings = load_yaml(params['settings_filename'])
        precursor = load_yaml(params['precursor_filename'])

        # Generate point exposure pattern
        printing_path = create_pattern(params)

        # Run simulation with cellular automata enabled
        start = timer()
        results_3d = self.run_3d_sim(ca, params, precursor, printing_path, settings, case_name, acceleration_enabled=acceleration_enabled)
        end = timer()

        # Print simulation results
        print(f"\nCA-enabled Simulation Results:")
        print(f"  - Total number of filled cells: {results_3d.process.filled_cells}")
        print(f"    Total run time: {(end - start):.2f} s")

    def examine_ca(self):
        """Utility method to run a CA-enabled simulation for manual examination (not a test)."""
        case_name = "CA_enabled"
        ca = True
        D_override = None
        print("CA examination run:")

        params = self.params

        # Load settings and precursor using ruamel.yaml
        settings = load_yaml(params['settings_filename'])
        precursor = load_yaml(params['precursor_filename'])

        # Override diffusion coefficient if specified
        if D_override is not None:
            precursor['diffusion_coefficient'] = D_override

        # Generate point exposure pattern
        printing_path = create_pattern(params)

        # Run simulation with cellular automata enabled
        results_3d = self.run_3d_sim(ca, params, precursor, printing_path, settings, case_name)

        # Print simulation results
        print(f"\nCA-enabled Simulation Results:")
        print(f"  - Total number of filled cells: {results_3d.process.filled_cells}")

    # ==================== CORE WORKFLOW METHODS ====================

    def get_or_run_simulation(self, case_name: str, D_override: float) -> tuple[SimulationResults1D, SimulationResults3D]:
        """
        Get cached simulation results or run new simulation.

        Args:
            case_name: Name of the test case
            D_override: Diffusion coefficient override (None = use file value)

        Returns:
            Tuple of (SimulationResults1D, SimulationResults3D)
        """
        cache_key = (case_name, D_override)

        # Check cache
        if cache_key not in self._simulation_cache:
            print("\n" + "="*80)
            print(f"CASE: {case_name}")
            print("="*80)

            # Run simulation
            results_1d, results_3d = self.run_simulation(D_override=D_override, case_name=case_name, ca=False)

            # Cache results
            self._simulation_cache[cache_key] = (results_1d, results_3d)

        return self._simulation_cache[cache_key]

    def extract_metrics(self, results_1d: SimulationResults1D, results_3d: SimulationResults3D) -> List[MetricResult]:
        """
        Extract three metrics from simulation results.

        Args:
            results_1d: SimulationResults1D object
            results_3d: SimulationResults3D object

        Returns:
            List of MetricResult objects [volume, center_coverage, edge_coverage]
        """
        metrics = []

        # Metric 1: Total deposited volume
        rel_error_vol = abs(results_1d.V - results_3d.V) / results_1d.V if results_1d.V != 0 else 0
        abs_error_vol = abs(results_1d.V - results_3d.V)
        passed_vol = rel_error_vol <= self.TOLERANCE

        metrics.append(MetricResult(
            metric_name="Deposited Volume",
            value_1d=results_1d.V,
            value_3d=results_3d.V,
            relative_error=rel_error_vol,
            absolute_error=abs_error_vol,
            tolerance=self.TOLERANCE,
            passed=passed_vol
        ))

        # Metric 2: Precursor coverage at center
        rel_error_center = abs(results_1d.theta_center - results_3d.theta_center) / results_1d.theta_center if results_1d.theta_center != 0 else 0
        abs_error_center = abs(results_1d.theta_center - results_3d.theta_center)
        passed_center = rel_error_center <= self.TOLERANCE

        metrics.append(MetricResult(
            metric_name="Coverage at Center",
            value_1d=results_1d.theta_center,
            value_3d=results_3d.theta_center,
            relative_error=rel_error_center,
            absolute_error=abs_error_center,
            tolerance=self.TOLERANCE,
            passed=passed_center
        ))

        # Metric 3: Precursor coverage at edge (far from beam)
        rel_error_edge = abs(results_1d.theta_edge - results_3d.theta_edge) / results_1d.theta_edge if results_1d.theta_edge != 0 else 0
        abs_error_edge = abs(results_1d.theta_edge - results_3d.theta_edge)
        passed_edge = rel_error_edge <= self.TOLERANCE

        metrics.append(MetricResult(
            metric_name="Coverage at Edge",
            value_1d=results_1d.theta_edge,
            value_3d=results_3d.theta_edge,
            relative_error=rel_error_edge,
            absolute_error=abs_error_edge,
            tolerance=self.TOLERANCE,
            passed=passed_edge
        ))

        return metrics

    def _assert_metric(self, metric: MetricResult):
        """
        Assert that a metric passes the tolerance test.
        (Private method - not a pytest test)

        Args:
            metric: MetricResult to test
        """
        assert metric.passed, (
            f"{metric.metric_name} failed: "
            f"1D={metric.value_1d:.6f}, 3D={metric.value_3d:.6f}, "
            f"rel_error={metric.relative_error*100:.3f}%, "
            f"tolerance={metric.tolerance*100:.3f}%"
        )

    def run_simulation(self, D_override: float = None, case_name: str = "", ca: bool = False) -> tuple[SimulationResults1D, SimulationResults3D]:
        """
        Run both 1D and 3D simulations and return results.

        Args:
            D_override: Override diffusion coefficient (None = use file value)
            case_name: Name for this test case
            ca: Enable cellular automata

        Returns:
            Tuple of (SimulationResults1D, SimulationResults3D)
        """
        params = self.params

        # Load settings and precursor using ruamel.yaml
        settings = load_yaml(params['settings_filename'])
        precursor = load_yaml(params['precursor_filename'])

        # Override diffusion coefficient if specified
        if D_override is not None:
            precursor['diffusion_coefficient'] = D_override

        # Generate point exposure pattern
        printing_path = create_pattern(params)

        # Setup 1D simulation for comparison
        results_1d = self.run_1d_sim(params, precursor, printing_path, settings, case_name)

        # Run 3D simulation
        results_3d = self.run_3d_sim(ca, params, precursor, printing_path, settings, case_name)

        # Return both results
        return results_1d, results_3d

    def run_3d_sim(self, ca, params, precursor, printing_path, settings, case_name: str = "", acceleration_enabled=True, structure=None) -> SimulationResults3D:
        print("\n" + "=" * 60)
        print("3D FEBID SIMULATION")
        print("=" * 60)

        # Setup simulation domain
        if structure is None:
            structure = self.setup_domain(params)

        # Prepare equation values for Process and MC_Simulation
        process = self.setup_rde_sim(params, precursor, settings, structure, acceleration_enabled=acceleration_enabled)

        # Create MC_Simulation
        sim = self.setup_mc_sim(precursor, settings, structure)

        # Setup nearest neighbors
        process.max_neib = int(np.max([sim.deponat.lambda_escape, sim.substrate.lambda_escape]))

        # Extract 3D parameters
        gauss_dev = settings.get('gauss_dev', 3.5)
        beam_current = settings.get('beam_current', 150e-12)
        S = precursor.get('sticking_coefficient', 1.0)
        n0 = precursor.get('max_density', 2.8)
        Phi = settings.get('precursor_flux', 1700)
        tau = precursor.get('residence_time', 100) * 1e-6
        sigma_param = precursor.get('cross_section', 0.022)
        D = precursor.get('diffusion_coefficient', 400000)
        Va = precursor.get('dissociated_volume', 0.094)
        e_charge = 1.602176634e-19
        se_yield = 0.67
        J0 = se_yield * beam_current / (2 * np.pi * (gauss_dev ** 2) * e_charge)

        cell_size = params['cell_size']
        R = max(params['width'], params['length']) / 2.0
        N = int(R / cell_size)
        dt_3d = process.dt
        t_end_3d = float(printing_path[:, 2].sum())

        params_3d = {
            'D': D,
            'S': S,
            'n0': n0,
            'Phi': Phi,
            'tau': tau,
            'sigma': sigma_param,
            'J0': J0,
            'beam_sigma': gauss_dev,
            'Va': Va,
            'R': R,
            'N': N,
            'dt': dt_3d,
            't_end': t_end_3d,
            'deposition_scaling': process.deposition_scaling,
            'grid_shape': structure.shape,
            'cell_size': structure.cell_size
        }

        # python
        # pretty print 3D parameters
        params_to_print = [
            ("D", D),
            ("S", S),
            ("n0", n0),
            ("Phi", Phi),
            ("tau", tau),
            ("sigma", sigma_param),
            ("J0", J0),
            ("beam_sigma", gauss_dev),
            ("Va", Va),
            ("R", R),
            ("N", N),
            ("dt", dt_3d),
            ("t_end", t_end_3d),
            ("deposition_scaling", process.deposition_scaling),
            ("grid_shape", structure.shape),
            ("cell_size", structure.cell_size),
        ]
        print("\n3D simulation key parameters:")
        maxk = max(len(k) for k, _ in params_to_print)
        for k, v in params_to_print:
            if isinstance(v, float):
                # use scientific notation for J0, compact general format otherwise
                val_str = f"{v:.3e}" if k == "J0" else f"{v:.6g}"
            else:
                val_str = str(v)
            print(f"  {k:<{maxk}} : {val_str}")

        self.run_sim(ca, printing_path, process, sim)

        # Extract precursor coverage profile for display
        s = structure.surface_bool.argmax(axis=0).max()
        l = structure.shape[1]
        c = structure.cell_size
        profile_3d = structure.precursor[s, l // 2, :]
        x = np.linspace(-l // 2 * c, l // 2 * c, l)

        # Get coverage at center and edge for display
        center_idx = l // 2
        edge_idx = 0
        theta_center_3d = profile_3d[center_idx]
        theta_edge_3d = structure.precursor[structure.substrate_height, 0, 0]

        print(f"\n3D Simulation Results:")
        print(f"  - Total deposited volume: {process._deposited_vol:.3f} nm³")
        print(f"  - Precursor coverage at center: {theta_center_3d:.4f} molecules/nm²")
        print(f"  - Precursor coverage at edge: {theta_edge_3d:.4f} molecules/nm²")
        print(f"  - Coverage ratio (center/edge): {theta_center_3d / theta_edge_3d:.4f}" if theta_edge_3d > 0 else "")

        return SimulationResults3D(
            D=D,
            case_name=case_name,
            params=params_3d,
            process=process,
            structure=structure
        )

    def run_1d_sim(self, params, precursor, printing_path, settings, case_name: str = "") -> SimulationResults1D:
        p1d = self.setup_1d_params(params, precursor, settings, printing_path)

        # Extract key parameters for reporting
        params_1d = {
            'D': p1d.D,
            'S': p1d.S,
            'n0': p1d.n0,
            'Phi': p1d.Phi,
            'tau': p1d.tau,
            'sigma': p1d.sigma,
            'J0': p1d.J0,
            'beam_sigma': p1d.beam_sigma,
            'Va': p1d.Va,
            'R': p1d.R,
            'N': p1d.N,
            'dt': p1d.dt,
            't_end': p1d.t_end
        }

        print("\n1D simulation key parameters:")
        print(f"  D={p1d.D}, S={p1d.S}, n0={p1d.n0}, Phi={p1d.Phi}, tau={p1d.tau}, sigma={p1d.sigma}")
        print(f"  J0={p1d.J0:.3e}, beam_sigma={p1d.beam_sigma}, Va={p1d.Va}")
        print(f"  R={p1d.R}, N={p1d.N}, dt={p1d.dt}, t_end={p1d.t_end}, snapshots={p1d.snapshots}")

        # Run 1D simulation
        print("\n" + "=" * 60)
        print("1D FEBID ANALYTICAL SOLUTION")
        print("=" * 60)
        V_1d, theta_center_1d, r, profile_snapshot = run_1d_simulation_metrics(p1d)
        n_replenished = p1d.n_replenished()
        n_depleted = p1d.n_depleted()

        print(f"\n1D Solution Results:")
        print(f"  - Total deposited volume: {V_1d:.2f} nm³")
        print(f"  - Final precursor coverage at r=0: {theta_center_1d:.4f} molecules/nm²")
        print(f"  - Replenished coverage n_r (far from beam): {n_replenished:.4f} molecules/nm²")
        print(f"  - Depleted coverage n_d (at beam center): {n_depleted:.4f} molecules/nm²")
        print(f"  - Depletion ratio n_d/n_r: {n_depleted / n_replenished:.4f}")
        print(f"  - Depletion factor: {(1 - n_depleted / n_replenished) * 100:.2f}%")

        return SimulationResults1D(
            D=p1d.D,
            case_name=case_name,
            params=params_1d,
            p1d=p1d,
            volume_calculated=V_1d,
            r=r,
            profile=profile_snapshot
        )

    def print_report(self, results: List[tuple[SimulationResults1D, SimulationResults3D]], all_metrics: List[List[MetricResult]]):
        """
        Print a comprehensive report comparing all cases in table format.

        Args:
            results: List of tuples (SimulationResults1D, SimulationResults3D) for each case
            all_metrics: List of metric lists for each case
        """
        print("\n" + "="*120)
        print("COMPREHENSIVE TEST REPORT: 1D vs 3D FEBID SIMULATION COMPARISON")
        print("="*120)

        # Print table header
        print("\n{:<20} {:<15} {:<20} {:<20} {:<15} {:<15} {:<10}".format(
            "Case", "Metric", "1D Value", "3D Value", "Abs Error", "Rel Error (%)", "Status"
        ))
        print("-" * 120)

        # Print metrics for each case
        for i, ((result_1d, result_3d), metrics) in enumerate(zip(results, all_metrics)):
            case_name = result_1d.case_name
            D_value = result_1d.D

            # Print case header
            print(f"\n{case_name} (D={D_value})")
            print("-" * 120)

            for metric in metrics:
                status = "✓ PASS" if metric.passed else "✗ FAIL"

                # Format values based on metric type
                if "Volume" in metric.metric_name:
                    val_1d_str = f"{metric.value_1d:.2f} nm³"
                    val_3d_str = f"{metric.value_3d:.2f} nm³"
                    abs_err_str = f"{metric.absolute_error:.2f} nm³"
                else:
                    val_1d_str = f"{metric.value_1d:.4f} mol/nm²"
                    val_3d_str = f"{metric.value_3d:.4f} mol/nm²"
                    abs_err_str = f"{metric.absolute_error:.4f} mol/nm²"

                rel_err_str = f"{metric.relative_error * 100:.3f}%"

                print("{:<20} {:<15} {:<20} {:<20} {:<15} {:<15} {:<10}".format(
                    "", metric.metric_name, val_1d_str, val_3d_str, abs_err_str, rel_err_str, status
                ))

        print("\n" + "="*120)
        print("SUMMARY")
        print("="*120)

        # Print summary statistics
        for i, ((result_1d, result_3d), metrics) in enumerate(zip(results, all_metrics)):
            passed_count = sum(1 for m in metrics if m.passed)
            total_count = len(metrics)
            print(f"{result_1d.case_name}: {passed_count}/{total_count} metrics passed")

        print("="*120 + "\n")

    def plot_comparison(self, results: List[tuple[SimulationResults1D, SimulationResults3D]]):
        """
        Plot precursor coverage profiles for all cases.

        Args:
            results: List of tuples (SimulationResults1D, SimulationResults3D) for each case
        """
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))

        if len(results) == 1:
            axes = [axes]

        for i, (result_1d, result_3d) in enumerate(results):
            ax = axes[i]

            # Plot 3D profile
            ax.plot(result_3d.x, result_3d.profile, 'b-', linewidth=2, label="3D simulation")

            # Plot 1D profile
            ax.plot(result_1d.r, result_1d.profile, 'r--', linewidth=2, label="1D simulation")

            # Plot 1D reference lines
            ax.axhline(y=result_1d.theta_edge, color='g', linestyle='--',
                      label=f'1D replenished: {result_1d.theta_edge:.4f}', alpha=0.7)
            ax.axhline(y=result_1d.theta_depleted, color='orange', linestyle='--',
                      label=f'1D depleted: {result_1d.theta_depleted:.4f}', alpha=0.7)
            ax.axhline(y=result_1d.theta_center, color='purple', linestyle=':',
                      label=f'1D center: {result_1d.theta_center:.4f}', alpha=0.7)

            # Plot 3D reference lines
            ax.axhline(y=result_3d.theta_edge, color='g', linestyle='-',
                      label=f'3D replenished: {result_3d.theta_edge:.4f}', alpha=0.5, linewidth=1.5)
            ax.axhline(y=result_3d.theta_center, color='purple', linestyle='-',
                      label=f'3D center: {result_3d.theta_center:.4f}', alpha=0.5, linewidth=1.5)
            # 3D depleted is the minimum value in the profile (at beam center)
            theta_depleted_3d = result_3d.profile.min()
            ax.axhline(y=theta_depleted_3d, color='orange', linestyle='-',
                      label=f'3D depleted: {theta_depleted_3d:.4f}', alpha=0.5, linewidth=1.5)

            ax.set_xlabel("Position [nm]")
            ax.set_ylabel("Precursor coverage [molecules/nm²]")
            ax.set_title(f"{result_1d.case_name}\nD={result_1d.D} nm²/s")
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_sim(self, ca, printing_path, process, sim):
        """
        Run the 3D FEBID simulation loop.
        This test loop follows the same algorithm as the real loop, but is modified to add extra external control.
        """
        # Setup progress bar
        progress_bar = self.setup_progress_bar(printing_path, process.deposition_scaling)
        # Simulation loop
        i = 0
        for x, y, dwell_time in printing_path:
            # Generate a sample beam intensity profile instead of the MC_Simulation
            beam_matrix = beam_matrix_test(x, y, sim=sim, pr=process)
            process.set_beam_matrix(beam_matrix)
            if i==0:
                print(f"Beam matrix peak: {beam_matrix.max()}")
                print(f"Beam matrix integral: {np.sum(beam_matrix) * process.cell_size**2:.1f}")
                print(f"Initial precursor_coverage: {process.structure.precursor.max():.5f}")
            process.x0, process.y0 = x, y
            time_passed = 0.0
            while time_passed < dwell_time:
                # Use test_beam_matrix instead of MC_Simulation
                process.deposition()
                process.precursor_density()
                process.t += process.dt * process.deposition_scaling
                time_passed += process.dt
                process.reset_dt()
                # process.dt *= 0.6
                if ca:
                    if process.check_cells_filled():
                        process.cell_filled_routine()
                        beam_matrix = beam_matrix_test(x, y, sim=sim, pr=process)
                        process.set_beam_matrix(beam_matrix)
                progress_bar.update((process.dt * process.deposition_scaling * 1e6))
                i += 1
        a=0

    def setup_progress_bar(self, printing_path, deposition_scaling=1):
        total_time = int(printing_path[:, 2].sum() * deposition_scaling * 1e6)
        bar_format = "{desc}: {percentage:.1f}%|{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        progress_bar = tqdm(total=total_time, desc='Patterning', position=0, unit='µs',
                            bar_format=bar_format)  # the execution speed is shown in µs of simulation time per s of real time
        return progress_bar

    def setup_mc_sim(self, precursor, settings, structure):
        mc_config = prepare_ms_config(precursor, settings, structure)
        sim = MC_Simulation(structure, mc_config)
        return sim

    def setup_rde_sim(self, params, precursor, settings, structure, acceleration_enabled=True):
        equation_values = prepare_equation_values(precursor, settings)
        process = Process(structure, equation_values, deposition_scaling=settings.get('deposition_scaling', 1),
                          temp_tracking=params['temperature_tracking'], acceleration_enabled=acceleration_enabled)
        return process

    def setup_domain(self, params):
        cell_size = params['cell_size']
        xdim = params['width'] // cell_size
        ydim = params['length'] // cell_size
        zdim = params['height'] // cell_size
        substrate_height = params['substrate_height'] // cell_size
        structure = Structure()
        structure.create_from_parameters(cell_size, xdim, ydim, zdim, substrate_height)
        return structure

    def setup_1d_params(self, params, precursor, settings, printing_path):
        """
        Create 1D FEBID simulation parameters from 3D simulation configuration.

        Args:
            params: Test case parameters
            precursor: Precursor YAML config
            settings: Settings YAML config
            printing_path: Pattern printing path array

        Returns:
            Params: 1D simulation parameters object
        """
        # Extract physical parameters from YAML configs
        gauss_dev = settings.get('gauss_dev', 3.5)  # nm, beam standard deviation
        beam_current = settings.get('beam_current', 150e-12)  # A
        beam_energy = settings.get('beam_energy', 30)  # keV

        # Precursor parameters
        S = precursor.get('sticking_coefficient', 1.0)
        n0 = precursor.get('max_density', 2.8)  # molecules/nm²
        Phi = settings.get('precursor_flux', 1700)  # molecules/(nm²·s)
        residence_time = precursor.get('residence_time', 100)  # µs
        tau = residence_time * 1e-6  # Convert to seconds
        sigma = precursor.get('cross_section', 0.022)  # nm²
        D = precursor.get('diffusion_coefficient', 400000)  # nm²/s
        Va = precursor.get('dissociated_volume', 0.094)  # nm³

        # Calculate electron flux J0 from beam current
        # J0 = I / (e * A), where A is the beam area
        # For Gaussian beam, peak flux J0 = I / (2*pi*sigma^2*e)
        e = 1.602176634e-19  # Elementary charge in C
        se_yield = 0.67 # Secondary electron yield
        total_pe = beam_current / e  # Primary electrons per second
        total_se = se_yield * total_pe  # Secondary electrons per second
        J0 = total_se / (2 * np.pi * gauss_dev**2)  # electrons/(nm²·s)
        print(f"Total secondary electrons per second: {total_se:.3e}, Peak flux J0: {J0:.3e} electrons/(nm²·s)")

        # Total dwell time from printing path
        total_dwell_time = printing_path[:, 2].sum()  # seconds

        # Simulation domain radius (should cover the beam + diffusion extent)
        R = max(params['width'], params['length']) / 2.0  # nm

        # Grid resolution
        cell_size = params['cell_size']
        N = int(R / cell_size)

        # Time step - use adaptive or based on expected dynamics
        dt = 1e-7  # seconds, conservative time step

        # Create 1D Params object
        p1d = Params(
            D=D,
            S=S,
            n0=n0,
            Phi=Phi,
            tau=tau,
            sigma=sigma,
            J0=J0,
            beam_sigma=gauss_dev,
            Va=Va,
            R=R,
            N=N,
            dt=dt,
            t_end=total_dwell_time,
            outer_bc="neumann",
            theta_outer=0.0,
            theta0=0.0,
            start_full=True,
            snapshots=(0.0,),
            max_candidates=100,
            num_profiles=2
        )

        return p1d


if __name__ == "__main__":
    tests = TestSimulationVolume()
    tests.examine_ca()
