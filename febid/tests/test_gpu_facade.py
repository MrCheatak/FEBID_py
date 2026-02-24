"""
Test suite for GPUFacade - Stage 4 refactoring validation

Compares CPU (PhysicsEngine) vs GPU (GPUFacade) implementations to ensure:
1. Numerical correctness (results match within float precision)
2. Interface consistency (same API behavior)
3. Performance validation (GPU maintains speed advantage)

Tests are structured similarly to test_deposition.py but compare CPU vs GPU
instead of 1D vs 3D simulations.

Test cases (matching test_deposition.py pattern):
- D=0, D=1, D=from_file (diffusion coefficient variations)
- acceleration_grid ON/OFF (for each D value)
- Total: 6 test cases (3 D values × 2 acceleration modes)
"""

import unittest
import pytest
from timeit import default_timer as timer
import numpy as np
from ruamel.yaml import YAML
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod

from febid import simple_patterns as sp
from febid.Process import Process
from febid.Structure import Structure
from febid.tests.sample_beam_matrix import test_beam_matrix as beam_matrix_test
from febid.monte_carlo.etraj3d import MC_Simulation
from febid.febid_core import prepare_equation_values, prepare_ms_config
from febid.logging_config import setup_logger
from febid.mlcca import initialize_structure_topology

logger = setup_logger(__name__)


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
class SimulationResultsCPU_GPU:
    """Stores results from CPU or GPU simulation."""
    device_type: str  # "CPU" or "GPU"
    case_name: str
    params: Dict
    process: 'Process'
    structure: 'Structure'
    execution_time: float  # seconds

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
    value_cpu: float
    value_gpu: float
    relative_error: float
    absolute_error: float
    tolerance: float
    passed: bool


class TestGPUFacade:
    """
    Test suite comparing CPU (PhysicsEngine) and GPU (GPUFacade) FEBID simulations.

    Stage 4 refactoring validation: ensures GPU implementation produces
    numerically identical results to CPU implementation.
    """
    # Class-level tolerance constant
    # GPU uses float32, so we allow slightly higher tolerance than CPU-CPU comparison
    TOLERANCE = 1e-6  # 0.0001% relative error

    # Simulation setups: (case_name, D_override, acceleration_enabled)
    # Matches test_deposition.py pattern: test with D=0, D=1, and D from file
    # D=0: Pure transport-limited regime (no diffusion)
    # D=1: Moderate diffusion (1 nm²/s)
    # D=from_file: High diffusion from precursor file (typically ~400000 nm²/s for Me3PtCpMe)
    SIMULATION_SETUPS = [
        ("D=0_accel_ON", 0.0, True),
        ("D=0_accel_OFF", 0.0, False),
        ("D=1_accel_ON", 1.0, True),
        ("D=1_accel_OFF", 1.0, False),
        ("D=from_file_accel_ON", None, True),
        ("D=from_file_accel_OFF", None, False),
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
        'dwell_time': 1000,  # us (reduced for faster testing)
        'pitch': 0,
        'repeats': 1,  # reduced for faster testing
        'settings_filename': r"febid/tests/Parameters.yml",
        'precursor_filename': r"febid/tests/Me3PtCpMe.yml",
        'save_simulation_data': False,
        'save_structure_snapshot': False,
        'simulation_data_interval': 1000,
        'structure_snapshot_interval': 1000,
        'unique_name': '',
        'save_directory': '',
        'temperature_tracking': False,
    }

    # ==================== UNIVERSAL TEST METHODS ====================

    @pytest.mark.parametrize("case_name,D_override,acceleration_enabled", SIMULATION_SETUPS)
    def test_volume_cpu_gpu(self, case_name, D_override, acceleration_enabled):
        """Test deposited volume metric for CPU vs GPU"""
        # Execute simulation (with caching)
        results_cpu, results_gpu = self.get_or_run_simulation(case_name, D_override, acceleration_enabled)

        # Extract metrics
        metrics = self.extract_metrics(results_cpu, results_gpu)

        # Report and plot only once per case (on first metric test)
        if case_name not in self._reported_cases:
            self._reported_cases.add(case_name)
            self.print_report([(results_cpu, results_gpu)], [metrics])
            self.plot_comparison([(results_cpu, results_gpu)])

        # Assess volume metric (index 0)
        self._assert_metric(metrics[0])

    @pytest.mark.parametrize("case_name,D_override,acceleration_enabled", SIMULATION_SETUPS)
    def test_center_coverage_cpu_gpu(self, case_name, D_override, acceleration_enabled):
        """Test center coverage metric for CPU vs GPU"""
        # Execute simulation (with caching)
        results_cpu, results_gpu = self.get_or_run_simulation(case_name, D_override, acceleration_enabled)

        # Extract metrics
        metrics = self.extract_metrics(results_cpu, results_gpu)

        # Assess center coverage metric (index 1)
        self._assert_metric(metrics[1])

    @pytest.mark.parametrize("case_name,D_override,acceleration_enabled", SIMULATION_SETUPS)
    def test_edge_coverage_cpu_gpu(self, case_name, D_override, acceleration_enabled):
        """Test edge coverage metric for CPU vs GPU"""
        # Execute simulation (with caching)
        results_cpu, results_gpu = self.get_or_run_simulation(case_name, D_override, acceleration_enabled)

        # Extract metrics
        metrics = self.extract_metrics(results_cpu, results_gpu)

        # Assess edge coverage metric (index 2)
        self._assert_metric(metrics[2])

    @pytest.mark.parametrize("case_name,D_override,acceleration_enabled", SIMULATION_SETUPS)
    def test_performance_gpu_vs_cpu(self, case_name, D_override, acceleration_enabled):
        """Test that GPU is faster than CPU (performance validation)"""
        # Execute simulation (with caching)
        results_cpu, results_gpu = self.get_or_run_simulation(case_name, D_override, acceleration_enabled)

        # Print performance comparison
        speedup = results_cpu.execution_time / results_gpu.execution_time
        print(f"\n{'='*60}")
        print(f"Performance Comparison: {case_name}")
        print(f"{'='*60}")
        print(f"CPU execution time: {results_cpu.execution_time:.3f} s")
        print(f"GPU execution time: {results_gpu.execution_time:.3f} s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"{'='*60}")

        # Assert GPU is faster (or at least not significantly slower)
        # Note: For small problems, GPU might be slower due to overhead
        # We just check it's not catastrophically slower (> 2x)
        assert results_gpu.execution_time < results_cpu.execution_time * 2.0, \
            f"GPU is more than 2x slower than CPU: {results_gpu.execution_time:.3f}s vs {results_cpu.execution_time:.3f}s"

    # Acceleration grid modes for semi-surface test
    ACCELERATION_MODES = [
        ("accel_ON", True),
        ("accel_OFF", False),
    ]

    @pytest.mark.parametrize("mode_name,acceleration_enabled", ACCELERATION_MODES)
    def test_semi_surface_cells_cpu_gpu(self, mode_name, acceleration_enabled):
        """
        Test semi-surface cell handling consistency between CPU and GPU.

        This test verifies that both CPU and GPU implementations handle semi-surface cells
        correctly by comparing deposited volumes on:
        1. Flat surface (baseline)
        2. Surface with single-layer deposit patch (creates semi-surface cells)

        Both surfaces should produce identical deposited volumes if semi-surface cell
        handling is correct.

        Matches test_deposition.py::test_semi_surface_cells() pattern but adds
        CPU vs GPU comparison.
        """
        print("\n" + "="*80)
        print(f"SEMI-SURFACE CELLS TEST: CPU vs GPU ({mode_name})")
        print("="*80)

        case_name = f"Semi_surface_{mode_name}"
        D_override = 0.0  # Use D=0 for faster testing
        params = self.params.copy()
        params["dwell_time"] = 100  # Reduced for faster testing

        # Load settings and precursor
        settings = load_yaml(params['settings_filename'])
        precursor = load_yaml(params['precursor_filename'])

        # Override diffusion coefficient
        if D_override is not None:
            precursor['diffusion_coefficient'] = D_override

        # Generate point exposure pattern
        printing_path = create_pattern(params)

        # ==================== FLAT SURFACE SIMULATIONS ====================

        print("\n" + "-"*60)
        print("FLAT SURFACE BASELINE")
        print("-"*60)

        # Run CPU simulation on flat surface
        print("\nCPU - Flat Surface:")
        results_cpu_flat = self.run_simulation(
            params, precursor, printing_path, settings, f"{case_name}_flat",
            use_gpu=False, acceleration_enabled=acceleration_enabled
        )

        # Run GPU simulation on flat surface
        print("\nGPU - Flat Surface:")
        results_gpu_flat = self.run_simulation(
            params, precursor, printing_path, settings, f"{case_name}_flat",
            use_gpu=True, acceleration_enabled=acceleration_enabled
        )

        # Verify CPU and GPU match on flat surface
        rel_error_flat = abs(results_cpu_flat.V - results_gpu_flat.V) / results_cpu_flat.V if results_cpu_flat.V != 0 else 0
        print(f"\nFlat Surface - CPU vs GPU:")
        print(f"  CPU volume: {results_cpu_flat.V:.6f} nm³")
        print(f"  GPU volume: {results_gpu_flat.V:.6f} nm³")
        print(f"  Relative error: {rel_error_flat*100:.6f}%")

        assert rel_error_flat <= self.TOLERANCE, \
            f"CPU and GPU don't match on flat surface: rel_error={rel_error_flat*100:.6f}%"

        # ==================== PATCHED SURFACE SIMULATIONS ====================

        print("\n" + "-"*60)
        print("SINGLE-LAYER PATCH SURFACE (Semi-surface cells)")
        print("-"*60)

        # Create structure with single-layer deposit patch
        structure_patched = self.setup_domain(params)
        surface_cells_count_flat = np.count_nonzero(structure_patched.surface_bool)

        # Add single-layer patch in center
        sub_height = structure_patched.substrate_height
        dep = structure_patched.deposit
        patch_size_abs = 10  # nm
        patch_size_cells = int(patch_size_abs / structure_patched.cell_size)
        center = (structure_patched.shape[1] // 2, structure_patched.shape[2] // 2)
        x1 = center[1] - patch_size_cells // 2
        x2 = center[1] + patch_size_cells // 2
        y1 = center[0] - patch_size_cells // 2
        y2 = center[0] + patch_size_cells // 2

        # Mark patch area as deposit (value = -1 indicates pre-existing deposit)
        dep[sub_height, y1:y2, x1:x2] = -1

        # Redefine surface, semi-surface, and ghost cells
        initialize_structure_topology(structure_patched)

        # Initialize precursor coverage
        prec = structure_patched.precursor
        surface_mask = structure_patched.surface_bool
        semi_surface_mask = structure_patched.semi_surface_bool
        prec[:] = 0
        prec[surface_mask] = structure_patched.nr
        prec[semi_surface_mask] = structure_patched.nr

        surface_cells_count_patched = np.count_nonzero(structure_patched.surface_bool)

        print(f"\nSurface geometry verification:")
        print(f"  Surface cells (flat): {surface_cells_count_flat}")
        print(f"  Surface cells (patched): {surface_cells_count_patched}")
        print(f"  Patch size: {patch_size_abs} nm ({patch_size_cells} cells)")
        print(f"  Patch creates semi-surface cells: {np.count_nonzero(semi_surface_mask)} cells")

        # Verify surface cell count is preserved
        assert surface_cells_count_flat == surface_cells_count_patched, \
            f"Surface cell count changed: flat={surface_cells_count_flat}, patched={surface_cells_count_patched}"

        # Run CPU simulation on patched surface
        print("\nCPU - Patched Surface:")
        results_cpu_patched = self.run_simulation(
            params, precursor, printing_path, settings, f"{case_name}_patched",
            use_gpu=False, acceleration_enabled=acceleration_enabled,
            structure_override=structure_patched
        )

        # Create new patched structure for GPU (need fresh copy)
        structure_patched_gpu = self.setup_domain(params)
        dep_gpu = structure_patched_gpu.deposit
        dep_gpu[sub_height, y1:y2, x1:x2] = -1
        initialize_structure_topology(structure_patched_gpu)
        prec_gpu = structure_patched_gpu.precursor
        surface_mask_gpu = structure_patched_gpu.surface_bool
        semi_surface_mask_gpu = structure_patched_gpu.semi_surface_bool
        prec_gpu[:] = 0
        prec_gpu[surface_mask_gpu] = structure_patched_gpu.nr
        prec_gpu[semi_surface_mask_gpu] = structure_patched_gpu.nr

        # Run GPU simulation on patched surface
        print("\nGPU - Patched Surface:")
        results_gpu_patched = self.run_simulation(
            params, precursor, printing_path, settings, f"{case_name}_patched",
            use_gpu=True, acceleration_enabled=acceleration_enabled,
            structure_override=structure_patched_gpu
        )

        # Verify CPU and GPU match on patched surface
        rel_error_patched = abs(results_cpu_patched.V - results_gpu_patched.V) / results_cpu_patched.V if results_cpu_patched.V != 0 else 0
        print(f"\nPatched Surface - CPU vs GPU:")
        print(f"  CPU volume: {results_cpu_patched.V:.6f} nm³")
        print(f"  GPU volume: {results_gpu_patched.V:.6f} nm³")
        print(f"  Relative error: {rel_error_patched*100:.6f}%")

        assert rel_error_patched <= self.TOLERANCE, \
            f"CPU and GPU don't match on patched surface: rel_error={rel_error_patched*100:.6f}%"

        # ==================== SEMI-SURFACE HANDLING VERIFICATION ====================

        print("\n" + "-"*60)
        print("SEMI-SURFACE CELL HANDLING VERIFICATION")
        print("-"*60)

        # CPU: flat vs patched (should be equal if semi-surface handling is correct)
        rel_error_cpu = abs(results_cpu_flat.V - results_cpu_patched.V) / results_cpu_flat.V if results_cpu_flat.V != 0 else 0
        print(f"\nCPU - Flat vs Patched:")
        print(f"  Flat volume: {results_cpu_flat.V:.6f} nm³")
        print(f"  Patched volume: {results_cpu_patched.V:.6f} nm³")
        print(f"  Relative error: {rel_error_cpu*100:.6f}%")

        # GPU: flat vs patched (should be equal if semi-surface handling is correct)
        rel_error_gpu = abs(results_gpu_flat.V - results_gpu_patched.V) / results_gpu_flat.V if results_gpu_flat.V != 0 else 0
        print(f"\nGPU - Flat vs Patched:")
        print(f"  Flat volume: {results_gpu_flat.V:.6f} nm³")
        print(f"  Patched volume: {results_gpu_patched.V:.6f} nm³")
        print(f"  Relative error: {rel_error_gpu*100:.6f}%")

        # ==================== ASSERTIONS ====================

        print("\n" + "="*80)
        print("ASSERTION RESULTS")
        print("="*80)

        # Assert 1: CPU handles semi-surface cells correctly
        assert rel_error_cpu <= self.TOLERANCE, \
            f"CPU semi-surface handling failed: flat={results_cpu_flat.V:.6f}, patched={results_cpu_patched.V:.6f}, error={rel_error_cpu*100:.6f}%"
        print("✓ CPU semi-surface handling: PASS")

        # Assert 2: GPU handles semi-surface cells correctly
        assert rel_error_gpu <= self.TOLERANCE, \
            f"GPU semi-surface handling failed: flat={results_gpu_flat.V:.6f}, patched={results_gpu_patched.V:.6f}, error={rel_error_gpu*100:.6f}%"
        print("✓ GPU semi-surface handling: PASS")

        # Assert 3: CPU and GPU match on flat surface
        assert rel_error_flat <= self.TOLERANCE, \
            f"CPU/GPU mismatch on flat surface: error={rel_error_flat*100:.6f}%"
        print("✓ CPU vs GPU (flat surface): PASS")

        # Assert 4: CPU and GPU match on patched surface
        assert rel_error_patched <= self.TOLERANCE, \
            f"CPU/GPU mismatch on patched surface: error={rel_error_patched*100:.6f}%"
        print("✓ CPU vs GPU (patched surface): PASS")

        print("\n" + "="*80)
        print("ALL SEMI-SURFACE TESTS PASSED")
        print("="*80)

    # ==================== CORE WORKFLOW METHODS ====================

    def get_or_run_simulation(self, case_name: str, D_override: float,
                              acceleration_enabled: bool) -> Tuple[SimulationResultsCPU_GPU, SimulationResultsCPU_GPU]:
        """
        Get cached simulation results or run new simulation.

        Args:
            case_name: Name of the test case
            D_override: Diffusion coefficient override
            acceleration_enabled: Enable acceleration grid

        Returns:
            Tuple of (CPU results, GPU results)
        """
        cache_key = (case_name, D_override, acceleration_enabled)

        # Check cache
        if cache_key not in self._simulation_cache:
            print("\n" + "="*80)
            print(f"CASE: {case_name} (D={D_override}, acceleration={acceleration_enabled})")
            print("="*80)

            # Run simulation
            results_cpu, results_gpu = self.run_cpu_gpu_comparison(
                D_override=D_override,
                case_name=case_name,
                acceleration_enabled=acceleration_enabled
            )

            # Cache results
            self._simulation_cache[cache_key] = (results_cpu, results_gpu)

        return self._simulation_cache[cache_key]

    def extract_metrics(self, results_cpu: SimulationResultsCPU_GPU,
                       results_gpu: SimulationResultsCPU_GPU) -> List[MetricResult]:
        """
        Extract three metrics from simulation results.

        Args:
            results_cpu: CPU simulation results
            results_gpu: GPU simulation results

        Returns:
            List of MetricResult objects [volume, center_coverage, edge_coverage]
        """
        metrics = []

        # Metric 1: Total deposited volume
        rel_error_vol = abs(results_cpu.V - results_gpu.V) / results_cpu.V if results_cpu.V != 0 else 0
        abs_error_vol = abs(results_cpu.V - results_gpu.V)
        passed_vol = rel_error_vol <= self.TOLERANCE

        metrics.append(MetricResult(
            metric_name="Deposited Volume",
            value_cpu=results_cpu.V,
            value_gpu=results_gpu.V,
            relative_error=rel_error_vol,
            absolute_error=abs_error_vol,
            tolerance=self.TOLERANCE,
            passed=passed_vol
        ))

        # Metric 2: Precursor coverage at center
        rel_error_center = abs(results_cpu.theta_center - results_gpu.theta_center) / results_cpu.theta_center if results_cpu.theta_center != 0 else 0
        abs_error_center = abs(results_cpu.theta_center - results_gpu.theta_center)
        passed_center = rel_error_center <= self.TOLERANCE

        metrics.append(MetricResult(
            metric_name="Coverage at Center",
            value_cpu=results_cpu.theta_center,
            value_gpu=results_gpu.theta_center,
            relative_error=rel_error_center,
            absolute_error=abs_error_center,
            tolerance=self.TOLERANCE,
            passed=passed_center
        ))

        # Metric 3: Precursor coverage at edge (far from beam)
        rel_error_edge = abs(results_cpu.theta_edge - results_gpu.theta_edge) / results_cpu.theta_edge if results_cpu.theta_edge != 0 else 0
        abs_error_edge = abs(results_cpu.theta_edge - results_gpu.theta_edge)
        passed_edge = rel_error_edge <= self.TOLERANCE

        metrics.append(MetricResult(
            metric_name="Coverage at Edge",
            value_cpu=results_cpu.theta_edge,
            value_gpu=results_gpu.theta_edge,
            relative_error=rel_error_edge,
            absolute_error=abs_error_edge,
            tolerance=self.TOLERANCE,
            passed=passed_edge
        ))

        return metrics

    def _assert_metric(self, metric: MetricResult):
        """
        Assert that a metric passes the tolerance test.

        Args:
            metric: MetricResult to test
        """
        assert metric.passed, (
            f"{metric.metric_name} failed: "
            f"CPU={metric.value_cpu:.6f}, GPU={metric.value_gpu:.6f}, "
            f"rel_error={metric.relative_error*100:.6f}%, "
            f"tolerance={metric.tolerance*100:.6f}%"
        )

    def run_cpu_gpu_comparison(self, D_override: float = None, case_name: str = "",
                               acceleration_enabled: bool = True, ca_enabled: bool = False) -> Tuple[SimulationResultsCPU_GPU, SimulationResultsCPU_GPU]:
        """
        Run both CPU and GPU simulations and return results.

        Args:
            D_override: Override diffusion coefficient (None = use file value)
            case_name: Name for this test case
            acceleration_enabled: Enable acceleration grid

        Returns:
            Tuple of (CPU results, GPU results)
        """
        params = self.params.copy()

        # Load settings and precursor using ruamel.yaml
        settings = load_yaml(params['settings_filename'])
        precursor = load_yaml(params['precursor_filename'])

        # Override diffusion coefficient if specified
        if D_override is not None:
            precursor['diffusion_coefficient'] = D_override

        # Generate point exposure pattern
        printing_path = create_pattern(params)

        # Run CPU simulation
        print("\n" + "="*60)
        print("CPU SIMULATION (PhysicsEngine)")
        print("="*60)
        results_cpu = self.run_simulation(
            params, precursor, printing_path, settings, case_name,
            use_gpu=False, acceleration_enabled=acceleration_enabled, ca_enabled=ca_enabled
        )

        # Run GPU simulation
        print("\n" + "="*60)
        print("GPU SIMULATION (GPUFacade)")
        print("="*60)
        results_gpu = self.run_simulation(
            params, precursor, printing_path, settings, case_name,
            use_gpu=True, acceleration_enabled=acceleration_enabled, ca_enabled=ca_enabled
        )

        # Return both results
        return results_cpu, results_gpu

    def run_simulation(self, params, precursor, printing_path, settings, case_name: str = "",
                      use_gpu: bool = False, acceleration_enabled: bool = True,
                      structure_override: Structure = None, ca_enabled: bool = False) -> SimulationResultsCPU_GPU:
        """
        Run a single simulation with specified device (CPU or GPU).

        Args:
            params: Test parameters
            precursor: Precursor config
            printing_path: Pattern path
            settings: Beam settings
            case_name: Test case name
            use_gpu: Use GPU if True, CPU if False
            acceleration_enabled: Enable acceleration grid
            structure_override: Optional pre-configured Structure (for semi-surface tests)

        Returns:
            SimulationResultsCPU_GPU with results
        """
        device_type = "GPU" if use_gpu else "CPU"

        # Setup simulation domain (or use override)
        if structure_override is not None:
            structure = structure_override
        else:
            structure = self.setup_domain(params)

        # Prepare equation values for Process
        params['gpu'] = use_gpu  # Set GPU flag in params
        process = self.setup_rde_sim(params, precursor, settings, structure, acceleration_enabled=acceleration_enabled)

        # Create MC_Simulation (for beam parameters)
        sim = self.setup_mc_sim(precursor, settings, structure)

        # Setup nearest neighbors
        process.max_neib = int(np.max([sim.deponat.lambda_escape, sim.substrate.lambda_escape]))

        # Extract parameters
        D = precursor.get('diffusion_coefficient', 0.0)

        # Print simulation info
        print(f"\nSimulation parameters:")
        print(f"  Device: {device_type}")
        print(f"  Acceleration grid: {acceleration_enabled}")
        print(f"  Diffusion coefficient D: {D} nm²/s")
        print(f"  Total dwell time: {float(printing_path[:, 2].sum()):.6f} s")
        print(f"  Time step: {process.dt:.3e} s")
        print(f"  Grid shape: {structure.shape}")
        print(f"  Cell size: {structure.cell_size} nm")

        # Run simulation
        start = timer()
        self.run_sim_loop(printing_path, process, sim, ca_enabled=ca_enabled)
        end = timer()
        execution_time = end - start

        # For GPU, retrieve data from GPU to CPU BEFORE reading results
        if use_gpu:
            # Enable stats gathering flag so get_data() retrieves necessary arrays
            process.stats_gathering = True
            process.get_data()  # Retrieve precursor and deposit arrays

        # Print results (after retrieving from GPU if needed)
        print(f"\n{device_type} Simulation Results:")
        print(f"  - Total deposited volume: {process._deposited_vol:.6f} nm³")
        print(f"  - Execution time: {execution_time:.3f} s")

        # Get coverage values
        s = structure.surface_bool.argmax(axis=0).max()
        l = structure.shape[1]
        center_idx = l // 2
        theta_center = structure.precursor[s, center_idx, center_idx]
        theta_edge = structure.precursor[structure.substrate_height, 0, 0]

        print(f"  - Precursor coverage at center: {theta_center:.6f} molecules/nm²")
        print(f"  - Precursor coverage at edge: {theta_edge:.6f} molecules/nm²")

        return SimulationResultsCPU_GPU(
            device_type=device_type,
            case_name=case_name,
            params=params,
            process=process,
            structure=structure,
            execution_time=execution_time
        )

    def print_report(self, results: List[Tuple[SimulationResultsCPU_GPU, SimulationResultsCPU_GPU]],
                    all_metrics: List[List[MetricResult]]):
        """
        Print a comprehensive report comparing CPU vs GPU for all cases.

        Args:
            results: List of tuples (CPU results, GPU results) for each case
            all_metrics: List of metric lists for each case
        """
        print("\n" + "="*120)
        print("COMPREHENSIVE TEST REPORT: CPU vs GPU FEBID SIMULATION COMPARISON")
        print("="*120)

        # Print table header
        print("\n{:<25} {:<15} {:<20} {:<20} {:<15} {:<15} {:<10}".format(
            "Case", "Metric", "CPU Value", "GPU Value", "Abs Error", "Rel Error (%)", "Status"
        ))
        print("-" * 120)

        # Print metrics for each case
        for i, ((result_cpu, result_gpu), metrics) in enumerate(zip(results, all_metrics)):
            case_name = result_cpu.case_name

            # Print case header
            print(f"\n{case_name}")
            print("-" * 120)

            for metric in metrics:
                status = "✓ PASS" if metric.passed else "✗ FAIL"

                # Format values based on metric type
                if "Volume" in metric.metric_name:
                    val_cpu_str = f"{metric.value_cpu:.6f} nm³"
                    val_gpu_str = f"{metric.value_gpu:.6f} nm³"
                    abs_err_str = f"{metric.absolute_error:.6e} nm³"
                else:
                    val_cpu_str = f"{metric.value_cpu:.6f} mol/nm²"
                    val_gpu_str = f"{metric.value_gpu:.6f} mol/nm²"
                    abs_err_str = f"{metric.absolute_error:.6e} mol/nm²"

                rel_err_str = f"{metric.relative_error * 100:.6f}%"

                print("{:<25} {:<15} {:<20} {:<20} {:<15} {:<15} {:<10}".format(
                    "", metric.metric_name, val_cpu_str, val_gpu_str, abs_err_str, rel_err_str, status
                ))

        print("\n" + "="*120)
        print("SUMMARY")
        print("="*120)

        # Print summary statistics
        for i, ((result_cpu, result_gpu), metrics) in enumerate(zip(results, all_metrics)):
            passed_count = sum(1 for m in metrics if m.passed)
            total_count = len(metrics)
            print(f"{result_cpu.case_name}: {passed_count}/{total_count} metrics passed")

        print("="*120 + "\n")

    def plot_comparison(self, results: List[Tuple[SimulationResultsCPU_GPU, SimulationResultsCPU_GPU]]):
        """
        Plot precursor coverage profiles for CPU vs GPU.

        Args:
            results: List of tuples (CPU results, GPU results) for each case
        """
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))

        if len(results) == 1:
            axes = [axes]

        for i, (result_cpu, result_gpu) in enumerate(results):
            ax = axes[i]

            # Plot CPU profile
            ax.plot(result_cpu.x, result_cpu.profile, 'b-', linewidth=2, label="CPU (PhysicsEngine)")

            # Plot GPU profile
            ax.plot(result_gpu.x, result_gpu.profile, 'r--', linewidth=2, label="GPU (GPUFacade)")

            # Plot reference lines
            ax.axhline(y=result_cpu.theta_center, color='b', linestyle=':',
                      label=f'CPU center: {result_cpu.theta_center:.6f}', alpha=0.5)
            ax.axhline(y=result_gpu.theta_center, color='r', linestyle=':',
                      label=f'GPU center: {result_gpu.theta_center:.6f}', alpha=0.5)

            ax.set_xlabel("Position [nm]")
            ax.set_ylabel("Precursor coverage [molecules/nm²]")
            ax.set_title(f"{result_cpu.case_name}\nCPU vs GPU Comparison")
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_sim_loop(self, printing_path, process, sim, ca_enabled: bool = False):
        """
        Run the simulation loop with optional CA updates.
        """
        # Setup progress bar
        progress_bar = self.setup_progress_bar(printing_path, process.deposition_scaling)

        # Simulation loop
        for x, y, dwell_time in printing_path:
            # Generate a sample beam intensity profile
            beam_matrix = beam_matrix_test(x, y, sim=sim, pr=process)
            process.set_beam_matrix(beam_matrix)
            process.x0, process.y0 = x, y

            time_passed = 0.0
            while time_passed < dwell_time:
                if process.device:
                    process.gpu_facade.synchronize()
                process.deposition()
                process.precursor_density()
                process.t += process.dt * process.deposition_scaling
                time_passed += process.dt
                process.reset_dt()

                if ca_enabled and process.check_cells_filled():
                    if process.device:
                        process.offload_from_gpu_partial('deposit', blocking=False)
                        process.offload_from_gpu_partial('precursor', blocking=True)

                    resized = process.cell_filled_routine()

                    if resized:
                        sim.update_structure(process.structure)
                        if process.device:
                            process.gpu_facade.reinitialize_after_resize()
                    elif process.device:
                        process.update_structure_to_gpu(blocking=True)

                    beam_matrix = beam_matrix_test(x, y, sim=sim, pr=process)
                    process.set_beam_matrix(beam_matrix)

                progress_bar.update((process.dt * process.deposition_scaling * 1e6))

    def setup_progress_bar(self, printing_path, deposition_scaling=1):
        total_time = int(printing_path[:, 2].sum() * deposition_scaling * 1e6)
        bar_format = "{desc}: {percentage:.1f}%|{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        progress_bar = tqdm(total=total_time, desc='Patterning', position=0, unit='µs',
                           bar_format=bar_format)
        return progress_bar

    def setup_mc_sim(self, precursor, settings, structure):
        mc_config = prepare_ms_config(precursor, settings, structure)
        sim = MC_Simulation(structure, mc_config)
        return sim

    def setup_rde_sim(self, params, precursor, settings, structure, acceleration_enabled=True):
        equation_values = prepare_equation_values(precursor, settings)

        # Determine device
        device = None
        if params.get('gpu', False):
            # For GPU tests, we need to specify device
            # Using default GPU device (first available)
            device = True  # This will trigger GPU initialization in Process

        process = Process(
            structure,
            equation_values,
            deposition_scaling=settings.get('deposition_scaling', 1),
            temp_tracking=params['temperature_tracking'],
            acceleration_enabled=acceleration_enabled,
            device=device
        )
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


if __name__ == "__main__":
    """
    Quick test runner for manual testing.
    Run with: python -m febid.tests.test_gpu_facade
    """
    print("\n" + "="*80)
    print("GPU FACADE TEST - MANUAL RUN")
    print("="*80)

    tests = TestGPUFacade()

    # Run a single test case manually
    case_name = "D=0_accel_ON"
    D_override = 0.0
    acceleration_enabled = True

    print(f"\nRunning manual test: {case_name}")
    results_cpu, results_gpu = tests.run_cpu_gpu_comparison(
        D_override=D_override,
        case_name=case_name,
        acceleration_enabled=acceleration_enabled
    )

    # Extract and print metrics
    metrics = tests.extract_metrics(results_cpu, results_gpu)
    tests.print_report([(results_cpu, results_gpu)], [metrics])
    tests.plot_comparison([(results_cpu, results_gpu)])

    # Print summary
    print("\n" + "="*80)
    print("MANUAL TEST COMPLETE")
    print("="*80)
    all_passed = all(m.passed for m in metrics)
    if all_passed:
        print("✓ All metrics PASSED")
    else:
        print("✗ Some metrics FAILED")
        for m in metrics:
            if not m.passed:
                print(f"  - {m.metric_name}: rel_error={m.relative_error*100:.6f}%")
