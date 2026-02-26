"""
SimulationStats - Statistics gathering and caching for FEBID simulation

This module provides passive statistics collection with external control:
- Statistics are calculated and cached only when gather() is called
- Daemon threads access pre-calculated values via properties (thread-safe)
- Can be disabled entirely without affecting physics

Design pattern: Cache-on-request with read-only properties
"""

from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatsFlags:
    """Control which statistics to calculate during gather().

    Allows selective calculation to skip expensive operations if
    consumers (UI/logging) don't need certain stats.
    """
    volume: bool = True           # Calculate deposited volume (expensive: array sum)
    coverage: bool = True          # Calculate min precursor coverage (expensive: array min)
    temperature: bool = False      # Calculate max temperature (only if temp enabled)
    rates: bool = True             # Calculate growth rates (cheap: arithmetic)


class SimulationStats:
    """Calculate and cache simulation statistics when requested.

    Passive data provider - external orchestrator (Process or SimulationPipeline)
    decides when to update statistics by calling gather(). Daemon threads and
    consumers access cached values via properties.

    Thread-safe by design: properties return pre-calculated values without
    computation, so readers never block.

    Attributes
    ----------
    state : SimulationState
        Reference to simulation data
    temp_manager : TemperatureManager or None
        Temperature manager (if thermal effects enabled)
    gathering_enabled : bool
        Master switch for statistics gathering
    calc_flags : StatsFlags
        Control which statistics to calculate
    stats_frequency : float
        Time interval between gather() calls (seconds)
    validate_on_gather : bool
        Enable validation checks for debugging (default: False)
    """

    def __init__(self, state, temp_manager=None, gathering_enabled=True,
                 stats_frequency=0.1, validate_on_gather=False):
        """
        Initialize SimulationStats.
        
        :param state: Reference to simulation data
        :type state: SimulationState
        :param temp_manager: Temperature manager (if thermal effects enabled)
        :type temp_manager: TemperatureManager, optional
        :param gathering_enabled: Master switch for statistics gathering
        :type gathering_enabled: bool, default=True
        :param stats_frequency: Time interval between gather() calls (seconds)
        :type stats_frequency: float, default=0.1
        :param validate_on_gather: Enable validation checks for debugging
        :type validate_on_gather: bool, default=False
        """
        self.state = state
        self.temp_manager = temp_manager
        self.gathering_enabled = gathering_enabled
        self.stats_frequency = stats_frequency
        self.validate_on_gather = validate_on_gather

        # Calculation control
        self.calc_flags = StatsFlags()
        if temp_manager is None or not temp_manager.enabled:
            self.calc_flags.temperature = False

        # Rate calculation settings
        self._min_rate_window = 0.001  # Minimum dt for rate calculations (seconds)

        # Cached statistics (updated by gather())
        self._cached_time: float = 0.0
        self._cached_filled_cells: int = 0
        self._cached_growth_rate: float = 0.0
        self._cached_deposited_volume: float = 0.0
        self._cached_min_precursor_coverage: float = 0.0
        self._cached_max_temperature: float = 0.0

        # Historical values for rate calculations
        self._t_prev: float = 0.0
        self._vol_prev: int = 0

        # Freshness tracking (Improvement #1)
        self._last_gather_time: float = 0.0
        self._gather_call_count: int = 0

        logger.info(f"SimulationStats initialized: gathering_enabled={gathering_enabled}, "
                   f"frequency={stats_frequency}s")

    def gather(self, t: float, filled_cells: int) -> None:
        """
        Calculate and cache all enabled statistics.
        
        Called by external orchestrator (Process.gather_stats() or SimulationPipeline)
        when it's time to update statistics. Performs expensive calculations once
        and caches results for daemon threads to read.
        
        :param t: Current simulation time (seconds)
        :type t: float
        :param filled_cells: Number of filled cells (deposit >= 1.0)
        :type filled_cells: int
        """
        if not self.gathering_enabled:
            return

        # Store passed-in values (cheap)
        self._cached_time = t
        self._cached_filled_cells = filled_cells

        # Calculate rates (cheap - arithmetic only)
        if self.calc_flags.rates:
            self._update_rates(t, filled_cells)

        # Calculate expensive statistics (only if enabled)
        if self.calc_flags.volume:
            self._cached_deposited_volume = self._calculate_volume()

        if self.calc_flags.coverage:
            self._cached_min_precursor_coverage = self._calculate_min_precursor()

        if self.calc_flags.temperature and self.temp_manager and self.temp_manager.enabled:
            self._cached_max_temperature = self.temp_manager.max_temperature

        # Track freshness (Improvement #1)
        self._last_gather_time = t
        self._gather_call_count += 1

        # Validation checks (Improvement #4 - debug mode only)
        if self.validate_on_gather:
            self._validate_stats()

    def _update_rates(self, t: float, filled_cells: int) -> None:
        """
        Update rate calculations with minimum time window validation.
        
        Improvement #3: Prevents noisy/infinite rates from tiny time steps.
        
        :param t: Current simulation time
        :type t: float
        :param filled_cells: Current number of filled cells
        :type filled_cells: int
        """
        dt = t - self._t_prev

        # Avoid division by zero or noisy rates from tiny time steps
        if dt < self._min_rate_window:
            return  # Keep previous rate

        # Calculate growth rate (cells/second)
        self._cached_growth_rate = (filled_cells - self._vol_prev) / dt

        # Update historical values for next calculation
        self._t_prev = t
        self._vol_prev = filled_cells

    def _calculate_volume(self) -> float:
        """
        Calculate total deposited volume.
        
        EXPENSIVE: Scans arrays over irradiated area.
        
        :return: (float) Total deposited volume (nm³)
        """
        # Get irradiated area slice
        s = self._get_irradiated_area_slice()

        # Volume from filled cells
        filled_volume = self._cached_filled_cells * self.state.cell_V

        # Volume from partial surface cells
        deposit = self.state.structure.deposit[s]
        surface = self.state.structure.surface_bool[s]
        partial_volume = deposit[surface].sum() * self.state.cell_V

        return filled_volume + partial_volume

    def _calculate_min_precursor(self) -> float:
        """
        Calculate minimum precursor coverage on surface.
        
        EXPENSIVE: Scans arrays over irradiated area.
        
        :return: (float) Minimum precursor density on surface (1/nm²)
        """
        # Get irradiated area slice
        s = self._get_irradiated_area_slice()

        precursor = self.state.structure.precursor[s]
        surface = self.state.structure.surface_bool[s]

        # Handle case where no surface cells exist
        surface_precursor = precursor[surface]
        if surface_precursor.size == 0:
            return 0.0

        return surface_precursor.min()

    def _get_irradiated_area_slice(self):
        """
        Get slice encapsulating the irradiated surface area.
        
        :return: (slice) 2D slice from substrate height to max z coordinate
        """
        return np.s_[self.state.substrate_height - 1:self.state.max_z, :, :]

    def _validate_stats(self) -> None:
        """Validate that statistics are physically reasonable.

        Improvement #4: Debug mode checks to catch physics bugs early.
        Only runs if validate_on_gather is True.
        """
        # Volume should be monotonic increasing (or equal)
        if self._gather_call_count > 1:
            prev_volume = (self._vol_prev * self.state.cell_V)
            if self._cached_deposited_volume < prev_volume - 1e-9:  # Small tolerance for FP errors
                logger.warning(
                    f"Volume decreased: {prev_volume:.6e} -> {self._cached_deposited_volume:.6e} nm³"
                )

        # Coverage should be non-negative
        if self._cached_min_precursor_coverage < 0:
            logger.warning(
                f"Negative precursor coverage: {self._cached_min_precursor_coverage:.6e}"
            )

        # Growth rate should be reasonable (not NaN or inf)
        if not np.isfinite(self._cached_growth_rate):
            logger.warning(
                f"Invalid growth rate: {self._cached_growth_rate}"
            )

        # Temperature should be reasonable (if tracking)
        if self.calc_flags.temperature:
            if self._cached_max_temperature < 0:
                logger.warning(
                    f"Negative temperature: {self._cached_max_temperature:.2f} K"
                )
            elif self._cached_max_temperature > 10000:  # Arbitrary high limit
                logger.warning(
                    f"Unreasonably high temperature: {self._cached_max_temperature:.2f} K"
                )

    # ========== Read-only Properties (return cached values) ==========

    @property
    def time(self) -> float:
        """Current simulation time (seconds)."""
        return self._cached_time

    @property
    def filled_cells(self) -> int:
        """Number of filled cells (deposit >= 1.0)."""
        return self._cached_filled_cells

    @property
    def growth_rate(self) -> float:
        """Growth rate (cells/second)."""
        return self._cached_growth_rate

    @property
    def deposited_volume(self) -> float:
        """Total deposited volume (nm³)."""
        return self._cached_deposited_volume

    @property
    def min_precursor_coverage(self) -> float:
        """Minimum precursor coverage on surface (1/nm²)."""
        return self._cached_min_precursor_coverage

    @property
    def max_temperature(self) -> float:
        """Maximum temperature in structure (K)."""
        return self._cached_max_temperature

    # ========== Freshness Tracking (Improvement #1) ==========

    @property
    def data_age(self) -> float:
        """
        Time elapsed since last gather (for debugging/logging).
        
        :return: (float) Time since last gather (seconds). Returns 0 if never gathered.
        """
        if self._gather_call_count == 0:
            return 0.0
        return self._cached_time - self._last_gather_time

    @property
    def is_stale(self) -> bool:
        """True if no gather has occurred yet.

        Useful for detecting uninitialized state during startup.
        """
        return self._gather_call_count == 0

    @property
    def gather_count(self) -> int:
        """Number of times gather() has been called."""
        return self._gather_call_count

    # ========== Differential Stats (Improvement #6) ==========

    @property
    def volume_delta(self) -> float:
        """Volume added since last gather (nm³).

        Useful for progress bars and incremental logging.
        """
        prev_volume = self._vol_prev * self.state.cell_V
        return self._cached_deposited_volume - prev_volume

    @property
    def cells_filled_delta(self) -> int:
        """Cells filled since last gather.

        Useful for progress tracking and adaptive time stepping.
        """
        return self._cached_filled_cells - self._vol_prev

    # ========== Convenience Methods ==========

    def get_monitoring_data(self) -> dict:
        """
        Get all cached statistics as a dictionary.
        
        Useful for visualization, logging, or serialization.
        
        :return: (dict) Dictionary with all cached statistics
        """
        return {
            'time': self._cached_time,
            'filled_cells': self._cached_filled_cells,
            'growth_rate': self._cached_growth_rate,
            'deposited_volume': self._cached_deposited_volume,
            'min_precursor_coverage': self._cached_min_precursor_coverage,
            'max_temperature': self._cached_max_temperature,
            'volume_delta': self.volume_delta,
            'cells_filled_delta': self.cells_filled_delta,
            'gather_count': self._gather_call_count,
        }

    def reset_rates(self) -> None:
        """Reset rate calculation history.

        Call this when simulation context changes (e.g., beam position change)
        to avoid calculating rates across discontinuous time periods.
        """
        self._t_prev = self._cached_time
        self._vol_prev = self._cached_filled_cells
        self._cached_growth_rate = 0.0
        logger.debug("Rate calculation history reset")
