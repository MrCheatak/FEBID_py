from dataclasses import dataclass, field
from febid.Process import Process
from febid.Structure import Structure
from febid.Statistics import Statistics, StructureSaver, SynchronizationHelper
from febid.monte_carlo.etraj3d import MC_Simulation
from typing import Optional, Tuple, Any

@dataclass
class SimulationParameters:
    """Store validated user-facing simulation input parameters."""

    structure_source: str = 'vtk'  # 'vtk', 'geom', or 'auto'
    vtk_filename: Optional[str] = None
    geom_parameters_filename: Optional[str] = None
    width: Optional[float] = None
    length: Optional[float] = None
    height: Optional[float] = None
    cell_size: Optional[float] = None
    substrate_height: Optional[float] = None
    pattern_source: str = 'simple'  # 'simple' or 'stream_file'
    pattern: Optional[str] = None
    param1: Optional[float] = None
    param2: Optional[float] = None
    dwell_time: Optional[float] = None
    pitch: Optional[float] = None
    repeats: Optional[int] = None
    stream_file_filename: Optional[str] = None
    hfw: Optional[float] = None
    settings_filename: Optional[str] = None
    precursor_filename: Optional[str] = None
    temperature_tracking: bool = False
    save_simulation_data: bool = False
    save_structure_snapshot: bool = False
    simulation_data_interval: Optional[float] = None
    structure_snapshot_interval: Optional[float] = None
    unique_name: Optional[str] = None
    save_directory: Optional[str] = None
    gpu: bool = False
    # Add more fields as needed

    def validate(self) -> None:
        """Validate enum-like parameter values for structure and pattern sources.

        :return: None
        """
        # Example validation logic
        if self.structure_source not in {'vtk', 'geom', 'auto'}:
            raise ValueError(f"Invalid structure_source: {self.structure_source}")
        if self.pattern_source not in {'simple', 'stream_file'}:
            raise ValueError(f"Invalid pattern_source: {self.pattern_source}")
        # Add more validation as needed

@dataclass
class SimulationContext:
    """Carry runtime objects and configuration shared across simulation components."""

    structure: Structure = None
    precursorParams: object = None
    settings: object = None
    simParams: object = None
    printingPath: str = None
    temperatureTracking: bool = False
    savingParams: object = None
    device: object = None

    process: Process = None
    mcSimulation: MC_Simulation = None
    syncHelper: SynchronizationHelper = None
