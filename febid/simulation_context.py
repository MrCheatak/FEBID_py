from dataclasses import dataclass
from febid.Process import Process
from febid.Structure import Structure
from febid.Statistics import Statistics, StructureSaver, SynchronizationHelper
from febid.monte_carlo.etraj3d import MC_Simulation

@dataclass
class SimulationContext:
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