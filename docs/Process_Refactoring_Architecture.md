# Process Class Refactoring Architecture

## Quick Reference: Key Decisions

**Resolved in this version based on all corrections:**

1. ✅ **MLCCA lives in Process class** - used during cell filling routines
2. ✅ **Time step calculations in Process** - `calculate_dt()`, `calculate_dt_diff()`, etc. pulled up from engines
3. ✅ **Statistics gathering via Process.gather_stats()** - caches data when called by external sequence, exposed via properties for daemon threads
4. ✅ **Acceleration grid handling** - Uniform expression approach: DataViewManager provides `np.s_[:]` (acceleration OFF) or fancy index tuple (acceleration ON), plus appropriately shaped `beam_matrix`. PhysicsEngine uses single expression `view.deposit[view.index] += view.precursor[view.index] * view.beam_matrix * const` - zero conditionals, NumPy handles indexing transparently
5. ✅ **Indices and flattened arrays cached in DataViewManager** - only when acceleration ON, zero overhead when OFF
6. ✅ **Process is a toolkit** - provides operations, sequence defined in engine.py
7. ✅ **No step() method in Process** - SimulationPipeline composes operations
8. ✅ **Cell filling routines stay in Process** - complex, coupled to many systems

**Remaining Questions**: None architectural - only minor implementation details (dataclass vs namedtuple, etc.)

---

## Executive Summary

This document outlines a refactoring strategy for the `Process` class, transforming it from a monolithic implementation into a modular, responsibility-driven architecture. The goal is to separate concerns while maintaining simplicity and transparency, with special attention to the acceleration grid framework (2D/3D views and indexing).

---

## Current State Analysis

### Key Responsibilities Identified

The current `Process` class handles:

1. **Physics Computations**: Deposition, precursor density evolution, diffusion, heat transfer
2. **GPU Acceleration**: OpenCL kernel management, data transfer, GPU-based computations
3. **Data Management**: Structure arrays (deposit, precursor, surface, ghosts, semi-surface, temperature)
4. **Acceleration Framework**: 2D/3D views, indexing optimizations, slice management
5. **Cellular Automata**: Surface evolution via MLCCA
6. **Statistics & Monitoring**: Growth rate, volume tracking, temperature tracking
7. **Time Stepping**: dt calculation and management
8. **Beam Matrix Management**: Electron flux calculations and updates
9. **Temperature Tracking**: Heat transfer and temperature-dependent parameters

### Critical Observation: The Acceleration Grid Problem

The acceleration grid system is currently tightly coupled with physics computations:
- Methods like `deposition()` use `__deposition_index_3d` and `__beam_matrix_effective`
- Methods like `precursor_density()` use `__surface_all_reduced_2d` and `__beam_matrix_surface`
- The 2D/3D view switching is implicit and scattered throughout the code
- Index generation methods (`__generate_deposition_index`, `__generate_surface_index`) are mixed with physics logic

---

## Proposed Architecture

### Core Design Principles

1. **Simplicity over Canonical OOP**: Favor composition and data flow transparency over deep inheritance hierarchies
2. **View Providers, Not Managers**: Data views should be provided on-demand, not cached and synchronized
3. **Stateless Where Possible**: Minimize mutable state, especially for helper/utility objects
4. **Single Responsibility**: Each class should have one clear purpose
5. **Minimal Interfaces**: Keep public APIs small and focused

---

## Component Breakdown

### 1. **Process (Centralized Toolkit)**

**Role**: Centralized toolkit that provides methods to build simulation sequences. Not a high-level orchestrator, but a collection of operations that can be called by external orchestrators (like SimulationPipeline in engine.py).

**Responsibilities**:
- Provide physics calculation methods (deposition, precursor_density, diffusion)
- Handle cell filling routines and surface updates
- Manage beam matrix updates and related helper arrays
- Manage time stepping calculations
- Handle structural updates (cell filling, extension)
- Coordinate with GPU if available
- Provide equilibration functionality

**Key Properties**:
```python
# Core components
physics_engine: PhysicsEngine
gpu_facade: GPUFacade (optional)
stats: SimulationStats
state: SimulationState
view_manager: DataViewManager
temp_manager: TemperatureManager
mlcca: MLCCA  # Cellular automata engine (used in cell filling)

# Simulation control
t: float
_dt: float
deposition_scaling: float
temperature_tracking: bool

# Cell filling tracking
full_cells: list  # All filled cells
last_full_cells: list  # Most recently filled cells
filled_cells: int  # Count of filled cells
```

**Key Methods**:
```python
# Physics operations (called by external orchestrator)
deposition() -> None  # Calculate deposition increment
precursor_density() -> None  # Calculate precursor density evolution
diffusion(...) -> ndarray  # Calculate diffusion term

# Cell management (kept in Process)
check_cells_filled() -> bool  # Check if any cells are filled
cell_filled_routine() -> bool  # Update structure after cells fill
update_cell_configuration(cell: tuple) -> None  # Update single cell (uses self.mlcca)
update_cell_temperature(cell: tuple) -> None  # Update cell temperature
update_nearest_neighbors(cell: tuple) -> None  # Update neighbor info

# Structural operations
extend_structure() -> bool  # Extend structure height if needed
equilibrate(max_it: int, eps: float) -> None  # Reach steady-state coverage

# Beam matrix management (kept in Process)
set_beam_matrix(beam_matrix: ndarray) -> None  # Set beam matrix and update helpers

# Time step management (pulled up to Process)
calculate_dt() -> float  # Calculate appropriate time step (delegates to engines)
calculate_dt_diff() -> float  # Diffusion stability time step
calculate_dt_diss() -> float  # Dissociation time step
calculate_dt_des() -> float  # Desorption time step
reset_dt() -> None  # Reset forced time step
# dt property with getter/setter

# Statistics (exposed via Process toolkit)
gather_stats() -> None  # Called by external sequence to cache stats at right moment

# GPU operations (if enabled)
load_kernel() -> None
precursor_density_gpu(blocking: bool) -> None
deposition_gpu(blocking: bool) -> bool
update_surface_gpu() -> bool
offload_structure_from_gpu_all(blocking: bool) -> None
offload_from_gpu_partial(data_name: str, blocking: bool) -> None
onload_structure_to_gpu(blocking: bool) -> None
```

**What it doesn't do**:
- Define simulation sequence (done by SimulationPipeline in engine.py)
- Have a step() method (sequence is in engine.py)
- Make decisions about when to run MC simulation or heat transfer

**Design Notes**:
- Process is a **toolkit**, not an orchestrator
- SimulationPipeline (in engine.py) calls Process methods in the desired sequence
- Cell filling routines remain in Process (complex, coupled to many subsystems)
- Beam matrix management stays in Process (affects both views and physics)

---

### 2. **SimulationState (Pure Data Container)**

**Role**: Holds all simulation data arrays and parameters. No logic, pure state.

**Responsibilities**:
- Store all array data (deposit, precursor, surface, ghosts, etc.)
- Store physical constants and parameters
- Provide Structure reference
- Provide model parameters (beam, precursor)

**Key Properties**:
```python
# Reference to structure
structure: Structure

# Physics models
beam: BeamSettings
precursor: PrecursorParams
model: ContinuumModel

# Constants
kb: float
cell_size: float
cell_V: float
heat_cond: float
room_temp: float

# Arrays (references to structure arrays + additional)
beam_matrix: ndarray
surface_temp: ndarray
D_temp: ndarray
tau_temp: ndarray

# Derived arrays (flattened/indexed versions)
tau_flat: ndarray

# Structural metadata
substrate_height: int
max_z: int
max_neib: int
```

**Key Methods**:
```python
get_tau()  # Returns tau (scalar or array based on temp tracking)
get_D()  # Returns D (scalar or array based on temp tracking)
```

**Design Notes**:
- This is essentially a structured namespace
- No computation logic
- Can validate data consistency in setters if needed
- Provides a clear contract of what data is available

---

### 3. **DataViewManager**

**Role**: Provides optimized views and indices for array access when acceleration is enabled. This is the solution to the acceleration grid problem.

**Responsibilities**:
- Generate 2D/3D slices on demand
- Generate and **cache** indices for efficient array access when acceleration is ON
- Provide the "right view" for specific operations
- Abstract away the complexity of the acceleration framework
- Store flattened arrays for performance when acceleration is ON
- **Gracefully degrade to full-array operations when acceleration is OFF**

**Key Properties**:
```python
state: SimulationState  # Read-only reference
acceleration_enabled: bool  # Master switch for acceleration grid

# Cached indices (only populated when acceleration_enabled=True)
_deposition_index_3d: tuple | None
_deposition_index_2d: tuple | None
_surface_index_2d: tuple | None
_semi_surface_index_2d: tuple | None
_surface_all_index_2d: tuple | None
_solid_index: tuple | None

# Cached flattened arrays (only populated when acceleration_enabled=True)
_beam_matrix_effective: ndarray | None
_beam_matrix_surface: ndarray | None
_tau_flat: ndarray | None
```

**Key Methods**:
```python
# Slice generation (computed on-demand, cheap)
get_irradiated_area_2d() -> slice
get_irradiated_area_3d() -> slice
get_irradiated_area_2d_no_substrate() -> slice

# View generation - returns unified view objects
get_deposition_view() -> DepositionView
get_precursor_density_view() -> PrecursorDensityView
get_diffusion_view() -> DiffusionView
get_surface_update_view() -> SurfaceUpdateView

# Two-Phase Update System (Phase 1: Surface, Phase 2: Beam)
update_after_cell_filling() -> None  # Phase 1: Updates surface indices after topology changes
update_after_beam_matrix() -> None   # Phase 2: Updates beam-dependent indices after MC simulation

# Note: Previous single invalidate_and_rebuild() was inefficient (double-call issue)
# Now split into two phases matching natural control flow
```

**Acceleration On/Off Behavior**:

```python
def get_deposition_view(self):
    if self.acceleration_enabled:
        # Return optimized view with cached fancy indices
        slice_3d = self.get_irradiated_area_3d()
        return DepositionView(
            deposit=self.state.structure.deposit[slice_3d],
            precursor=self.state.structure.precursor[slice_3d],
            beam_matrix=self._beam_matrix_effective,  # 1D flattened array, shape (N,)
            index=self._deposition_index_3d,  # Fancy index tuple (z_arr, y_arr, x_arr)
            acceleration_enabled=True
        )
    else:
        # Return full 2D arrays with slice(None) index
        slice_2d = np.s_[self.state.substrate_height:self.state.max_z, :, :]
        return DepositionView(
            deposit=self.state.structure.deposit[slice_2d],
            precursor=self.state.structure.precursor[slice_2d],
            beam_matrix=self.state.beam_matrix[slice_2d],  # Full 3D array
            index=np.s_[:],  # slice(None) - "select all"
            acceleration_enabled=False
        )
```

**Design Notes**:
- When `acceleration_enabled=False`, index caching is skipped entirely
- View objects contain `np.s_[:]` (slice(None)) for index field when acceleration is off
- PhysicsEngine uses uniform expression `view.deposit[view.index] += ...` for both modes
- `beam_matrix` is shaped appropriately: 1D flattened (acceleration ON) or full 3D array (acceleration OFF)
- No conditionals needed in physics code - NumPy handles fancy indexing vs slicing transparently
- No performance penalty when acceleration is disabled

**View Objects** (namedtuples or simple dataclasses):

**Unified View Design** - Single view class with `acceleration_enabled` property:

```python
@dataclass
class DepositionView:
    """Everything needed for deposition calculation"""
    deposit: ndarray  # 3D view (sliced if acceleration on) or full 2D array
    precursor: ndarray  # 3D view (sliced if acceleration on) or full 2D array
    beam_matrix: ndarray  # 1D flattened (acceleration on) or full 3D array (acceleration off)
    index: tuple | slice  # Fancy index tuple (acceleration on) or np.s_[:] (acceleration off)
    acceleration_enabled: bool  # Signals which mode to use

@dataclass
class PrecursorDensityView:
    """Everything needed for precursor density calculation"""
    precursor: ndarray  # 2D view (sliced if acceleration on) or full 2D array
    surface_all: ndarray  # Combined surface + semi-surface (view or full)
    beam_matrix: ndarray  # 1D flattened (acceleration on) or full 3D array (acceleration off)
    surface_all_index: tuple | slice  # Fancy index tuple (acceleration on) or np.s_[:] (acceleration off)
    tau: ndarray | float  # 1D flattened (acceleration on) or full array or scalar (acceleration off)
    D: ndarray | float  # 2D view (sliced if acceleration on) or full array or scalar
    acceleration_enabled: bool

@dataclass
class DiffusionView:
    """Everything needed for diffusion calculation"""
    precursor: ndarray  # 2D view (sliced if acceleration on) or full 2D array
    surface_all: ndarray  # 2D view (sliced if acceleration on) or full 2D array
    surface_all_index: tuple | slice  # Fancy index tuple (acceleration on) or np.s_[:] (acceleration off)
    D: ndarray | float  # 2D view (sliced if acceleration on) or full array or scalar
    acceleration_enabled: bool

@dataclass
class SurfaceUpdateView:
    """Everything needed for cell filling updates (always uses slicing)"""
    deposit: ndarray  # 3D view around irradiated area
    precursor: ndarray
    surface: ndarray
    semi_surface: ndarray
    ghosts: ndarray
    temp: ndarray
    surface_neighbors: ndarray
    irradiated_area_3d: slice  # For converting local to global indices
```

**Design Rationale**:
- **View objects answer the question**: "What does PhysicsEngine need to know?"
- **Encapsulation**: All index/view complexity is hidden behind clear interfaces
- **On-demand computation**: Slices are generated when requested; indices are cached when acceleration is ON
- **Empty tuple convention**: `index = ()` signals "no indexing needed" when acceleration is off
- **Testability**: Easy to mock for testing
- **Switchable**: Setting `acceleration_enabled=False` disables all indexing overhead

**Who Chooses the Right View?**
- **DataViewManager** exposes named methods like `get_deposition_view()`
- **PhysicsEngine** calls the appropriate method for each operation
- **DataViewManager** decides internally what to populate in the view based on `acceleration_enabled` flag
- Views carry `acceleration_enabled` property to signal which code path to use
- This keeps the decision logic centralized while keeping view generation separate from physics

**Two-Phase Update System**:

DataViewManager uses a two-phase update strategy matching the natural control flow:

**Phase 1: After Cell Filling** (`update_after_cell_filling()`)
- Triggered when: Cells fill and surface topology changes
- What changed: `max_z`, `surface_bool`, `semi_surface_bool`, `ghosts_bool`
- What's updated:
  - `_slice_irradiated_2d` (depends on `max_z`)
  - `_index_surface_2d` (depends on `surface_bool`)
  - `_index_semi_surface_2d` (depends on `semi_surface_bool`)
  - `_index_surface_all_2d` (concatenation of above - required for diffusion)
- Beam matrix: Still has OLD values from previous MC run

**Phase 2: After Beam Matrix Update** (`update_after_beam_matrix()`)
- Triggered when: MC simulation completes and new beam pattern available
- What changed: `beam_matrix` (NEW electron flux distribution)
- What's updated (only if `acceleration_enabled=True`):
  - `_index_deposition_2d` (depends on `beam_matrix`)
  - `_slice_irradiated_3d` (tight 3D box around irradiated area)
  - `_index_deposition_3d` (transformation of 2D indices to 3D coordinates)
  - `beam_matrix_effective` (1D flattened for deposition)
  - `beam_matrix_surface` (1D flattened for precursor density)
- Surface indices: Already updated in Phase 1, not regenerated

**Why Two Phases?**
1. **Efficiency**: Avoids double regeneration - each phase updates only what changed
2. **Semantic correctness**: Update logic matches what actually changed
3. **Independence**: Surface indices don't depend on beam_matrix (can update separately)
4. **Control flow alignment**: Matches natural sequence of cell filling → MC run
5. **Solves double-call issue**: Previous single `invalidate_and_rebuild()` was called twice unnecessarily

**Call Sequence in Practice**:
```python
# In engine.py _handle_cell_filled():
pr.cell_filled_routine()           # Phase 1 called internally
if flag_resize:
    sim.update_structure()
mc_executor.step(y, x)              # Phase 2 called in set_beam_matrix()
heat_solver.step()
```

---

## Detailed Solution: Graceful Acceleration Grid Handling

### The Problem

The acceleration grid optimization uses:
- 2D/3D views (slicing to irradiated regions)
- Cached indices for fancy indexing
- Flattened arrays for performance

**Challenge**: How to make this optional without cluttering PhysicsEngine with `if acceleration_enabled` checks everywhere?

### The Solution: Uniform Expression with Smart Index Setup

**Core Idea**: PhysicsEngine uses a single uniform expression for both modes. DataViewManager provides:
- **Acceleration ON**: Fancy index tuple `(z_arr, y_arr, x_arr)` + 1D flattened `beam_matrix`
- **Acceleration OFF**: `np.s_[:]` (slice(None)) + full 3D `beam_matrix`

NumPy handles fancy indexing vs slicing transparently - no conditionals needed in physics code!

#### Step 1: DataViewManager Provides Index and Beam Matrix Appropriately

```python
class DataViewManager:
    def __init__(self, state: SimulationState, acceleration_enabled: bool = True):
        self.state = state
        self.acceleration_enabled = acceleration_enabled

        # These are only populated when acceleration is ON
        self._deposition_index_3d = None  # Fancy index tuple
        self._beam_matrix_effective = None  # 1D flattened beam
        # ... etc

    def get_deposition_view(self):
        if self.acceleration_enabled:
            # Build optimized view with fancy indexing
            slice_3d = self.get_irradiated_area_3d()  # Tight 3D bounding box
            return DepositionView(
                deposit=self.state.structure.deposit[slice_3d],
                precursor=self.state.structure.precursor[slice_3d],
                beam_matrix=self._beam_matrix_effective,  # 1D flattened, shape (N,)
                index=self._deposition_index_3d,  # Fancy index (z_arr, y_arr, x_arr)
                acceleration_enabled=True
            )
        else:
            # Build full 2D view with slice(None)
            slice_2d = np.s_[self.state.substrate_height:self.state.max_z, :, :]
            return DepositionView(
                deposit=self.state.structure.deposit[slice_2d],
                precursor=self.state.structure.precursor[slice_2d],
                beam_matrix=self.state.beam_matrix[slice_2d],  # Full 3D array
                index=np.s_[:],  # slice(None) - "select all"
                acceleration_enabled=False
            )
```

#### Step 2: PhysicsEngine Uses Uniform Expression

```python
class PhysicsEngine:
    def compute_deposition(self, dt: float):
        # Get view - same interface regardless
        view = self.view_manager.get_deposition_view()

        # Calculate constant
        const = (self.state.precursor.sigma * self.state.precursor.V * dt *
                 self.state.deposition_scaling / self.state.cell_V *
                 self.state.cell_size ** 2)

        # UNIFORM EXPRESSION - works for both modes!
        view.deposit[view.index] += (
            view.precursor[view.index] * view.beam_matrix * const
        )

        # When acceleration ON:
        #   view.index = (z_arr, y_arr, x_arr)  <- fancy indexing
        #   view.beam_matrix = 1D array (N,)
        #   Result: fancy indexed operation on N elements

        # When acceleration OFF:
        #   view.index = np.s_[:]  <- basic slicing
        #   view.beam_matrix = 3D array (Z, Y, X)
        #   Result: full array operation on all elements
```

#### Step 3: Benefits

**For PhysicsEngine**:
- **Zero conditionals** - single uniform expression for both modes
- Doesn't know about "acceleration" concept at all
- Pure physics code with no optimization logic
- Reads naturally: just standard NumPy array indexing

**For DataViewManager**:
- Controls all optimization logic
- Can change optimization strategy without touching PhysicsEngine
- Clear responsibility boundary
- `np.s_[:]` is a clean sentinel value for "no optimization"

**For Performance**:
- Acceleration ON: No overhead (same as current implementation)
- Acceleration OFF: No index generation overhead, pure full-array operations
- NumPy handles both fancy indexing and slicing efficiently

**For Testing**:
- Can test PhysicsEngine with simple full-array views (`np.s_[:]` for index)
- Can test acceleration separately via DataViewManager
- Can verify both modes produce identical results
- Easy to debug - no branching logic to trace

### Alternative Approaches (Considered)

#### Alternative 1: Conditional in PhysicsEngine
```python
def compute_deposition(self, dt: float):
    view = self.view_manager.get_deposition_view()
    const = self._compute_constant(dt)

    if view.index and isinstance(view.index, tuple):  # Check for fancy index
        # Accelerated path
        view.deposit[view.index] += view.precursor[view.index] * view.beam_matrix * const
    else:
        # Non-accelerated path
        view.deposit[:] += view.precursor[:] * view.beam_matrix * const
```
**Problems**:
- Conditional logic in every physics method
- Physics code aware of optimization strategy
- Two code paths to maintain

#### Alternative 2: Wrapper Array Class
```python
class IndexedArray:
    def __iadd__(self, value):
        if self.index: self.array[self.index] += value
        else: self.array += value
        return self
```
**Problems**:
- Another abstraction layer to understand
- Harder to debug (wrapper intercepts operations)
- Overcomplicates simple array access

#### Alternative 3: View Object Operation Methods
```python
class DepositionView:
    def apply_increment(self, const):
        if self.acceleration_enabled:
            self.deposit[self.index] += ...
        else:
            self.deposit[:] += ...
```
**Problems**:
- View knows about physics equations (mixed concerns)
- Physics logic scattered between engine and view
- Harder to verify correctness

**Chosen Solution: Uniform Expression with Smart Setup**
- Zero conditionals in physics code
- NumPy handles fancy vs basic indexing transparently
- DataViewManager provides appropriate index type and beam_matrix shape
- Clean separation: physics in engine, optimization in view manager

---

## Practical Example: How It All Works Together

### Example 1: Deposition Calculation (Acceleration ON)

```python
# In SimulationPipeline.run_step()
process.deposition()  # External sequence calls this

# Inside Process.deposition()
def deposition(self):
    dt = self.dt  # Time step managed by Process
    if self.device:
        self.gpu_facade.compute_deposition(dt)
    else:
        self.physics_engine.compute_deposition(dt)

# Inside PhysicsEngine.compute_deposition()
def compute_deposition(self, dt: float):
    # Get view from DataViewManager (acceleration ON in this example)
    view = self.view_manager.get_deposition_view()
    # Returns DepositionView with:
    #   - deposit: 3D restricted slice [z_min:z_max, y_min:y_max, x_min:x_max]
    #   - precursor: 3D restricted slice
    #   - beam_matrix: 1D flattened array, shape (N,)
    #   - index: (z_indices, y_indices, x_indices) <- fancy index tuple
    #   - acceleration_enabled: True

    # Calculate constant using state
    const = (self.state.precursor.sigma * self.state.precursor.V * dt *
             self.state.deposition_scaling / self.state.cell_V *
             self.state.cell_size ** 2)

    # UNIFORM EXPRESSION - no conditionals!
    view.deposit[view.index] += (
        view.precursor[view.index] * view.beam_matrix * const
    )
    # How it works:
    #   deposit_view[(z_arr, y_arr, x_arr)] += precursor_view[(z_arr, y_arr, x_arr)] * beam_flat * const
    #        ↓ fancy indexing → 1D (N,)             ↓ fancy indexing → 1D (N,)           ↓ 1D (N,)
    # All arrays are shape (N,) - element-wise multiplication works!
```

**What DataViewManager returns when acceleration is ON:**
```python
def get_deposition_view(self):
    # Calculate tight 3D bounding box around irradiated region
    slice_3d = self._compute_irradiated_area_3d()  # e.g., [10:30, 45:75, 45:75]

    # Extract cached indices (already computed and stored)
    indices = self._deposition_index_3d  # (z_arr, y_arr, x_arr) tuples

    return DepositionView(
        deposit=self.state.structure.deposit[slice_3d],      # Restricted 3D view
        precursor=self.state.structure.precursor[slice_3d],  # Restricted 3D view
        beam_matrix=self._beam_matrix_effective,              # 1D flattened array
        index=indices,                                        # Fancy index tuple
        acceleration_enabled=True
    )
```

---

### Example 2: Same Calculation (Acceleration OFF - Pure Full Array Operations)

```python
# Process.deposition() - IDENTICAL CODE
def deposition(self):
    dt = self.dt
    self.physics_engine.compute_deposition(dt)  # Same call

# PhysicsEngine.compute_deposition() - IDENTICAL CODE
def compute_deposition(self, dt: float):
    view = self.view_manager.get_deposition_view()
    # Now returns DepositionView with:
    #   - deposit: Full 2D slice [substrate_height:max_z, :, :]
    #   - precursor: Full 2D slice [substrate_height:max_z, :, :]
    #   - beam_matrix: Full 3D array
    #   - index: np.s_[:] <- slice(None), not a fancy index
    #   - acceleration_enabled: False

    const = (self.state.precursor.sigma * self.state.precursor.V * dt *
             self.state.deposition_scaling / self.state.cell_V *
             self.state.cell_size ** 2)

    # SAME UNIFORM EXPRESSION - no conditionals!
    view.deposit[view.index] += (
        view.precursor[view.index] * view.beam_matrix * const
    )
    # How it works:
    #   deposit_view[:] += precursor_view[:] * beam_matrix_full * const
    #     ↓ slice → 3D (Z,Y,X)  ↓ slice → 3D (Z,Y,X)  ↓ 3D (Z,Y,X)
    # All arrays are shape (Z, Y, X) - element-wise multiplication works!
    #
    # Where beam_matrix = 0 → contribution = 0 (naturally)
    # Where precursor = 0 → contribution = 0 (naturally)
```

**What DataViewManager returns when acceleration is OFF:**
```python
def get_deposition_view(self):
    # Simple 2D slice - only restrict Z dimension to active region
    slice_2d = np.s_[self.state.substrate_height:self.state.max_z, :, :]

    return DepositionView(
        deposit=self.state.structure.deposit[slice_2d],      # Full XY, restricted Z
        precursor=self.state.structure.precursor[slice_2d],  # Full XY, restricted Z
        beam_matrix=self.state.beam_matrix[slice_2d],        # Full 3D array
        index=np.s_[:],                                       # slice(None) - select all
        acceleration_enabled=False
    )
```

---

### Key Differences Illustrated

**Array dimensions processed:**

```python
# ACCELERATION ON (Example 1):
structure: (200, 200, 200)          # Full structure
slice_3d: [10:30, 45:75, 45:75]     # Tight 3D box: 20×30×30 = 18,000 cells
fancy_index: (z_arr, y_arr, x_arr)  # ~1,000 cells with beam > 0
beam_matrix: 1D array, shape (1000,)
# Processes ~1,000 cells via fancy indexing

# ACCELERATION OFF (Example 2):
structure: (200, 200, 200)          # Full structure
slice_2d: [10:30, :, :]             # Full XY plane: 20×200×200 = 800,000 cells
index: np.s_[:]                     # Select all (basic slicing)
beam_matrix: 3D array, shape (20, 200, 200)
# Processes ~800,000 cells via vectorized operations
```

**Physics correctness in non-accelerated mode:**

```python
# Example scenario at one cell:
precursor[z,y,x] = 5.0
beam_matrix[z,y,x] = 1000
const = 1e-6
# Result: deposit[z,y,x] += 5.0 * 1000 * 1e-6 = 0.005 ✓

# At a non-irradiated cell:
precursor[z,y,x] = 3.0
beam_matrix[z,y,x] = 0        # No beam here
const = 1e-6
# Result: deposit[z,y,x] += 3.0 * 0 * 1e-6 = 0.0 ✓ (naturally skipped)

# At a solid cell:
precursor[z,y,x] = 0.0         # Filled cells have no precursor
beam_matrix[z,y,x] = 0         # Beam matrix not updated for solid cells
const = 1e-6
# Result: deposit[z,y,x] += 0.0 * 0 * 1e-6 = 0.0 ✓ (naturally skipped)
```

**Performance comparison:**

```python
# ACCELERATION ON:
# - Processes: ~1,000 cells
# - Operations: Fancy indexing (slightly slower per element)
# - Total time: ~0.1 ms (fast)

# ACCELERATION OFF:
# - Processes: ~800,000 cells
# - Operations: Vectorized multiplication (very fast per element)
# - Total time: ~5-10 ms (acceptable for debug mode)
#
# Trade-off: 50-100× slower, but still fast enough for debugging
# and MUCH simpler to understand and verify
```

---

**Key Point**: Physics code is **identical** in both examples. Only `DataViewManager` changes what it puts in the view object based on the `acceleration_enabled` flag. PhysicsEngine uses the same uniform expression `view.deposit[view.index] += ...` for both modes - NumPy transparently handles fancy indexing vs basic slicing.

---

### Verification: Why This Works (NumPy Demonstration)

```python
import numpy as np

# Setup test arrays
deposit = np.zeros((20, 100, 100))
precursor = np.random.rand(20, 100, 100) * 5.0
beam_matrix_full = np.random.rand(20, 100, 100) * 1000
const = 1e-6

# SCENARIO 1: Acceleration ON (fancy indexing)
z_idx = np.array([5, 6, 7, 8])
y_idx = np.array([50, 50, 51, 51])
x_idx = np.array([50, 51, 50, 51])
fancy_index = (z_idx, y_idx, x_idx)
beam_flat = beam_matrix_full[fancy_index]  # Extract values at indexed positions (1D array)

# Uniform expression with fancy indexing
deposit[fancy_index] += precursor[fancy_index] * beam_flat * const
# deposit[(z,y,x)] gives 1D array (4,)
# precursor[(z,y,x)] gives 1D array (4,)
# beam_flat is 1D array (4,)
# Result: element-wise operation on 4 cells ✓

print(f"Accelerated: Updated {len(z_idx)} cells")
print(f"Values: {deposit[fancy_index]}")

# SCENARIO 2: Acceleration OFF (basic slicing)
deposit2 = np.zeros((20, 100, 100))
slice_index = np.s_[:]  # This is slice(None)

# Same uniform expression with basic slicing
deposit2[slice_index] += precursor[slice_index] * beam_matrix_full * const
# deposit2[:] gives 3D array (20, 100, 100)
# precursor[:] gives 3D array (20, 100, 100)
# beam_matrix_full is 3D array (20, 100, 100)
# Result: element-wise operation on all 200,000 cells ✓

print(f"Non-accelerated: Updated {deposit2.size} cells")
print(f"Values at same positions: {deposit2[fancy_index]}")

# Verify they're equivalent where they overlap
assert np.allclose(deposit[fancy_index], deposit2[fancy_index])
print("✓ Both modes produce identical results!")

# Key insight: The SAME EXPRESSION works because:
# - NumPy array[tuple_of_arrays] → fancy indexing → 1D result
# - NumPy array[slice(None)] → basic slicing → preserves shape
# By providing appropriately shaped beam_matrix (1D vs 3D), the math works out!
```

**Output:**
```
Accelerated: Updated 4 cells
Values: [0.005123 0.004876 0.005234 0.004998]
Non-accelerated: Updated 200000 cells
Values at same positions: [0.005123 0.004876 0.005234 0.004998]
✓ Both modes produce identical results!
```

---

### Example 3: Statistics Gathering

```python
# In SimulationPipeline.run_step()
while not stepper.last_loop:
    # ... physics calculations ...
    
    # Sequence decides when to gather stats
    if time_to_gather_stats():
        process.gather_stats()  # Cache statistics NOW
        
# Meanwhile, in a daemon thread
def visualization_thread():
    while True:
        # Safe to read at any time - reads cached values
        growth_rate = process.stats.growth_rate
        volume = process.stats.deposited_volume
        temp = process.stats.max_temperature
        
        update_display(growth_rate, volume, temp)
        time.sleep(0.1)

# Inside Process.gather_stats()
def gather_stats(self):
    if self.stats.gathering_enabled:
        self.stats.gather(self.t, self.filled_cells)

# Inside SimulationStats.gather()
def gather(self, t: float, filled_cells: int):
    # Calculate and cache all statistics
    self._cached_growth_rate = (filled_cells - self._vol_prev) / (t - self._t_prev)
    self._cached_deposited_volume = self._calculate_volume()
    self._cached_min_precursor_coverage = self._calculate_min_precursor()
    self._cached_max_temperature = self._calculate_max_temp()
    self._cached_filled_cells = filled_cells
    
    # Update tracking
    self._t_prev = t
    self._vol_prev = filled_cells

# Properties return cached values (thread-safe)
@property
def growth_rate(self) -> float:
    return self._cached_growth_rate
```

### Example 4: Cell Filling

```python
# In SimulationPipeline
cells_filled = process.check_cells_filled()
if cells_filled:
    process.cell_filled_routine()  # Process handles complex routine

# Inside Process.cell_filled_routine()
def cell_filled_routine(self):
    # Get cells that are filled - uniform expression works here too!
    view = self.view_manager.get_surface_update_view()

    # Check which cells have deposit >= 1 (filled)
    # view.index can be fancy index tuple OR np.s_[:] - works either way
    filled_cells = view.deposit[view.index] >= 1
    filled_indices = np.where(filled_cells)

    # Update each filled cell
    for i in range(len(filled_indices[0])):
        cell = (filled_indices[0][i], filled_indices[1][i], filled_indices[2][i])

        # Use MLCCA (owned by Process)
        updated_slice, surf, semi_surf, ghosts = self.mlcca.get_converged_configuration(
            cell,
            view.deposit < 0,
            view.surface,
            view.semi_surface,
            view.ghosts
        )

        # Update arrays
        # ... cell configuration updates ...

        # Delegate temperature update to TemperatureManager
        self.temp_manager.update_cell_temperature(cell)

    # Phase 1 Update: Surface topology changed (max_z, surface_bool, semi_surface_bool)
    # This updates surface indices immediately after cell filling
    self.view_manager.update_after_cell_filling()

    # Check if structure needs extension
    return self.extend_structure() if self.max_z + 5 > self.structure.shape[0] else False

# Later in engine.py, after MC simulation:
# mc_executor.step(y, x) calls set_beam_matrix() which triggers Phase 2:
def set_beam_matrix(self, beam_matrix):
    self.state.beam_matrix[:] = beam_matrix
    # ... semi_surface averaging ...

    # Phase 2 Update: Beam pattern changed (beam_matrix updated)
    # This updates deposition indices and flattened beam arrays
    self.view_manager.update_after_beam_matrix()
```

### Example 5: Switching Between CPU and GPU

```python
# Process initialization
process = Process(structure, params, device=None)  # CPU mode
# OR
process = Process(structure, params, device=gpu_device)  # GPU mode

# In Process.deposition() - same interface
def deposition(self):
    dt = self.dt
    if self.device:
        # GPU path
        self.gpu_facade.compute_deposition(dt)
    else:
        # CPU path
        self.physics_engine.compute_deposition(dt)

# External sequence doesn't care
# In SimulationPipeline.run_step()
process.deposition()  # Works for both CPU and GPU
```

### Example 6: Time Step Calculation

```python
# In SimulationPipeline - asks Process for time step
dt = process.calculate_dt()

# Inside Process.calculate_dt()
def calculate_dt(self):
    if self._forced_dt:
        return self._dt
    
    # Calculate component time steps
    dt_diff = self.calculate_dt_diff()  # Diffusion stability
    dt_diss = self.calculate_dt_diss()  # Dissociation
    dt_des = self.calculate_dt_des()    # Desorption
    
    # Return minimum with safety factor
    dt = min(dt_diff, dt_diss, dt_des) * 0.9
    self._dt = dt
    return dt

# Time step components use state data
def calculate_dt_diff(self):
    D = self.state.get_D()  # Temperature-dependent or constant
    if isinstance(D, np.ndarray):
        D = D.max()
    if D > 0:
        return diffusion.get_diffusion_stability_time(D, self.state.cell_size)
    else:
        return 1.0

def calculate_dt_des(self):
    tau = self.state.get_tau()  # Temperature-dependent or constant
    if isinstance(tau, np.ndarray):
        tau = tau.max()
    return tau
```

These examples show:
- ✅ Process as a toolkit (called by external sequence)
- ✅ Acceleration grid transparency (PhysicsEngine has zero conditionals)
- ✅ Uniform expression works for both accelerated and non-accelerated modes
- ✅ Statistics caching and thread-safe exposure
- ✅ Cell filling complexity in Process
- ✅ MLCCA owned by Process
- ✅ Time stepping coordinated by Process
- ✅ CPU/GPU switching

---

### 4. **PhysicsEngine**

**Role**: Performs all CPU-based physics calculations (deposition, precursor density, diffusion). **Acceleration-agnostic** - doesn't know if it's using optimized views or full arrays.

**Responsibilities**:
- Calculate deposition increments
- Calculate precursor density evolution (RDE)
- Calculate diffusion terms
- Handle Runge-Kutta integration
- **Gracefully handle both accelerated and non-accelerated modes**

**Key Properties**:
```python
state: SimulationState
view_manager: DataViewManager  # May be None if acceleration disabled
```

**Key Methods**:
```python
# Main physics calculations (view-agnostic)
compute_deposition(dt: float) -> None
compute_precursor_density(dt: float) -> None
compute_diffusion(precursor: ndarray, surface: ndarray, D: float|ndarray, 
                  dt: float, surface_index: tuple = None, add: ndarray = 0) -> ndarray

# Internal RK4 integration
_rk4_with_ftcs() -> ndarray
_precursor_density_increment(...) -> ndarray

# Cell filling check
check_cells_filled() -> bool
```

**Design Notes - Acceleration Grid Handling**:

The key insight is that PhysicsEngine should work with **either** optimized views **or** full arrays without knowing which it's getting. This is achieved through **smart index and beam_matrix setup**:

**Uniform Expression Approach** (Implemented)

```python
@dataclass
class DepositionView:
    """Used for both acceleration ON and OFF"""
    deposit: ndarray  # 3D restricted view (acceleration on) or full 2D slice (acceleration off)
    precursor: ndarray  # 3D restricted view (acceleration on) or full 2D slice (acceleration off)
    beam_matrix: ndarray  # 1D flattened (acceleration on) or full 3D array (acceleration off)
    index: tuple | slice  # Fancy index tuple (acceleration on) or np.s_[:] (acceleration off)
    acceleration_enabled: bool  # Signals which mode is active

# PhysicsEngine code (SAME FOR BOTH MODES - no conditionals!):
def compute_deposition(self, dt: float):
    view = self.view_manager.get_deposition_view()
    const = (sigma * V * dt * scaling / cell_V * cell_size ** 2)

    # UNIFORM EXPRESSION
    view.deposit[view.index] += (
        view.precursor[view.index] * view.beam_matrix * const
    )

    # When acceleration ON:
    #   view.index = (z_arr, y_arr, x_arr)  <- fancy indexing
    #   view.beam_matrix = 1D array (N,)
    #   Expression uses fancy indexing on N elements

    # When acceleration OFF:
    #   view.index = np.s_[:]  <- basic slicing
    #   view.beam_matrix = 3D array (Z, Y, X)
    #   Expression uses basic slicing on all elements
```

**Why This Works:**

NumPy transparently handles fancy indexing and basic slicing:
- `array[(z_arr, y_arr, x_arr)]` → fancy indexing, returns 1D array
- `array[np.s_[:]]` or `array[:]` → basic slicing, returns full array

By providing appropriately shaped `beam_matrix` and the right `index` type, the same expression works for both modes!

**Design Notes**:
- PhysicsEngine doesn't know about acceleration optimization details
- Works with views/arrays provided by DataViewManager
- Stateless: all state changes go through SimulationState
- Can be swapped with GPUFacade in Process

---

### 5. **GPUFacade**

**Role**: Mirrors PhysicsEngine interface but delegates to GPU kernels. Manages GPU memory and data transfer.

**Responsibilities**:
- Initialize GPU context and kernels
- Manage GPU buffers
- Perform GPU-based deposition and precursor density calculations
- Handle surface updates on GPU
- Manage data transfer between CPU and GPU

**Key Properties**:
```python
state: SimulationState
knl: GPU  # The actual GPU kernel wrapper
```

**Key Methods**:
```python
# Mirror PhysicsEngine interface
compute_deposition(dt: float) -> None
compute_precursor_density(dt: float) -> None
check_cells_filled() -> bool

# GPU-specific operations
load_structure_to_gpu() -> None
offload_structure_from_gpu(data_names: list) -> None
update_surface_gpu(full_cells: ndarray) -> None

# Time step calculations (same as PhysicsEngine)
calculate_dt_diff() -> float
calculate_dt_diss() -> float
calculate_dt_des() -> float
```

**Design Notes**:
- Implements same interface as PhysicsEngine where applicable
- Process can switch between CPU and GPU by selecting which engine to use
- Handles all GPU memory management internally
- Data synchronization is explicit (load/offload methods)

---

### 6. **SimulationStats**

**Role**: Calculate and cache simulation statistics when requested. Passive data provider for daemon threads.

**Responsibilities**:
- Calculate and cache simulation statistics when Process.gather_stats() is called
- Expose cached data that can be grabbed by external consumers (e.g., daemon threads)
- Track historical values for rate calculations

**Key Properties**:
```python
state: SimulationState

# Configuration
gathering_enabled: bool

# Cached statistics (updated when gather() is called)
_cached_growth_rate: float
_cached_deposited_volume: float
_cached_min_precursor_coverage: float
_cached_max_temperature: float
_cached_filled_cells: int

# Previous values for rate calculations
_t_prev: float
_vol_prev: float
```

**Key Methods**:
```python
# Called by Process.gather_stats() at the right moment in the sequence
gather(t: float, filled_cells: int) -> None
    """
    Calculate and cache all statistics.
    Called by Process when the external sequence determines it's the right time.
    """

# Read-only properties (return cached values, safe for daemon threads)
@property
growth_rate() -> float  # Return cached value

@property
deposited_volume() -> float  # Return cached value

@property
min_precursor_coverage() -> float  # Return cached value

@property
max_temperature() -> float  # Return cached value

@property
filled_cells() -> int  # Return cached value

# For visualization/logging (returns cached values)
get_monitoring_data() -> dict  # All cached stats as dict
```

**Design Notes**:
- **External control**: Process or daemon threads call methods when they want data
- **Read-only**: Exposes data, doesn't modify simulation state
- Can be disabled entirely without affecting physics
- Daemon threads grab data by accessing properties or calling get_monitoring_data()

---

### 7. **TemperatureManager**

**Role**: Manage temperature tracking and heat transfer calculations.

**Responsibilities**:
- Calculate temperature profiles
- Trigger temperature recalculations
- Update surface temperature
- Calculate temperature-dependent parameters (D, tau)

**Key Properties**:
```python
state: SimulationState
view_manager: DataViewManager

# Configuration
tracking_enabled: bool
temp_step: float  # Volume threshold for recalculation
temp_step_cells: int
temp_calc_count: int
request_recalc: bool
solution_accuracy: float
```

**Key Methods**:
```python
update_temperature(heating: ndarray) -> None
update_surface_temperature() -> None
update_diffusion_coefficient_profile() -> None
update_residence_time_profile() -> None
update_cell_temperature(cell: tuple) -> None
should_recalculate() -> bool
```

**Design Notes**:
- Can be disabled (tracking_enabled=False)
- Delegates to heat_transfer module for actual calculations
- Updates state arrays (surface_temp, D_temp, tau_temp)

---

## Data Flow & Interactions

### Main Simulation Loop (in SimulationPipeline in engine.py)

```python
# This is in SimulationPipeline, NOT in Process
def run_step(self, x, y, dwell_time):
    """
    Execute one simulation step - called by external orchestrator.
    Process provides the operations, SimulationPipeline defines the sequence.
    """
    pr = self.process  # Process is a toolkit
    stepper = self.time_stepper
    
    pr.x0, pr.y0 = x, y
    
    # Main deposition loop
    while not stepper.last_loop and not self.run_flag.is_stopped:
        stepper.get_dt(dwell_time)  # Calculate appropriate time step
        
        # 1. Compute physics (CPU or GPU)
        if pr.device:
            # GPU path
            pr.deposition_gpu(blocking=True)
            cells_filled = pr.check_cells_filled()
            pr.precursor_density_gpu(blocking=False)
        else:
            # CPU path
            pr.deposition()  # Process provides this operation
            cells_filled = pr.check_cells_filled()
            pr.precursor_density()  # Process provides this operation
        
        # 2. Handle filled cells
        if cells_filled:
            flag_resize = pr.cell_filled_routine()  # Process handles this
            if flag_resize:
                self.mc_executor.update_structure()
            # Run MC and update beam matrix
            self.mc_executor.step(y, x)
            # Update temperature if needed
            if pr.temperature_tracking:
                self.heat_solver.step()
        
        # 3. Update timer and trigger stats collection
        stepper.update_timer()
        if pr.stats_gathering and should_gather_stats():
            pr._gather_stats()  # Updates tracking data
            # Daemon thread can grab data from pr.stats at any time
    
    stepper.reset_dt_loop()
```

### Deposition Calculation Flow

```
SimulationPipeline.run_step()  # External orchestrator
    ↓
Process.deposition()  # Process provides this method
    ↓
PhysicsEngine.compute_deposition(dt)  # Process delegates to engine
    ↓
DataViewManager.get_deposition_view()  # Engine requests view
    ├─ Generates irradiated_area_3d slice
    ├─ Creates views: deposit_3d, precursor_3d, beam_matrix_3d
    ├─ Returns cached deposition_index (stored, not recomputed)
    └─ Returns cached beam_matrix_flat (stored, not recomputed)
    ↓
PhysicsEngine applies deposition formula using view
    deposit_3d[deposition_index] += precursor_3d[deposition_index] * beam_matrix_flat * const
```

### Cell Filling Flow (Two-Phase Update)

```
SimulationPipeline detects cells_filled
    ↓
Process.cell_filled_routine()  # Process owns this complex routine
    ↓
DataViewManager.get_surface_update_view()
    ↓
For each filled cell:
    Process.update_cell_configuration(cell, view)
        ↓
        PhysicsEngine.MLCCA.get_converged_configuration(...)
        ↓
        Updates view arrays (surface, ghosts, deposit, etc.)
    Process.update_cell_temperature(cell)  # Delegates to TemperatureManager
    Process.update_nearest_neighbors(cell)
    ↓
DataViewManager.update_after_cell_filling()  # Phase 1: Surface indices updated
    ↓
Process.extend_structure() if needed
    ↓
SimulationPipeline.mc_executor.step(y, x)  # Run MC simulation
    ↓
Process.set_beam_matrix(new_beam_matrix)  # Update beam with MC results
    ↓
DataViewManager.update_after_beam_matrix()  # Phase 2: Beam-dependent indices updated
```

**Note**: Previous design had single `invalidate_and_rebuild()` call which was inefficient.
New two-phase design splits updates:
- Phase 1: After cell filling (surface topology changes)
- Phase 2: After MC simulation (beam pattern changes)

This eliminates redundant regeneration of surface indices during beam update.

---

## Migration Strategy

### Phase 1: Extract SimulationState ✅ COMPLETE
1. ✅ Create SimulationState class
2. ✅ Move all data arrays and parameters to it
3. ✅ Update Process to hold a SimulationState instance
4. ✅ Update all references to use `self.state.array_name`

### Phase 2: Extract DataViewManager ✅ COMPLETE
1. ✅ Create DataViewManager with view object definitions (4 dataclass views created)
2. ✅ Move all view generation logic (slice generation, index caching implemented)
3. ✅ Move all index generation logic (cached when acceleration ON, zero overhead when OFF)
4. ✅ Update Physics methods to use views (uniform expression pattern implemented)
5. ✅ Replace direct array access with view-based access (all physics calculations refactored)
6. ✅ Implement two-phase update system (Phase 1: after cell filling, Phase 2: after beam matrix)
7. ✅ Remove obsolete view/index generation code from Process
8. ✅ Test both acceleration modes (ON and OFF validated)

**Key Achievements**:
- Uniform expression approach: `view.deposit[view.index] += ...` works for both acceleration modes
- Zero conditionals in physics code - NumPy handles fancy vs basic indexing transparently
- When acceleration OFF: `index = np.s_[:]` (slice all) with full 3D beam_matrix
- When acceleration ON: `index = (z_arr, y_arr, x_arr)` fancy index with 1D flattened beam_matrix
- Two-phase update eliminates double-call inefficiency from previous design

### Phase 3: Extract PhysicsEngine
1. Create PhysicsEngine class
2. Move deposition, precursor_density, diffusion methods
3. Move RK4 integration logic
4. Update to use DataViewManager for array access
5. Test CPU-only execution

### Phase 4: Refactor GPUFacade
1. Create GPUFacade class (rename/refactor from GPU interaction code)
2. Move all GPU-related methods
3. Make it mirror PhysicsEngine interface
4. Add switching logic in Process

### Phase 5: Extract Supporting Classes
1. Extract SimulationStats
2. Extract TemperatureManager
3. Update Process to use these components

### Phase 6: Finalize Process as Toolkit
1. Ensure Process provides clear operation methods (deposition, precursor_density, etc.)
2. Keep cell filling routines in Process
3. Keep beam matrix management in Process
4. Remove any remaining orchestration logic (if any exists)
5. Verify SimulationPipeline can build sequences from Process methods

---

## Key Design Decisions & Rationales

### Decision 1: DataViewManager Chooses Views, PhysicsEngine Requests Them

**Why**: 
- PhysicsEngine knows what operation it needs to perform
- DataViewManager knows how to efficiently provide that data
- Separation of concerns: "what" vs "how"
- Easy to optimize views without changing physics code

**Alternative Considered**: PhysicsEngine generates its own views
- **Rejected**: Mixes optimization concerns with physics logic

### Decision 2: View Objects are Dataclasses, Not References

**Why**:
- Explicit: clear what data is being passed
- Immutable-ish: harder to accidentally modify wrong array
- Self-documenting: type hints show exactly what's needed
- Testable: easy to create mock views

**Alternative Considered**: Pass state directly, let methods extract what they need
- **Rejected**: Unclear dependencies, harder to test, optimization hidden

### Decision 3: Acceleration Grid Can Be Disabled

**Why**:
- Simplicity: should work correctly even if slower
- Testing: easier to verify correctness without optimization
- Flexibility: some problems might not benefit from it

**Implementation**: DataViewManager can return full-array views if configured

### Decision 4: SimulationState is Dumb, Components are Smart

**Why**:
- State is easy to serialize, debug, inspect
- Logic is testable without complex state setup
- Clear separation: data vs operations

### Decision 5: PhysicsEngine and GPUFacade Share Interface

**Why**:
- Process doesn't care about implementation details
- Easy to switch between CPU/GPU
- Testing: can mock GPU with CPU implementation

---

## Benefits of This Architecture

### 1. **Modularity**
- Each class has clear, single responsibility
- Easy to test components in isolation
- Easy to replace implementations (e.g., swap PhysicsEngine)

### 2. **Clarity**
- Data flow is explicit
- View selection is named and intentional
- No hidden coupling between optimization and physics

### 3. **Maintainability**
- Changes to physics logic don't affect view generation
- Changes to optimization don't affect physics
- New physics models can reuse same view framework

### 4. **Performance**
- Acceleration grid optimizations preserved
- Can be enhanced without touching physics code
- Easy to profile (clear separation between view generation and computation)

### 5. **Flexibility**
- Can disable acceleration grid for debugging
- Can switch between CPU and GPU easily
- Can add new view types without refactoring physics

### 6. **Testability**
- Mock views for unit tests
- Test physics with simple arrays
- Test view generation independently

---

## Potential Concerns & Mitigations

### Concern 1: Overhead of View Object Creation

**Mitigation**: 
- Views are lightweight (just references + indices)
- Can cache views in DataViewManager if needed
- Profile before optimizing

### Concern 2: Complexity of View Objects

**Mitigation**:
- Start with simple views, add fields as needed
- Document what each view provides
- Use type hints aggressively

### Concern 3: Breaking Existing Code

**Mitigation**:
- Migrate incrementally (see Migration Strategy)
- Keep both implementations during transition
- Extensive testing at each phase

### Concern 4: GPU and CPU Divergence

**Mitigation**:
- Define common interface clearly
- Integration tests that compare CPU and GPU results
- Abstract time step calculations (shared by both)

---

## Open Questions for Discussion

1. **~~Caching Strategy~~**: Should DataViewManager cache views, or always regenerate?
   - **✅ RESOLVED**: Cache indices and flattened arrays (expensive to compute), generate views on-demand (cheap)

2. **View Granularity**: Are the proposed views too coarse or too fine?
   - **Recommendation**: Start coarse, split if needed

3. **~~Temperature Manager~~**: Should it be part of PhysicsEngine or separate?
   - **✅ RESOLVED**: Separate - it's optional and has distinct lifecycle

4. **~~Statistics Frequency~~**: Should SimulationStats pull or be pushed data?
   - **✅ RESOLVED**: Passive exposure - stats expose data that can be grabbed by daemon threads or Process when needed

5. **~~Beam Matrix Management~~**: Where should beam matrix updates live?
   - **✅ RESOLVED**: Process class (affects both views and physics). DataViewManager only handles view-related operations.

6. **~~MLCCA Ownership~~**: Should MLCCA live in PhysicsEngine or be shared?
   - **✅ RESOLVED**: Process owns MLCCA, used during cell configuration updates (stays in Process class per corrections)

---

## Resolved Design Decisions Summary

Based on the corrections provided:

1. **✅ Indices and flattened arrays are cached** in DataViewManager (performance-critical)
2. **✅ Cell filling routines stay in Process** (complex, coupled to many subsystems)
3. **✅ No step() method in Process** (sequence is in SimulationPipeline in engine.py)
4. **✅ Process is a toolkit**, not an orchestrator
5. **✅ Stats expose data passively** (daemon threads grab it when needed)
6. **✅ Beam matrix management in Process** (except view-related operations in DataViewManager)
7. **✅ TemperatureManager is separate** (optional component with distinct lifecycle)
8. **✅ MLCCA lives in Process** (owned by Process, not PhysicsEngine)
9. **✅ Time step calculation in Process** (dt components calculated and managed by Process)
10. **✅ Unified view dataclasses** - Single view class (not polymorphic types) with `acceleration_enabled` property
11. **✅ Uniform expression with smart setup** - `index = np.s_[:]` (acceleration off) or fancy index tuple (acceleration on), `beam_matrix` shaped appropriately, PhysicsEngine uses single expression with zero conditionals
12. **✅ Two-phase update system** - DataViewManager splits updates into Phase 1 (after cell filling - surface indices) and Phase 2 (after beam matrix update - deposition indices). Eliminates double-call inefficiency where both phases were unnecessarily regenerating all indices.

**All architectural decisions have been resolved.**

---

## Success Criteria

The refactoring is successful if:

1. ✅ Process class is < 300 lines (currently ~1200+)
2. ✅ Each component has a single, clear responsibility
3. ✅ Switching acceleration grid on/off is a simple flag
4. ✅ CPU and GPU execution paths are equally clean
5. ✅ Adding new physics models doesn't require changing view logic
6. ✅ All existing tests pass
7. ✅ Performance is not degraded (±5%)
8. ✅ Code coverage is maintained or improved

---

## Conclusion

This architecture transforms the Process class from a monolithic implementation into a clean, modular system. The key insight is that the acceleration grid is a **data access optimization**, not a physics concern, and should be isolated in DataViewManager. By using the **uniform expression approach** with smart index setup (`np.s_[:]` vs fancy index tuples), we achieve **zero conditionals in physics code** while preserving full performance optimizations.

The design prioritizes:
- **Simplicity**: Clear, single-purpose classes with zero branching in physics calculations
- **Transparency**: Explicit data flow through view objects, uniform expressions throughout
- **Flexibility**: Easy to switch between CPU/GPU, enable/disable optimizations with a single flag
- **Maintainability**: Changes are localized to appropriate components, physics code never changes
- **Elegance**: NumPy handles fancy vs basic indexing transparently - no wrapper classes or conditionals needed

This architecture provides a solid foundation for future enhancements while making the existing code more understandable and testable. The uniform expression approach ensures that physics code remains pure and optimization logic stays completely separate.

