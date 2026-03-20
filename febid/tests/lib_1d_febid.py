"""
Core implementation library for 1D FEBID radial reaction-diffusion solver.

This module contains all the core functionality for the 1D FEBID simulation,
separated from demo/execution code to allow safe importing.

FEBID 1D radial (cylindrical) reaction–diffusion solver for surface coverage θ(r,t).

PDE (planar surface, radial symmetry):
    ∂θ/∂t = D * (1/r) ∂/∂r ( r ∂θ/∂r ) + S*Φ*(1 - θ) - θ/τ - σ*J(r)*θ

- D            : surface diffusivity [m^2/s]
- S*Φ          : adsorption flux to the surface in monolayer/s (with θ normalized to [0,1])
- 1/τ          : thermal desorption rate [1/s]
- σ*J(r)       : electron-induced dissociation (effective first-order) [1/s]
- θ            : normalized surface coverage (0..1)

Discretization:
- Space: finite difference on [0, R], N+1 nodes, uniform Δr.
  Radial Laplacian handled with symmetry at r=0 (∂θ/∂r = 0) and user-chosen outer BC.
- Time: IMEX (Backward Euler for diffusion, Explicit for reactions):
    (I - dt*D*L) θ^{n+1} = θ^n + dt * R(θ^n, r)

Where L is the discrete radial Laplacian operator and R is the reaction source term.
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple, Dict, List

OuterBC = Literal["neumann", "dirichlet"]

@dataclass
class Params:
    """
    Parameter container for the 1D FEBID radial reaction-diffusion solver.
    
    Key attributes:
    * ``D`` (float): Surface diffusivity [nm^2/s]
    * ``S`` (float): Sticking coefficient (dimensionless)
    * ``Phi`` (float): Precursor flux [molecules/(m^2 s)]
    * ``tau`` (float): Thermal desorption time [s]
    * ``sigma`` (float): Electron-induced dissociation cross-section [nm^2]
    * ``J0`` (float): Peak electron flux [1/m^2]
    * ``beam_sigma`` (float): Beam width (1-sigma) [nm]
    * ``R`` (float): Simulation domain radius [m]
    * ``N`` (int): Number of spatial grid points
    * ``dt`` (float): Time step [s]
    * ``t_end`` (float): End time [s]
    * ``outer_bc`` (OuterBC): Outer boundary condition type
    * ``theta_outer`` (float): Value for Dirichlet BC at r=R
    * ``theta0`` (float): Initial coverage (if not start_full)
    * ``start_full`` (bool): If True, start with theta=1 everywhere
    * ``snapshots`` (Tuple[float, ...]): Times to save snapshots
    """
    D: float = 2e6
    S: float = 1
    n0: float = 2.7
    Phi: float = 1700
    tau: float = 0.0001
    sigma: float = 0.022
    J0: float = 9.2e5
    beam_sigma: float = 8
    Va: float = 0.95
    R: float = 100
    N: int = 800
    dt: float = 1e-6
    t_end: float = 0.002
    outer_bc: OuterBC = "neumann"
    theta_outer: float = 0.0
    theta0: float = 0.0
    start_full: bool = True
    snapshots: Tuple[float, ...] = (0.0, 1e-5, 2e-5, 5e-5, 0.001)
    max_candidates: int = 100       # how many candidate snapshots to save during run
    num_profiles: int = 5          # if >0, choose this many profiles from the candidate batch

    def n_replenished(self) -> float:
        """
        Calculate the replenished (steady-state) precursor coverage far from the beam.
        
        This is the equilibrium coverage where adsorption balances thermal desorption
        (no electron-induced depletion).
        
        Formula: n_r = S*Phi / (S*Phi/n0 + 1/tau)
        
        :return: float: Replenished precursor coverage [molecules/nm²]
        """
        return self.S * self.Phi / (self.S * self.Phi / self.n0 + 1.0 / self.tau)

    def n_depleted(self) -> float:
        """
        Calculate the depleted (steady-state) precursor coverage at the beam center.
        
        This is the equilibrium coverage at r=0 where adsorption balances both
        thermal desorption and electron-induced dissociation.
        
        Formula: n_d = S*Phi / (S*Phi/n0 + 1/tau + sigma*J0)
        
        :return: float: Depleted precursor coverage at beam center [molecules/nm²]
        """
        return self.S * self.Phi / (self.S * self.Phi / self.n0 + 1.0 / self.tau + self.sigma * self.J0)

def generate_round_candidates(t_end: float, n_candidates: int = 100):
    """
    Generate up to `n_candidates` candidate times in [0, t_end] snapped to
    'round' values (mantissas 1,2,5 times powers of ten). Always includes 0 and t_end.
    """
    if t_end <= 0.0:
        return [0.0]
    nice = {0.0, t_end}
    exp_min = int(np.floor(np.log10(t_end))) - 3
    exp_max = int(np.ceil(np.log10(t_end))) + 3
    for e in range(exp_min, exp_max + 1):
        for m in range(10):
            val = m * (10.0 ** e)
            if 0.0 <= val <= t_end:
                nice.add(val)
    nice_list = sorted(nice)

    uniq_sorted = nice_list
    return uniq_sorted

def select_snapshots_by_center(snaps_all: Dict[float, np.ndarray],
                               candidate_times: List[float],
                               num_profiles: int,
                               ensure_final: float = None) -> Tuple[List[float], Dict[float, np.ndarray]]:
    """
    Choose `num_profiles` times from `candidate_times` so that the center values
    (snaps_all[t][0]) are approximately evenly spaced.

    Returns (selected_times, selected_snaps_dict).

    Behavior:
    - If center values are monotonic in time, invert center->time by selecting
      target center levels and taking nearest candidates.
    - If centers are not monotonic, fall back to evenly spaced indices in time.
    - Always prefers unique times; if duplicates arise the nearest unused is chosen.
    - If `ensure_final` is provided and available in snaps_all, it will be set as the last selected time.
    """
    times = [t for t in sorted(candidate_times) if t in snaps_all]
    if len(times) == 0:
        return [], {}

    centers = np.array([snaps_all[t][0] for t in times])

    # trivial cases
    if num_profiles <= 0 or num_profiles >= len(times):
        selected = times.copy()
        if ensure_final is not None and ensure_final in snaps_all and selected[-1] != ensure_final:
            if ensure_final in selected:
                # move ensure_final to end
                selected.remove(ensure_final)
            selected.append(ensure_final)
        return selected, {t: snaps_all[t] for t in selected}

    # check monotonicity (allow tiny numerical noise)
    d = np.diff(centers)
    monotonic_increasing = np.all(d >= -1e-12)
    monotonic_decreasing = np.all(d <= 1e-12)

    if monotonic_increasing or monotonic_decreasing:
        # target center levels evenly spaced between endpoints
        targets = np.linspace(centers[0], centers[-1], num_profiles)
        selected_idxs = []
        used = set()
        for tgt in targets:
            idx = int(np.argmin(np.abs(centers - tgt)))
            if idx in used:
                # pick nearest unused
                order = np.argsort(np.abs(centers - tgt))
                for j in order:
                    if int(j) not in used:
                        idx = int(j)
                        break
            used.add(idx)
            selected_idxs.append(idx)
        selected = [times[i] for i in selected_idxs]
    else:
        # fallback: evenly spaced in time
        idxs = np.linspace(0, len(times) - 1, num_profiles).round().astype(int)
        selected = [times[i] for i in idxs]

    # ensure final time if requested and available
    if ensure_final is not None and ensure_final in snaps_all:
        if ensure_final not in selected:
            selected[-1] = ensure_final

    selected = sorted(dict.fromkeys(selected), key=lambda x: x)  # preserve order, unique
    return selected, {t: snaps_all[t] for t in selected}


def gaussian_beam_J(r, J0, sigma_b):
    """
    Returns the electron flux profile J(r) for a Gaussian beam.
    
    :param r: Radial grid points.
    :type r: np.ndarray
    :param J0: Peak electron flux [A/m^2].
    :type J0: float
    :param sigma_b: Beam width (1-sigma) [m].
    :type sigma_b: float
    
    :return: np.ndarray: Electron flux profile at each r.
    """
    return J0 * np.exp(-0.5 * (r / sigma_b) ** 2)

def build_radial_laplacian(R, N):
    """
    Constructs the tridiagonal coefficients for the radial Laplacian operator
    with symmetry at r=0 and user-chosen outer boundary.
    
    :param R: Simulation domain radius [m].
    :type R: float
    :param N: Number of spatial grid points.
    :type N: int
    
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: r: grid points, a: lower diagonal, b: main diagonal, c: upper diagonal.
    """
    dr = R / N
    r = np.linspace(0.0, R, N + 1)
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    b[0] = -4.0 / dr**2
    c[0] =  4.0 / dr**2
    for i in range(1, N):
        ri = r[i]
        r_imh = ri - 0.5 * dr
        r_iph = ri + 0.5 * dr
        fac = 1.0 / (ri * dr**2)
        a[i] =  r_imh * fac
        c[i] =  r_iph * fac
        b[i] = -(r_imh + r_iph) * fac
    return r, a, b, c

def apply_outer_bc_tridiag(a, b, c, r, dr, outer_bc, theta_outer=None):
    """
    Modifies the tridiagonal Laplacian coefficients for the outer boundary.
    Supports Neumann (zero-flux) and Dirichlet (fixed value) BCs.
    
    :param a: Lower diagonal of Laplacian.
    :type a: np.ndarray
    :param b: Main diagonal of Laplacian.
    :type b: np.ndarray
    :param c: Upper diagonal of Laplacian.
    :type c: np.ndarray
    :param r: Radial grid points.
    :type r: np.ndarray
    :param dr: Grid spacing.
    :type dr: float
    :param outer_bc: Boundary condition type ("neumann" or "dirichlet").
    :type outer_bc: str
    :param theta_outer: Value for Dirichlet BC at r=R.
    :type theta_outer: float, optional
    
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray]: Modified diagonals (a, b, c).
    """
    N = len(r) - 1
    if outer_bc == "neumann":
        rN   = r[N]
        r_nmh = rN - 0.5 * dr
        facN = 1.0 / (rN * dr**2) if rN > 0 else 0.0
        a[N] =  r_nmh * facN
        b[N] = -r_nmh * facN
        c[N] =  0.0
    else:  # dirichlet
        a[N] = 0.0
        b[N] = 1.0
        c[N] = 0.0
    return a, b, c

def solve_tridiagonal(a, b, c, d):
    """
    Thomas algorithm for solving a tridiagonal system Ax = d.
    
    :param a: Lower diagonal (a[0] unused).
    :type a: np.ndarray
    :param b: Main diagonal.
    :type b: np.ndarray
    :param c: Upper diagonal (c[-1] unused).
    :type c: np.ndarray
    :param d: Right-hand side.
    :type d: np.ndarray
    
    :return: np.ndarray: Solution array x.
    """
    n = len(d)
    ac, bc, cc, dc = map(np.array, (a.copy(), b.copy(), c.copy(), d.copy()))
    for i in range(1, n):
        m = ac[i] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]
    x = np.zeros_like(d)
    x[-1] = dc[-1] / bc[-1]
    for i in reversed(range(n - 1)):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x

def advance_imex(theta, r, aL, bL, cL, p, implicit_linear_sinks=True):
    """
    Advances the solution by one IMEX time step.
    Diffusion is treated implicitly, reactions explicitly.
    Optionally, linear sinks (desorption, dissociation) can be treated implicitly.
    
    :param theta: Current surface coverage.
    :type theta: np.ndarray
    :param r: Radial grid points. aL, bL, cL (np.ndarray): Laplacian tridiagonal coefficients.
    :type r: np.ndarray
    :param p: Simulation parameters.
    :type p: Params
    :param implicit_linear_sinks: If True, treat linear sinks implicitly.
    :type implicit_linear_sinks: bool
    
    :return: np.ndarray: Updated surface coverage after one time step.
    """
    dt, D = p.dt, p.D
    if implicit_linear_sinks:
        J = gaussian_beam_J(r, p.J0, p.beam_sigma)
        K = (1.0 / p.tau) + p.sigma * J
    else:
        K = np.zeros_like(r)
    a = -dt * D * aL
    b =  1.0 - dt * D * bL + dt * K
    c = -dt * D * cL
    rhs = theta + dt * (p.S * p.Phi * (1.0 - theta/p.n0))
    if p.outer_bc == "dirichlet":
        a[-1] = 0.0; b[-1] = 1.0; c[-1] = 0.0
        rhs[-1] = p.theta_outer
    theta_new = solve_tridiagonal(a, b, c, rhs)
    return np.maximum(theta_new, 0.0)  # Only prevent negative values, don't cap at 1.0

def snapshots_no_diffusion(p: Params, r: np.ndarray, times: Tuple[float, ...], theta_init: np.ndarray) -> Dict[float, np.ndarray]:
    """
    Returns analytical solution snapshots for the ODE case (D=0).
    
    :param p: Simulation parameters.
    :type p: Params
    :param r: Radial grid points.
    :type r: np.ndarray
    :param times: Times at which to compute snapshots.
    :type times: Tuple[float, ...]
    :param theta_init: Initial surface coverage.
    :type theta_init: np.ndarray
    
    :return: Dict[float, np.ndarray]: Dictionary mapping time to theta profile.
    """
    J = gaussian_beam_J(r, p.J0, p.beam_sigma)
    k = p.S * p.Phi / p.n0 + (1.0 / p.tau) + p.sigma * J
    theta_inf = np.divide(p.S * p.Phi, k, out=np.zeros_like(k), where=k>0)
    snaps = {}
    for t in sorted(times):
        theta_t = theta_inf + (theta_init - theta_inf) * np.exp(-k * t)
        if p.outer_bc == "dirichlet":
            theta_t[-1] = p.theta_outer
        snaps[t] = np.maximum(theta_t, 0.0)  # Only prevent negative values
    return snaps

def run_simulation(p: Params):
    """
    Runs the full simulation for the given parameters.
    
    :param p: Simulation parameters.
    :type p: Params
    
    :return: Tuple[np.ndarray, Dict[float, np.ndarray], Params]: r: grid points, snaps: dict of {time: theta profile}, p: Params object.
    """
    r, aL, bL, cL = build_radial_laplacian(p.R, p.N)
    dr = p.R / p.N
    aL, bL, cL = apply_outer_bc_tridiag(aL, bL, cL, r, dr, p.outer_bc, p.theta_outer)

    if p.start_full:
        k_min = p.S * p.Phi / p.n0 + 1.0 / p.tau
        theta_max = p.S * p.Phi / k_min
        theta = np.full(p.N + 1, theta_max, dtype=float)
    else:
        theta = np.zeros(p.N + 1, dtype=float)

    if p.outer_bc == "dirichlet":
        theta[-1] = p.theta_outer

    if p.D == 0.0:
        times = tuple(sorted(set((0.0,) + p.snapshots + (p.t_end,))))
        snaps = snapshots_no_diffusion(p, r, times, theta.copy())
        return r, snaps, p

    # Prepare candidate times (snapped, round values)
    candidate_times = generate_round_candidates(p.t_end, n_candidates=p.max_candidates)
    # Ensure 0 first
    if candidate_times[0] != 0.0:
        candidate_times.insert(0, 0.0)
    # For D==0 use analytic snapshots at candidate times
    if p.D == 0.0:
        snaps_all = snapshots_no_diffusion(p, r, tuple(candidate_times), theta.copy())
        # snaps_all keys are times from candidate_times possibly clipped; select final set
        if p.num_profiles and p.num_profiles > 0:
            sel_times, snaps = select_snapshots_by_center(snaps_all, candidate_times, p.num_profiles,
                                                          ensure_final=p.t_end)
        else:
            # Use intersection of requested p.snapshots with available candidates (or all candidates)
            requested = sorted(set(p.snapshots))
            snaps = {}
            for t in requested:
                # pick nearest candidate time
                nearest = min(candidate_times, key=lambda x: abs(x - t))
                snaps[nearest] = snaps_all[nearest]
        # always ensure final time present
        snaps[p.t_end] = snaps_all[candidate_times[-1]]
        return r, snaps, p

    # Numeric IMEX time stepping: save theta at candidate times when reached
    snaps_all = {}
    t = 0.0
    snaps_all[candidate_times[0]] = theta.copy()
    next_idx = 1
    nsteps = int(np.ceil(p.t_end / p.dt))
    for n in range(1, nsteps + 1):
        theta = advance_imex(theta, r, aL, bL, cL, p, implicit_linear_sinks=True)
        t = n * p.dt
        # save any candidate times that have been reached
        while next_idx < len(candidate_times) and t + 1e-12 >= candidate_times[next_idx]:
            snaps_all[candidate_times[next_idx]] = theta.copy()
            next_idx += 1
    # ensure final time saved
    snaps_all[candidate_times[-1]] = theta.copy()

    # Choose final set of snapshots
    if p.num_profiles and p.num_profiles > 0:
        indices = np.linspace(0, len(candidate_times) - 1, p.num_profiles).round().astype(int)
        sel_times, snaps = select_snapshots_by_center(snaps_all, candidate_times, p.num_profiles, ensure_final=p.t_end)
    else:
        # fallback: use p.snapshots mapped to nearest candidate times
        snaps = {}
        for t_req in sorted(set(p.snapshots)):
            nearest = min(candidate_times, key=lambda x: abs(x - t_req))
            snaps[nearest] = snaps_all[nearest]
    # ensure final time present
    snaps[p.t_end] = snaps_all[candidate_times[-1]]
    return r, snaps, p

def calculate_deposited_volume(p: Params, return_height_profile: bool = False):
    """
    Calculate the total deposited volume from a radially symmetric 1D FEBID simulation.
    
    The function tracks the height profile h(r,t) over time by accumulating height increments
    at each time step according to:
        dh[i] = θ[i] * σ * J[i] * Va * dt
    
    where:
        - θ[i]: surface coverage at radial position i [molecules/nm²]
        - σ: electron-induced dissociation cross-section [nm²]
        - J[i]: electron flux at position i [electrons/(nm²·s)]
        - Va: volume deposited per precursor molecule [nm³]
        - dt: time step [s]
    
    The total volume is then integrated radially over the cylindrically symmetric domain:
        V_total = 2π ∫₀^R h(r) × r × dr
    
    :param p: Simulation parameters containing all physical constants and numerical settings.
    :type p: Params
    :param return_height_profile: If True, returns the final height profile h(r) along with volume.
    :type return_height_profile: bool
    
    :return: If return_height_profile is False: float: Total deposited volume [nm³] If return_height_profile is True: Tuple[float, np.ndarray, np.ndarray]: (V_total [nm³], r [nm], h(r) [nm]) Physics: - The deposition rate per unit area is: σ * J(r) * θ(r) [molecules/(nm²·s)] - Multiplying by Va gives height growth rate: dh/dt = σ * J(r) * θ(r) * Va [nm/s] - Integration over cylindrical annuli accounts for radial symmetry Example: >>> p = Params(D=2e6, t_end=0.002, Va=0.95) >>> V_total = calculate_deposited_volume(p) >>> print(f"Total deposited volume: {V_total:.2f} nm³") >>> V_total, r, h = calculate_deposited_volume(p, return_height_profile=True) >>> # plt.plot(r, h) >>> # plt.xlabel("r [nm]"); plt.ylabel("h [nm]")
    """
    # Build spatial grid and operators
    r, aL, bL, cL = build_radial_laplacian(p.R, p.N)
    dr = p.R / p.N
    aL, bL, cL = apply_outer_bc_tridiag(aL, bL, cL, r, dr, p.outer_bc, p.theta_outer)

    # Initialize surface coverage θ(r, t=0)
    if p.start_full:
        k_min = p.S * p.Phi / p.n0 + 1.0 / p.tau
        theta_max = p.S * p.Phi / k_min
        theta = np.full(p.N + 1, theta_max, dtype=float)
    else:
        theta = np.full(p.N + 1, p.theta0, dtype=float)

    if p.outer_bc == "dirichlet":
        theta[-1] = p.theta_outer

    # Initialize height profile h(r, t=0)
    h = np.zeros(p.N + 1, dtype=float)

    # Precompute electron beam flux profile J(r)
    J = gaussian_beam_J(r, p.J0, p.beam_sigma)

    # Time stepping
    nsteps = int(np.ceil(p.t_end / p.dt))
    dt = p.dt

    for n in range(nsteps):
        # Calculate height increment at current time step
        # dh = θ(r) * σ * J(r) * Va * dt
        # Units: [molecules/nm²] × [nm²] × [electrons/(nm²·s)] × [nm³] × [s] = [nm]
        dh = theta * p.sigma * J * p.Va * dt

        # Accumulate height
        h += dh

        # Advance surface coverage to next time step (if not at final step)
        if n < nsteps - 1:
            if p.D == 0.0:
                # Use analytical update for D=0 case
                k = p.S * p.Phi / p.n0 + (1.0 / p.tau) + p.sigma * J
                theta_inf = np.divide(p.S * p.Phi, k, out=np.zeros_like(k), where=k>0)
                theta = theta_inf + (theta - theta_inf) * np.exp(-k * dt)
            else:
                # Use IMEX time stepping for D>0
                theta = advance_imex(theta, r, aL, bL, cL, p, implicit_linear_sinks=True)

            if p.outer_bc == "dirichlet":
                theta[-1] = p.theta_outer

    # Integrate radially to get total volume
    # V_total = 2π ∫₀^R h(r) × r × dr
    # Using trapezoidal rule with uniform grid:
    integrand = h * r
    V_total = 2.0 * np.pi * np.trapz(integrand, r)

    if return_height_profile:
        return V_total, r, h
    else:
        return V_total

def run_1d_simulation_metrics(p: Params) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run the full 1D FEBID simulation and return key scalar metrics.

    Returns a tuple:
        (V_total_nm3, theta_center_final, r, theta_profile_final)

    Where:
        - V_total_nm3 (float): Total deposited volume integrated radially [nm^3]
        - theta_center_final (float): Final precursor surface coverage at r=0 [molecules/nm^2]

    Notes:
        - Uses run_simulation(p) to obtain the final coverage profile and reads θ(r=0, t=t_end).
        - Uses calculate_deposited_volume(p) to integrate the deposited volume over time and radius.
    """
    # Run coverage simulation (analytic for D=0, IMEX for D>0)
    r, snaps, _ = run_simulation(p)
    # Central coverage at final time
    theta_center_final = float(snaps[p.t_end][0])
    # Total deposited volume
    V_total_nm3 = float(calculate_deposited_volume(p, return_height_profile=False))
    return V_total_nm3, theta_center_final, r, snaps[p.t_end]


def estimate_se_flux_prefactor(i, a, yld):
    """
    Estimate the prefactor for electron flux J0 given
    desired dissociation rate at beam center.
    
    :param i: Beam current, A.
    :type i: float
    :param a: Gaussian standard deviation, nm.
    :type a: float
    :param yld: Secondary electron yield.
    :type yld: float
    
    :return: float: Estimated pre-factor for Gaussian secondary electron flux J0 [1/(nm²·s)].
    """
    e = 1.602e-19  # elementary charge, C
    pe_flux = i / e  # primary electrons per second
    area_factor = 2 * np.pi * (a ** 2)  # area under Gaussian
    f0_pe = pe_flux / area_factor  # primary electron flux prefactor
    f0_se = f0_pe * yld  # secondary electron flux prefactor
    return f0_se
