/*
 * dep_prec_den.cl
 *
 * GPU kernels for precursor coverage and deposition updates.
 *
 * Numerical model implemented here:
 * - Precursor update: RK4 integration of reaction term
 *   + FTCS diffusion term evaluated at every RK stage.
 * - Deposition update: explicit volume increment from local precursor and beam flux.
 *
 * Notes:
 * - Arrays are flattened 3D grids (z-major indexing).
 * - "surface_all" includes surface + semi-surface cells (diffusion domain).
 * - "surface" includes only true surface cells (reaction/deposition domain).
 */

/* Return the RK stage state value for one cell:
 * n_stage = base + addon_scale * addon (for active surface_all cells),
 * otherwise return base unchanged.
 */
inline double state_value(__global const double *base, __global const double *addon,
                          __global const bool *surface_all, int idx, double addon_scale) {
    if (surface_all[idx]) {
        return base[idx] + addon[idx] * addon_scale;
    }
    return base[idx];
}

/* Compute discrete Laplacian (6-neighbor stencil) for one RK stage.
 *
 * Boundary and "solid/empty neighbor" handling follows the CPU stencil logic:
 * - Missing/zero neighbors contribute to zero_count.
 * - center * zero_count is added back to preserve edge behavior parity.
 */
inline double laplace_stage(__global const double *base, __global const double *addon,
                            __global const bool *surface_all, int idx,
                            int zdim, int ydim, int xdim, int z_min, double addon_scale) {
    int z = idx / (ydim * xdim);
    int y = (idx - z * ydim * xdim) / xdim;
    int x = idx - z * ydim * xdim - y * xdim;

    double center = state_value(base, addon, surface_all, idx, addon_scale);
    double acc = center * -6.0;
    double zero_count = 0.0;

    if (z > zdim - 2) {
        zero_count += 1.0;
    } else {
        double n = state_value(base, addon, surface_all, idx + ydim * xdim, addon_scale);
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    if (z < z_min + 1) {
        zero_count += 1.0;
    } else {
        double n = state_value(base, addon, surface_all, idx - ydim * xdim, addon_scale);
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    if (y > ydim - 2) {
        zero_count += 1.0;
    } else {
        double n = state_value(base, addon, surface_all, idx + xdim, addon_scale);
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    if (y < 1) {
        zero_count += 1.0;
    } else {
        double n = state_value(base, addon, surface_all, idx - xdim, addon_scale);
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    if (x > xdim - 2) {
        zero_count += 1.0;
    } else {
        double n = state_value(base, addon, surface_all, idx + 1, addon_scale);
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    if (x < 1) {
        zero_count += 1.0;
    } else {
        double n = state_value(base, addon, surface_all, idx - 1, addon_scale);
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    return acc + center * zero_count;
}

/* Compute Laplacian for the base field only (k1 specialization).
 *
 * This removes addon/surface-dependent state reconstruction and is equivalent to
 * laplace_stage(..., addon_scale=0) while reducing memory reads and branches.
 */
inline double laplace_base(__global const double *base, int idx,
                           int zdim, int ydim, int xdim, int z_min) {
    int z = idx / (ydim * xdim);
    int y = (idx - z * ydim * xdim) / xdim;
    int x = idx - z * ydim * xdim - y * xdim;

    double center = base[idx];
    double acc = center * -6.0;
    double zero_count = 0.0;

    if (z > zdim - 2) {
        zero_count += 1.0;
    } else {
        double n = base[idx + ydim * xdim];
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    if (z < z_min + 1) {
        zero_count += 1.0;
    } else {
        double n = base[idx - ydim * xdim];
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    if (y > ydim - 2) {
        zero_count += 1.0;
    } else {
        double n = base[idx + xdim];
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    if (y < 1) {
        zero_count += 1.0;
    } else {
        double n = base[idx - xdim];
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    if (x > xdim - 2) {
        zero_count += 1.0;
    } else {
        double n = base[idx + 1];
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    if (x < 1) {
        zero_count += 1.0;
    } else {
        double n = base[idx - 1];
        if (n != 0.0) {
            acc += n;
        } else {
            zero_count += 1.0;
        }
    }

    return acc + center * zero_count;
}

/* Kernel: rk4_stage_scalar_k1
 * Purpose:
 *   Compute RK4 stage k1 for constant (scalar) diffusion and residence-time coefficients.
 *
 * Inputs:
 *   - precur_old: precursor field at the start of the RK step.
 *   - beam_matrix: electron flux map (flattened grid).
 *   - offset/zdim/ydim/xdim/z_min: flattened-domain indexing bounds.
 *   - F, n0, tau, sigma: reaction model parameters.
 *   - stage_dt: RK stage time increment.
 *   - a_stage: FTCS diffusion factor for this stage, stage_dt * D / dx^2.
 *   - surface_all: active diffusion domain (surface + semi-surface).
 *   - surface: active reaction/deposition domain (surface only).
 *
 * Output:
 *   - k_out[ind]: stage increment k1 for each active cell, zero otherwise.
 *
 * Notes:
 *   - Diffusion branch is skipped when a_stage == 0.
 */
__kernel void rk4_stage_scalar_k1(__global const double *precur_old, __global double *k_out,
                                  __global const int *beam_matrix, int offset, int zdim, int ydim, int xdim,
                                  double F, double n0, double tau, double sigma, double stage_dt, double a_stage,
                                  __global const bool *surface_all, __global const bool *surface, int z_min) {
    int ind = get_global_id(0) + offset;

    if (!surface_all[ind]) {
        k_out[ind] = 0.0;
        return;
    }

    double n = precur_old[ind];
    double diff_term = 0.0;
    if (a_stage != 0.0) {
        diff_term = laplace_base(precur_old, ind, zdim, ydim, xdim, z_min) * a_stage;
    }
    double reaction_term = 0.0;
    if (surface[ind]) {
        reaction_term = F * stage_dt * (1.0 - n / n0) - n * stage_dt / tau - n * sigma * ((double)beam_matrix[ind]) * stage_dt;
    }
    k_out[ind] = reaction_term + diff_term;
}

/* Kernel: rk4_stage_array_k1
 * Purpose:
 *   Compute RK4 stage k1 for spatially varying diffusion and residence-time arrays.
 *
 * Inputs:
 *   - precur_old: precursor field at the start of the RK step.
 *   - D_array/tau_array: per-cell diffusion and residence-time coefficients.
 *   - beam_matrix, geometry parameters, model constants: same role as scalar k1.
 *   - cell_size_sq: dx^2 for FTCS diffusion scaling.
 *
 * Output:
 *   - k_out[ind]: stage increment k1 for each active cell, zero otherwise.
 */
__kernel void rk4_stage_array_k1(__global const double *precur_old, __global double *k_out,
                                 __global const int *beam_matrix, int offset, int zdim, int ydim, int xdim,
                                 double F, double n0, double sigma, double stage_dt, double cell_size_sq,
                                 __global const double *D_array, __global const double *tau_array,
                                 __global const bool *surface_all, __global const bool *surface, int z_min) {
    int ind = get_global_id(0) + offset;

    if (!surface_all[ind]) {
        k_out[ind] = 0.0;
        return;
    }

    double n = precur_old[ind];
    double D_loc = D_array[ind];
    double a_stage = stage_dt * D_loc / cell_size_sq;
    double diff_term = laplace_base(precur_old, ind, zdim, ydim, xdim, z_min) * a_stage;
    double reaction_term = 0.0;
    if (surface[ind]) {
        double tau_loc = tau_array[ind];
        reaction_term = F * stage_dt * (1.0 - n / n0) - n * stage_dt / tau_loc - n * sigma * ((double)beam_matrix[ind]) * stage_dt;
    }
    k_out[ind] = reaction_term + diff_term;
}

/* Kernel: rk4_stage_scalar
 * Purpose:
 *   Compute intermediate RK4 stages (k2 or k3) for scalar D/tau.
 *
 * Inputs:
 *   - precur_old: precursor field at RK step start.
 *   - addon: previous RK stage (k1 or k2) used to form stage state.
 *   - addon_scale: 0.5 for k2/k3-style midpoint evaluation.
 *   - Remaining parameters follow rk4_stage_scalar_k1 semantics.
 *
 * Output:
 *   - k_out[ind]: stage increment for the current stage.
 *
 * Notes:
 *   - Stage state is computed as state_value(precur_old, addon, ...).
 *   - Diffusion is evaluated on surface_all, reaction only on surface.
 */
__kernel void rk4_stage_scalar(__global const double *precur_old, __global const double *addon,
                               __global double *k_out, __global const int *beam_matrix,
                               int offset, int zdim, int ydim, int xdim,
                               double F, double n0, double tau, double sigma, double stage_dt, double a_stage,
                               __global const bool *surface_all, __global const bool *surface, int z_min,
                               double addon_scale) {
    int ind = get_global_id(0) + offset;

    if (!surface_all[ind]) {
        k_out[ind] = 0.0;
        return;
    }

    double n = state_value(precur_old, addon, surface_all, ind, addon_scale);
    double diff_term = 0.0;
    if (a_stage != 0.0) {
        diff_term = laplace_stage(precur_old, addon, surface_all, ind, zdim, ydim, xdim, z_min, addon_scale) * a_stage;
    }
    double reaction_term = 0.0;

    if (surface[ind]) {
        reaction_term = F * stage_dt * (1.0 - n / n0) - n * stage_dt / tau - n * sigma * ((double)beam_matrix[ind]) * stage_dt;
    }

    k_out[ind] = reaction_term + diff_term;
}

/* Kernel: rk4_stage_array
 * Purpose:
 *   Compute intermediate RK4 stages (k2 or k3) for per-cell D(x), tau(x).
 *
 * Inputs:
 *   - Same as rk4_stage_scalar, plus D_array/tau_array and cell_size_sq.
 *
 * Output:
 *   - k_out[ind]: stage increment for the current stage.
 *
 * Notes:
 *   - Intended for temperature-dependent transport/desorption coefficients.
 */
__kernel void rk4_stage_array(__global const double *precur_old, __global const double *addon,
                              __global double *k_out, __global const int *beam_matrix,
                              int offset, int zdim, int ydim, int xdim,
                              double F, double n0, double sigma, double stage_dt, double cell_size_sq,
                              __global const double *D_array, __global const double *tau_array,
                              __global const bool *surface_all, __global const bool *surface, int z_min,
                              double addon_scale) {
    int ind = get_global_id(0) + offset;

    if (!surface_all[ind]) {
        k_out[ind] = 0.0;
        return;
    }

    double n = state_value(precur_old, addon, surface_all, ind, addon_scale);
    double D_loc = D_array[ind];
    double a_stage = stage_dt * D_loc / cell_size_sq;
    double diff_term = 0.0;
    if (a_stage != 0.0) {
        diff_term = laplace_stage(precur_old, addon, surface_all, ind, zdim, ydim, xdim, z_min, addon_scale) * a_stage;
    }
    double reaction_term = 0.0;

    if (surface[ind]) {
        double tau_loc = tau_array[ind];
        reaction_term = F * stage_dt * (1.0 - n / n0) - n * stage_dt / tau_loc - n * sigma * ((double)beam_matrix[ind]) * stage_dt;
    }

    k_out[ind] = reaction_term + diff_term;
}

/* Kernel: rk4_stage_scalar_final
 * Purpose:
 *   Compute RK4 final stage (k4) for scalar D/tau and directly combine stages into n_new.
 *
 * Inputs:
 *   - precur_old: RK step base state.
 *   - addon: k3 stage (full-step addon).
 *   - k1/k2/k3: previously computed stage buffers.
 *   - Remaining scalar-physics and geometry parameters match rk4_stage_scalar.
 *
 * Output:
 *   - precur_new[ind]: updated precursor after RK4 combine:
 *       n_new = n_old + (k1 + k4)/6 + (k2 + k3)/3
 *
 * Notes:
 *   - Fused k4+combine path avoids separate combine kernel launch/pass.
 */
__kernel void rk4_stage_scalar_final(__global const double *precur_old, __global const double *addon,
                                     __global const double *k1, __global const double *k2, __global const double *k3,
                                     __global double *precur_new, __global const int *beam_matrix,
                                     int offset, int zdim, int ydim, int xdim,
                                     double F, double n0, double tau, double sigma, double stage_dt, double a_stage,
                                     __global const bool *surface_all, __global const bool *surface, int z_min,
                                     double addon_scale) {
    int ind = get_global_id(0) + offset;

    double out = precur_old[ind];
    if (!surface_all[ind]) {
        precur_new[ind] = out;
        return;
    }

    double n = state_value(precur_old, addon, surface_all, ind, addon_scale);
    double diff_term = 0.0;
    if (a_stage != 0.0) {
        diff_term = laplace_stage(precur_old, addon, surface_all, ind, zdim, ydim, xdim, z_min, addon_scale) * a_stage;
    }
    double reaction_term = 0.0;
    if (surface[ind]) {
        reaction_term = F * stage_dt * (1.0 - n / n0) - n * stage_dt / tau - n * sigma * ((double)beam_matrix[ind]) * stage_dt;
    }
    double k4 = reaction_term + diff_term;
    out += (k1[ind] + k4) / 6.0 + (k2[ind] + k3[ind]) / 3.0;
    precur_new[ind] = out;
}

/* Kernel: rk4_stage_array_final
 * Purpose:
 *   Compute RK4 final stage (k4) for per-cell D(x), tau(x) and directly write n_new.
 *
 * Inputs:
 *   - Same stage inputs as rk4_stage_scalar_final, with D_array/tau_array and cell_size_sq.
 *
 * Output:
 *   - precur_new[ind]: updated precursor after fused final stage + RK4 combine.
 */
__kernel void rk4_stage_array_final(__global const double *precur_old, __global const double *addon,
                                    __global const double *k1, __global const double *k2, __global const double *k3,
                                    __global double *precur_new, __global const int *beam_matrix,
                                    int offset, int zdim, int ydim, int xdim,
                                    double F, double n0, double sigma, double stage_dt, double cell_size_sq,
                                    __global const double *D_array, __global const double *tau_array,
                                    __global const bool *surface_all, __global const bool *surface, int z_min,
                                    double addon_scale) {
    int ind = get_global_id(0) + offset;

    double out = precur_old[ind];
    if (!surface_all[ind]) {
        precur_new[ind] = out;
        return;
    }

    double n = state_value(precur_old, addon, surface_all, ind, addon_scale);
    double D_loc = D_array[ind];
    double a_stage = stage_dt * D_loc / cell_size_sq;
    double diff_term = laplace_stage(precur_old, addon, surface_all, ind, zdim, ydim, xdim, z_min, addon_scale) * a_stage;
    double reaction_term = 0.0;
    if (surface[ind]) {
        double tau_loc = tau_array[ind];
        reaction_term = F * stage_dt * (1.0 - n / n0) - n * stage_dt / tau_loc - n * sigma * ((double)beam_matrix[ind]) * stage_dt;
    }
    double k4 = reaction_term + diff_term;
    out += (k1[ind] + k4) / 6.0 + (k2[ind] + k3[ind]) / 3.0;
    precur_new[ind] = out;
}

/* Kernel: rk4_combine
 * Purpose:
 *   Legacy standalone RK4 combine pass.
 *
 * Inputs:
 *   - precur_old and stage buffers k1..k4.
 *   - surface_all mask for active update domain.
 *
 * Output:
 *   - precur_new[ind] with RK4 weighted stage sum.
 *
 * Notes:
 *   - Kept for compatibility/testing; fused final-stage kernels are preferred in hot path.
 */
__kernel void rk4_combine(__global const double *precur_old, __global double *precur_new,
                          __global const double *k1, __global const double *k2,
                          __global const double *k3, __global const double *k4,
                          int offset, __global const bool *surface_all) {
    int ind = get_global_id(0) + offset;

    double out = precur_old[ind];
    if (surface_all[ind]) {
        out += (k1[ind] + k4[ind]) / 6.0 + (k2[ind] + k3[ind]) / 3.0;
    }
    precur_new[ind] = out;
}

/* Kernel: deposition
 * Purpose:
 *   Apply explicit deposited-volume increment from current precursor coverage and beam flux.
 *
 * Inputs:
 *   - precur: current precursor field.
 *   - beam_matrix: beam flux field; positive values contribute to deposition.
 *   - val: precomputed scalar factor for dt, molecular volume, scaling, and geometry.
 *   - deposit: accumulated deposited volume fraction per cell.
 *   - flag: global integer flag/counter for fully filled cells (deposit >= 1).
 *
 * Output:
 *   - deposit[ind] incremented for irradiated cells.
 *   - beam_matrix[ind] set to -1 for newly filled cells.
 *   - flag incremented when any cell crosses fill threshold.
 */
__kernel void deposition(__global double *precur, __global int *beam_matrix,
 int offset, double val, __global double *deposit, __global int *flag)
 {
    int ind = get_global_id(0) + offset;
    double* deposit_cell = &deposit[ind];

    if (beam_matrix[ind] > 0) {
        *deposit_cell += precur[ind] * beam_matrix[ind] * val / 1000000;
        if (*deposit_cell >= 1){
            flag[0] += 1;
            beam_matrix[ind] = -1;
        }
    }
}
