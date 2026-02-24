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

/* k1 specialization (scalar D/tau). */
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
    double diff_term = laplace_base(precur_old, ind, zdim, ydim, xdim, z_min) * a_stage;
    double reaction_term = 0.0;
    if (surface[ind]) {
        reaction_term = F * stage_dt * (1.0 - n / n0) - n * stage_dt / tau - n * sigma * ((double)beam_matrix[ind]) * stage_dt;
    }
    k_out[ind] = reaction_term + diff_term;
}

/* k1 specialization (array D/tau). */
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

/* RK stage kernel for scalar coefficients D and tau.
 *
 * For each active cell:
 * k = reaction(surface-only) + diffusion(surface_all)
 * where diffusion uses FTCS coefficient a_stage = dt_stage * D / dx^2.
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
    double diff_term = laplace_stage(precur_old, addon, surface_all, ind, zdim, ydim, xdim, z_min, addon_scale) * a_stage;
    double reaction_term = 0.0;

    if (surface[ind]) {
        reaction_term = F * stage_dt * (1.0 - n / n0) - n * stage_dt / tau - n * sigma * ((double)beam_matrix[ind]) * stage_dt;
    }

    k_out[ind] = reaction_term + diff_term;
}

/* RK stage kernel for per-cell coefficient arrays D(x) and tau(x).
 *
 * This path supports temperature-dependent coefficients.
 * Diffusion coefficient is sampled per-cell for stage coefficient evaluation.
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
    double diff_term = laplace_stage(precur_old, addon, surface_all, ind, zdim, ydim, xdim, z_min, addon_scale) * a_stage;
    double reaction_term = 0.0;

    if (surface[ind]) {
        double tau_loc = tau_array[ind];
        reaction_term = F * stage_dt * (1.0 - n / n0) - n * stage_dt / tau_loc - n * sigma * ((double)beam_matrix[ind]) * stage_dt;
    }

    k_out[ind] = reaction_term + diff_term;
}

/* Final RK stage for scalar coefficients:
 * computes k4 and immediately writes n_new to avoid an extra combine pass.
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
    double diff_term = laplace_stage(precur_old, addon, surface_all, ind, zdim, ydim, xdim, z_min, addon_scale) * a_stage;
    double reaction_term = 0.0;
    if (surface[ind]) {
        reaction_term = F * stage_dt * (1.0 - n / n0) - n * stage_dt / tau - n * sigma * ((double)beam_matrix[ind]) * stage_dt;
    }
    double k4 = reaction_term + diff_term;
    out += (k1[ind] + k4) / 6.0 + (k2[ind] + k3[ind]) / 3.0;
    precur_new[ind] = out;
}

/* Final RK stage for per-cell D/tau arrays:
 * computes k4 and immediately writes n_new to avoid an extra combine pass.
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

/* Final RK4 combination:
 * n_new = n_old + (k1 + k4)/6 + (k2 + k3)/3
 * applied only on the active diffusion domain (surface_all).
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


__kernel void deposition(__global double *precur, __global int *beam_matrix,
 int offset, double val, __global double *deposit, __global int *flag)
 {
    /* Explicit deposition increment:
     * deposit += precursor * beam_flux * val / 1e6
     * Cells reaching deposit >= 1 are marked filled:
     * - increment global flag
     * - mark beam cell as -1 (used by topology update routines)
     */
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
