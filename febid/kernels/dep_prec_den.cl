double roll(__global double *precur_old, double cur_val, int zdim, int ydim,
 int xdim, int ind, int z_min) {

    int z = ind / (ydim*xdim);
    int y = (ind-z*ydim*xdim) / xdim;
    int x = ind - (z*ydim*xdim) - (y*xdim);

    double zero_count = 0;

    if (z>zdim-2){
        zero_count += 1;
    }
    else{
        if (precur_old[ind+ydim*xdim] != 0){
            cur_val = cur_val + precur_old[ind+ydim*xdim];
        }
        else{
            zero_count += 1;
        }
    }
    if (z<z_min+1){
        zero_count += 1;
    }
    else{
        if (precur_old[ind-ydim*xdim] != 0){
            cur_val = cur_val + precur_old[ind-ydim*xdim];
        }
        else{
            zero_count += 1;
        }
    }
    if (y>ydim-2){
        zero_count += 1;
    }
    else{
        if (precur_old[ind+xdim] != 0){
            cur_val = cur_val + precur_old[ind+xdim];
        }
        else{
            zero_count += 1;
        }
    }
    if (y<1){
        zero_count += 1;
    }
    else{
        if (precur_old[ind-xdim] != 0){
            cur_val = cur_val + precur_old[ind-xdim];
        }
        else{
            zero_count += 1;
        }
    }
    if (x>xdim-2){
        zero_count += 1;
    }
    else{
        if (precur_old[ind+1] != 0){
            cur_val = cur_val + precur_old[ind+1];
        }
        else{
            zero_count += 1;
        }
    }
    if (x<1){
        zero_count += 1;
    }
    else{
        if (precur_old[ind-1] != 0){
            cur_val = cur_val + precur_old[ind-1];
        }
        else{
            zero_count += 1;
        }
    }
    cur_val = cur_val + (precur_old[ind]) * zero_count;


    return cur_val;
}


double rk4(double precur_val, int beam_val, double F_dt, 
double F_dt_n0_1_tau_dt, double sigma_dt) {
    double k1 = F_dt - precur_val *
        (F_dt_n0_1_tau_dt + sigma_dt * beam_val);
    double k2 = F_dt/2 - (precur_val + k1/2) *
        (F_dt_n0_1_tau_dt/2 + sigma_dt/2 * beam_val);
    double k3 = F_dt/2 - (precur_val + k2/2) *
        (F_dt_n0_1_tau_dt/2 + sigma_dt/2 * beam_val);
    double k4 = F_dt - (precur_val + k3) *
        (F_dt_n0_1_tau_dt + sigma_dt * beam_val);

    return (k1+k4)/6 + (k2+k3)/3;
}


__kernel void precursor_coverage(__global double *precur_old, __global double *precur_new, __global int *beam_matrix,
 int offset, int zdim, int ydim, int xdim, double F_dt, double F_dt_n0_1_tau_dt, double sigma_dt,
__global bool *surface_all, double a, __global int *flag, __global bool *surface, int z_min)
{
    int ind = get_global_id(0) + offset;

    double curr_precur = precur_old[ind];
    double old_precur = precur_old[ind];
    if (surface_all[ind]){
        double diff_mat = roll(precur_old, curr_precur * -6, zdim, ydim, xdim, ind, z_min) * a;
        if (surface[ind]) {
            curr_precur += rk4(curr_precur, beam_matrix[ind], F_dt, F_dt_n0_1_tau_dt, sigma_dt);
        }
        curr_precur += diff_mat;
    }
    precur_new[ind] = curr_precur;
}


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