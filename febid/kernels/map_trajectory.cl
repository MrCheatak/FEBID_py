void get_direction(double *e_direction, double e_stheta, double e_ctheta, double e_psi)
{
    double AM = - e_direction[2] / e_direction[0];
    double AN = 1.0 / sqrt(1.0);
    float V1 = AN * e_stheta;
    float V2 = AN * AM * e_stheta;
    float V3 = cos(e_psi);
    float V4 = sin(e_psi);

    float ca = e_direction[2] * e_ctheta + V1 * V3 + e_direction[1] * V2 * V4;
    float cb = e_direction[1] * e_ctheta + V4 * (e_direction[0] * V1 - e_direction[2] * V2);
    float cc = e_direction[0] * e_ctheta + V2 * V3 - e_direction[1] * V1 * V4;

    if (ca == 0) { ca = 0.0000001; }
    if (cb == 0) { cb = 0.0000001; }
    if (cc == 0) { cc = 0.0000001; }
}

void get_angles(double a)
{
    double rnd1 = 0.5;
    double rnd2 = 0.5;
    float ctheta = (1.0 - 2.0 * a * rnd1 / (1.0 + a - rnd1));
    double stheta = sqrt(1.0 - ctheta * ctheta);
    double psi = 2.0 * M_PI_F * rnd2;
}

double[3] check_boundaries(double[3] e_point, double *dims)
{
    double min = 1e-6;
    if (min > e_point[2])
    {
        e_point[2] = 0.000001;
    } else if (dims[2] < e_point[2])
    {
        e_point[2] = dims[2] - 0.000001;
    }

    if (min > e_point[1])
    {
        e_point[1] = 0.000001;
    } else if (dims[1] < e_point[1])
    {
        e_point[1] = dims[2] - 0.000001;
    }

    if (min > e_point[0])
    {
        e_point[0] = 0.000001;
    } else if (dims[0] < e_point[0])
    {
        e_point[0] = dims[0] - 0.000001;
    }
    return e_point;
}

bool get_solid_crossing() {
    int ind;
    float next_t;
    while (true) {
        ind = -1;
        next_t = INFINITY; // get index
        for (int i=0; i<3; i++) {
            if (t[i] < next_t) {
                next_t = t[i];
                ind = i;
            }
        }
        if (next_t > 1.0f) {
            *flag = 1;
            return true;
        }
        for (int i = 0; i < 3; i++) {
            coord[i] = p0[i] + next_t * direction[i];
            index[i] = (uint)(coord[i] / *cell_dim);
        }
        if (grid[index[0]*3*3+index[1]*3+index[2]] <= -1.0f) {
            *flag = -1;
            return false;
        }
        t[ind] += step_t;
    }
}

bool get_surface_crossing() {
    while (1) {  // iterating until all the cells are traversed by the ray
        float next_t = INFINITY;
        int ind = 0;
        for (int i=0; i<3; i++) {
            if (t[i] < next_t) {
                next_t = t[i];
                ind = i;
            }
        }
        if (next_t > 1.0) {
            *result = 1;
            return (true);
        }
        for (int i=0; i<3; i++) {
            coord[i] = p0[i] + next_t * direction[i];
            index[i] = (int) (coord[i] / cell_dim);
        }
        if (grid[index[0] + index[1]*index[2] + index[2]*grid_dim] <= -1) {  // rewrite
            *result = 0;
            return (false);
        }
        t[ind] = t[ind] + step_t[ind];  // going to the next wall
    }
}

__kernel void my_kernel(__global float *x0,
                      __global float *y0,
                      __global float *energy,
                      __global float *mask,
                      __global float *trajectory,
                      __global float *delta,
                      __global int *grid,
                      __global int *materials,
                      __global int *crossing,
                      __global int *crossing1,
                      __global float *trajectories,
                      __global float *energies,
                      __global float *mask,
                      double E_min, double E0,
                      double E0, double material_Z, double A, double rho, double Z,
                      int g, __global float *dims,
                      int grid_z, int grid_y, int grid_z)
{
    int ind = get_global_id(0);

    double e_ctheta;
    double e_stheta;
    double e_psi;

    int curr_energy = 0;
    int curr_traj = 0;
    int curr_mask = 0;

    float rnd = 0;

    int flag = 0;

    float p0c[3], directionc[3], step[3], signc[3], temp[3], tc[3], pn[3], deltac[3], temp1[3], step_tc[3];

    // Create a new Electron instance
    float coord[3] = {grid_z - 0.001, y0[g], x0[g]};
    float e_point[3] = {coord[0], coord[1], coord[2]};
    float e_point_prev[3] = {coord[0], coord[1], coord[2]};
    float e_direction[3] = {1.0, 0.0, 0.0};
    float E = E0;

    // Store initial point and energy value
    for (int i = 0; i < 3; i++) {
        trajectory[g*3 + i] = e_point[i];
    }
    energy[g] = E;

    // get indicies
    int i = e_point[0] / grid_cell_dim;
    int j = e_point[1] / grid_cell_dim;
    int k = e_point[2] / grid_cell_dim;

    if (grid[i*grid_y*grid_x + j*grid_x + k] > -1)
    {
        // max index less double
        trajectory[g*3] = coord[0];
        trajectory[g*3 + 1] = coord[1];
        trajectory[g*3 + 2] = coord[2];
        energy[curr_energy] = E;
        mask[curr_mask] = 0.0;
        if (e_point[0] == grid_cell_dim) { return; }
    }

    while (e_E > Emin)
    {
        double a = 3.4E-3*material_Z**0.67/e_E;
        double sigma = 5.21E-7 * pow(Z, 2) / pow(e_E, 2) * 4.0 * 3.14159 / (a * (1.0 + a)) * pow(
                (e_E + 511.0) / (e_E + 1022.0), 2);
        double lambda = A / (6.022141E23 * rho * 1.0E-21 * sigma);
        double step = - log(rnd) * lambda;
        e_point[0] = e_point[0] - e_direction[0] * step;
        e_point[1] = e_point[1] + e_direction[1] * step;
        e_point[2] = e_point[2] + e_direction[2] * step;
        float e_point_copy[3];
        e_point_copy = check_boundaries(e_point, dims);
        if (e_point_copy != e_point)
        {
            flag = 1;
            e_point = e_point_copy;
            step = sqrt(sqrt(e_point[0] - e_point_prev[0]) +
                         sqrt(e_point[1] - e_point_prev[1]) + sqrt(e_point[2] - e_point_prev[2]))
        }
        if (grid[i,j,k] < 0)
        {
            e_E = e_E + (-7.85E-3 * material.rho * material.Z / (material.A * e_E) *
                    log(1.166 * (e_E / material.J + 0.85))) * step;
            trajectory[curr_traj] = e_point;
            energy[curr_energy] = e_E;
            mask[curr_mask] = 1.0;
            // change materials
        }
        else
        {
            p0c[0] = p0[2];
            p0c[1] = p0[1];
            p0c[2] = p0[0];
            directionc[0] = -vec[2];
            directionc[1] = vec[1];
            directionc[2] = vec[0];
            step[0] = grid_shape_abs[2];
            step[1] = grid_shape_abs[1];
            step[2] = grid_shape_abs[0];

            sign_double(directionc, signc);

            for (int i = 0; i < 3; i++) {
                if (signc[i] == 1) {
                    temp[i] = 1;
                } else {
                    temp[i] = 0;
                }
            }
            for (int i = 0; i < 3; i++) {
                tc[i] = abs((-p0c[i] + temp[i]*step[i])/directionc[i]);
            }
            t[0], _ = arr_min(t);
            for (int i = 0; i < 3; i++) {
                pn[i] = p0c[i] + t[0] * directionc[i];
            }
            for (int i = 0; i < 3; i++) {
                if (pn[i] >= step[i]) {
                    pn[i] = step[i] - 0.000001;
                } else if (pn[i] < min) {
                    pn[i] = 0.000001;
                }
            }
            sub_double(pn, p0c, directionc);
            for (int i = 0; i < 3; i++) {
                if (directionc[i] == 0) {
                    directionc[i] = rnd_uniform(-0.000001, 0.000001);
                }
                step[i] = signc[i] * grid_cell_dim[0];
            }
            for (int i = 0; i < 3; i++) {
                step_tc[i] = step[i]/directionc[i];
            }
            for (int i = 0; i < 3; i++) {
                deltac[i] = -(p0c[i]%grid_cell_dim[0]);
            }
            for (int i = 0; i < 3; i++) {
                if (delta[i] == 0) {
                    temp1[i] = 1;
                } else {
                    temp1[i] = 0;
                }
            }
            for (int i = 0; i < 3; i++) {
                tc[i] = abs((deltac[i] + temp[i]*grid_cell_dim[0] + temp1[i]*step[i])/directionc[i]);
                if (signc[i] == 1) { signc[i] = 0; }
            }
            bool flag = get_surface_crossing();
            if (flag)
            {
                crossing = pnc;
                flag = 2;
            }
            else
            {
                crossing[0] -= sign[0] * 0.001;
                crossing[1] -= sign[1] * 0.001;
                crossing[2] -= sign[2] * 0.001;
                flag = get_solid_crossing()
                if (flag)
                {
                    crossing1 = pnc;
                }
                else
                {
                    crossing1[0] += sign[0] * 0.001;
                    crossing1[1] += sign[1] * 0.001;
                    crossing1[2] += sign[2] * 0.001;
                }
            }
            // from_mv
            e_E = e_E + (-7.85E-3 * material.rho * material.Z / (material.A * e_E) *
                         log(1.166 * (e_E / material.J + 0.85))) * sqrt(sqrt(e_point[0] - e_point_prev[0]) +
                         sqrt(e_point[1] - e_point_prev[1]) + sqrt(e_point[2] - e_point_prev[2]));
            trajectory[curr_traj] = e_point;
            energy[curr_energy] = e_E;
            if (flag == 2)
            {
                mask[curr_mask] = 0.0;
            }
            if (flag < 2)
            {
                mask[curr_mask] = 1.0;
                // add point
                trajectory[curr_traj] = e_point;
                energy[curr_energy] = e_E;
                mask[curr_mask] = 0.0;
            }
        }
        if (flag > 0) { return; }
    }
}



