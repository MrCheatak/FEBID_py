bool is_side(int i, int j, int k) {
    int cond = (i==0) + (j==0) + (k==0);
    return(cond==2);
}

bool is_edge(int i, int j, int k) {
    int cond = (i==0) + (j==0) + (k==0);
    return(cond==1);
}


__kernel void update(__global double *precursor, __global double *deposit, int zdim,
 int ydim, int xdim, __global bool *surface, __global int *index, __global bool *surface_all,
 __global bool *semi_surface, __global bool *ghost) {

    int ind = index[get_global_id(0)];

    int z = ind / (ydim*xdim);
    int y = (ind-z*ydim*xdim) / xdim;
    int x = ind - (z*ydim*xdim) - (y*xdim);

    double surplus = deposit[ind] - 1;
    deposit[ind] = -1;
    precursor[ind] = 0;
    surface[ind] = 0;
    surface_all[ind] = 0;
    semi_surface[ind] = 0;
    int count = 0;

    for (int i = -2; i < 3; ++i) {
        for (int j = -2; j < 3; ++j) {
            for (int k = -2; k < 3; ++k) {
                if (!(z+i > zdim-1 || z+i < 0 || y+j > ydim-1 || y+j < 0 || x+k > xdim-1 || x+k < 0)) {
                    int curr_ind = (z+i)*ydim*xdim + (y+j)*xdim + (x+k);
                    if (abs(i)<2 && abs(j)<2 && abs(k)<2){
                        if (deposit[curr_ind] == 0 && is_side(i,j,k)) {
                            surface[curr_ind] = 1;
                            surface_all[curr_ind] = 1;
                            semi_surface[curr_ind] = 0;
                        }
                        if (deposit[curr_ind] == 0 && !(surface[curr_ind]) && is_edge(i,j,k)) {
                            semi_surface[curr_ind] = 1;
                        }
                        ghost[curr_ind] = 0;
                        if (surface[curr_ind]){
                            count += 1;
                        }
                    }
                    int cond = surface[curr_ind];
                    int semi_cond = semi_surface[curr_ind];
                    if (cond+semi_cond == 0) {
                        ghost[curr_ind] = 1;
                    }
                }
            }
        }
    }

    for (int i = -1; i < 2; ++i) {
        for (int j = -1; j < 2; ++j) {
            for (int k = -1; k < 2; ++k) {
                if (!(z+i > zdim-1 || z+i < 0 || y+j > ydim-1 || y+j < 0 || x+k > xdim-1 || x+k < 0)) {
                    int curr_ind = (z+i)*ydim*xdim + (y+j)*xdim + (x+k);
                    if (surface[curr_ind]){
                        deposit[curr_ind] += surplus/count;
                    }
                }
            }
        }
    }

}