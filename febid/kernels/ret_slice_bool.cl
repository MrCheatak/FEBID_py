__kernel void ret(__global bool *target_array, global bool *return_array, __global int *index_array, int shape,
int ydim, int xdim) {
    int ind = get_global_id(0);

    int z = index_array[ind];
    int y = index_array[ind+shape];
    int x = index_array[ind+2*shape];

    return_array[ind] = target_array[z*ydim*xdim+y*xdim+x];
}