import numpy as np
import pyopencl as cl
import warnings
import os
from pynvml import *
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

class GPU:
    def __init__(self, device=None):
        f = open(os.path.dirname(os.path.realpath(__file__)) + r'\kernels\dep_prec_den.cl', 'r', encoding='utf-8')
        kernels_prec_den = ''.join(f.readlines()) # get path to kernel code
        f.close()

        f = open(os.path.dirname(os.path.realpath(__file__)) + r'\kernels\up_surface.cl', 'r', encoding='utf-8')
        kernels_up_surf = ''.join(f.readlines()) # get path to kernel code
        f.close()

        f = open(os.path.dirname(os.path.realpath(__file__)) + r'\kernels\return_slice.cl', 'r', encoding='utf-8')
        kernels_ret_slice = ''.join(f.readlines()) # get path to kernel code
        f.close()

        f = open(os.path.dirname(os.path.realpath(__file__)) + r'\kernels\ret_slice_bool.cl', 'r', encoding='utf-8')
        kernels_ret_slice_b = ''.join(f.readlines()) # get path to kernel code
        f.close()

        """
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        self.ctx = cl.Context(devices=my_gpu_devices)
        """
        """
        platforms = cl.get_platforms()
        self.device = platforms[device[0]].get_devices()[device[1]]
        self.ctx = cl.Context([self.device])
        """
        
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.max_storage = self.queue.device.max_mem_alloc_size
        
        prg_prec_den = cl.Program(self.ctx, kernels_prec_den).build()  # compile kernel program
        self.knl_prec_den = prg_prec_den.precursor_coverage  # create shortcut to main function of kernel
        self.knl_dep = prg_prec_den.deposition  # create shortcut to main function of kernel
        prg_up_surf = cl.Program(self.ctx, kernels_up_surf).build()  # compile kernel program
        self.knl_up_surf = prg_up_surf.update  # create shortcut to main function of kernel
        prg_ret_slice = cl.Program(self.ctx, kernels_ret_slice).build()  # compile kernel program
        self.knl_ret_slice = prg_ret_slice.ret  # create shortcut to main function of kernel
        prg_ret_slice_b = cl.Program(self.ctx, kernels_ret_slice_b).build()  # compile kernel program
        self.knl_ret_slice_b = prg_ret_slice_b.ret  # create shortcut to main function of kernel

        self.dt = None


    def load_vals(self, precursor, surface_all, flux_matrix, deposit, surf_bool,
    surface_ind_lap, semi_surface, ghost):
        """
        Load all data, necessary for deposition and reaction equation calculation, to host. 
        Multidimensional arrays will be flattened
        """
        self.xdim = precursor.shape[2]
        self.ydim = precursor.shape[1]
        self.zdim = precursor.shape[0]
        self.zdim_max = surface_ind_lap[1]
        self.zdim_min = surface_ind_lap[0]
        self.len_lap = (surface_ind_lap[1] - surface_ind_lap[0]) * self.xdim * self.ydim
        self.offset = surface_ind_lap[0] * self.xdim * self.ydim
        self.surface_all = surface_all.reshape(-1).astype(np.bool_)
        self.surface = surf_bool.reshape(-1).astype(np.bool_)
        self.semi_surface = semi_surface.reshape(-1).astype(np.bool_)
        self.precursor = precursor.reshape(-1).astype(np.double)
        self.flux_matrix = flux_matrix.reshape(-1).astype(np.int32)
        self.deposit = deposit.reshape(-1).astype(np.double)
        self.ghost = ghost.reshape(-1).astype(np.bool_)

        # calculate if storage size is sufficient
        self.req_space = self.precursor.size * self.precursor.itemsize * 2 + self.deposit.size * self.deposit.itemsize
        + self.surface.size * self.surface.itemsize * 4 + self.flux_matrix.size *  self.flux_matrix.itemsize + 32

        """
        # remove surface all!!!
        if self.req_space > self.max_storage:
            raise MemoryError("Device doesn't have sufficient memory ressources to resume calculations")
        """

        # create buffers
        mf = cl.mem_flags
        self.cur_prec = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.precursor)
        self.next_prec = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.precursor)
        self.surface_all_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.surface_all)
        self.deposit_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.deposit)
        self.flux_mat_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.flux_matrix)
        self.surf_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.surface)
        self.semi_surf_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.semi_surface)
        self.ghost_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.ghost)
        self.flag = np.array([0])  # flag determining if at least one cell is full
        self.flag_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.flag)

        self.queue.finish()


    def set_updated_structure(self, precursor, deposit, surface, semi_surface, flux_matrix, irr_ind, ghost):
        """
        If grid height is not sufficient and needs to be extended, all buffers need to be reconstructed with
        the adapted size
        """
        self.queue.finish()
        self.cur_prec.release()
        self.next_prec.release()
        self.surface_all_buf.release()
        self.deposit_buf.release()
        self.flux_mat_buf.release()
        self.surf_buf.release()
        self.semi_surf_buf.release()
        self.ghost_buf.release()
        self.flag_buf.release()
        self.load_vals(precursor, np.logical_or(surface, semi_surface), flux_matrix, deposit, surface,
        irr_ind, semi_surface, ghost)


    def get_updated_structure(self):
        """
        return necessary arrays for structure monitoring and dumping
        """
        self.queue.finish()
        try:
            cl.enqueue_copy(self.queue, self.surface, self.surf_buf)
            cl.enqueue_copy(self.queue, self.semi_surface, self.semi_surf_buf)
            cl.enqueue_copy(self.queue, self.deposit, self.deposit_buf)
            cl.enqueue_copy(self.queue, self.precursor, self.cur_prec)
            cl.enqueue_copy(self.queue, self.ghost, self.ghost_buf)
        except:
            return None

        return self.surface.reshape(self.zdim, self.ydim, self.xdim), self.semi_surface.reshape(self.zdim, self.ydim, self.xdim), self.deposit.reshape(self.zdim, self.ydim, self.xdim), self.precursor.reshape(self.zdim, self.ydim, self.xdim), self.ghost.reshape(self.zdim, self.ydim, self.xdim), self.zdim_max

    
    def return_slice(self, index, index_shape):
        """
        return necessary arrays for definition of surface neighbors 
        """
        self.queue.finish()
        mf = cl.mem_flags
        deposit = np.zeros(index_shape).astype(np.double)
        surface = np.zeros(index_shape).astype(np.bool_)
        dep_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=deposit)
        surf_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=surface)
        index_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=index)
        event_1 = self.knl_ret_slice(self.queue, (index_shape,), None, self.deposit_buf, dep_buf, index_buf, np.int32(index_shape),
        np.int32(self.ydim), np.int32(self.xdim))
        event_2 = self.knl_ret_slice_b(self.queue, (index_shape,), None, self.surf_buf, surf_buf, index_buf, np.int32(index_shape),
        np.int32(self.ydim), np.int32(self.xdim))
        cl.enqueue_copy(self.queue, deposit, dep_buf, wait_for = [event_1])
        cl.enqueue_copy(self.queue, surface, surf_buf, wait_for = [event_2])
        index_buf.release
        dep_buf.release
        surf_buf.release

        return deposit, surface


    def precur_den(self, add, a, F_dt, F_dt_n0_1_tau_dt, sigma_dt):
        """
        Recalculate precursor coverage using reaction equation with diffusion.
        """
        event = self.knl_prec_den(self.queue, (self.len_lap,), None,
        self.cur_prec, self.next_prec, self.flux_mat_buf, 
        np.int32(self.offset), np.int32(self.zdim_max), np.int32(self.ydim),
        np.int32(self.xdim), np.double(F_dt), np.double(F_dt_n0_1_tau_dt),
        np.double(sigma_dt), self.surface_all_buf, np.double(a),
        self.flag_buf, self.surf_buf, np.int32(self.zdim_min))
        # event.wait()

        self.cur_prec, self.next_prec = self.next_prec, self.cur_prec

    def deposition(self, const):
        """
        Perform deposition and check if any cell is filled.

        :return: True, if at least one cell is full
        """
        event = self.knl_dep(self.queue, (self.len_lap,), None,
        self.cur_prec, self.flux_mat_buf, np.int32(self.offset), np.double(const),
        self.deposit_buf, self.flag_buf)
        # event.wait()

        cl.enqueue_copy(self.queue, self.flag, self.flag_buf)

        flag = self.flag[0]
        if self.flag[0] != 0:
            self.flag[0] = 0
            cl.enqueue_copy(self.queue, self.flag_buf, self.flag)
        return flag

    def update_surface(self, full_cells):
        """
        update surface directly on compute device, definition of surface neighbors is not
        included and is calculated on host device 
        """
        self.queue.finish()
        mf = cl.mem_flags
        full_cell_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=full_cells)
        event = self.knl_up_surf(self.queue, full_cells.shape, None,
        self.cur_prec, self.deposit_buf, np.int32(self.zdim), np.int32(self.ydim),
        np.int32(self.xdim), self.surf_buf, full_cell_buf, self.surface_all_buf, self.semi_surf_buf, self.ghost_buf)
        event.wait()
        full_cell_buf.release()

    
    def get_updated_structure_partial(self):
        """
        return necessary arrays for Monte Carlo calcuations 
        """
        self.queue.finish()
        cl.enqueue_copy(self.queue, self.surface, self.surf_buf)
        cl.enqueue_copy(self.queue, self.deposit, self.deposit_buf)

        return self.surface.reshape(self.zdim, self.ydim, self.xdim), self.deposit.reshape(self.zdim, self.ydim, self.xdim)


    def return_beam_matrix(self):
        """
        return beam_matrix, including negative values for fully deposited cells
        """
        self.queue.finish()
        cl.enqueue_copy(self.queue, self.flux_matrix, self.flux_mat_buf)
        return self.flux_matrix

    
    def update_beam_matrix(self, beam_matrix):
        """
        update beam matrix calculated by Monte Carlo module
        """
        self.queue.finish()
        self.flux_matrix = beam_matrix.reshape(-1)
        cl.enqueue_copy(self.queue, self.flux_mat_buf, self.flux_matrix)


if __name__ == "__main__":
    pass
