import numpy as np
import pyopencl as cl
# import warnings
import os
# from pynvml import *
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
        self.buffers = {}
        self.beam_matrix = None

    def load_structure(self, precursor, surface_all, deposit, surf_bool,
    irr_ind_2d, semi_surface, ghost, blocking=True):
        """
        Load all structure data related to deposition and reaction equation calculation, to host.
        Multidimensional arrays will be flattened
        """
        self.xdim = precursor.shape[2]
        self.ydim = precursor.shape[1]
        self.zdim = precursor.shape[0]
        self.zdim_max = irr_ind_2d[1]
        self.zdim_min = irr_ind_2d[0]
        self.len_lap = (irr_ind_2d[1] - irr_ind_2d[0]) * self.xdim * self.ydim
        self.offset = irr_ind_2d[0] * self.xdim * self.ydim
        self.surface_all = surface_all.reshape(-1).astype(np.bool_)
        self.surface_bool = surf_bool.reshape(-1).astype(np.bool_)
        self.semi_surface_bool = semi_surface.reshape(-1).astype(np.bool_)
        self.precursor = precursor.reshape(-1).astype(np.double)
        self.deposit = deposit.reshape(-1).astype(np.double)
        self.ghosts_bool = ghost.reshape(-1).astype(np.bool_)

        req_space = self._get_required_space()

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
        self.surf_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.surface_bool)
        self.semi_surf_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.semi_surface_bool)
        self.ghost_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.ghosts_bool)
        self.flag = np.array([0])  # flag determining if at least one cell is full
        self.flag_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.flag)

        self.buffers = {
            'surface_bool': (self.surface_bool, self.surf_buf),
            'deposit': (self.deposit, self.deposit_buf),
            'precursor': (self.precursor, self.cur_prec),
            'ghosts_bool': (self.ghosts_bool, self.ghost_buf),
            'semi_surface_bool': (self.semi_surface_bool, self.semi_surf_buf),
            'surface_all': (self.surface_all, self.surface_all_buf)
        }
        if blocking:
            self.queue.finish()


    def update_structure(self, precursor, deposit, surf_bool, semi_surface, irr_ind_2d, ghost, blocking=True):
        """
        Upload all structure arrays to the existing buffers on GPU.

        """

        self.xdim = precursor.shape[2]
        self.ydim = precursor.shape[1]
        self.zdim = precursor.shape[0]
        self.zdim_max = irr_ind_2d[1]
        self.zdim_min = irr_ind_2d[0]
        self.len_lap = (irr_ind_2d[1] - irr_ind_2d[0]) * self.xdim * self.ydim
        self.offset = irr_ind_2d[0] * self.xdim * self.ydim
        surface_all = np.logical_or(surf_bool, semi_surface)
        self.surface_all = surface_all.reshape(-1).astype(np.bool_)
        self.surface_bool = surf_bool.reshape(-1).astype(np.bool_)
        self.semi_surface_bool = semi_surface.reshape(-1).astype(np.bool_)
        self.precursor = precursor.reshape(-1).astype(np.double)
        self.deposit = deposit.reshape(-1).astype(np.double)
        self.ghosts_bool = ghost.reshape(-1).astype(np.bool_)

        cl.enqueue_copy(self.queue, self.cur_prec, self.precursor, is_blocking=False)
        # cl.enqueue_copy(self.queue, self.next_prec, self.precursor)
        cl.enqueue_copy(self.queue, self.surface_all_buf, self.surface_all, is_blocking=False)
        cl.enqueue_copy(self.queue, self.deposit_buf, self.deposit, is_blocking=False)
        cl.enqueue_copy(self.queue, self.surf_buf, self.surface_bool, is_blocking=False)
        cl.enqueue_copy(self.queue, self.semi_surf_buf, self.semi_surface_bool, is_blocking=False)
        # cl.enqueue_copy(self.queue, self.ghost_buf, self.ghosts_bool)
        if blocking:
            self.queue.finish()

    def _get_required_space(self):
        # calculate if storage size is sufficient
        self.req_space = self.precursor.size * self.precursor.itemsize * 2 + self.deposit.size * self.deposit.itemsize
        + self.surface_bool.size * self.surface_bool.itemsize * 4 + 32
        if self.beam_matrix is not None:
            self.req_space += self.beam_matrix.size * self.beam_matrix.itemsize
        else:
            self.req_space += self.xdim * self.ydim * self.zdim * np.array([1]).itemsize * 4 # assuming beam matrix is int32, as it should be

    def set_updated_structure(self, precursor, deposit, surface, semi_surface, irr_ind, ghost, blocking=True):
        """
        If grid height is not sufficient and needs to be extended, all buffers need to be reconstructed with
        the adapted size
        """
        # self.queue.finish()
        self.cur_prec.release()
        self.next_prec.release()
        self.surface_all_buf.release()
        self.deposit_buf.release()
        self.surf_buf.release()
        self.semi_surf_buf.release()
        self.ghost_buf.release()
        self.flag_buf.release()
        self.load_structure(precursor, np.logical_or(surface, semi_surface), deposit, surface, irr_ind, semi_surface,
                            ghost, blocking)


    def get_updated_structure(self, blocking=True):
        """
        Get all data arrays from the GPU

        :return: dictionary with all arrays with restored shape
        """
        # self.queue.finish()
        names_all = self.buffers.keys()
        retrieved = {}
        for name in names_all:
            arr = self.get_structure_partial(name, blocking).reshape(self.zdim, self.ydim, self.xdim)
            retrieved[name] = arr

        return retrieved

    def get_structure_partial(self, name, blocking=True):
        """
        Get a buffer of the array by its name.

        :param name: name of the data
        :param blocking: flag to indicate if the operation should be blocking

        :return: array with restored shape
        """
        # self.queue.finish()
        if name in self.buffers:
            array, buffer = self.buffers[name]
            cl.enqueue_copy(self.queue, array, buffer)
            if blocking:
                self.queue.finish()
            return array
        else:
            raise ValueError(f"Array with name {name} not found")
    
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

    def precur_den_gpu(self, add, a, F_dt, F_dt_n0_1_tau_dt, sigma_dt, blocking=True):
        """
        Recalculate precursor coverage using reaction equation with diffusion.
        """
        event = self.knl_prec_den(self.queue, (self.len_lap,), None,
                                  self.cur_prec, self.next_prec, self.beam_matrix_buf,
                                  np.int32(self.offset), np.int32(self.zdim_max), np.int32(self.ydim),
                                  np.int32(self.xdim), np.double(F_dt), np.double(F_dt_n0_1_tau_dt),
                                  np.double(sigma_dt), self.surface_all_buf, np.double(a),
                                  self.flag_buf, self.surf_buf, np.int32(self.zdim_min))
        if blocking:
            event.wait()

        self.cur_prec, self.next_prec = self.next_prec, self.cur_prec

    def deposit_gpu(self, const, blocking=True):
        """
        Perform deposition and check if any cell is filled.

        :return: True, if at least one cell is full
        """
        event = self.knl_dep(self.queue, (self.len_lap,), None,
                             self.cur_prec, self.beam_matrix_buf, np.int32(self.offset), np.double(const),
                             self.deposit_buf, self.flag_buf)
        if blocking:
            event.wait()

        cl.enqueue_copy(self.queue, self.flag, self.flag_buf)

        flag = self.flag[0]
        if self.flag[0] != 0:
            self.flag[0] = 0
            cl.enqueue_copy(self.queue, self.flag_buf, self.flag)
        return flag

    def return_beam_matrix(self, blocking=True):
        """
        Get the beam matrix from the GPU.

        :return: 1D array of int32
        """
        beam_matrix = self.get_structure_partial('beam_matrix', blocking=blocking)
        return beam_matrix

    def reload_beam_matrix(self, beam_matrix, blocking=True):
        """
        Reload the beam matrix calculated by Monte Carlo module to the GPU. This operation is for when the size is changed.
        """
        self.beam_matrix_buf.release()
        self.load_beam_matrix(beam_matrix, blocking=blocking)

    def load_beam_matrix(self, beam_matrix, blocking=True):
        """
        Load the beam matrix calculated by Monte Carlo module to the GPU. This operation is for the initialization.
        """
        # self.queue.finish()
        mf = cl.mem_flags
        self.beam_matrix_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=beam_matrix.reshape(-1))
        self.update_beam_matrix(beam_matrix, blocking=blocking)

    def update_beam_matrix(self, beam_matrix, resized=False, blocking=True):
        """
        Update beam matrix calculated by Monte Carlo module on the GPU. This operation is for the update.
        """
        # self.queue.finish()
        cl.enqueue_copy(self.queue, self.beam_matrix_buf, beam_matrix)
        self.beam_matrix = beam_matrix.reshape(-1)
        self.buffers['beam_matrix'] = (self.beam_matrix, self.beam_matrix_buf)
        if blocking:
            self.queue.finish()


    def index_1d_to_3d(self, index):
        """
        Convert 1D index to 3D index based on array shape

        :param index: 1D index

        """
        x = index % self.xdim
        y = (index // self.xdim) % self.ydim
        z = index // (self.xdim * self.ydim)
        return z, y, x

    def index_3d_to_1d(self, z, y, x):
        """
        convert 3D index to 1D index
        """
        return z * self.xdim * self.ydim + y * self.xdim + x


if __name__ == "__main__":
    pass
