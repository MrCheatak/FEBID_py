"""
This module contains the GPU class that is used to perform the calculations on the GPU, optimized memory transfer,
and utility functions for 3D<->1D index conversion.
"""
import numpy as np
import pyopencl as cl
# import warnings
import os
# from pynvml import *
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

class GPU:
    """Manage OpenCL context, buffers, and kernels for FEBID GPU computations."""

    def __init__(self, device=None):
        """
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        self.ctx = cl.Context(devices=my_gpu_devices)
        """
        self.len_lap = None  # global work size
        self.offset = None  # offset for the data arrays
        self.flag_buf = None  # buffer for the flag
        self.flag = None  # flag determining if at least one cell is full

        """
        platforms = cl.get_platforms()
        self.device = platforms[device[0]].get_devices()[device[1]]
        self.ctx = cl.Context([self.device])
        """

        # Create a context with the first available GPU
        platforms = cl.get_platforms()  # initialize OpenCL platform
        platform = platforms[0]  # select the first platform
        devices = platform.get_devices(device_type=cl.device_type.GPU)  # Get GPU devices
        self.ctx = cl.Context(devices=devices)  # Create a context with the selected devices

        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.max_storage = self.queue.device.max_mem_alloc_size
        self._timing_enabled = False
        self._timings = {}
        self.reset_timing()

        self.__open_and_compile_kernels()

        self.buffers = {}  # dictionary relating the arrays to the buffers
        self.beam_matrix = None

    def reset_timing(self):
        self._timings = {
            'gpu_kernel_dep_s': 0.0,
            'gpu_kernel_rk4_k1_s': 0.0,
            'gpu_kernel_rk4_k2_s': 0.0,
            'gpu_kernel_rk4_k3_s': 0.0,
            'gpu_kernel_rk4_k4_s': 0.0,
            'gpu_kernel_rk4_combine_s': 0.0,
            'gpu_kernel_rk4_total_s': 0.0,
        }

    def enable_timing(self, enabled=True, reset=True):
        self._timing_enabled = bool(enabled)
        if reset:
            self.reset_timing()

    def get_timing(self):
        return dict(self._timings)

    def timing_enabled(self):
        return self._timing_enabled

    @staticmethod
    def _event_seconds(event):
        try:
            return (event.profile.end - event.profile.start) * 1e-9
        except Exception:
            return 0.0

    def __open_and_compile_kernels(self):
        """
        Open and compile the kernels with necessary functions for the GPU.
        Compiled functions are stored in the class as attributes and can be called directly.
        kernels must be stored in the 'kernels' folder in the same directory as this file.
        """
        f = open(os.path.dirname(os.path.realpath(__file__)) + r'\kernels\dep_prec_den.cl', 'r', encoding='utf-8')
        kernels_prec_den = ''.join(f.readlines())  # get path to kernel code
        f.close()
        f = open(os.path.dirname(os.path.realpath(__file__)) + r'\kernels\up_surface.cl', 'r', encoding='utf-8')
        kernels_up_surf = ''.join(f.readlines())  # get path to kernel code
        f.close()
        f = open(os.path.dirname(os.path.realpath(__file__)) + r'\kernels\return_slice.cl', 'r', encoding='utf-8')
        kernels_ret_slice = ''.join(f.readlines())  # get path to kernel code
        f.close()
        f = open(os.path.dirname(os.path.realpath(__file__)) + r'\kernels\ret_slice_bool.cl', 'r', encoding='utf-8')
        kernels_ret_slice_b = ''.join(f.readlines())  # get path to kernel code
        f.close()

        # 1. Compile the kernel
        # 2. Create a shortcut to the main function of the kernel
        prg_prec_den = cl.Program(self.ctx, kernels_prec_den).build()  # compile kernel program
        self.knl_prec_stage_scalar_k1 = prg_prec_den.rk4_stage_scalar_k1  # scalar k1 specialization
        self.knl_prec_stage_array_k1 = prg_prec_den.rk4_stage_array_k1  # array k1 specialization
        self.knl_prec_stage_scalar = prg_prec_den.rk4_stage_scalar  # scalar D/tau RK stages
        self.knl_prec_stage_array = prg_prec_den.rk4_stage_array  # array D/tau RK stages
        self.knl_prec_stage_scalar_final = prg_prec_den.rk4_stage_scalar_final  # fused final stage + combine
        self.knl_prec_stage_array_final = prg_prec_den.rk4_stage_array_final  # fused final stage + combine
        self.knl_dep = prg_prec_den.deposition  # deposition function
        prg_up_surf = cl.Program(self.ctx, kernels_up_surf).build()  # compile kernel program
        self.knl_up_surf = prg_up_surf.update  # cellular automaton update function
        prg_ret_slice = cl.Program(self.ctx, kernels_ret_slice).build()  # compile kernel program
        self.knl_ret_slice = prg_ret_slice.ret  #  function to return slice of 3D array GPU memory
        prg_ret_slice_b = cl.Program(self.ctx, kernels_ret_slice_b).build()  # compile kernel program
        self.knl_ret_slice_b = prg_ret_slice_b.ret  # function to return slice of 3D boolean array GPU memory

    def load_structure(self, precursor, deposit, surface_all, surf_bool, semi_surface, ghost, irr_ind_2d, blocking=True):
        """
        Load all structure data related to deposition and reaction equation calculation to host.
        This function creates buffers for all arrays on GPU and uploads them to the buffers.
        Multidimensional arrays will be flattened.

        :param precursor: precursor coverage array
        :param surface_all: combined surface and semi-surface array
        :param deposit: deposition array
        :param surf_bool: surface array
        :param irr_ind_2d: slice of the irradiated area along 0 axis
        :param semi_surface: semi-surface array
        :param ghost: ghost array
        :param blocking: flag to indicate if the operation should be blocking
        """
        self.__set_structure(precursor, deposit, surface_all, surf_bool, semi_surface, ghost, irr_ind_2d)

        req_space = self._get_required_space()

        """
        if self.req_space > self.max_storage:
            raise MemoryError("Device doesn't have sufficient memory ressources to resume calculations")
        """
        # create buffers
        self.__create_buffers()

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

    def __create_buffers(self):
        """
        Create buffers for all arrays on GPU.
        Two precursor arrays are created to store the current and the next state.
        A flag is created to determine if at least one cell is full after a deposition step.
        """
        mf = cl.mem_flags
        self.cur_prec = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.precursor)
        self.next_prec = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.precursor)
        self.k1 = np.zeros_like(self.precursor)
        self.k2 = np.zeros_like(self.precursor)
        self.k3 = np.zeros_like(self.precursor)
        self.zero_add = np.zeros_like(self.precursor)
        self.k1_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.k1)
        self.k2_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.k2)
        self.k3_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.k3)
        self.zero_add_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.zero_add)
        self.D_coeff = np.zeros_like(self.precursor)
        self.tau_coeff = np.ones_like(self.precursor)
        self.D_coeff_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.D_coeff)
        self.tau_coeff_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.tau_coeff)
        self.surface_all_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.surface_all)
        self.deposit_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.deposit)
        self.surf_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.surface_bool)
        self.semi_surf_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.semi_surface_bool)
        self.ghost_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.ghosts_bool)
        self.flag = np.array([0])  # flag determining if at least one cell is full
        self.flag_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.flag)

    def update_structure(self, precursor, deposit, surface_all, surf_bool, semi_surface, ghost, irr_ind_2d, cells=None, blocking=True):
        """
        Upload all structure arrays to the existing buffers on GPU.

        If a list of cells is provided, only the data of these cells will be updated.
        Due to the fact that the updated view (5x5x5) is not continuous in the flattened array, the continuous chunk
        containing the view will be copied to the GPU.

        :param precursor: precursor coverage array
        :param deposit: deposition array
        :param surface_all: combined surface and semi-surface array
        :param surf_bool: surface array
        :param semi_surface: semi-surface array
        :param irr_ind_2d: slice of the irradiated area along 0 axis
        :param ghost: ghost array
        :param cells: list of 3D indices of cells that need to be updated
        :param blocking: flag to indicate if the operation should be blocking

        """
        self.__set_structure(precursor, deposit, surface_all, surf_bool, semi_surface, ghost, irr_ind_2d)

        if cells is not None:
            for cell in cells:
                ind_start, ind_end = self.get_continuous_chunk(cell)
                cl.enqueue_copy(self.queue, self.cur_prec, self.precursor[ind_start:ind_end], dst_offset=ind_start * self.precursor.itemsize, is_blocking=False)
                # cl.enqueue_copy(self.queue, self.next_prec[ind_start:ind_end], self.precursor, is_blocking=False)
                cl.enqueue_copy(self.queue, self.surface_all_buf, self.surface_all[ind_start:ind_end], dst_offset=ind_start * self.surface_all.itemsize, is_blocking=False)
                cl.enqueue_copy(self.queue, self.deposit_buf, self.deposit[ind_start:ind_end], dst_offset=ind_start * self.deposit.itemsize, is_blocking=False)
                cl.enqueue_copy(self.queue, self.surf_buf, self.surface_bool[ind_start:ind_end], dst_offset=ind_start * self.surface_bool.itemsize, is_blocking=False)
                cl.enqueue_copy(self.queue, self.semi_surf_buf, self.semi_surface_bool[ind_start:ind_end], dst_offset=ind_start * self.semi_surface_bool.itemsize, is_blocking=False)
                # cl.enqueue_copy(self.queue, self.ghost_buf[ind_start:ind_end], self.ghosts_bool, is_blocking=False)
        else:
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
        """Estimate GPU memory required for current structure and beam buffers.

        :return: Estimated byte count required for GPU allocations.
        """
        # calculate if storage size is sufficient
        self.req_space = self.precursor.size * self.precursor.itemsize * 2 + self.deposit.size * self.deposit.itemsize
        + self.surface_bool.size * self.surface_bool.itemsize * 4 + 32
        if self.beam_matrix is not None:
            self.req_space += self.beam_matrix.size * self.beam_matrix.itemsize
        else:
            self.req_space += self.xdim * self.ydim * self.zdim * np.array([1]).itemsize * 4 # assuming beam matrix is int32, as it should be
        return self.req_space

    def set_updated_structure(self, precursor, deposit, surface_all, surface, semi_surface, ghost, irr_ind, blocking=True):
        """
        If grid height is not sufficient and needs to be extended, all buffers need to be reconstructed with
        the adapted size.
        """
        # self.queue.finish()
        self.__release_all_buffers()
        self.load_structure(precursor, deposit, surface_all, surface, semi_surface, ghost, irr_ind, blocking)

    def __release_all_buffers(self):
        """
        Release all buffers in the GPU memory.
        """
        try:
            self.cur_prec.release()
            self.next_prec.release()
            self.k1_buf.release()
            self.k2_buf.release()
            self.k3_buf.release()
            self.zero_add_buf.release()
            self.D_coeff_buf.release()
            self.tau_coeff_buf.release()
            self.surface_all_buf.release()
            self.deposit_buf.release()
            self.surf_buf.release()
            self.semi_surf_buf.release()
            self.ghost_buf.release()
            self.flag_buf.release()
        except AttributeError as e:
            raise AttributeError("Cannot release buffers from GPU, because they were not created.") from e

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

    def precur_den_gpu(self, dt, D, F, n0, tau, sigma, cell_size, blocking=False, wait_for=None):
        """
        Recalculate precursor coverage using RK4 + FTCS diffusion at each stage.
        """
        dt_full = np.double(dt)
        dt_half = np.double(dt * 0.5)
        addon_zero = np.double(0.0)
        addon_half = np.double(0.5)
        addon_full = np.double(1.0)

        common_dims = (
            np.int32(self.offset), np.int32(self.zdim_max), np.int32(self.ydim), np.int32(self.xdim)
        )
        common_phys = (np.double(F), np.double(n0), np.double(sigma))

        use_array_coeffs = isinstance(D, np.ndarray) or isinstance(tau, np.ndarray)
        e_k1 = e_k2 = e_k3 = e_final = None
        if use_array_coeffs:
            D_flat = D.reshape(-1).astype(np.double, copy=False) if isinstance(D, np.ndarray) else None
            tau_flat = tau.reshape(-1).astype(np.double, copy=False) if isinstance(tau, np.ndarray) else None

            if D_flat is None:
                self.D_coeff.fill(float(D))
            else:
                self.D_coeff[...] = D_flat
            if tau_flat is None:
                self.tau_coeff.fill(float(tau))
            else:
                self.tau_coeff[...] = tau_flat

            cl.enqueue_copy(self.queue, self.D_coeff_buf, self.D_coeff, is_blocking=False)
            cl.enqueue_copy(self.queue, self.tau_coeff_buf, self.tau_coeff, is_blocking=False)

            cell_size_sq = np.double(cell_size * cell_size)
            e_k1 = self.knl_prec_stage_array_k1(
                self.queue, (self.len_lap,), None,
                self.cur_prec, self.k1_buf, self.beam_matrix_buf,
                *common_dims, *common_phys, dt_full, cell_size_sq,
                self.D_coeff_buf, self.tau_coeff_buf, self.surface_all_buf, self.surf_buf, np.int32(self.zdim_min),
                wait_for=wait_for
            )
            e_k2 = self.knl_prec_stage_array(
                self.queue, (self.len_lap,), None,
                self.cur_prec, self.k1_buf, self.k2_buf, self.beam_matrix_buf,
                *common_dims, *common_phys, dt_half, cell_size_sq,
                self.D_coeff_buf, self.tau_coeff_buf, self.surface_all_buf, self.surf_buf, np.int32(self.zdim_min),
                addon_half, wait_for=[e_k1]
            )
            e_k3 = self.knl_prec_stage_array(
                self.queue, (self.len_lap,), None,
                self.cur_prec, self.k2_buf, self.k3_buf, self.beam_matrix_buf,
                *common_dims, *common_phys, dt_half, cell_size_sq,
                self.D_coeff_buf, self.tau_coeff_buf, self.surface_all_buf, self.surf_buf, np.int32(self.zdim_min),
                addon_half, wait_for=[e_k2]
            )
            e_final = self.knl_prec_stage_array_final(
                self.queue, (self.len_lap,), None,
                self.cur_prec, self.k3_buf, self.k1_buf, self.k2_buf, self.k3_buf, self.next_prec, self.beam_matrix_buf,
                *common_dims, *common_phys, dt_full, cell_size_sq,
                self.D_coeff_buf, self.tau_coeff_buf, self.surface_all_buf, self.surf_buf, np.int32(self.zdim_min),
                addon_full, wait_for=[e_k3]
            )
        else:
            a_full = np.double(dt * D / (cell_size * cell_size))
            a_half = np.double(0.5 * a_full)
            tau_scalar = np.double(tau)

            e_k1 = self.knl_prec_stage_scalar_k1(
                self.queue, (self.len_lap,), None,
                self.cur_prec, self.k1_buf, self.beam_matrix_buf,
                *common_dims, np.double(F), np.double(n0), tau_scalar, np.double(sigma), dt_full, a_full,
                self.surface_all_buf, self.surf_buf, np.int32(self.zdim_min), wait_for=wait_for
            )
            e_k2 = self.knl_prec_stage_scalar(
                self.queue, (self.len_lap,), None,
                self.cur_prec, self.k1_buf, self.k2_buf, self.beam_matrix_buf,
                *common_dims, np.double(F), np.double(n0), tau_scalar, np.double(sigma), dt_half, a_half,
                self.surface_all_buf, self.surf_buf, np.int32(self.zdim_min), addon_half, wait_for=[e_k1]
            )
            e_k3 = self.knl_prec_stage_scalar(
                self.queue, (self.len_lap,), None,
                self.cur_prec, self.k2_buf, self.k3_buf, self.beam_matrix_buf,
                *common_dims, np.double(F), np.double(n0), tau_scalar, np.double(sigma), dt_half, a_half,
                self.surface_all_buf, self.surf_buf, np.int32(self.zdim_min), addon_half, wait_for=[e_k2]
            )
            e_final = self.knl_prec_stage_scalar_final(
                self.queue, (self.len_lap,), None,
                self.cur_prec, self.k3_buf, self.k1_buf, self.k2_buf, self.k3_buf, self.next_prec, self.beam_matrix_buf,
                *common_dims, np.double(F), np.double(n0), tau_scalar, np.double(sigma), dt_full, a_full,
                self.surface_all_buf, self.surf_buf, np.int32(self.zdim_min), addon_full, wait_for=[e_k3]
            )

        if blocking:
            e_final.wait()
        if self._timing_enabled and blocking:
            t_k1 = self._event_seconds(e_k1)
            t_k2 = self._event_seconds(e_k2)
            t_k3 = self._event_seconds(e_k3)
            t_k4 = self._event_seconds(e_final)
            t_combine = 0.0
            self._timings['gpu_kernel_rk4_k1_s'] += t_k1
            self._timings['gpu_kernel_rk4_k2_s'] += t_k2
            self._timings['gpu_kernel_rk4_k3_s'] += t_k3
            self._timings['gpu_kernel_rk4_k4_s'] += t_k4
            self._timings['gpu_kernel_rk4_combine_s'] += t_combine
            self._timings['gpu_kernel_rk4_total_s'] += (t_k1 + t_k2 + t_k3 + t_k4 + t_combine)

        self.cur_prec, self.next_prec = self.next_prec, self.cur_prec
        return e_final

    def deposit_gpu(self, const, blocking=False, wait_for=None):
        """
        Perform deposition and check if any cell is filled.

        :return: True, if at least one cell is full
        """
        event = self.knl_dep(self.queue, (self.len_lap,), None,
                             self.cur_prec, self.beam_matrix_buf, np.int32(self.offset), np.double(const),
                             self.deposit_buf, self.flag_buf, wait_for=wait_for)
        if blocking:
            event.wait()
        if self._timing_enabled and blocking:
            self._timings['gpu_kernel_dep_s'] += self._event_seconds(event)
        return event

    def read_flag(self, wait_for=None, clear=True, blocking=True):
        """
        Read and optionally clear the filled-cell flag from GPU memory.
        """
        ev = cl.enqueue_copy(self.queue, self.flag, self.flag_buf, wait_for=wait_for)
        if blocking:
            ev.wait()

        flag = int(self.flag[0])
        if clear and flag != 0:
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

    def get_continuous_chunk(self, index):
        """
        Select a 5x5x5 chunk around the index in the 3D array and return the start and end indices for the flattened array
        to extract a continuous chunk that contains the 5x5x5 chunk.

        :param index: 3D index of the center of the chunk

        :return: start and end indices for the flattened array
        """

        # Calculate the start and end indices for each dimension
        start_indices = [max(0, idx - 2) for idx in index]
        end_indices = [min(self.shape[i], idx + 3) for i, idx in enumerate(index)]

        start_1d = self.index_3d_to_1d(*start_indices)
        end_1d = self.index_3d_to_1d(*end_indices)

        return start_1d, end_1d


    @property
    def xdim(self):
        """Return X dimension of current structure.

        :return: Size of X axis in cells.
        """
        return self.shape[2]

    @property
    def ydim(self):
        """Return Y dimension of current structure.

        :return: Size of Y axis in cells.
        """
        return self.shape[1]

    @property
    def zdim(self):
        """Return Z dimension of current structure.

        :return: Size of Z axis in cells.
        """
        return self.shape[0]

    def __set_structure(self, precursor, deposit, surface_all, surf_bool, semi_surface, ghost, irr_ind_2d):
        """
        Set the structure arrays for the GPU calculations.

        :param precursor: precursor coverage array
        :param deposit: deposition array
        :param surface_all: combined surface and semi-surface array
        :param surf_bool: surface array
        :param semi_surface: semi-surface array
        :param ghost: ghost array
        :param irr_ind_2d: slice of the irradiated area along 0 axis
        """
        self.surface_all = surface_all.reshape(-1).astype(np.bool_)
        self.surface_bool = surf_bool.reshape(-1).astype(np.bool_)
        self.semi_surface_bool = semi_surface.reshape(-1).astype(np.bool_)
        self.precursor = precursor.reshape(-1).astype(np.double)
        self.deposit = deposit.reshape(-1).astype(np.double)
        self.ghosts_bool = ghost.reshape(-1).astype(np.bool_)
        self.shape = deposit.shape
        self.zdim_max = irr_ind_2d[1]
        self.zdim_min = irr_ind_2d[0]
        self.global_work_size(irr_ind_2d)
        self.data_offset(irr_ind_2d)

    def data_offset(self, irr_ind_2d):
        """
        Calculate the offset for the data arrays based on the irradiated area.
        The offset is used to skip elements that are not processed.

        :param irr_ind_2d: slice of the irradiated area along 0 axis
        :return: offset for the data arrays
        """
        self.offset = irr_ind_2d[0] * self.xdim * self.ydim
        return self.offset

    def global_work_size(self, irr_ind_2d):
        """
        Calculate the global work size for the kernel based on the irradiated area.

        :param irr_ind_2d: slice of the irradiated area along 0 axis
        :return: global work size
        """
        self.len_lap = (irr_ind_2d[1] - irr_ind_2d[0]) * self.xdim * self.ydim
        return self.len_lap


if __name__ == "__main__":
    pass
