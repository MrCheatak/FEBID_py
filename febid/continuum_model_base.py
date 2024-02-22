from pickle import dump

import numpy as np


class BaseParameterCollection:
    def __init__(self):
        self.KB = 0.00008617

    def _print_params(self, params):
        text = ''
        for param in params:
            val = getattr(self, param)
            val_format = self.__custom_format(val)
            text += param + ': ' + val_format + '\n'
        print(text)

    def __custom_format(self, number):
        if type(number) is not type(None) or str:
            if number > 1000:
                return "{:.3e}".format(number)
            elif number >= 0.01:
                return "{:.3g}".format(number)
            elif number < 0.01:
                return "{:.3e}".format(number)
        else:
            return str(number)


class BeamSettings(BaseParameterCollection):
    def __init__(self):
        super().__init__()
        self.beam_settings = ['st_dev', 'fwhm', 'beam_type', 'order', 'f0']
        self._st_dev = 1
        self._fwhm = 2.355 * self._st_dev
        self.beam_type = 'gauss'
        self.order = 1
        self.f0 = 1.0  # 1/(nm^2*s)

    @property
    def st_dev(self):
        """
        Standard deviation of the Gaussian beam.
        :return:
        """
        fwhm = self._fwhm
        n = 1
        if self.beam_type == 'super_gauss':
            n = self.order
        # noinspection PyTestUnpassedFixture
        self._st_dev = fwhm / (2 * np.sqrt(2) * (np.log(2)) ** (1 / 2 / n))
        return self._st_dev

    @st_dev.setter
    def st_dev(self, val):
        self._st_dev = val
        _ = self.fwhm

    @property
    def fwhm(self):
        """
        FWHM of the Gaussian beam, nm.
        :return:
        """
        s = self._st_dev
        n = 1
        if self.beam_type == 'super_gauss':
            n = self.order
        self._fwhm = s * 2 * np.sqrt(2) * (np.log(2)) ** (1 / 2 / n)
        return self._fwhm

    @fwhm.setter
    def fwhm(self, val):
        self._fwhm = val
        _ = self.st_dev

    def print_beam_settings(self):
        self._print_params(self.beam_settings)


class PrecursorParams(BaseParameterCollection):
    def __init__(self):
        super().__init__()
        self.name = ''
        self.base_params = ['s', 'F', 'n0', 'tau', 'sigma', 'D', 'V']
        self.s = 1.0
        self.F = 1.0  # 1/nm^2/s
        self.n0 = 1.0  # 1/nm^2
        self.tau = 1.0  # s
        self.sigma = 1.0  # nm^2
        self.D = 0.0  # nm^2/s
        self.V = 1.0  # nm^3
        self.k0 = 1.0  # Hz
        self.Ea = 1.0  # eV
        self.D0 = 1.0  # nm^2/s
        self.Ed = 1.0  # eV

    def diffusion_coefficient_at_T(self, temp=294):
        """
        Calculate surface diffusion coefficient at a specified temperature.

        :param temp: temperature, K
        :return:
        """
        return self.D0 * np.exp(-self.Ed / self.KB / temp)

    def residence_time_at_T(self, temp=294):
        """
        Calculate residence time at the given temperature.

        :param temp: , K
        :return:
        """
        return 1 / self.k0 * np.exp(self.Ea / self.KB / temp)

    def print_precursor_params(self):
        self._print_params(self.base_params)


class ContinuumModel(BaseParameterCollection):
    """
    The class represents single precursor Continuum Model with diffusion.
    It contains all parameters including base precursor, beam and process parameters.
    Process parameters such as depletion or replenishment rate are calculated according to the model.
    The class also provides an appropriate time step for numerical solution.
    """

    def __init__(self):
        super().__init__()
        self.precursor = PrecursorParams()
        self.beam = BeamSettings()

        self.step = 1.0  # nm

        self._dt = 1.0
        self._dt_diff = np.nan
        self._dt_des = np.nan
        self._dt_diss = np.nan

        self.process_attrs = ['kd', 'kr', 'nd', 'nr', 'tau_in', 'tau_out', 'tau_r', 'p_in', 'p_out', 'p_i', 'p_o',
                              'phi1', 'phi2']
        self._kd = np.nan
        self._kr = np.nan
        self._nd = np.nan
        self._nr = np.nan
        self._tau_in = np.nan
        self._tau_out = np.nan
        self._tau_r = np.nan
        self._p_in = np.nan
        self._p_out = np.nan
        self._p_i = np.nan
        self._p_o = np.nan
        self._phi1 = np.nan
        self._phi2 = np.nan

    def set_precursor_params(self, params: PrecursorParams):
        self.precursor = params

    @property
    def dt(self):
        """
        Maximal time step for the solution of the reaction-diffusion equation, s.
        :return:
        """
        self._dt = np.min([self.dt_des, self.dt_diss, self.dt_diff])
        self._dt -= self._dt * 0.1
        return self._dt

    @dt.setter
    def dt(self, val):
        dt = self.dt
        if val > dt:
            print(f'Not allowed to increase time step. \nTime step larger than {dt} s will crash the solution.')
        else:
            self._dt = val

    @property
    def dt_diff(self):
        """
        Maximal time step for the diffusion process, s.
        :return:
        """
        if self.precursor.D > 0:
            self._dt_diff = self.step ** 2 / (2 * self.precursor.D)
        else:
            self._dt_diff = 1
        return self._dt_diff

    @property
    def dt_diss(self):
        """
        Maximal time step for the dissociation process, s.
        :return:
        """
        self._dt_diss = 1 / self.precursor.sigma / self.beam.f0
        return self._dt_diss

    @property
    def dt_des(self):
        """
        Maximal time step for the desorption process, s.
        :return:
        """
        self._dt_des = self.precursor.tau
        return self._dt_des

    @property
    def kd(self):
        """
        Depletion rate (under beam irradiation), Hz.
        :return:
        """
        self._kd = (self.precursor.s * self.precursor.F / self.precursor.n0 + 1 / self.precursor.tau +
                    self.precursor.sigma * self.beam.f0)
        return self._kd

    @property
    def kr(self):
        """
        Replenishment rate (without beam irradiation), Hz.
        :return:
        """
        self._kr = self.precursor.s * self.precursor.F / self.precursor.n0 + 1 / self.precursor.tau
        return self._kr

    @property
    def nd(self):
        """
        Depleted precursor coverage (under beam irradiation).
        :return:
        """
        self._nd = self.precursor.s * self.precursor.F / self.kd
        return self._nd

    @property
    def nr(self):
        """
        Replenished precursor coverage (without beam irradiation).
        :return:
        """
        self._nr = self.precursor.s * self.precursor.F / self.kr
        return self._nr

    @property
    def tau_in(self):
        """
        Effective residence time in the center of the beam, s.
        :return:
        """
        self._tau_in = 1 / self.kd
        return self._tau_in

    @property
    def tau_out(self):
        """
        Effective residence time outside the beam, s.
        :return:
        """
        self._tau_out = 1 / self.kr
        return self._tau_out

    @property
    def tau_r(self):
        """
        Relative depletion or just Delpetion. Defined as ratio between effective residence time in the center and outside the beam.
        :return:
        """
        self._tau_r = self.tau_out / self.tau_in
        return self._tau_r

    @property
    def p_in(self):
        """
        Precursor molecule diffusion path in the center of the beam, nm.
        :return:
        """
        self._p_in = np.sqrt(self.precursor.D * self.tau_in)
        return self._p_in

    @property
    def p_out(self):
        """
        Precursor molecule diffusion path outside the beam, nm.
        :return:
        """
        self._p_out = np.sqrt(self.precursor.D * self.tau_out)
        return self._p_out

    @property
    def p_i(self):
        """
        Normalized precursor molecule diff. path in the center of the beam.
        :return:
        """
        self._p_i = 2 * self.p_in / self.beam.fwhm
        return self._p_i

    @property
    def p_o(self):
        """
        Diffusive replenishment. Normalized precursor molecule diff. path outside the beam.
        :return:
        """
        self._p_o = 2 * self.p_out / self.beam.fwhm
        return self._p_o

    @property
    def phi1(self):
        """
        Deposit size relative to beam size without surface diffusion.
        First scaling law. Applies only to gaussian beams.
        :return:
        """
        self._phi1 = np.power(np.log2(1 + self.tau_r), 1 / 2 / self.beam.order)
        return self._phi1

    @property
    def phi2(self):
        """
        Deposit size relative to beam size with surface diffusion.
        Second scaling law. Applies only to gaussian beams.
        :return:
        """
        if self.p_o != 0:
            self._phi2 = np.power(np.log2(2 + (self.tau_r - 1) / (1 + self.p_o ** 2)), 1 / 2 / self.beam.order)
        else:
            self._phi2 = np.nan
        return self._phi2

    def print_initial_parameters(self):
        self.precursor.print_precursor_params()

    def print_process_attributes(self):
        self._print_params(self.process_attrs)

    def save_to_file(self, filename):
        """
        Save experiment to a file.
        :param filename: full file name (including path and extension)
        :return:
        """
        with open(filename, mode='wb') as f:
            dump(self, f)
