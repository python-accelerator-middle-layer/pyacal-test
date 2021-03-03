"""."""

import numpy as np

from ..optimization import PSO, SimulAnneal, GA


class SHBPSO(PSO):
    """."""

    C = 299792458
    E0 = 0.51099895e6
    EMIN = E0 + 90e3
    DRIFT = 615e-3
    FREQUENCY = 499.658e6
    WAVELEN = C/FREQUENCY
    NPOINT = 51
    BUN_LEN = np.pi

    def initialization(self):
        """."""
        self._upper_limits = np.array([np.pi, 40e3])
        self._lower_limits = np.array([-np.pi, 10e3])
        self.beta0 = self.calc_beta(self.EMIN)
        self._nswarm = 10 + 2 * int(np.sqrt(len(self._upper_limits)))

    def phase_drift(self, phi_c, vg):
        """."""
        phi_min = phi_c - self.BUN_LEN / 2
        phi_max = phi_c + self.BUN_LEN / 2
        phi0 = np.linspace(phi_min, phi_max, self.NPOINT)

        E = self.EMIN - vg * np.sin(phi0)
        beta = self.calc_beta(E)
        dphi = (2 * np.pi / self.WAVELEN) * (1/self.beta0 - 1/beta)
        phif = phi0 + self.DRIFT * dphi
        return phi0, phif

    def calc_beta(self, E):
        """."""
        gamma = E/self.E0
        return np.sqrt(1 - 1/gamma**2)

    def calc_merit_function(self):
        """."""
        f_out = np.zeros(self._nswarm)

        for i in range(self._nswarm):
            phi_c, vg = self._position[i]
            phi_init, phi_final = self.phase_drift(phi_c, vg)
            t0 = np.std(phi_init)
            tf = np.std(phi_final)
            f_out[i] = t0/tf
        return - f_out


class SHBSimulAnneal(SimulAnneal):
    """."""

    C = 299792458
    E0 = 0.51099895e6
    EMIN = E0 + 90e3
    DRIFT = 615e-3
    FREQUENCY = 499.658e6
    WAVELEN = C/FREQUENCY
    NPOINT = 51
    BUN_LEN = np.pi

    def initialization(self):
        """."""
        self._upper_limits = np.array([np.pi, 40e3])
        self._lower_limits = np.array([-np.pi, 10e3])
        self._max_delta = np.array([2*np.pi, 30e3])
        self.beta0 = self.calc_beta(self.EMIN)
        self._temperature = 0

    def phase_drift(self, phi_c, vg):
        """."""
        phi_min = phi_c - self.BUN_LEN / 2
        phi_max = phi_c + self.BUN_LEN / 2
        phi0 = np.linspace(phi_min, phi_max, self.NPOINT)

        E = self.EMIN - vg * np.sin(phi0)
        beta = self.calc_beta(E)
        dphi = (2 * np.pi / self.WAVELEN) * (1/self.beta0 - 1/beta)
        phif = phi0 + self.DRIFT * dphi
        return phi0, phif

    def calc_beta(self, E):
        """."""
        gamma = E/self.E0
        return np.sqrt(1 - 1/gamma**2)

    def calc_merit_function(self):
        """."""
        phi_c, vg = self._position[0], self._position[1]
        phi_init, phi_final = self.phase_drift(phi_c, vg)
        t0 = np.std(phi_init)
        tf = np.std(phi_final)
        return - t0/tf


class SHBGA(GA):
    """."""

    C = 299792458
    E0 = 0.51099895e6
    EMIN = E0 + 90e3
    DRIFT = 615e-3
    FREQUENCY = 499.658e6
    WAVELEN = C/FREQUENCY
    NPOINT = 51
    BUN_LEN = np.pi

    def __init__(self, npop, nparents, mutrate):
        """."""
        super().__init__(npop=npop, nparents=nparents, mutrate=mutrate)

    def initialization(self):
        """."""
        self._upper_limits = np.array([np.pi, 40e3])
        self._lower_limits = np.array([-np.pi, 10e3])
        self.beta0 = self.calc_beta(self.EMIN)
        self._temperature = 0

    def phase_drift(self, phi_c, vg):
        """."""
        phi_min = phi_c - self.BUN_LEN / 2
        phi_max = phi_c + self.BUN_LEN / 2
        phi0 = np.linspace(phi_min, phi_max, self.NPOINT)

        E = self.EMIN - vg * np.sin(phi0)
        beta = self.calc_beta(E)
        dphi = (2 * np.pi / self.WAVELEN) * (1/self.beta0 - 1/beta)
        phif = phi0 + self.DRIFT * dphi
        return phi0, phif

    def calc_beta(self, E):
        """."""
        gamma = E/self.E0
        return np.sqrt(1 - 1/gamma**2)

    def calc_merit_function(self):
        """."""
        f_out = np.zeros(self._npop)

        for i in range(self._npop):
            phi_c, vg = self._indiv[i, 0], self._indiv[i, 1]
            phi_init, phi_final = self.phase_drift(phi_c, vg)
            t0 = np.std(phi_init)
            tf = np.std(phi_final)
            f_out[i] = t0/tf
        return - f_out
