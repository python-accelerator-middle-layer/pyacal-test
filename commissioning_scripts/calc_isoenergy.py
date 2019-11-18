#!/usr/bin/env python-sirius
"""."""

import numpy as np


class Isoenergy():
    """."""

    def __init__(self, qinit=None, qfinal=None):
        """."""
        self.qinit = qinit
        self.qfinal = qfinal
        self.get_params_charge()

    def bias2charge(self, b):
        """."""
        p = np.array(
            [7.90472090e-04, 1.34199022e-01, 7.62099148e+00, 1.44833137e+02])
        return np.polyval(p, b)

    def charge2bias(self, q):
        """."""
        p = np.array(
            [7.90472090e-04, 1.34199022e-01, 7.62099148e+00, 1.44833137e+02])
        newp = np.poly1d(p)
        roots = (newp-q).roots
        return roots[~np.iscomplex(roots)].real[0]

    def charge2energy(self, q):
        """."""
        p = np.array(
            [1.09315839e-03, -3.16996192e-02, -1.72815644e-01, 1.46492452e+02])
        return np.polyval(p, q)

    def charge2spread(self, q):
        """."""
        p = np.array([0.00178367, 0.0177915, 0.05214362, 0.12375059])
        return np.polyval(p, q)

    def kly2en(self, amp):
        """."""
        p = [0.925, 77.05]
        return np.polyval(p, amp)

    def en2kly(self, en):
        """."""
        p = np.array([1.01026423, 71.90322743])
        return (en - p[1])/p[0]

    def calc_deltak2(self, delta_en):
        """."""
        p = np.array([1.01026423, 71.90322743])
        return delta_en/p[0]

    def calc_deltaen(self, delta_q):
        """."""
        p = np.array(
            [1.09315839e-03, -3.16996192e-02, -1.72815644e-01, 1.46492452e+02])
        return p[0]*delta_q**3 + p[1]*delta_q**2 + p[2]*delta_q

    def get_params_charge(self):
        """."""
        energy_init = self.charge2energy(q=self.qinit)
        energy_final = self.charge2energy(q=self.qfinal)
        denergy = energy_final - energy_init
        bias_init = self.charge2bias(q=self.qinit)
        spread_init = self.charge2spread(q=self.qinit)
        bias_final = self.charge2bias(q=self.qfinal)
        spread_final = self.charge2spread(q=self.qfinal)
        delta_k2 = self.calc_deltak2(-denergy)
        print(
            'Charge: {0:4f} nC --> {1:4f} nC'.format(
                self.qinit, self.qfinal))
        print(
            'Bias: {0:4f} V --> {1:4f} V'.format(
                bias_init, bias_final))
        print(
            'Energy: {0:4f} MeV --> {1:4f} MeV'.format(
                energy_init, energy_final-denergy))
        print(
            'Spread: {0:4f} % --> {1:4f} %'.format(
                spread_init, spread_final))
        print(
            'Apply Delta Klystron 2 Amp: {0:4f} %'.format(delta_k2))
