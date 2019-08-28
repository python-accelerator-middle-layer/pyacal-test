#!/usr/bin/env python-sirius
"""."""

import time as _time
from epics import PV as _PV
import numpy as np
from apsuite.optimization import PSO


class PSOLinac(PSO):

    ENERGY = 0
    SPREAD = 1
    TRANSMIT = 2

    # def __init__(self, low_lim, up_lim):
    #     self._upper_limits = np.array(up_lim)
    #     self._lower_limits = np.array(low_lim)
    #     self._ndim = len(self._upper_limits)
    #     self._nswarm = 10 + 2 * int(np.sqrt(self._ndim))
    #     PSO.__init__(self, nswarm=self._nswarm)

    def initialization(self):
        # prefix = 'murilo-lnls558-linux-'

        self._lower_limits = np.array([-180, 0, -180, 0, -180, 0])
        self._upper_limits = np.array([180, 100, 180, 100, 180, 100])

        self._wait = 3
        self.p_energy = 1
        self.p_spread = 75
        self.p_transmit = 150

        _pv_phase_shb = _PV('LA-RF:LLRF:BUN1:SET_PHASE')
        _pv_phase_kly1 = _PV('LA-RF:LLRF:KLY1:SET_PHASE')
        _pv_phase_kly2 = _PV('LA-RF:LLRF:KLY2:SET_PHASE')

        _pv_amp_shb = _PV('LA-RF:LLRF:BUN1:SET_AMP')
        _pv_amp_kly1 = _PV('LA-RF:LLRF:KLY1:SET_AMP')
        _pv_amp_kly2 = _PV('LA-RF:LLRF:KLY2:SET_AMP')

        self.params = [_pv_phase_shb,
                       _pv_amp_shb,
                       _pv_phase_kly1,
                       _pv_amp_kly1,
                       _pv_phase_kly2,
                       _pv_amp_kly2]

        _pv_energy = _PV('LA-BI:PRF4:X:Gauss:Peak')
        _pv_spread = _PV('LA-BI:PRF4:X:Gauss:Sigma')
        _pv_transmit = _PV('LI-Glob:AP-TranspEff:Eff-Mon')

        self.diag = [_pv_energy,
                     _pv_spread,
                     _pv_transmit]

        self._ndim = len(self._upper_limits)
        self._nswarm = 10 + 2 * int(np.sqrt(self._ndim))

    def calc_merit_function(self):
        f_out = np.zeros(self._nswarm)

        for i in range(self._nswarm):
            for k in range(self._ndim):
                self.params[k].value = self._position[i, k]

            # print('Waiting ' + str(self._wait) + ' seconds')
            print('Particle ' + str(i+1) + '|' + str(self._nswarm))
            _time.sleep(self._wait)
            f_out[i] = self.p_energy * self.diag[self.ENERGY].value + \
                       self.p_transmit * self.diag[self.TRANSMIT].value
            if self.diag[self.SPREAD].value:
                f_out[i] += self.p_spread / self.diag[self.SPREAD].value
        return - f_out

# if __name__ == "__main__":
#     import argparse as _argparse

#     parser = _argparse.ArgumentParser(
#         description="PSO script for LLRF Linac Optimization.")

#     parser.add_argument(
#         '-niter', '--niter', type=int, default=30,
#         help='Number of Iteractions (30).')
#     parser.add_argument(
#         '-upper_lim', '--up_lim', nargs='+', type=float,
#         help='Upper limits [SHB_Phase, SHB_Amp, Kly1_Phase, Kly1_Amp, Kly2_Phase, Kly2_Amp]')
#     parser.add_argument(
#         '-lower_lim', '--low_lim', nargs='+', type=float,
#         help='Lower limits [SHB_Phase, SHB_Amp, Kly1_Phase, Kly1_Amp, Kly2_Phase, Kly2_Amp]')

#     args = parser.parse_args()
#     Pso = PSOLinac(low_lim=args.low_lim, up_lim=args.up_lim)
#     Pso._start_optimization(niter=args.niter)
