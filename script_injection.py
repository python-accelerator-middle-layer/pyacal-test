#!/usr/bin/env python-sirius
"""."""

import time as _time
from epics import PV as _PV
import numpy as np
from optimization import PSO


class PSOInjection(PSO):

    def initialization(self):
        self._upper_limits = np.array([2e-3, 2e-3, 2e-3, 2e-3, 1e-3])
        self._lower_limits = - self._upper_limits
        self._sum_lim = 0
        self._nswarm = 10 + 2 * int(np.sqrt(len(self._upper_limits)))
        self._p_bpm = 1.0
        self._p_orb = 1.0e3

        # Single Pass acquisition
        _pv_sp_orbx = _PV('BO-Glob:AP-SOFB:SPassOrbX-Mon')  # um
        _pv_sp_orby = _PV('BO-Glob:AP-SOFB:SPassOrbY-Mon')  # um
        _pv_sp_sum = _PV('BO-Glob:AP-SOFB:SPassSum-Mon')    # counts

        sp_diag = [_pv_sp_orbx,
                   _pv_sp_orby,
                   _pv_sp_sum]

        # # Multi Turn acquisition
        # _pv_mt_orbx = _PV('BO-Glob:AP-SOFB:MTurnIdxOrbX-Mon')  # um
        # _pv_mt_orby = _PV('BO-Glob:AP-SOFB:MTurnIdxOrbY-Mon')  # um
        # _pv_mt_sum = _PV('BO-Glob:AP-SOFB:MTurnSum-Mon')    # counts

        # mt_diag = [_pv_mt_orbx,
        #            _pv_mt_orby,
        #            _pv_mt_sum]

        self.diag = sp_diag

        _pv_dx = _PV('TB-Glob:AP-PosAng:DeltaPosX-SP')   # mm
        _pv_dxl = _PV('TB-Glob:AP-PosAng:DeltaAngX-SP')  # mrad
        _pv_dy = _PV('TB-Glob:AP-PosAng:DeltaPosY-SP')   # mm
        _pv_dyl = _PV('TB-Glob:AP-PosAng:DeltaAngY-SP')  # mrad
        _pv_kckr = _PV('BO-01D:PM-InjKckr:Kick-SP')      # mrad
        self.corrs = [_pv_dx,
                      _pv_dxl,
                      _pv_dy,
                      _pv_dyl,
                      _pv_kckr]

        _pv_resetposang = _PV('TB-Glob:AP-PosAng:SetNewRefKick-Cmd')
        _pv_resetposang.value = 1

        self.reference = np.array([cval.value for cval in self.corrs])

        _pv_buffer = _PV('BO-Glob:AP-SOFB:SmoothNrPts-RB')
        self._wait = _pv_buffer.value

    def calc_merit_function(self):
        f_out = np.zeros(self._nswarm)
        ind_bpm = np.arange(1, len(self.diag[-1].value) + 1)

        for i in range(self._nswarm):
            change = self.reference + self._position[i, :]

            for k in range(len(self.corrs)):
                self.corrs[k].value = change[k]

            _time.sleep(self._wait * 0.5 + 1)
            f_bpm = np.dot(ind_bpm, self.diag[-1].value)
            bpm_sel = self.diag[-1].value < self._sum_lim
            ind_sel = np.where(bpm_sel)
            orbx = self.diag[0].value
            orby = self.diag[1].value
            sum_quad = np.sum(
                orbx[:ind_sel[0]] ** 2 +
                orby[:ind_sel[0]] ** 2)
            f_orb = np.sqrt(sum_quad)
            if f_orb.size > 0 and f_orb != 0:
                f_out[i] = self._p_bpm * f_bpm + self._p_orb / f_orb
            else:
                f_out[i] = self._p_bpm * f_bpm
        return - f_out

# if __name__ == "__main__":
#     import argparse as _argparse

#     parser = _argparse.ArgumentParser(
#         description="PSO script for Booster Injection Optimization.")

#     parser.add_argument(
#         '-sum', '--sum_lim', type=float, default=1e3,
#         help='Minimum BPM Sum Signal to calculate merit function (1 kcount).')
#     parser.add_argument(
#         '-niter', '--niter', type=int, default=30,
#         help='Number of Iteractions (30).')
#     parser.add_argument(
#         '-lim', '--limits', nargs='+', type=float,
#         help='Upper limits [XPos, XAng, YPos, YAng, Kckr]')

#     args = parser.parse_args()
#     Pso = PSOInjection(limits=args.limits, sum_lim=args.sum_lim)
#     Pso._start_optimization(niter=args.niter)
