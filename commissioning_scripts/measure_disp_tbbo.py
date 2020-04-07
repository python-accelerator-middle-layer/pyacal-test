#!/usr/bin/env python-sirius
"""."""

import time as _time
from threading import Thread as _Thread, Event as _Event
import numpy as np

import pyaccel
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import SOFB, LLRF

from apsuite.optimization import SimulAnneal
from apsuite.commissioning_scripts.base import BaseClass


class ParamsDisp:
    """."""

    def __init__(self):
        """."""
        self.klystron_delta = -2
        self.wait_time = 40
        self.timeout_orb = 10
        self.num_points = 10
        # self.klystron_excit_coefs = [1.098, 66.669]  # old
        # self.klystron_excit_coefs = [1.01026423, 71.90322743]  # > 2.5nC
        self.klystron_excit_coefs = [0.80518365, 87.56545895]  # < 2.5nC


class MeasureDispTBBO(BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__(ParamsDisp())
        self.devices = {
            'bo_sofb': SOFB(SOFB.DEVICES.BO),
            'tb_sofb': SOFB(SOFB.DEVICES.TB),
            'kly2': LLRF(LLRF.DEVICES.LI_KLY2),
            }

    @property
    def energy(self):
        """."""
        return np.polyval(
            self.params.klystron_excit_coefs, self.devices['kly2'].amplitude)

    @property
    def trajx(self):
        """."""
        return np.hstack(
            [self.devices['tb_sofb'].trajx, self.devices['bo_sofb'].trajx_idx])

    @property
    def trajy(self):
        """."""
        return np.hstack(
            [self.devices['tb_sofb'].trajy, self.devices['bo_sofb'].trajy_idx])

    @property
    def nr_points(self):
        """."""
        return min(
            self.devices['tb_sofb'].nr_points,
            self.devices['bo_sofb'].nr_points)

    @nr_points.setter
    def nr_points(self, value):
        self.devices['tb_sofb'].nr_points = int(value)
        self.devices['bo_sofb'].nr_points = int(value)

    def wait(self, timeout=10):
        """."""
        self.devices['tb_sofb'].wait_buffer(timeout=timeout)
        self.devices['bo_sofb'].wait_buffer(timeout=timeout)

    def reset(self, wait=0):
        """."""
        _time.sleep(wait)
        self.devices['tb_sofb'].cmd_reset()
        self.devices['bo_sofb'].cmd_reset()
        _time.sleep(1)

    def measure_dispersion(self):
        """."""
        self.nr_points = self.params.num_points
        delta = self.params.klystron_delta

        self.reset(3)
        self.wait(self.params.timeout_orb)
        orb = [-np.hstack([self.trajx, self.trajy]), ]
        ene0 = self.energy

        origamp = self.devices['kly2'].amplitude
        self.devices['kly2'].amplitude = origamp + delta

        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_orb)
        orb.append(np.hstack([self.trajx, self.trajy]))
        ene1 = self.energy

        self.devices['kly2'].amplitude = origamp

        d_ene = ene1/ene0 - 1
        return np.array(orb).sum(axis=0) / d_ene


class ParamsDispMat:
    """."""

    def __init__(self):
        """."""
        self.deltas = {'QF': 0.1, 'QD': 0.1}


class MeasureDispMatTBBO(BaseClass):
    """."""

    def __init__(self, quads):
        """."""
        super().__init__(ParamsDispMat())
        self._all_corrs = quads
        self.measdisp = MeasureDispTBBO()
        self._matrix = dict()
        self._corrs_to_measure = []
        self._thread = _Thread(target=self._measure_matrix_thread)
        self._stopped = _Event()

    @property
    def connected(self):
        """."""
        conn = self.measdisp.connected
        conn &= all(map(lambda x: x.connected, self._all_corrs.values()))
        return conn

    @property
    def corr_names(self):
        """."""
        return sorted(self._all_corrs.keys())

    @property
    def corrs_to_measure(self):
        """."""
        if not self._corrs_to_measure:
            return sorted(self._all_corrs.keys() - self._matrix.keys())
        return self._corrs_to_measure

    @corrs_to_measure.setter
    def corrs_to_measure(self, value):
        self._corrs_to_measure = sorted([_PVName(n) for n in value])

    @property
    def matrix(self):
        """."""
        mat = np.zeros(
            [len(self._all_corrs), 2*self.measdisp.trajx.size], dtype=float)
        for i, cor in enumerate(self.corr_names):
            line = self._matrix.get(cor)
            if line is not None:
                mat[i, :] = line
        return mat

    @property
    def measuring(self):
        """."""
        return self._thread.is_alive()

    def start(self):
        """."""
        if not self._thread.is_alive():
            self._thread = _Thread(
                target=self._measure_matrix_thread, daemon=True)
            self._stopped.clear()
            self._thread.start()

    def stop(self):
        """."""
        self._stopped.set()

    def _measure_matrix_thread(self):
        corrs = self.corrs_to_measure
        print('Starting...')
        for i, cor in enumerate(corrs):
            print('{0:2d}|{1:2d}: {20:s}'.format(i, len(corrs), cor), end='')
            orb = []
            delta = self._get_delta(cor)
            origkl = self._all_corrs[cor].strength
            for sig in (1, -1):
                print('  pos' if sig > 0 else '  neg\n', end='')
                self._all_corrs[cor].strength = origkl + sig * delta / 2
                orb.append(sig*self.measdisp.measure_dispersion())
                if self._stopped.is_set():
                    break
            else:
                self._matrix[cor] = np.array(orb).sum(axis=0)/delta
            self._all_corrs[cor].strength = origkl
            if self._stopped.is_set():
                print('Stopped!')
                break
        else:
            print('Finished!')

    def _get_delta(self, cor):
        for k, v in self.params.deltas.items():
            if cor.dev.startswith(k):
                return v
        print('ERR: delta not found!')
        return 0.0

def calc_model_dispersionTBBO(model, bpms):
    """."""
    dene = 0.0001
    rin = np.array([
        [0, 0, 0, 0, dene/2, 0],
        [0, 0, 0, 0, -dene/2, 0]]).T
    rout, *_ = pyaccel.tracking.line_pass(
        model, rin, bpms)
    dispx = (rout[0, 0, :] - rout[0, 1, :]) / dene
    dispy = (rout[2, 0, :] - rout[2, 1, :]) / dene
    return np.hstack([dispx, dispy])


def calc_model_dispmatTBBO(tb_mod, bo_mod, corr_names, elems, nturns=3,
                           dKL=None):
    """."""
    dKL = 0.0001 if dKL is None else dKL

    model = tb_mod + nturns*bo_mod
    bpms = pyaccel.lattice.find_indices(model, 'fam_name', 'BPM')

    matrix = np.zeros((len(corr_names), 2*len(bpms)))
    for idx, corr in enumerate(corr_names):
        elem = elems[corr]
        model = tb_mod + nturns*bo_mod
        disp0 = calc_model_dispersionTBBO(model, bpms)
        elem.model_strength += dKL
        model = tb_mod + nturns*bo_mod
        disp = calc_model_dispersionTBBO(model, bpms)
        elem.model_strength -= dKL
        matrix[idx, :] = (disp-disp0)/dKL
    return matrix


class FindSeptQuad(SimulAnneal):
    """."""

    def __init__(self, tb_model, bo_model, corr_names, elems,
                 respmat, nturns=5, save=False, in_sept=True):
        """."""
        super().__init__(save=save)
        self.tb_model = tb_model
        self.bo_model = bo_model
        self.corr_names = corr_names
        self.elems = elems
        self.nturns = nturns
        self.respmat = respmat
        self.in_sept = in_sept

    def initialization(self):
        """."""
        return

    def calc_obj_fun(self):
        """."""
        if self.in_sept:
            sept_idx = pyaccel.lattice.find_indices(
                self.tb_model, 'fam_name', 'InjSept')
        else:
            sept_idx = self.elems['TB-04:MA-CV-2'].model_indices
        kxl, kyl, ksxl, ksyl = self._position
        pyaccel.lattice.set_attribute(self.tb_model, 'KxL', sept_idx, kxl)
        pyaccel.lattice.set_attribute(self.tb_model, 'KyL', sept_idx, kyl)
        pyaccel.lattice.set_attribute(self.tb_model, 'KsxL', sept_idx, ksxl)
        pyaccel.lattice.set_attribute(self.tb_model, 'KsyL', sept_idx, ksyl)
        respmat = calc_model_dispmatTBBO(
            self.tb_model, self.bo_model, self.corr_names, self.elems,
            nturns=self.nturns)
        respmat -= self.respmat
        return np.sqrt(np.mean(respmat*respmat))
