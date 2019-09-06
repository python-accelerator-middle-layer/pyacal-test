import time as _time
from threading import Thread as _Thread, Event as _Event
import numpy as np

import pyaccel
from siriuspy.epics import PV
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.csdevice.orbitcorr import SOFBFactory

from apsuite.optimization import SimulAnneal


class SOFB:

    def __init__(self, acc):
        self.data = SOFBFactory.create(acc)
        orbtp = 'MTurnIdx' if acc == 'BO' else 'SPass'
        self._trajx = PV(acc+'-Glob:AP-SOFB:'+orbtp+'OrbX-Mon')
        self._trajy = PV(acc+'-Glob:AP-SOFB:'+orbtp+'OrbY-Mon')
        self._rst = PV(acc+'-Glob:AP-SOFB:SmoothReset-Cmd')
        self._npts_sp = PV(acc+'-Glob:AP-SOFB:SmoothNrPts-SP')
        self._npts_rb = PV(acc+'-Glob:AP-SOFB:BufferCount-Mon')

    @property
    def connected(self):
        conn = self._trajx.connected
        conn &= self._trajx.connected
        conn &= self._rst.connected
        conn &= self._npts_sp.connected
        conn &= self._npts_rb.connected
        return conn

    @property
    def trajx(self):
        return self._trajx.get()

    @property
    def trajy(self):
        return self._trajy.get()

    @property
    def nr_points(self):
        return self._npts_rb.value

    @nr_points.setter
    def nr_points(self, value):
        self._npts_sp.value = int(value)

    def wait(self, timeout=10):
        inter = 0.05
        n = int(timeout/inter)
        for _ in range(n):
            if self._npts_rb.value >= self._npts_sp.value:
                break
            _time.sleep(inter)
        else:
            print('WARN: Timed out waiting orbit.')

    def reset(self):
        self._rst.value = 1


class Klystron:

    def __init__(self):
        self._sp = PV('LA-RF:LLRF:KLY2:SET_AMP')
        self._rb = PV('LA-RF:LLRF:KLY2:GET_AMP')

    @property
    def amplitude(self):
        return self._rb.value

    @amplitude.setter
    def amplitude(self, value):
        self._sp.value = value

    @property
    def connected(self):
        return self._sp.connected & self._rb.connected


class ParamsDisp:

    def __init__(self):
        self.klystron_delta = -2
        self.wait_time = 40
        self.timeout_orb = 10
        self.num_points = 10
        self.klystron_excit_coefs = [1.098, 66.669]


class MeasureDispTBBO:

    def __init__(self):
        self.bo_sofb = SOFB('BO')
        self.tb_sofb = SOFB('TB')
        self.params = ParamsDisp()
        self.kly = Klystron()

    @property
    def energy(self):
        return np.polyval(self.params.klystron_excit_coefs, self.kly.amplitude)

    @property
    def connected(self):
        conn = self.bo_sofb.connected
        conn &= self.tb_sofb.connected
        return conn

    @property
    def trajx(self):
        return np.hstack([self.tb_sofb.trajx, self.bo_sofb.trajx])

    @property
    def trajy(self):
        return np.hstack([self.tb_sofb.trajy, self.bo_sofb.trajy])

    @property
    def nr_points(self):
        return min(self.tb_sofb.nr_points, self.bo_sofb.nr_points)

    @nr_points.setter
    def nr_points(self, value):
        self.tb_sofb.nr_points = int(value)
        self.bo_sofb.nr_points = int(value)

    def wait(self, timeout=10):
        self.tb_sofb.wait(timeout=timeout)
        self.bo_sofb.wait(timeout=timeout)

    def reset(self, wait=0):
        _time.sleep(wait)
        self.tb_sofb.reset()
        self.bo_sofb.reset()
        _time.sleep(1)

    def measure_dispersion(self):
        self.nr_points = self.params.num_points
        delta = self.params.klystron_delta

        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_orb)
        orb = [-np.hstack([self.trajx, self.trajy]), ]
        ene0 = self.energy

        origamp = self.kly.amplitude
        self.kly.amplitude = origamp + delta

        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_orb)
        orb.append(np.hstack([self.trajx, self.trajy]))
        ene1 = self.energy

        self.kly.amplitude = origamp

        d_ene = ene1/ene0 - 1
        return np.array(orb).sum(axis=0) / d_ene


class ParamsDispMat:

    def __init__(self):
        self.deltas = {'QF': 0.1, 'QD': 0.1}


class MeasureDispMatTBBO:

    def __init__(self, quads):
        self._all_corrs = quads
        self.measdisp = MeasureDispTBBO()
        self.params = ParamsDispMat()
        self._matrix = dict()
        self._corrs_to_measure = []
        self._thread = _Thread(target=self._measure_matrix_thread)
        self._stopped = _Event()

    @property
    def connected(self):
        conn = self.measdisp.connected
        conn &= all(map(lambda x: x.connected, self._all_corrs.values()))
        return conn

    @property
    def corr_names(self):
        return sorted(self._all_corrs.keys())

    @property
    def corrs_to_measure(self):
        if not self._corrs_to_measure:
            return sorted(self._all_corrs.keys() - self._matrix.keys())
        return self._corrs_to_measure

    @corrs_to_measure.setter
    def corrs_to_measure(self, value):
        self._corrs_to_measure = sorted([_PVName(n) for n in value])

    @property
    def matrix(self):
        mat = np.zeros(
            [len(self._all_corrs), 2*self.measdisp.trajx.size], dtype=float)
        for i, cor in enumerate(self.corr_names):
            line = self._matrix.get(cor)
            if line is not None:
                mat[i, :] = line
        return mat

    @property
    def measuring(self):
        return self._thread.is_alive()

    def start(self):
        if not self._thread.is_alive():
            self._thread = _Thread(
                target=self._measure_matrix_thread, daemon=True)
            self._stopped.clear()
            self._thread.start()

    def stop(self):
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
                print('  pos' if sig>0 else '  neg\n', end='')
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
    dene = 0.0001
    rout, *_ = pyaccel.tracking.line_pass(
        model,
        [[0, 0, 0, 0, dene/2, 0],
         [0, 0, 0, 0, -dene/2, 0]],
        bpms)
    dispx = (rout[0, 0, :] - rout[1, 0, :]) / dene
    dispy = (rout[0, 2, :] - rout[1, 2, :]) / dene
    return np.hstack([dispx, dispy])


def calc_model_dispmatTBBO(tb_mod, bo_mod, corr_names, elems, nturns=3,
                           dKL=None):
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

    def __init__(self, tb_model, bo_model, corr_names, elems,
                 respmat, nturns=5, save=False, in_sept=True):
        super().__init__(save=save)
        self.tb_model = tb_model
        self.bo_model = bo_model
        self.corr_names = corr_names
        self.elems = elems
        self.nturns = nturns
        self.respmat = respmat
        self.in_sept = in_sept

    def initialization(self):
        return

    def calc_obj_fun(self):
        if self.in_sept:
            sept_idx = pyaccel.lattice.find_indices(
                self.tb_model, 'fam_name', 'InjSept')
        else:
            sept_idx = self.elems['TB-04:MA-CV-2'].model_indices
        k, ks = self._position
        pyaccel.lattice.set_attribute(self.tb_model, 'K', sept_idx, k)
        pyaccel.lattice.set_attribute(self.tb_model, 'Ks', sept_idx, ks)
        respmat = calc_model_dispmatTBBO(
            self.tb_model, self.bo_model, self.corr_names, self.elems,
            nturns=self.nturns)
        respmat -= self.respmat
        return np.sqrt(np.mean(respmat*respmat))
