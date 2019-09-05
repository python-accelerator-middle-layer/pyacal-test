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


class Params:

    def __init__(self):
        self.deltas = {
            'CH': 0.3e-3, 'CV': 0.15e-3, 'InjSept': 0.3e-3, 'InjKckr': 0.3e-3}
        self.wait_time = 2
        self.timeout_orb = 10
        self.num_points = 10


class MeasureRespMatTBBO:

    def __init__(self, all_corrs):
        self.bo_sofb = SOFB('BO')
        self.tb_sofb = SOFB('TB')
        self._all_corrs = all_corrs
        self.params = Params()
        self._matrix = dict()
        self._corrs_to_measure = []
        self._thread = _Thread(target=self._measure_matrix_thread)
        self._stopped = _Event()

    @property
    def connected(self):
        conn = self.bo_sofb.connected
        conn &= self.tb_sofb.connected
        conn &= all(map(lambda x: x.connected, self._all_corrs.values()))
        return conn

    @property
    def trajx(self):
        return np.hstack([self.tb_sofb.trajx, self.bo_sofb.trajx])

    @property
    def trajy(self):
        return np.hstack([self.tb_sofb.trajy, self.bo_sofb.trajy])

    def wait(self, timeout=10):
        self.tb_sofb.wait(timeout=timeout)
        self.bo_sofb.wait(timeout=timeout)

    def reset(self, wait=0):
        if self._stopped.wait(wait):
            return False
        self.tb_sofb.reset()
        self.bo_sofb.reset()
        if self._stopped.wait(wait):
            return False
        return True

    @property
    def corr_names(self):
        corrs = sorted([
            c for c in self._all_corrs if not c.dev.startswith('CV')])
        corrs.extend(sorted([
            c for c in self._all_corrs if c.dev.startswith('CV')]))
        return corrs

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
        mat = np.zeros([len(self._all_corrs), 2*self.trajx.size], dtype=float)
        for i, cor in enumerate(self.corr_names):
            line = self._matrix.get(cor)
            if line is not None:
                mat[i, :] = line
        return mat

    @property
    def measuring(self):
        return self._thread.is_alive()

    @property
    def nr_points(self):
        return min(self.tb_sofb.nr_points, self.bo_sofb.nr_points)

    @nr_points.setter
    def nr_points(self, value):
        self.tb_sofb.nr_points = int(value)
        self.bo_sofb.nr_points = int(value)

    def start(self):
        if not self._thread.is_alive():
            self._thread = _Thread(
                target=self._measure_matrix_thread, daemon=True)
            self._stopped.clear()
            self._thread.start()

    def stop(self):
        self._stopped.set()

    def _measure_matrix_thread(self):
        self.nr_points = self.params.num_points
        corrs = self.corrs_to_measure
        for cor in corrs:
            orb = []
            delta = self.params.deltas[cor.dev]
            origkick = self._all_corrs[cor].strength
            for sig in (1, -1):
                self._all_corrs[cor].strength = origkick + sig * delta / 2
                if not self.reset(self.params.wait_time):
                    break
                self.wait(self.params.timeout_orb)
                orb.append(sig*np.hstack([self.trajx, self.trajy]))
            else:
                self._matrix[cor] = np.array(orb).sum(axis=0)/delta
            self._all_corrs[cor].strength = origkick
            if self._stopped.is_set():
                break


def calc_model_respmatTBBO(tb_mod, bo_mod, corr_names, data, nturns=3,
                           meth='middle'):
    model = tb_mod + nturns*bo_mod
    bpms = pyaccel.lattice.find_indices(model, 'fam_name', 'BPM')[1:]
    _, cumulmat = pyaccel.tracking.find_m44(model)

    matrix = np.zeros((len(corr_names), 2*len(bpms)))
    for idx, corr in enumerate(corr_names):
        indcs = np.array(data[corr]['index'])
        if corr.sec == 'BO':
            print('Booster ', corr)
            indcs += len(tb_mod)
        leng = sum([model[i].length for i in indcs])
        ksl = sum([model[i].KsL for i in indcs])
        kl = sum([model[i].KL for i in indcs])
        matrix[idx, :] = _get_respmat_line(
            cumulmat, indcs, bpms, length=leng, kl=kl, ksl=ksl,
            cortype=data[corr]['magnet_type'],
            meth=meth)
    return matrix


def _get_respmat_line(cumul_mat, indcs, bpms, length, kl=0, ksl=0,
                      cortype='vertical', meth='middle'):

    idx = 3 if cortype.startswith('vertical') else 1
    cor = indcs[-1]+1
    if meth.lower().startswith('begin'):
        cor = indcs[0]
    elif meth.lower().startswith('mid'):
        drift = np.eye(4, dtype=float)
        drift[0, 1] = length/2
        drift[2, 3] = length/2
        # quadrupoles do not affect the response matrix in first order.
        # I kept this here only to demonstrate this:
        quads = np.eye(4, dtype=float)
        quads[1, 0] = -kl/2
        quads[3, 2] = kl/2
        quads[1, 2] = -ksl/2
        quads[3, 0] = -ksl/2
        half_cor = np.dot(drift, quads)

    m0c = cumul_mat[cor]
    respx = []
    respy = []
    for bpm in bpms:
        if bpm < indcs[0]:
            respx.append(0)
            respy.append(0)
            continue
        bpm_mat = cumul_mat[bpm]
        mat = np.linalg.solve(m0c.T, bpm_mat.T).T
        if meth.lower().startswith('mid'):
            mat = np.dot(mat, half_cor)
        respx.append(mat[0, idx])
        respy.append(mat[2, idx])

    return np.hstack([respx, respy])


class FindSeptQuad(SimulAnneal):

    def __init__(self, tb_model, bo_model, corr_names, data,
                 respmat, nturns=5, save=False, in_sept=True):
        super().__init__(save=save)
        self.tb_model = tb_model
        self.bo_model = bo_model
        self.corr_names = corr_names
        self.data = data
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
        respmat = calc_model_respmatTBBO(
            self.tb_model, self.bo_model, self.corr_names, self.data,
            nturns=self.nturns)
        respmat -= self.respmat
        return np.sqrt(np.mean(respmat*respmat))
