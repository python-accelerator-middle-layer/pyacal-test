import time as _time
from threading import Thread as _Thread, Event as _Event
import numpy as np

from siriuspy.epics import PV
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.search import MASearch
from siriuspy.csdevice.orbitcorr import SOFBFactory


class Quad:

    def __init__(self, name):
        self._sp = PV(name+':KL-SP')
        self._rb = PV(name+':KL-Mon')

    @property
    def kl(self):
        return self._rb.value

    @kl.setter
    def kl(self, value):
        self._sp.value = value

    @property
    def connected(self):
        return self._sp.connected & self._rb.connected


class Kicker:

    def __init__(self):
        self._sp = PV('BO-01D:PM-InjKckr:Kick-SP')
        self._rb = PV('BO-01D:PM-InjKckr:Kick-RB')

    @property
    def kl(self):
        return self._rb.value * 1000  # from mrad to urad

    @kl.setter
    def kl(self, value):
        self._sp.value = value / 1000  # from urad to mrad

    @property
    def connected(self):
        return self._sp.connected & self._rb.connected


class Septum:

    def __init__(self):
        self._sp = PV('TB-04:PM-InjSept:Kick-SP')
        self._rb = PV('TB-04:PM-InjSept:Kick-RB')

    @property
    def kl(self):
        return self._rb.value * 1000  # from mrad to urad

    @kl.setter
    def kl(self, value):
        self._sp.value = value / 1000  # from urad to mrad

    @property
    def connected(self):
        return self._sp.connected & self._rb.connected


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
        self.klystron_delta = -0.05
        self.wait_time = 2
        self.timeout_orb = 10
        self.num_points = 10
        self.klystron_excit_coefs = [0.01, 0.2]


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
        _time.sleep(wait)

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
        self.deltas = {'QF': 0.1, 'QD': 0.1, 'InjSept': 300, 'InjKckr': 300}


class MeasureDispMatTBBO:

    def __init__(self):
        quads = MASearch.get_manames(filters={'sec': 'TB', 'dev': 'Q'})
        dic = {n: Quad(n) for n in quads}
        dic['TB-04:PM-InjSept'] = Septum()
        dic['BO-01D:PM-InjKckr'] = Kicker()
        self._all_corrs = {_PVName(n): v for n, v in dic.items()}
        self.measdisp = MeasureDispTBBO()
        self.params = ParamsDispMat()
        self._matrix = dict()
        self._corrs_to_measure = []
        self._thread = _Thread(target=self._measure_matrix_thread)
        self._stoped = _Event()

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
            self._stoped.clear()
            self._thread.start()

    def stop(self):
        self._stoped.set()

    def _measure_matrix_thread(self):
        corrs = self.corrs_to_measure
        for cor in corrs:
            orb = []
            delta = self._get_delta(cor)
            origkl = self._all_corrs[cor].kl
            for sig in (1, -1):
                self._all_corrs[cor].kl = origkl + sig * delta / 2
                orb.append(sig*self.measdisp.measure_dispersion())
                if self._stoped.is_set():
                    break
            else:
                self._matrix[cor] = np.array(orb).sum(axis=0)/delta
            self._all_corrs[cor].kl = origkl
            if self._stoped.is_set():
                break

    def _get_delta(self, cor):
        for k, v in self.params.deltas.items():
            if cor.dev.startswith(k):
                return v
        print('ERR: delta not found!')
        return 0.0
