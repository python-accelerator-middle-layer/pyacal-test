import time as _time
from threading import Thread as _Thread, Event as _Event
import numpy as np

from siriuspy.epics import PV
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.csdevice.orbitcorr import SOFBFactory


class CHCV:

    def __init__(self, name):
        self._sp = PV(name+':Kick-SP')
        self._rb = PV(name+':KickRef-Mon')

    @property
    def kick(self):
        return self._rb.value

    @kick.setter
    def kick(self, value):
        self._sp.value = value

    @property
    def connected(self):
        return self._sp.connected & self._rb.connected


class Kicker:

    def __init__(self):
        self._sp = PV('BO-01D:PM-InjKckr:Kick-SP')
        self._rb = PV('BO-01D:PM-InjKckr:Kick-RB')

    @property
    def kick(self):
        return self._rb.value * 1000  # from mrad to urad

    @kick.setter
    def kick(self, value):
        self._sp.value = value / 1000  # from urad to mrad

    @property
    def connected(self):
        return self._sp.connected & self._rb.connected


class Septum:

    def __init__(self):
        self._sp = PV('TB-04:PM-InjSept:Kick-SP')
        self._rb = PV('TB-04:PM-InjSept:Kick-RB')

    @property
    def kick(self):
        return self._rb.value * 1000  # from mrad to urad

    @kick.setter
    def kick(self, value):
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


class Params:

    def __init__(self):
        self.deltas = {'CH': 300, 'CV': 150, 'InjSept': 300, 'InjKckr': 300}
        self.wait_time = 2
        self.timeout_orb = 10
        self.num_points = 10


class MeasureRespMatTBBO:

    def __init__(self):
        self.bo_sofb = SOFB('BO')
        self.tb_sofb = SOFB('TB')
        dic = {n: CHCV(n) for n in self.tb_sofb.data.CH_NAMES}
        dic.update({n: CHCV(n) for n in self.tb_sofb.data.CV_NAMES})
        dic['TB-04:PM-InjSept'] = Septum()
        dic['BO-01D:PM-InjKckr'] = Kicker()
        self._all_corrs = {_PVName(n): v for n, v in dic.items()}
        self.params = Params()
        self._matrix = dict()
        self._corrs_to_measure = []
        self._thread = _Thread(target=self._measure_matrix_thread)
        self._stoped = _Event()

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
        if self._stoped.wait(wait):
            return False
        self.tb_sofb.reset()
        self.bo_sofb.reset()
        if self._stoped.wait(wait):
            return False
        return True

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
            self._stoped.clear()
            self._thread.start()

    def stop(self):
        self._stoped.set()

    def _measure_matrix_thread(self):
        self.nr_points = self.params.num_points
        corrs = self.corrs_to_measure
        for cor in corrs:
            orb = []
            delta = self.params.deltas[cor.dev]
            origkick = self._all_corrs[cor].kick
            for sig in (1, -1):
                self._all_corrs[cor].kick = origkick + sig * delta / 2
                if not self.reset(self.params.wait_time):
                    break
                self.wait(self.params.timeout_orb)
                orb.append(sig*np.hstack([self.trajx, self.trajy]))
            else:
                self._matrix[cor] = np.array(orb).sum(axis=0)/delta
            self._all_corrs[cor].kick = origkick
            if self._stoped.is_set():
                break
