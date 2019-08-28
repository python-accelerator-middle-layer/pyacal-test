import time as _time
from threading import Thread as _Thread, Event as _Event
import numpy as np

from siriuspy.epics import PV
from siriuspy.namesys import SiriusPVName as _PVName


class Septum:

    def __init__(self):
        self._sp = PV('TB-04:TI-InjSept:Delay-SP')
        self._rb = PV('TB-04:TI-InjSept:Delay-RB')

    @property
    def delay(self):
        return self._rb.value

    @delay.setter
    def delay(self, value):
        self._sp.value = value

    @property
    def connected(self):
        return self._sp.connected & self._rb.connected


class Kicker:

    def __init__(self):
        self._sp = PV('BO-01D:TI-InjKckr:Delay-SP')
        self._rb = PV('BO-01D:TI-InjKckr:Delay-RB')

    @property
    def delay(self):
        return self._rb.value

    @delay.setter
    def delay(self, value):
        self._sp.value = value

    @property
    def connected(self):
        return self._sp.connected & self._rb.connected


class Screen:

    def __init__(self):
        self._centerx = PV('BO-01D:DI-Scrn-1:CenterXDimFei-Mon')
        self._centery = PV('BO-01D:DI-Scrn-1:CenterYDimFei-Mon')
        self._sigmax = PV('BO-01D:DI-Scrn-1:SigmaXDimFei-Mon')
        self._sigmay = PV('BO-01D:DI-Scrn-1:SigmaYDimFei-Mon')

    @property
    def centerx(self):
        return self._centerx.value

    @property
    def centery(self):
        return self._centery.value

    @property
    def sigmax(self):
        return self._sigmax.value

    @property
    def sigmay(self):
        return self._sigmay.value

    @property
    def connected(self):
        conn = self._centerx.connected
        conn &= self._centery.connected
        conn &= self._sigmax.connected
        conn &= self._sigmay.connected
        return conn


class Params:

    def __init__(self):
        self.deltas = {'InjSept': 5, 'InjKckr': 1}  # delay in [us]
        self.num_points = 10
        self.wait_time = 2
        self.num_mean_scrn = 10
        self.wait_mean_scrn = 0.5


class FindMaxPulsedMagnets:

    def __init__(self):
        dic = dict()
        dic['TB-04:TI-InjSept'] = Septum()
        dic['BO-01D:TI-InjKckr'] = Kicker()
        self.screen = Screen()
        self._all_corrs = {_PVName(n): v for n, v in dic.items()}
        self._corrs_to_measure = []
        self._data = []
        self.params = Params()
        self._thread = _Thread(target=self._findmax_thread)
        self._stoped = _Event()

    @property
    def corr_names(self):
        return sorted(self._all_corrs.keys())

    @property
    def data(self):
        return self._data

    def start(self):
        if not self._thread.is_alive():
            self._thread = _Thread(
                target=self._findmax_thread, daemon=True)
            self._stoped.clear()
            self._thread.start()

    def stop(self):
        self._stoped.set()

    def _findmax_thread(self):
        corrs = self.corr_names
        for cor in corrs:
            delta = self.params.deltas[cor.dev]
            origdelay = self._all_corrs[cor].delay
            rangedelay = np.linspace(-delta, delta, self.params.num_points)
            for dly in rangedelay:
                self._all_corrs[cor].delay = origdelay + dly
                _time.sleep(self.params.wait_time)
                self._data.append(self._all_corrs[cor].delay)

                for _ in range(self.params.num_mean_scrn):
                    self._data.append(self.screen.centerx)
                    _time.sleep(self.params.wait_mean_scrn)

                if self._stoped.is_set():
                    break
            self._all_corrs[cor].delay = origdelay
