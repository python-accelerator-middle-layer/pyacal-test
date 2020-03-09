"""."""

import time as _time
from threading import Thread as _Thread, Event as _Event
import pickle as _pickle
import numpy as np

from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupplyPU, Screen, BPM


class Params:

    def __init__(self):
        self.delay_span = {'Sept': 3, 'Kckr': 2}  # delay in [us]
        self.num_points = 10
        self.wait_for_pm = 3
        self.num_mean_scrn = 20
        self.wait_mean_scrn = 0.5

    def __str__(self):
        st = '{0:30s}= '.format('Delay Span [us]')
        st += ',  '.join([
            '{0:s}: {1:7.3f}'.format(k, v) for k, v in self.delay_span.items()
            ]) + '\n'
        st += '{0:30s}= {1:9.4f}\n'.format(
            'Scan Num. of points', self.num_points)
        st += '{0:30s}= {1:9.4f}\n'.format(
            'Mean Num. of points', self.num_mean_scrn)
        st += '{0:30s}= {1:9.4f}\n'.format(
            'Wait for PM [s]', self.wait_for_pm)
        st += '{0:30s}= {1:9.4f}\n'.format(
            'Wait Time [s]', self.wait_mean_scrn)
        return st


class FindMaxPulsedMagnets:

    def __init__(self, pulsed_mags, screen, bpm):
        dic = dict()
        for mag in pulsed_mags:
            mag = _PVName(mag)
            dic[mag] = PowerSupplyPU(mag)
        self.screen = Screen(screen)
        self.bpm = BPM(bpm)
        self._all_mags = dic
        self._mags_to_measure = self.magnets
        self._data_delay = {mag: [] for mag in dic}
        self._data_bpmx = {mag: [] for mag in dic}
        self._data_bpmy = {mag: [] for mag in dic}
        self._data_x = {mag: [] for mag in dic}
        self._data_y = {mag: [] for mag in dic}
        self._data_sx = {mag: [] for mag in dic}
        self._data_sy = {mag: [] for mag in dic}
        self.params = Params()
        self._thread = _Thread(target=self._findmax_thread)
        self._stopped = _Event()

    @property
    def connected(self):
        """."""
        conn = all([v.connected for v in self._all_mags.values()])
        conn &= self.screen.connected
        conn &= self.bpm.connected
        return conn

    @property
    def magnets(self):
        """."""
        return sorted(self._all_mags.keys())

    @property
    def magnets_to_measure(self):
        """."""
        return self._mags_to_measure

    @magnets_to_measure.setter
    def magnets_to_measure(self, mags):
        """."""
        self._mags_to_measure = [_PVName(mag) for mag in mags if mag in self._all_mags]

    @property
    def datax(self):
        return self._data_x

    @property
    def datay(self):
        return self._data_y

    @property
    def datasx(self):
        return self._data_sx

    @property
    def datasy(self):
        return self._data_sy

    def start(self):
        if not self._thread.is_alive():
            self._thread = _Thread(
                target=self._findmax_thread, daemon=True)
            self._stopped.clear()
            self._thread.start()

    @property
    def measuring(self):
        return self._thread.is_alive()

    def stop(self):
        self._stopped.set()

    def save_data(self, fname):
        data = {
            'params': self.params,
            'data_bpmx': self._data_bpmx,
            'data_bpmy': self._data_bpmy,
            'datax': self._data_x,
            'datay': self._data_y,
            'datasx': self._data_sx,
            'datasy': self._data_sy,
            'data_delay': self._data_delay}
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'wb') as fil:
            _pickle.dump(data, fil)

    @staticmethod
    def load_data(fname):
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'rb') as fil:
            data = _pickle.load(fil)
        return data

    def _findmax_thread(self):
        corrs = self.magnets_to_measure
        for cor in corrs:
            print(cor)
            mag = self._all_mags[cor]
            delta = self.params.delay_span['Sept' if 'Sept' in cor.dev else 'Kckr']
            origdelay = mag.delay
            rangedelay = np.linspace(-delta/2, delta/2, self.params.num_points)
            self._data_delay[cor] = []
            self._data_bpmx[cor] = []
            self._data_bpmy[cor] = []
            self._data_x[cor] = []
            self._data_y[cor] = []
            self._data_sx[cor] = []
            self._data_sy[cor] = []
            for dly in rangedelay:
                print('{0:9.4f}: Measuring'.format(origdelay+dly), end='')
                mag.delay = origdelay + dly
                _time.sleep(self.params.wait_for_pm)
                for _ in range(self.params.num_mean_scrn):
                    print('.', end='')
                    self._data_delay[cor].append(mag.delay)
                    self._data_bpmx[cor].append(self.bpm.spposx)
                    self._data_bpmy[cor].append(self.bpm.spposy)
                    self._data_x[cor].append(self.screen.centerx)
                    self._data_y[cor].append(self.screen.centery)
                    self._data_sx[cor].append(self.screen.sigmax)
                    self._data_sy[cor].append(self.screen.sigmay)
                    _time.sleep(self.params.wait_mean_scrn)
                    if self._stopped.is_set():
                        break
                print('done!')
                if self._stopped.is_set():
                    break
            mag.delay = origdelay
            if self._stopped.is_set():
                break
        print('Finished!')
