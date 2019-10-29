import time as _time
from collections import namedtuple
import pickle as _pickle
import numpy as np

from pymodels.middlelayer.devices import Kicker, Septum, Screen, BPM


class Params:

    SWEEPORDER = namedtuple(
        'SWeepOrder', ['KickerFirst', 'SeptumFirst', 'Together'])(0, 1, 2)

    def __init__(self):
        self.ejekckr_ini = 220
        self.ejekckr_fin = 250
        self.ejekckr_step = 2
        self.ejeseptf_ini = 49
        self.ejeseptf_fin = 57
        self.ejeseptf_step = 0.5
        self.ejeseptg_volt_offset = 0
        self._sweep_order = 0
        self.nrpulses = 5
        self.wait_pm = 4

    @property
    def sweep_order_string(self):
        return self.SWEEPORDER._fields[self._sweep_order]

    @property
    def sweep_order(self):
        return self._sweep_order

    @sweep_order.setter
    def sweep_order(self, value):
        if int(value) in self.SWEEPORDER:
            self._sweep_order = int(value)
        elif value in self.SWEEPORDER._fields:
            self._sweep_order = self.SWEEPORDER._fields.index(value)

    def __str__(self):
        st = '{0:30s}= {1:.2f}:{2:.2f}:{3:.2f}\n'.format(
            'EjeKckr Sweep [V]',
            self.ejekckr_ini, self.ejekckr_fin, self.ejekckr_step)
        st += '{0:30s}= {1:.2f}:{2:.2f}:{3:.2f}\n'.format(
            'EjeSeptF Sweep [V]',
            self.ejeseptf_ini, self.ejeseptf_fin, self.ejeseptf_step)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'EjeSeptG Voltage offset [V]', self.ejeseptg_volt_offset)
        st += '{0:30s}= {1:s}\n'.format('Sweep Order', self.sweep_order_string)
        st += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'Wait Pulsed Magnets [s]', self.wait_pm)
        return st


class FindEjeBO:
    def __init__(self):
        self.params = Params()
        self.screen = Screen('TS-01:DI-Scrn')
        self.bpm = BPM('TS-01:DI-BPM')
        self.ejekckr = Kicker('BO-48D:PU-EjeKckr')
        self.ejeseptf = Septum('TS-01:PU-EjeSeptF')
        self.ejeseptg = Septum('TS-01:PU-EjeSeptG')
        self.data_image = []
        self.data_ejekckr = []
        self.data_ejeseptf = []
        self.data_ejeseptg = []
        self.data_bpm_anta = []
        self.data_bpm_antb = []
        self.data_bpm_antc = []
        self.data_bpm_antd = []

    @property
    def connected(self):
        conn = self.screen.connected
        conn &= self.bpm.connected
        conn &= self.ejekckr.connected
        conn &= self.ejeseptf.connected
        conn &= self.ejeseptg.connected
        return conn

    @property
    def span_ejekckr(self):
        ini = self.params.ejekckr_ini
        fin = self.params.ejekckr_fin
        dlt = self.params.ejekckr_step
        return self._calc_span(ini, fin, dlt)

    @property
    def span_ejeseptf(self):
        ini = self.params.ejeseptf_ini
        fin = self.params.ejeseptf_fin
        dlt = self.params.ejeseptf_step
        return self._calc_span(ini, fin, dlt)

    @property
    def span_ejeseptg(self):
        return self.span_ejeseptf + self.params.ejeseptg_volt_offset

    @staticmethod
    def _calc_span(ini, fin, dlt):
        npts = abs(int((fin - ini)/dlt)) + 1
        return np.linspace(ini, fin, npts)

    def do_scan(self):
        kckr = self.span_ejekckr
        septf = self.span_ejeseptf
        septg = self.span_ejeseptg
        lkckr = len(kckr)
        kckr = np.repeat(kckr[:, None], len(septf), axis=1)
        septf = np.repeat(septf[:, None], lkckr, axis=1).T
        septg = np.repeat(septg[:, None], lkckr, axis=1).T
        if self.params.sweep_order == self.params.SWEEPORDER.SeptumFirst:
            kckr = kckr.T
            septf = septf.T
            septg = septg.T

        self.data_image = []
        self.data_ejekckr = []
        self.data_ejeseptf = []
        self.data_ejeseptg = []
        self.data_bpm_anta = []
        self.data_bpm_antb = []
        self.data_bpm_antc = []
        self.data_bpm_antd = []

        print('Starting Loop')
        print('{0:9s}{1:^7s}, {2:^7s}, {3:^7s} '.format(
            '', 'Kckr', 'SeptF', 'SeptG'))
        for idx2 in range(kckr.shape[1]):
            for idx1 in range(kckr.shape[0]):
                i = idx1
                j = idx2
                if self.params.sweep_order == self.params.SWEEPORDER.Together:
                    j = (idx1+idx2) % kckr.shape[1]
                self.ejekckr.voltage = kckr[i, j]
                self.ejeseptf.voltage = septf[i, j]
                self.ejeseptg.voltage = septg[i, j]
                print(
                    '{0:03d}/{1:03d} :{2:7.2f}, {3:7.2f}, {4:7.2f} '.format(
                        idx1 + idx2*kckr.shape[0],
                        kckr.size, kckr[i, j], septf[i, j], septg[i, j]),
                    end='')
                _time.sleep(self.params.wait_pm)
                print('   Measuring', end='')
                for _ in range(self.params.nrpulses):
                    print('.', end='')
                    self.data_image.append(self.screen.image)
                    self.data_bpm_anta.append(self.bpm.sp_anta)
                    self.data_bpm_antb.append(self.bpm.sp_antb)
                    self.data_bpm_antc.append(self.bpm.sp_antc)
                    self.data_bpm_antd.append(self.bpm.sp_antd)
                    self.data_ejekckr.append(self.ejekckr.voltage)
                    self.data_ejeseptf.append(self.ejeseptf.voltage)
                    self.data_ejeseptg.append(self.ejeseptg.voltage)
                    _time.sleep(0.5)
                print('done!')
        print('Finished!')

    def save_data(self, fname):
        data = dict(
            params=self.params,
            data_bpm_anta=self.data_bpm_anta,
            data_bpm_antb=self.data_bpm_antb,
            data_bpm_antc=self.data_bpm_antc,
            data_bpm_antd=self.data_bpm_antd,
            data_image=self.data_image,
            data_ejekckr=self.data_ejekckr,
            data_ejeseptf=self.data_ejeseptf,
            data_ejeseptg=self.data_ejeseptg,
            )
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'wb') as f:
            _pickle.dump(data, f)

    @staticmethod
    def load_data(fname):
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'rb') as fil:
            data = _pickle.load(fil)
        return data
