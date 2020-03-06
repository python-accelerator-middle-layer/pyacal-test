"""."""

import time as _time
from collections import namedtuple
import numpy as np

from siriuspy.devices import Kicker, Septum, Screen, BPM
from apsuite.commissioning_scripts.base import BaseClass


class Params:
    """."""

    SWEEPORDER = namedtuple(
        'SWeepOrder', ['KickerFirst', 'SeptumFirst', 'Together'])(0, 1, 2)

    def __init__(self):
        """."""
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
        """."""
        return self.SWEEPORDER._fields[self._sweep_order]

    @property
    def sweep_order(self):
        """."""
        return self._sweep_order

    @sweep_order.setter
    def sweep_order(self, value):
        if int(value) in self.SWEEPORDER:
            self._sweep_order = int(value)
        elif value in self.SWEEPORDER._fields:
            self._sweep_order = self.SWEEPORDER._fields.index(value)

    def __str__(self):
        """."""
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


class FindEjeBO(BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__(Params())
        self.devices = {
            'screen': Screen('TS-01:DI-Scrn'),
            'bpm': BPM('TS-01:DI-BPM'),
            'ejekckr': Kicker('BO-48D:PU-EjeKckr'),
            'ejeseptf': Septum('TS-01:PU-EjeSeptF'),
            'ejeseptg': Septum('TS-01:PU-EjeSeptG'),
            }
        self.data = {
            'image': [],
            'ejekckr': [], 'ejeseptf': [], 'ejeseptg': [],
            'bpm_anta': [], 'bpm_antb': [], 'bpm_antc': [], 'bpm_antd': [],
            }

    @property
    def span_ejekckr(self):
        """."""
        ini = self.params.ejekckr_ini
        fin = self.params.ejekckr_fin
        dlt = self.params.ejekckr_step
        return self._calc_span(ini, fin, dlt)

    @property
    def span_ejeseptf(self):
        """."""
        ini = self.params.ejeseptf_ini
        fin = self.params.ejeseptf_fin
        dlt = self.params.ejeseptf_step
        return self._calc_span(ini, fin, dlt)

    @property
    def span_ejeseptg(self):
        """."""
        return self.span_ejeseptf + self.params.ejeseptg_volt_offset

    @staticmethod
    def _calc_span(ini, fin, dlt):
        npts = abs(int((fin - ini)/dlt)) + 1
        return np.linspace(ini, fin, npts)

    def do_scan(self):
        """."""
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

        self.data['image'] = []
        self.data['ejekckr'] = []
        self.data['ejeseptf'] = []
        self.data['ejeseptg'] = []
        self.data['bpm_anta'] = []
        self.data['bpm_antb'] = []
        self.data['bpm_antc'] = []
        self.data['bpm_antd'] = []

        print('Starting Loop')
        print('{0:9s}{1:^7s}, {2:^7s}, {3:^7s} '.format(
            '', 'Kckr', 'SeptF', 'SeptG'))
        for idx2 in range(kckr.shape[1]):
            for idx1 in range(kckr.shape[0]):
                i = idx1
                j = idx2
                if self.params.sweep_order == self.params.SWEEPORDER.Together:
                    j = (idx1+idx2) % kckr.shape[1]
                self.devices['ejekckr'].voltage = kckr[i, j]
                self.devices['ejeseptf'].voltage = septf[i, j]
                self.devices['ejeseptg'].voltage = septg[i, j]
                print(
                    '{0:03d}/{1:03d} :{2:7.2f}, {3:7.2f}, {4:7.2f} '.format(
                        idx1 + idx2*kckr.shape[0],
                        kckr.size, kckr[i, j], septf[i, j], septg[i, j]),
                    end='')
                _time.sleep(self.params.wait_pm)
                print('   Measuring', end='')
                for _ in range(self.params.nrpulses):
                    print('.', end='')
                    self.data['image'].append(self.devices['screen'].image)
                    self.data['bpm_anta'].append(self.devices['bpm'].sp_anta)
                    self.data['bpm_antb'].append(self.devices['bpm'].sp_antb)
                    self.data['bpm_antc'].append(self.devices['bpm'].sp_antc)
                    self.data['bpm_antd'].append(self.devices['bpm'].sp_antd)
                    self.data['ejekckr'].append(
                        self.devices['ejekckr'].voltage)
                    self.data['ejeseptf'].append(
                        self.devices['ejeseptf'].voltage)
                    self.data['ejeseptg'].append(
                        self.devices['ejeseptg'].voltage)
                    _time.sleep(0.5)
                print('done!')
        print('Finished!')
