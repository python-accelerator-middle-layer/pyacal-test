#!/usr/bin/env python-sirius
"""."""

import time as _time
import numpy as np

from pymodels.middlelayer.devices import SOFB, Kicker
from apsuite.commissioning_scripts.base import BaseClass


class Params:
    """."""

    def __init__(self):
        """."""
        self.delay_ini = 0
        self.delay_fin = 1.7
        self.delay_step = 0.05
        self.kckr_voltage = 20
        self.nrpulses = 20
        self.sofb_timeout = 30
        self.wait_kckr = 1

    def __str__(self):
        """."""
        st = '{0:30s}= {1:9.3f}\n'.format('initial delay [us]', self.delay_ini)
        st += '{0:30s}= {1:9.3f}\n'.format('final delay [us]', self.delay_fin)
        st += '{0:30s}= {1:9.3f}\n'.format('step delay [us]', self.delay_step)
        st += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'SOFB timeout [s]', self.sofb_timeout)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'Kicker Voltage [V]', self.kckr_voltage)
        st += '{0:30s}= {1:9.3f}\n'.format('Wait Kicker [s]', self.wait_kckr)
        return st


class FindKickerDelay(BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__(Params())
        self.devices = {
            'sofb': SOFB('BO'),
            'kicker': Kicker('BO-48D:PU-EjeKckr'),
            }
        self.data = {
            'sum': [],
            'orbx': [],
            'orby': [],
            'delay': [],
            }

    @property
    def delay_span(self):
        """."""
        ini = self.params.delay_ini
        fin = self.params.delay_fin
        dlt = self.params.delay_step
        return self._calc_span(ini, fin, dlt)

    @staticmethod
    def _calc_span(ini, fin, dlt):
        npts = abs(int((fin - ini)/dlt)) + 1
        return np.linspace(ini, fin, npts)

    def do_scan(self):
        """."""
        self.devices['sofb'].nr_points = self.params.nrpulses

        var_span = self.delay_span
        self.data['sum'] = []
        self.data['orbx'] = []
        self.data['orby'] = []
        self.data['delay'] = []
        self.devices['kicker'].voltage = self.params.kckr_voltage
        print('Starting Loop')
        for val in var_span:
            print('delay -> {0:9.4f} '.format(val), end='')
            self.devices['kicker'].delay = val
            print(' turn off pulse ', end='')
            self.devices['kicker'].turnoff_pulse()
            _time.sleep(self.params.wait_kckr)
            print(' reset sofb ', end='')
            self.devices['sofb'].reset()
            _time.sleep(2)
            self.devices['sofb'].wait(self.params.sofb_timeout)
            print(' measure orbit ', end='')
            data_sum = [self.devices['sofb'].sum, ]
            data_orbx = [self.devices['sofb'].trajx, ]
            data_orby = [self.devices['sofb'].trajy, ]

            print(' turn on pulse ', end='')
            self.devices['kicker'].turnon_pulse()
            _time.sleep(self.params.wait_kckr)
            print(' reset sofb ', end='')
            self.devices['sofb'].reset()
            _time.sleep(2)
            self.devices['sofb'].wait(self.params.sofb_timeout)
            print(' measure orbit ', end='')
            data_sum.append(self.devices['sofb'].sum)
            data_orbx.append(self.devices['sofb'].trajx)
            data_orby.append(self.devices['sofb'].trajy)
            print(' end')

            self.data['sum'].append(data_sum)
            self.data['orbx'].append(data_orbx)
            self.data['orby'].append(data_orby)
            self.data['delay'].append(self.devices['kicker'].delay)
            print('')
        print('Finished!')
