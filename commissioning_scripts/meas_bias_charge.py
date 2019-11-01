#!/usr/bin/env python-sirius
"""."""

import time as _time
import numpy as np

from pymodels.middlelayer.devices import Bias, ICT, TranspEff
from apsuite.commissioning_scripts.base import BaseClass


class Params:
    """."""

    def __init__(self):
        """."""
        self.bias_ini = -41
        self.bias_fin = -55
        self.bias_step = 1
        self.nrpulses = 20
        self.wait_bias = 10

    def __str__(self):
        """."""
        st = '{0:30s}= {1:9.3f}\n'.format('initial bias [V]', self.bias_ini)
        st += '{0:30s}= {1:9.3f}\n'.format('final bias [V]', self.bias_fin)
        st += '{0:30s}= {1:9.3f}\n'.format('step bias [V]', self.bias_step)
        st += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        st += '{0:30s}= {1:9.3f}\n'.format('Wait Bias [s]', self.wait_bias)
        return st


class MeasCharge(BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__(Params())
        self.devices = {
            'bias': Bias(),
            'ict': ICT('ICT-1'),
            'transpeff': TranspEff(),
            }
        self.data = {
            'eff': [],
            'charge': [],
            'bias': [],
            }

    @property
    def bias_span(self):
        """."""
        ini = self.params.bias_ini
        fin = self.params.bias_fin
        dlt = self.params.bias_step
        return self._calc_span(ini, fin, dlt)

    @staticmethod
    def _calc_span(ini, fin, dlt):
        npts = abs(int((fin - ini)/dlt)) + 1
        return np.linspace(ini, fin, npts)

    def do_meas(self):
        """."""
        var_span = self.bias_span
        self.data['eff'] = []
        self.data['charge'] = []
        self.data['bias'] = []

        print('Setting Initial Value...')
        self.devices['bias'].voltage = var_span[0]
        _time.sleep(self.params.wait_bias)
        print('Starting Loop')
        for val in var_span:
            eff = np.zeros(self.params.nrpulses)
            chrg = np.zeros(self.params.nrpulses)
            bias_val = np.zeros(self.params.nrpulses)
            self.devices['bias'].voltage = val
            _time.sleep(self.params.wait_bias)
            for k in range(self.params.nrpulses):
                eff[k] = self.devices['transpeff'].efficiency
                chrg[k] = self.devices['ict'].charge
                bias_val[k] = self.devices['bias'].voltage
                _time.sleep(0.5)
        self.data['eff'].append(eff)
        self.data['charge'].append(chrg)
        self.data['bias'].append(bias_val)
        print('Bias [V]: {0:8.3f} -> Charge [nC]: {1:8.3f}'.format(
            self.devices['bias'].voltage, self.devices['ict'].charge))
    print('Finished!')
