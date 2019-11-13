#!/usr/bin/env python-sirius
"""."""

import time as _time
import numpy as np

from pymodels.middlelayer.devices import DCCT, RF, SOFB
from apsuite.commissioning_scripts.base import BaseClass


class Params:
    """."""

    def __init__(self):
        """."""
        self.phase_ini = -177.5
        self.phase_fin = 177.5
        self.phase_delta = 5
        self.voltage_ini = 50
        self.voltage_fin = 150
        self.voltage_delta = 1
        self.nrpulses = 20
        self.freq_pulses = 2
        self.dcct_nrsamples = 50
        self.dcct_period = 0.05
        self.sofb_timeout = 10
        self.dcct_timeout = 10
        self.rf_timeout = 10
        self.wait_rf = 2

    def __str__(self):
        """."""
        st = '{0:30s}= {1:9.3f}\n'.format('initial phase [°]', self.phase_ini)
        st += '{0:30s}= {1:9.3f}\n'.format('final phase [°]', self.phase_fin)
        st += '{0:30s}= {1:9.3f}\n'.format('delta phase [°]', self.phase_delta)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'initial voltage [mV]', self.voltage_ini)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'final voltage [mV]', self.voltage_fin)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'delta voltage [mV]', self.voltage_delta)
        st += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'pulses freq [Hz]', self.freq_pulses)
        st += '{0:30s}= {1:9d}\n'.format(
            'DCCT number of samples', self.dcct_nrsamples)
        st += '{0:30s}= {1:9.3f}\n'.format('DCCT period', self.dcct_period)
        st += '{0:30s}= {1:9.3f}\n'.format('SOFB timeout', self.sofb_timeout)
        st += '{0:30s}= {1:9.3f}\n'.format('DCCT timeout', self.dcct_timeout)
        st += '{0:30s}= {1:9.3f}\n'.format('RF timeout', self.rf_timeout)
        st += '{0:30s}= {1:9.3f}\n'.format('Wait RF', self.wait_rf)
        return st


class ControlRF(BaseClass):
    """."""

    def __init__(self, is_cw=True):
        """."""
        super().__init__(Params())
        self.devices = {
            'dcct': DCCT(),
            'rf': RF(is_cw=is_cw),
            'sofb': SOFB('BO'),
            }
        self.data = {
            'phase': [],
            'voltage': [],
            'power': [],
            'dcct': [],
            'sum': [],
            'orbx': [],
            'orby': [],
            }

    @property
    def phase_span(self):
        """."""
        ini = self.params.phase_ini
        fin = self.params.phase_fin
        dlt = self.params.phase_delta
        return self._calc_span(ini, fin, dlt)

    @property
    def voltage_span(self):
        """."""
        ini = self.params.voltage_ini
        fin = self.params.voltage_fin
        dlt = self.params.voltage_delta
        return self._calc_span(ini, fin, dlt)

    @staticmethod
    def _calc_span(ini, fin, dlt):
        npts = abs(int((fin - ini)/dlt)) + 1
        return np.linspace(ini, fin, npts)

    def do_phase_scan(self):
        """."""
        self._do_scan(isphase=True)

    def do_voltage_scan(self):
        """."""
        self._do_scan(isphase=False)

    def _do_scan(self, isphase=True):
        nrpul = self.params.nrpulses
        freq = self.params.freq_pulses

        self.devices['sofb'].nr_points = nrpul
        print('Turning DCCT Off')
        self.devices['dcct'].turn_off(self.params.dcct_timeout)
        print('Setting DCCT params')
        self.devices['dcct'].nrsamples = self.params.dcct_nrsamples
        self.devices['dcct'].period = self.params.dcct_period
        _time.sleep(2)
        print('Turning DCCT On')
        self.devices['dcct'].turn_on(self.params.dcct_timeout)

        var_span = self.phase_span if isphase else self.voltage_span
        self.data['phase'] = []
        self.data['voltage'] = []
        self.data['power'] = []
        self.data['dcct'] = []
        self.data['sum'] = []
        self.data['orbx'] = []
        self.data['orby'] = []
        print('Starting Loop')
        for val in var_span:
            self._vary(val, isphase=isphase)
            _time.sleep(self.params.wait_rf)
            dcct_data = np.zeros(nrpul)
            phase_data = np.zeros(nrpul)
            voltage_data = np.zeros(nrpul)
            power_data = np.zeros(nrpul)
            self.devices['sofb'].reset()
            for k in range(nrpul):
                dcct_data[k] = np.mean(self.devices['dcct'].current)
                phase_data[k] = self.devices['rf'].phase
                voltage_data[k] = self.devices['rf'].voltage
                power_data[k] = self.devices['rf'].power
                _time.sleep(1/freq)
            self.devices['sofb'].wait(self.params.sofb_timeout)
            self.data['phase'].append(phase_data)
            self.data['voltage'].append(voltage_data)
            self.data['power'].append(power_data)
            self.data['dcct'].append(dcct_data)
            self.data['sum'].append(self.devices['sofb'].sum)
            self.data['orbx'].append(self.devices['sofb'].trajx)
            self.data['orby'].append(self.devices['sofb'].trajy)
            if isphase:
                print('Phase [°]: {0:8.3f} -> Current [uA]: {1:8.3f}'.format(
                    self.devices['rf'].phase, np.mean(dcct_data)*1e3))
            else:
                print(
                    'Voltage [mV]: {0:8.3f} -> Current [uA]: {1:8.3f}'.format(
                        self.devices['rf'].voltage, np.mean(dcct_data)*1e3))
        print('Finished!')

    def _vary(self, val, isphase=True):
        if isphase:
            self.devices['rf'].set_phase(val, timeout=self.params.rf_timeout)
        else:
            self.devices['rf'].set_voltage(val, timeout=self.params.rf_timeout)
