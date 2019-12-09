#!/usr/bin/env python-sirius
"""."""

import time as _time
import numpy as np

from pymodels.middlelayer.devices import RF, SOFB, DCCT, Timing
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
        self.sofb_timeout = 10
        self.rf_timeout = 10
        self.tim_timeout = 10
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
        st += '{0:30s}= {1:9.3f}\n'.format('SOFB timeout', self.sofb_timeout)
        st += '{0:30s}= {1:9.3f}\n'.format('RF timeout', self.rf_timeout)
        st += '{0:30s}= {1:9.3f}\n'.format('Wait RF', self.wait_rf)
        st += '{0:30s}= {1:9.3f}\n'.format('Timing timeout', self.tim_timeout)
        return st


class ControlRF(BaseClass):
    """."""

    def __init__(self, acc=None, is_cw=True):
        """."""
        super().__init__(Params())
        if acc is not None:
            self.acc = acc
        else:
            raise Exception('Set BO or SI')
        self.devices = {
            'tim': Timing(),
            'rf': RF(acc=acc, is_cw=is_cw),
            'sofb': SOFB(acc=acc),
            }
        self.data = {
            'phase': [],
            'voltage': [],
            'power': [],
            'sum': [],
            'orbx': [],
            'orby': [],
            }
        if acc == 'BO':
            self.devices['dcct'] = DCCT('BO')
            self.data['dcct'] = []
        elif acc == 'SI':
            self.devices['dcct-1'] = DCCT('SI-1')
            self.devices['dcct-2'] = DCCT('SI-2')
            self.data['dcct-1'] = []
            self.data['dcct-2'] = []



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

        var_span = self.phase_span if isphase else self.voltage_span
        self.data['phase'] = []
        self.data['voltage'] = []
        self.data['power'] = []
        self.data['sum'] = []
        self.data['orbx'] = []
        self.data['orby'] = []
        if self.acc == 'BO':
            self.data['dcct'] = []
        elif self.acc == 'SI':
            self.data['dcct-1'] = []
            self.data['dcct-2'] = []
        print('Starting Loop')
        for val in var_span:
            print('Turning pulses off --> ', end='')
            self.devices['tim'].turn_pulses_off(self.params.tim_timeout)
            print('varying phase --> ', end='')
            self._vary(val, isphase=isphase)
            _time.sleep(self.params.wait_rf)
            phase_data = np.zeros(nrpul)
            voltage_data = np.zeros(nrpul)
            power_data = np.zeros(nrpul)
            if self.acc == 'BO':
                dcct_data = np.zeros(nrpul)
            elif self.acc == 'SI':
                dcct1_data = np.zeros(nrpul)
                dcct2_data = np.zeros(nrpul)
            print('turning pulses on --> ', end='')
            self.devices['tim'].turn_pulses_on(self.params.tim_timeout)
            print('Getting data ', end='')
            self.devices['sofb'].reset()
            for k in range(nrpul):
                print('.', end='')
                phase_data[k] = self.devices['rf'].phase
                voltage_data[k] = self.devices['rf'].voltage
                power_data[k] = self.devices['rf'].power
                if self.acc == 'BO':
                    dcct_data[k] = np.mean(self.devices['dcct'].current)
                elif self.acc == 'SI':
                    dcct1_data[k] = np.mean(self.devices['dcct-1'].current)
                    dcct2_data[k] = np.mean(self.devices['dcct-2'].current)
                _time.sleep(1/freq)
            self.devices['sofb'].wait(self.params.sofb_timeout)
            self.data['phase'].append(phase_data)
            self.data['voltage'].append(voltage_data)
            self.data['power'].append(power_data)
            self.data['sum'].append(self.devices['sofb'].sum)
            self.data['orbx'].append(self.devices['sofb'].trajx)
            self.data['orby'].append(self.devices['sofb'].trajy)
            if self.acc == 'BO':
                self.data['dcct'].append(dcct_data)
            elif self.acc == 'SI':
                self.data['dcct-1'].append(dcct1_data)
                self.data['dcct-2'].append(dcct2_data)
            if isphase:
                print('Phase [°]: {0:8.3f}'.format(self.devices['rf'].phase))
            else:
                print('Voltage [mV]: {0:8.3f}'.format(
                    self.devices['rf'].voltage))
        print('Finished!')

    def _vary(self, val, isphase=True):
        if isphase:
            self.devices['rf'].set_phase(val, timeout=self.params.rf_timeout)
        else:
            self.devices['rf'].set_voltage(val, timeout=self.params.rf_timeout)
