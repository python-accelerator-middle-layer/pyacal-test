"""."""

import time as _time
from threading import Thread, Event

import numpy as np

from siriuspy.devices import RFCav, SOFB, DCCT, EVG
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
        strep = '{0:30s}= {1:9.3f}\n'.format(
            'initial phase [°]', self.phase_ini)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'final phase [°]', self.phase_fin)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'delta phase [°]', self.phase_delta)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'initial voltage [mV]', self.voltage_ini)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'final voltage [mV]', self.voltage_fin)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'delta voltage [mV]', self.voltage_delta)
        strep += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'pulses freq [Hz]', self.freq_pulses)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'SOFB timeout', self.sofb_timeout)
        strep += '{0:30s}= {1:9.3f}\n'.format('RF timeout', self.rf_timeout)
        strep += '{0:30s}= {1:9.3f}\n'.format('Wait RF', self.wait_rf)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'Timing timeout', self.tim_timeout)
        return strep


class ControlRF(BaseClass):
    """."""

    def __init__(self, acc=None, is_cw=True):
        """."""
        super().__init__(Params())
        if acc is not None:
            self.acc = acc
        else:
            raise Exception('Set BO or SI')

        devname_rf, devname_sofb = ControlRF._get_devnames(acc)
        self.devices = {
            'tim': EVG(),
            'rf': RFCav(devname_rf, is_cw=is_cw),
            'sofb': SOFB(devname_sofb),
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
        self._thread = Thread(target=self._do_scan)
        self._stopped = Event()

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
        if not self._thread.is_alive():
            self._thread = Thread(
                target=self._do_scan, kwargs={'isphase': True}, daemon=True)
            self._stopped.clear()
            self._thread.start()

    def do_voltage_scan(self):
        """."""
        if not self._thread.is_alive():
            self._thread = Thread(
                target=self._do_scan, kwargs={'isphase': False}, daemon=True)
            self._stopped.clear()
            self._thread.start()

    def stop(self):
        """."""
        self._stopped.set()

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
            self.devices['tim'].cmd_turn_pulses_off(self.params.tim_timeout)
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
                phase_data[k] = self.devices['rf'].dev_rfll.phase
                voltage_data[k] = self.devices['rf'].dev_rfll.voltage
                power_data[k] = self.devices['rf'].dev_rfpowmon.power
                if self.acc == 'BO':
                    dcct_data[k] = np.mean(self.devices['dcct'].current)
                elif self.acc == 'SI':
                    dcct1_data[k] = np.mean(self.devices['dcct-1'].current)
                    dcct2_data[k] = np.mean(self.devices['dcct-2'].current)
                _time.sleep(1/freq)
                if self._stopped.is_set():
                    break
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
                print('Phase [°]: {0:8.3f}'.format(
                    self.devices['rf'].dev_rfll.phase))
            else:
                print('Voltage [mV]: {0:8.3f}'.format(
                    self.devices['rf'].dev_rfll.voltage))
            if self._stopped.is_set():
                print('Stopped!')
                break
        self.devices['tim'].turn_pulses_off(self.params.tim_timeout)
        print('Finished!')

    def _vary(self, val, isphase=True):
        if isphase:
            self.devices['rf'].cmd_set_phase(
                val, timeout=self.params.rf_timeout)
        else:
            self.devices['rf'].cmd_set_voltage(
                val, timeout=self.params.rf_timeout)

    @staticmethod
    def _get_devnames(acc):
        if acc is None:
            devname_rf, devname_sofb = None, None
        elif acc.upper() == 'SI':
            devname_rf, devname_sofb = RFCav.DEVICES.SI, SOFB.DEVICES.SI
        elif acc.upper() == 'BO':
            devname_rf, devname_sofb = RFCav.DEVICES.BO, SOFB.DEVICES.BO
        else:
            devname_rf, devname_sofb = None, None
        return devname_rf, devname_sofb
