import time as _time
import pickle as _pickle
import numpy as np

from pymodels.middlelayer.devices import DCCT, RF, SOFB


class Params:

    def __init__(self):
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
        st = '{0:30s}= {1:9.3f}\n'.format('initial phase [°]', self.phase_ini)
        st += '{0:30s}= {1:9.3f}\n'.format('final phase [°]', self.phase_fin)
        st += '{0:30s}= {1:9.3f}\n'.format('delta phase [°]', self.phase_delta)
        st += '{0:30s}= {1:9.3f}\n'.format('initial voltage [mV]', self.voltage_ini)
        st += '{0:30s}= {1:9.3f}\n'.format('final voltage [mV]', self.voltage_fin)
        st += '{0:30s}= {1:9.3f}\n'.format('delta voltage [mV]', self.voltage_delta)
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


class ControlRF:
    def __init__(self):
        self.params = Params()
        self.dcct = DCCT()
        self.rf = RF()
        self.sofb = SOFB('BO')
        self.data_phase = []
        self.data_voltage = []
        self.data_power = []
        self.data_dcct = []
        self.data_sum = []
        self.data_orbx = []
        self.data_orby = []

    @property
    def connected(self):
        conn = self.dcct.connected
        conn &= self.rf.connected
        conn &= self.sofb.connected
        return conn

    @property
    def phase_span(self):
        ini = self.params.phase_ini
        fin = self.params.phase_fin
        dlt = self.params.phase_delta
        return self._calc_span(ini, fin, dlt)

    @property
    def voltage_span(self):
        ini = self.params.voltage_ini
        fin = self.params.voltage_fin
        dlt = self.params.voltage_delta
        return self._calc_span(ini, fin, dlt)

    @staticmethod
    def _calc_span(ini, fin, dlt):
        npts = abs(int((fin - ini)/dlt)) + 1
        return np.linspace(ini, fin, npts)

    def do_phase_scan(self):
        self._do_scan(isphase=True)

    def do_voltage_scan(self):
        self._do_scan(isphase=False)

    def _do_scan(self, isphase=True):
        nrpul = self.params.nrpulses
        freq = self.params.freq_pulses

        self.sofb.nr_points = nrpul
        print('Turning DCCT Off')
        self.dcct.turn_off(self.params.dcct_timeout)
        print('Setting DCCT params')
        self.dcct.nrsamples = self.params.dcct_nrsamples
        self.dcct.period = self.params.dcct_period
        _time.sleep(2)
        print('Turning DCCT On')
        self.dcct.turn_on(self.params.dcct_timeout)

        var_span = self.phase_span if isphase else self.voltage_span
        self.data_phase = []
        self.data_voltage = []
        self.data_power = []
        self.data_dcct = []
        self.data_sum = []
        self.data_orbx = []
        self.data_orby = []
        print('Starting Loop')
        for val in var_span:
            self._vary(val, isphase=isphase)
            _time.sleep(self.params.wait_rf)
            dcct_data = np.zeros(nrpul)
            phase_data = np.zeros(nrpul)
            voltage_data = np.zeros(nrpul)
            power_data = np.zeros(nrpul)
            self.sofb.reset()
            for k in range(nrpul):
                dcct_data[k] = np.mean(self.dcct.current)
                phase_data[k] = self.rf.phase
                voltage_data[k] = self.rf.voltage
                power_data[k] = self.rf.power
                _time.sleep(1/freq)
            self.sofb.wait(self.params.sofb_timeout)
            self.data_phase.append(phase_data)
            self.data_voltage.append(voltage_data)
            self.data_power.append(power_data)
            self.data_dcct.append(dcct_data)
            self.data_sum.append(self.sofb.sum)
            self.data_orbx.append(self.sofb.trajx)
            self.data_orby.append(self.sofb.trajy)
            if isphase:
                print('Phase [°]: {0:8.3f} -> Current [uA]: {1:8.3f}'.format(
                    self.rf.phase, np.mean(dcct_data)*1e3))
            else:
                print('Voltage [mV]: {0:8.3f} -> Current [uA]: {1:8.3f}'.format(
                    self.rf.voltage, np.mean(dcct_data)*1e3))
        print('Finished!')

    def _vary(self, val, isphase=True):
        if isphase:
            self.rf.set_phase(val, timeout=self.params.rf_timeout)
        else:
            self.rf.set_voltage(val, timeout=self.params.rf_timeout)

    def save_data(self, fname):
        data = dict(
            params=self.params,
            data_dcct=self.data_dcct,
            data_phase=self.data_phase,
            data_voltage=self.data_voltage,
            data_power=self.data_power,
            data_orbx=self.data_orbx,
            data_orby=self.data_orby,
            data_sum=self.data_sum,
            phase_span=self.phase_span,
            voltage_span=self.voltage_span,
            )
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'wb') as f:
            _pickle.dump(data, f)

    @staticmethod
    def load_data(fname):
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'rb') as f:
            data = _pickle.load(f)
        return data
