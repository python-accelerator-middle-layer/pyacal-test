import time as _time
import pickle as _pickle
from threading import Thread as _Thread, Event as _Event
import numpy as np

import pyaccel
from siriuspy.epics import PV
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.csdevice.orbitcorr import SOFBFactory

from apsuite.optimization import SimulAnneal


class SOFB:

    def __init__(self):
        self._trajx = PV('BO-Glob:AP-SOFB:MTurnOrbX-Mon')
        self._trajy = PV('BO-Glob:AP-SOFB:MTurnOrbY-Mon')
        self._sum = PV('BO-Glob:AP-SOFB:MTurnSum-Mon')
        self._rst = PV('BO-Glob:AP-SOFB:SmoothReset-Cmd')
        self._npts_sp = PV('BO-Glob:AP-SOFB:SmoothNrPts-SP')
        self._npts_rb = PV('BO-Glob:AP-SOFB:BufferCount-Mon')

    @property
    def connected(self):
        conn = self._trajx.connected
        conn &= self._trajy.connected
        conn &= self._sum.connected
        conn &= self._rst.connected
        conn &= self._npts_sp.connected
        conn &= self._npts_rb.connected
        return conn

    @property
    def trajx(self):
        return self._trajx.get()

    @property
    def trajy(self):
        return self._trajy.get()

    @property
    def sum(self):
        return self._sum.get()

    @property
    def nr_points(self):
        return self._npts_rb.value

    @nr_points.setter
    def nr_points(self, value):
        self._npts_sp.value = int(value)

    def wait(self, timeout=10):
        inter = 0.05
        n = int(timeout/inter)
        for _ in range(n):
            if self._npts_rb.value >= self._npts_sp.value:
                break
            _time.sleep(inter)
        else:
            print('WARN: Timed out waiting orbit.')

    def reset(self):
        self._rst.value = 1


class DCCT:

    def __init__(self):
        self._current = PV('BO-35D:DI-DCCT:RawReadings-Mon')
        self._meas_per_sp = PV('BO-35D:DI-DCCT:FastMeasPeriod-SP')
        self._meas_per_rb = PV('BO-35D:DI-DCCT:FastMeasPeriod-RB')
        self._nr_samples_sp = PV('BO-35D:DI-DCCT:FastSampleCnt-SP')
        self._nr_samples_rb = PV('BO-35D:DI-DCCT:FastSampleCnt-RB')
        self._acq_ctrl_sp = PV('BO-35D:DI-DCCT:MeasTrg-Sel')
        self._acq_ctrl_rb = PV('BO-35D:DI-DCCT:MeasTrg-Sts')
        self._on_state = 1
        self._off_state = 0

    @property
    def connected(self):
        conn = self._current.connected
        conn &= self._meas_per_sp.connected
        conn &= self._meas_per_rb.connected
        conn &= self._nr_samples_sp.connected
        conn &= self._nr_samples_rb.connected
        conn &= self._acq_ctrl_sp.connected
        conn &= self._acq_ctrl_rb.connected
        return conn

    @property
    def nrsamples(self):
        return self._nr_samples_rb.value

    @nrsamples.setter
    def nrsamples(self, value):
        self._nr_samples_sp.value = value

    @property
    def period(self):
        return self._meas_per_rb.value

    @period.setter
    def period(self, value):
        self._meas_per_sp.value = value

    @property
    def acq_ctrl(self):
        return self._acq_ctrl_rb.value

    @acq_ctrl.setter
    def acq_ctrl(self, value):
        self._acq_ctrl_sp.value = self._on_state if value else self._off_state

    @property
    def current(self):
        return self._current.get()

    def wait(self, timeout=10):
        nrp = int(timeout/0.1)
        for _ in range(nrp):
            _time.sleep(0.1)
            if self._isok():
                break
        else:
            print('timed out waiting DCCT.')

    def _isok(self):
        if self._acq_ctrl_sp.value:
            return self.acq_ctrl == self._on_state
        else:
            return self.acq_ctrl != self._on_state

    def turn_on(self, timeout=10):
        self.acq_ctrl = self._on_state
        self.wait(timeout)

    def turn_off(self, timeout=10):
        self.acq_ctrl = self._off_state
        self.wait(timeout)


class RF:

    def __init__(self):
        self._phase_sp = PV('BR-RF-DLLRF-01:PL:REF:S')
        self._phase_rb = PV('BR-RF-DLLRF-01:SL:INP:PHS')
        self._voltage_sp = PV('BR-RF-DLLRF-01:mV:AL:REF:S')
        self._voltage_rb = PV('BR-RF-DLLRF-01:SL:REF:AMP')
        self._power_mon = PV('RA-RaBO01:RF-LLRFCalSys:PwrW1-Mon')

    @property
    def connected(self):
        conn = self._phase_rb.connected
        conn &= self._phase_sp.connected
        conn &= self._voltage_sp.connected
        conn &= self._voltage_rb.connected
        conn &= self._power_mon.connected
        return conn

    @property
    def power(self):
        return self._power_mon.value

    @property
    def phase(self):
        return self._phase_rb.value

    @phase.setter
    def phase(self, value):
        self._phase_sp.value = value

    @property
    def voltage(self):
        return self._voltage_rb.value

    @voltage.setter
    def voltage(self, value):
        self._voltage_sp.value = value

    def set_voltage(self, value, timeout=10):
        self.voltage = value
        self.wait(timeout, isphase=False)

    def set_phase(self, value, timeout=10):
        self.phase = value
        self.wait(timeout)

    def wait(self, timeout=10, isphase=True):
        nrp = int(timeout / 0.1)
        for _ in range(nrp):
            _time.sleep(0.1)
            if isphase:
                if abs(self.phase - self._phase_sp.value) < 0.1:
                    break
            else:
                if abs(self.voltage - self._voltage_sp.value) < 0.1:
                    break
        else:
            print('timed out waiting RF.')


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
        self.sofb = SOFB()
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
    def phase_spam(self):
        ini = self.params.phase_ini
        fin = self.params.phase_fin
        dlt = self.params.phase_delta
        return self._calc_spam(ini, fin, dlt)

    @property
    def voltage_spam(self):
        ini = self.params.voltage_ini
        fin = self.params.voltage_fin
        dlt = self.params.voltage_delta
        return self._calc_spam(ini, fin, dlt)

    @staticmethod
    def _calc_spam(ini, fin, dlt):
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

        var_spam = self.phase_spam if isphase else self.voltage_spam
        self.data_phase = []
        self.data_voltage = []
        self.data_power = []
        self.data_dcct = []
        self.data_sum = []
        self.data_orbx = []
        self.data_orby = []
        print('Starting Loop')
        for val in var_spam:
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
            phase_spam=self.phase_spam,
            voltage_spam=self.voltage_spam,
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