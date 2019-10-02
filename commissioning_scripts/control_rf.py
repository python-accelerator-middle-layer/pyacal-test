import time as _time
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
        conn &= self._meas_per_sp
        conn &= self._meas_per_rb
        conn &= self._nr_samples_sp
        conn &= self._nr_samples_rb
        conn &= self._acq_ctrl_sp
        conn &= self._acq_ctrl_rb
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
        self._phase_sp = PV('BR-RF-DLLRF-01:PL:REF:S', auto_monitor=True)
        self._phase_rb = PV('BR-RF-DLLRF-01:SL:INP:PHS', auto_monitor=True)

    @property
    def connected(self):
        conn = self._phase_rb.connected
        conn &= self._phase_sp.connected
        return conn

    @property
    def phase(self):
        return self._phase_rb.value

    @phase.setter
    def phase(self, value):
        self._phase_sp.value = value

    def set_phase(self, value, timeout=10):
        self.phase = value
        self.wait(timeout)

    def wait(self, timeout=10):
        nrp = int(timeout / 0.1)
        for _ in range(nrp):
            _time.sleep(0.1)
            if abs(self.phase - self._phase_sp.value) < 0.1:
                break
        else:
            print('timed out waiting RF.')


class Params:
    def __init__(self):
        self.phase_ini = -177.5
        self.phase_fin = 177.5
        self.phase_delta = 5
        self.nrpulses = 20
        self.freq_pulses = 2
        self.dcct_nrsamples = 50
        self.dcct_period = 0.05
        self.sofb_timeout = 10
        self.dcct_timeout = 10
        self.rf_timeout = 10

    def __str__(self):
        st = '{0:30s}= {1:9.3f}\n'.format('initial phase [°]', self.phase_ini)
        st += '{0:30s}= {1:9.3f}\n'.format('final phase [°]', self.phase_fin)
        st += '{0:30s}= {1:9.3f}\n'.format('delta phase [°]', self.phase_delta)
        st += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'pulses freq [Hz]', self.freq_pulses)
        st += '{0:30s}= {1:9d}\n'.format(
            'DCCT number of samples', self.dcct_nrsamples)
        st += '{0:30s}= {1:9.3f}\n'.format('DCCT period', self.dcct_period)
        st += '{0:30s}= {1:9.3f}\n'.format('SOFB timeout', self.sofb_timeout)
        st += '{0:30s}= {1:9.3f}\n'.format('DCCT timeout', self.dcct_timeout)
        st += '{0:30s}= {1:9.3f}\n'.format('RF timeout', self.rf_timeout)
        return st


class ControlRF:
    def __init__(self):
        self.params = Params()
        self.dcct = DCCT()
        self.rf = RF()
        self.sofb = SOFB()
        self.data_phase = []
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
        pha_ini = self.params.phase_ini
        pha_fin = self.params.phase_fin
        dpha = self.params.phase_delta
        npts = abs(int((pha_fin - pha_ini)/dpha)) + 1
        return np.linspace(pha_ini, pha_fin, npts)

    def do_phase_scan(self):
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

        phase_spam = self.phase_spam
        self.data_phase = []
        self.data_dcct = []
        self.data_sum = []
        self.data_orbx = []
        self.data_orby = []
        print('Starting Loop')
        for pha in phase_spam:
            self.rf.set_phase(pha, timeout=self.params.rf_timeout)
            dcct_data = np.zeros(nrpul)
            phase_data = np.zeros(nrpul)
            self.sofb.reset()
            for k in range(nrpul):
                dcct_data[k] = np.mean(self.dcct.current)
                phase_data[k] = np.mean(self.rf.phase)
                _time.sleep(1/freq)
            self.sofb.wait(self.params.sofb_timeout)
            self.data_phase.append(phase_data)
            self.data_dcct.append(dcct_data)
            self.data_sum.append(self.sofb.sum)
            self.data_orbx.append(self.sofb.trajx)
            self.data_orby.append(self.sofb.trajy)
            print('Phase [°]: {0:0.4f} -> Current [uA]: {1:0.4f}\n'.format(
                self.rf.phase, np.mean(dcct_data)*1e3))
        print('Finished!')
