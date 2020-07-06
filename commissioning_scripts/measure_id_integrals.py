"""."""
from threading import Thread as _Thread, Event as _Event
import time as _time
import numpy as _np
from siriuspy.devices import SOFB, APU, Tune

from epics import PV
from pymodels import si
from .base import BaseClass


class IDParams:
    """."""

    def __init__(self, phases=None, meas_type='static'):
        """."""
        self.phases = phases
        self.meas_type = meas_type
        if self.meas_type == 'static':
            self.phase_speed = 0.5
            self.sofb_mode = 'SlowOrb'
            self.sofb_buffer = 20
            self.wait_sofb = 1
            self.wait_to_move = 0
        elif self.meas_type == 'dynamic':
            self.phase_speed = 0.5
            self.sofb_mode = 'Monit1'
            self.sofb_buffer = 1
            self.wait_sofb = 10
            self.wait_to_move = 1

    def __str__(self):
        """."""
        ftmp = '{0:26s} = {1:9.6f}  {2:s}\n'.format
        stmp = '{0:35s}: {1:}  {2:s}\n'.format
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stg = ftmp('phase_speed', self.phase_speed, '')
        stg += stmp('sofb_mode', self.sofb_mode, '')
        stg += dtmp('sofb_buffer', self.sofb_buffer, '')
        return stg


class MeasIDIntegral(BaseClass):
    """."""

    def __init__(self,
                 model,
                 id_name=None, phases=None, meas_type='static'):
        """."""
        super().__init__()
        self.model = model
        self.famdata = si.get_family_data(model)
        self.params = IDParams(phases, meas_type)
        self.id_name = id_name
        self.devices['apu'] = APU(APU.DEVICES.APU22_09SA)
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.devices['study_event'] = PV('AS-RaMO:TI-EVG:StudyExtTrig-Cmd')
        self.devices['current_info'] = PV('SI-Glob:AP-CurrInfo:Current-Mon')
        self.ph_dyn_tstamp = []
        self.ph_dyn_mon = []
        self.data['measure'] = dict()
        self._stopevt = _Event()
        if self.params.meas_type == 'static':
            self._meas_func = self._meas_integral_static
        elif self.params.meas_type == 'dynamic':
            self._meas_func = self._meas_integral_dynamic
        self._thread = _Thread(
            target=self._meas_func, daemon=True)

    def start(self):
        """."""
        if self._thread.is_alive():
            return
        self._stopevt.clear()
        self._thread = _Thread(target=self._meas_func, daemon=True)
        self._thread.start()

    def stop(self):
        """."""
        self._stopevt.set()

    @property
    def ismeasuring(self):
        """."""
        return self._thread.is_alive()

    def get_orbit(self):
        """."""
        # reset SOFB buffer and wait for filling
        sofb = self.devices['sofb']
        sofb.cmd_reset()
        sofb.wait_buffer()
        # get orbit
        orb = _np.vstack((sofb.orbx, sofb.orby))
        return orb

    def get_mt_traj(self):
        """."""
        # reset SOFB buffer and wait for filling
        sofb = self.devices['sofb']
        sofb.cmd_reset()
        sofb.wait_buffer()
        # get trajectory
        traj = _np.vstack((sofb.mt_trajx, sofb.mt_trajy))
        return traj, sofb.mt_time

    def get_tunes(self):
        """."""
        return self.devices['tune'].tunex, self.devices['tune'].tuney

    def get_stored_curr(self):
        """."""
        return self.devices['current_info'].value

    def apu_move(self, phase, phase_speed):
        """."""
        print('- moving to phase {} ... '.format(phase), end='')
        apu = self.devices['apu']
        apu.phase_speed = phase_speed
        apu.phase = phase
        apu.cmd_move()
        apu.wait_move()
        print('ok')

    def cmd_trigger_study(self):
        """."""
        self.devices['study_event'].value = 1

    def _meas_integral_static(self):
        ph_spd = self.params.phase_speed
        # sending to initial phase
        self.apu_move(self.params.phases[0], ph_spd)
        orb0 = self.get_orbit()
        nux, nuy = self.get_tunes()
        curr = self.get_stored_curr()
        orb = []
        phs_rb = []

        for phs in self.params.phases:
            _time.sleep(self.params.wait_to_move)
            self.apu_move(phs, ph_spd)
            _time.sleep(self.params.wait_sofb)
            orb.append(self.get_orbit())
            phs_rb.append(self.devices['apu'].phase)

        _time.sleep(self.params.wait_sofb)
        orbf = self.get_orbit()
        meas = dict()
        meas['initial_orbit'] = orb0
        meas['final_orbit'] = orbf
        meas['tunex'] = nux
        meas['tuney'] = nuy
        meas['stored_current'] = curr
        meas['phases'] = phs_rb
        meas['orbits'] = orb
        self.data['measure'] = meas
        print('finished!')

    def _meas_integral_dynamic(self):
        ph_spd = self.params.phase_speed
        # sending to initial phase
        self.apu_move(self.params.phases[0], ph_spd)

        apu_phase_mon = self.devices['apu'].pv_object['Phase-Mon']
        ph_mon = []
        ph_tstamp = []

        def phase_cb(**kwargs):
            nonlocal ph_mon, ph_tstamp
            ph_mon.append(kwargs['value'])
            ph_tstamp.append(kwargs['timestamp'])

        nux, nuy = self.get_tunes()
        curr = self.get_stored_curr()

        self.cmd_trigger_study()
        apu_phase_mon.add_callback(phase_cb)
        _time.sleep(self.params.wait_sofb)
        traj0, tstamp0 = self.get_mt_traj()

        self.cmd_trigger_study()
        _time.sleep(self.params.wait_to_move)
        self.apu_move(self.params.phases[-1], ph_spd)
        _time.sleep(self.params.wait_sofb)
        traj, tstamp = self.get_mt_traj()

        self.cmd_trigger_study()
        _time.sleep(self.params.wait_sofb)
        trajf, tstampf = self.get_mt_traj()
        apu_phase_mon.clear_callbacks()

        meas = dict()
        meas['initial_traj'] = traj0
        meas['initial_timestamp'] = tstamp0
        meas['final_traj'] = trajf
        meas['final_timestamp'] = tstampf
        meas['tunex'] = nux
        meas['tuney'] = nuy
        meas['stored_current'] = curr
        meas['phases'] = ph_mon
        meas['phases_timestamp'] = ph_tstamp
        meas['traj'] = traj
        meas['timestamp'] = tstamp
        self.data['measure'] = meas
        print('finished!')
