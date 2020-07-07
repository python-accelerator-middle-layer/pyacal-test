"""."""
from threading import Thread as _Thread, Event as _Event
import time as _time
import numpy as _np
from siriuspy.devices import SOFB, APU, Tune

from epics import PV
import pyaccel
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
        self.id_idx = _np.array(self.famdata[self.id_name]['index']).flatten()
        self.bpm_idx = _np.array(self.famdata['BPM']['index']).flatten()
        self.devices['apu'] = APU(self.id_name)
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

# measurement
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

        apu_phase_mon = self.devices['apu'].pv_object('Phase-Mon')
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

# analysis
    def _add_id_correctors(self, corr_len=1e-6):
        corr1 = self.id_idx[0] - 1
        corr2 = self.id_idx[-1] + 1

        self.model[corr1].length += corr_len
        self.model[corr2].length += corr_len
        self.model[corr1 + 1].length -= corr_len
        self.model[corr2 - 1].length -= corr_len

        self.model[corr1].pass_method = 'str_mpole_symplectic4_pass'
        self.model[corr2].pass_method = 'str_mpole_symplectic4_pass'
        return [corr1, corr2]

    def _calc_id_kickmat(self, corr_idx, dkick=1e-8):
        mat = _np.zeros((2*self.bpm_idx.size, 2*len(corr_idx)))

        for num, id_knb in enumerate(corr_idx):
            # apply +delta/2
            MeasIDIntegral.apply_id_deltakick(self.model, id_knb, dkick/2, 'x')
            codp = pyaccel.tracking.find_orbit6(
                self.model, indices=self.bpm_idx)
            # apply -delta/2
            MeasIDIntegral.apply_id_deltakick(self.model, id_knb, -dkick, 'x')
            codn = pyaccel.tracking.find_orbit6(
                self.model, indices=self.bpm_idx)
            # recover model
            MeasIDIntegral.apply_id_deltakick(self.model, id_knb, dkick/2, 'x')
            diffcod = codp - codn
            dcodxy = _np.hstack((diffcod[0, :], diffcod[2, :]))
            mat[:, num] = dcodxy/dkick

        for num, id_knb in enumerate(corr_idx):
            # apply +delta/2
            MeasIDIntegral.apply_id_deltakick(self.model, id_knb, dkick/2, 'y')
            codp = pyaccel.tracking.find_orbit6(
                self.model, indices=self.bpm_idx)
            # apply -delta/2
            MeasIDIntegral.apply_id_deltakick(self.model, id_knb, -dkick, 'y')
            codn = pyaccel.tracking.find_orbit6(
                self.model, indices=self.bpm_idx)
            # recover model
            MeasIDIntegral.apply_id_deltakick(self.model, id_knb, dkick/2, 'y')
            diffcod = codp - codn
            dcodxy = _np.hstack((diffcod[0, :], diffcod[2, :]))
            mat[:, len(corr_idx) + num] = dcodxy/dkick
        return mat

    def fit_kicks(self, diff_orbit, idmat=None, nr_iter=5, nsv=None):
        """."""
        corr_idx = self._add_id_correctors()
        if idmat is None:
            idmat = self._calc_id_kickmat(corr_idx)
        imat = MeasIDIntegral._svd_invert_matrix(idmat, nsv=nsv)
        cod0 = pyaccel.tracking.find_orbit6(self.model, indices=self.bpm_idx)
        cod0xy = _np.hstack((cod0[0, :], cod0[2, :]))
        err = cod0xy - diff_orbit
        print(
            'initial error: {:.2f} um'.format(
                MeasIDIntegral._calc_error(err)*1e6))

        for _ in range(nr_iter):
            dkick = _np.dot(imat, err)
            MeasIDIntegral.apply_id_deltakick(
                self.model, corr_idx[0], dkick[0], 'x')
            MeasIDIntegral.apply_id_deltakick(
                self.model, corr_idx[1], dkick[1], 'x')
            MeasIDIntegral.apply_id_deltakick(
                self.model, corr_idx[0], dkick[2], 'y')
            MeasIDIntegral.apply_id_deltakick(
                self.model, corr_idx[1], dkick[3], 'y')
            cod = pyaccel.tracking.find_orbit6(
                self.model, indices=self.bpm_idx)
            codxy = _np.hstack((cod[0, :], cod[2, :]))
            err = codxy - diff_orbit
        print(
            'final error: {:.2f} um'.format(
                MeasIDIntegral._calc_error(err)*1e6))

# static methods
    @staticmethod
    def _calc_error(err):
        return _np.sqrt(_np.sum(err*err/err.size))

    @staticmethod
    def apply_id_deltakick(model, idx, dkick, plane):
        """."""
        if plane == 'x':
            model[idx].hkick_polynom += dkick
        elif plane == 'y':
            model[idx].vkick_polynom += dkick
        else:
            raise ValueError('Plane must be x or y.')

    @staticmethod
    def _svd_invert_matrix(id_mat, nsv=None):
        umat, smat, vhmat = _np.linalg.svd(id_mat, full_matrices=False)
        ismat = 1/smat
        ismat[_np.isnan(ismat)] = 0
        ismat[_np.isinf(ismat)] = 0
        if nsv is not None:
            ismat[nsv:] = 0
        ismat = _np.diag(ismat)
        invmat = -1 * _np.dot(_np.dot(vhmat.T, ismat), umat.T)
        return invmat
