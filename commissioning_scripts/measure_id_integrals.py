"""."""
from threading import Thread as _Thread, Event as _Event
from collections import namedtuple as _namedtuple
from copy import deepcopy as _dcopy
import time as _time
import numpy as _np
from siriuspy.devices import SOFB, APU, Tune
from apsuite.optics_analysis.tune_correction import TuneCorr
from siriuspy.namesys import SiriusPVName as _SiriusPVName

from epics import PV
import pyaccel
from pymodels import si
from .base import BaseClass


class IDParams:
    """."""

    def __init__(self, phases, meas_type):
        """."""
        self.phases = phases
        self.meas_type = meas_type
        if self.meas_type == MeasIDIntegral.MEAS_TYPE.Static:
            self.phase_speed = 0.5
            self.sofb_mode = 'SlowOrb'
            self.sofb_buffer = 20
            self.wait_sofb = 1
            self.wait_to_move = 0
        elif self.meas_type == MeasIDIntegral.MEAS_TYPE.Dynamic:
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
        stg = ftmp('phases', self.phases, '[mm]')
        stg += stmp('meas_type', self.meas_type, '')
        stg += ftmp('phase_speed', self.phase_speed, '[mm/s]')
        stg += stmp('sofb_mode', self.sofb_mode, '')
        stg += dtmp('sofb_buffer', self.sofb_buffer, '')
        stg += ftmp('wait_sofb', self.wait_sofb, '[s]')
        stg += ftmp('wait_to_move', self.wait_to_move, '[s]')
        return stg


class MeasIDIntegral(BaseClass):
    """."""

    DEFAULT_CORR_LEN = 1e-6  # [m]
    MEAS_TYPE = _namedtuple('MeasType', ['Static', 'Dynamic'])(0, 1)

    def __init__(self,
                 model,
                 id_name=None, phases=None, meas_type=None):
        """."""
        super().__init__()
        self.model = model
        self.famdata = si.get_family_data(model)
        if meas_type is None:
            self._meas_type = MeasIDIntegral.MEAS_TYPE.Static
        else:
            self.meas_type = meas_type
        self.params = IDParams(phases, self.meas_type)
        self.id_name = _SiriusPVName(id_name)
        self.id_idx = _np.array(
            self.famdata[self.id_name.dev]['index']).flatten()
        self.bpm_idx = _np.array(self.famdata['BPM']['index']).flatten()
        self.devices['apu'] = APU(self.id_name)
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.devices['study_event'] = PV('AS-RaMO:TI-EVG:StudyExtTrig-Cmd')
        self.devices['current_info'] = PV('SI-Glob:AP-CurrInfo:Current-Mon')
        self.ph_dyn_tstamp = []
        self.ph_dyn_mon = []
        self.analysis = dict()
        self.data['measure'] = dict()
        self._stopevt = _Event()
        if self.params.meas_type == MeasIDIntegral.MEAS_TYPE.Static:
            self._meas_func = self._meas_integral_static
        elif self.params.meas_type == MeasIDIntegral.MEAS_TYPE.Dynamic:
            self._meas_func = self._meas_integral_dynamic
        self._thread = _Thread(
            target=self._meas_func, daemon=True)

    @property
    def meas_type(self):
        """."""
        return self._meas_type

    @meas_type.setter
    def meas_type(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._meas_type = int(value in MeasIDIntegral.MEAS_TYPE._fields[1])
        elif int(value) in MeasIDIntegral.MEAS_TYPE:
            self._meas_type = int(value)

    @property
    def meas_type_str(self):
        """."""
        return MeasIDIntegral.MEAS_TYPE._fields[self._meas_type]

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
        abs_tstamp = _time.time()
        return traj, sofb.mt_time, abs_tstamp

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
        return _time.time()

# measurement
    def _meas_integral_static(self):
        ph_spd = self.params.phase_speed
        # sending to initial phase
        self.apu_move(self.params.phases[0], ph_spd)
        orb0 = self.get_orbit()
        nux, nuy = self.get_tunes()
        curr = self.get_stored_curr()
        orb = []
        phs_mon = []

        for phs in self.params.phases:
            _time.sleep(self.params.wait_to_move)
            self.apu_move(phs, ph_spd)
            _time.sleep(self.params.wait_sofb)
            orb.append(self.get_orbit())
            phs_mon.append(self.devices['apu'].phase)

        _time.sleep(self.params.wait_sofb)
        orbf = self.get_orbit()
        meas = dict()
        meas['initial_orbit'] = orb0
        meas['final_orbit'] = orbf
        meas['tunex'] = nux
        meas['tuney'] = nuy
        meas['stored_current'] = curr
        meas['phases'] = phs_mon
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
            ph_tstamp.append(_time.time())

        nux0, nuy0 = self.get_tunes()
        curr0 = self.get_stored_curr()

        trigger_stamp0 = self.cmd_trigger_study()
        apu_phase_mon.add_callback(phase_cb)
        _time.sleep(self.params.wait_sofb)
        traj0, sofb_tstamp0, abs_tstamp0 = self.get_mt_traj()

        trigger_stampm = self.cmd_trigger_study()
        _time.sleep(self.params.wait_to_move)
        self.apu_move(self.params.phases[-1], ph_spd)
        _time.sleep(self.params.wait_sofb)
        trajm, sofb_tstampm, abs_tstampm = self.get_mt_traj()

        trigger_stampf = self.cmd_trigger_study()
        _time.sleep(self.params.wait_sofb)
        trajf, sofb_tstampf, abs_tstampf = self.get_mt_traj()
        apu_phase_mon.clear_callbacks()

        nuxf, nuyf = self.get_tunes()
        currf = self.get_stored_curr()

        meas = dict()
        meas['initial'] = dict()
        meas['initial']['traj'] = traj0
        meas['initial']['sofb_timestamp'] = sofb_tstamp0
        meas['initial']['abs_timestamp'] = abs_tstamp0
        meas['initial']['trigger_timestamp'] = trigger_stamp0
        meas['initial']['tunex'] = nux0
        meas['initial']['tuney'] = nuy0
        meas['initial']['stored_current'] = curr0

        meas['moving'] = dict()
        meas['moving']['traj'] = trajm
        meas['moving']['sofb_timestamp'] = sofb_tstampm
        meas['moving']['abs_timestamp'] = abs_tstampm
        meas['moving']['trigger_timestamp'] = trigger_stampm
        meas['moving']['phase'] = ph_mon
        meas['moving']['phase_timestamp'] = ph_tstamp

        meas['final'] = dict()
        meas['final']['traj'] = trajf
        meas['final']['sofb_timestamp'] = sofb_tstampf
        meas['final']['abs_timestamp'] = abs_tstampf
        meas['final']['trigger_timestamp'] = trigger_stampf
        meas['final']['tunex'] = nuxf
        meas['final']['tuney'] = nuyf
        meas['final']['stored_current'] = currf
        self.data['measure'] = meas
        print('finished!')

# analysis
    def _add_id_correctors(self, model=None, corr_len=None):
        if corr_len is None:
            corr_len = MeasIDIntegral.DEFAULT_CORR_LEN
        corr1 = self.id_idx[0] - 1
        corr2 = self.id_idx[-1] + 1
        if model is None:
            model = self.model

        model[corr1].length += corr_len
        model[corr2].length += corr_len
        model[corr1 + 1].length -= corr_len
        model[corr2 - 1].length -= corr_len

        model[corr1].pass_method = 'str_mpole_symplectic4_pass'
        model[corr2].pass_method = 'str_mpole_symplectic4_pass'
        return [corr1, corr2]

    def _calc_id_kickmat(self, corr_idx, model=None, dkick=1e-8):
        mat = _np.zeros((2*self.bpm_idx.size, 2*len(corr_idx)))
        if model is None:
            model = self.model

        for num, id_knb in enumerate(corr_idx):
            # apply +delta/2
            MeasIDIntegral.apply_id_deltakick(model, id_knb, dkick/2, 'x')
            codp = pyaccel.tracking.find_orbit6(
                self.model, indices=self.bpm_idx)
            # apply -delta/2
            MeasIDIntegral.apply_id_deltakick(model, id_knb, -dkick, 'x')
            codn = pyaccel.tracking.find_orbit6(
                self.model, indices=self.bpm_idx)
            # recover model
            MeasIDIntegral.apply_id_deltakick(model, id_knb, dkick/2, 'x')
            diffcod = codp - codn
            dcodxy = _np.hstack((diffcod[0, :], diffcod[2, :]))
            mat[:, num] = dcodxy/dkick

        for num, id_knb in enumerate(corr_idx):
            # apply +delta/2
            MeasIDIntegral.apply_id_deltakick(model, id_knb, dkick/2, 'y')
            codp = pyaccel.tracking.find_orbit6(
                self.model, indices=self.bpm_idx)
            # apply -delta/2
            MeasIDIntegral.apply_id_deltakick(model, id_knb, -dkick, 'y')
            codn = pyaccel.tracking.find_orbit6(
                self.model, indices=self.bpm_idx)
            # recover model
            MeasIDIntegral.apply_id_deltakick(model, id_knb, dkick/2, 'y')
            diffcod = codp - codn
            dcodxy = _np.hstack((diffcod[0, :], diffcod[2, :]))
            mat[:, len(corr_idx) + num] = dcodxy/dkick
        return mat

    def fit_kicks(self,
                  diff_orbit, model=None, corr_idx=None,
                  invmat=None, nr_iter=5, nsv=None):
        """."""
        if model is None:
            model = self.model
            corr_idx = self._add_id_correctors(model=model)
        if invmat is None:
            idmat = self._calc_id_kickmat(corr_idx, model=model)
            invmat = MeasIDIntegral._svd_invert_matrix(idmat, nsv=nsv)
        cod0 = pyaccel.tracking.find_orbit6(model, indices=self.bpm_idx)
        cod0xy = _np.hstack((cod0[0, :], cod0[2, :]))
        err = cod0xy - diff_orbit
        print(
            'initial error: {:.2f} um'.format(
                MeasIDIntegral._calc_error(err)*1e6))

        for _ in range(nr_iter):
            dkick = _np.dot(invmat, err)
            MeasIDIntegral.apply_id_deltakick(
                model, corr_idx[0], dkick[0], 'x')
            MeasIDIntegral.apply_id_deltakick(
                model, corr_idx[1], dkick[1], 'x')
            MeasIDIntegral.apply_id_deltakick(
                model, corr_idx[0], dkick[2], 'y')
            MeasIDIntegral.apply_id_deltakick(
                model, corr_idx[1], dkick[3], 'y')
            cod = pyaccel.tracking.find_orbit6(
                model, indices=self.bpm_idx)
            codxy = _np.hstack((cod[0, :], cod[2, :]))
            err = codxy - diff_orbit
        print(
            'final error: {:.2f} um'.format(
                MeasIDIntegral._calc_error(err)*1e6))
        return model

    def calc_field_integrals(self):
        """."""
        meas = self.data['measure']
        phs = meas['phases']
        npts = len(phs)
        orbs = meas['orbits']

        if phs[0] > phs[-1]:
            orbref = orbs[0]
        else:
            orbref = orbs[-1]

        mod = _dcopy(self.model[:])
        nux_goal = 49 + self.data['measure']['tunex']
        nuy_goal = 14 + self.data['measure']['tunex']
        MeasIDIntegral._adjust_tune(mod, nux_goal, nuy_goal)

        corr_idx = self._add_id_correctors(model=mod)
        idmat = self._calc_id_kickmat(corr_idx, model=mod)
        invmat = MeasIDIntegral._svd_invert_matrix(idmat)
        kickx = _np.zeros((npts, 2))
        kicky = _np.zeros((npts, 2))

        for phidx in range(npts):
            diffx = orbs[phidx][0, :] - orbref[0, :]
            diffy = orbs[phidx][1, :] - orbref[1, :]
            diff_orb = _np.hstack((diffx, diffy))
            mod = self.fit_kicks(
                diff_orb, model=mod, corr_idx=corr_idx, invmat=invmat)
            kickx[phidx, 0] = mod[corr_idx[0]].hkick_polynom
            kicky[phidx, 0] = mod[corr_idx[0]].vkick_polynom
            kickx[phidx, 1] = mod[corr_idx[1]].hkick_polynom
            kicky[phidx, 1] = mod[corr_idx[1]].vkick_polynom
            mod[corr_idx[0]].hkick_polynom = 0
            mod[corr_idx[0]].vkick_polynom = 0
            mod[corr_idx[1]].hkick_polynom = 0
            mod[corr_idx[1]].vkick_polynom = 0

        intx = (kicky[:, 0] + kicky[:, 1]) * mod.brho
        inty = (kickx[:, 0] + kickx[:, 1]) * mod.brho
        spos = pyaccel.lattice.find_spos(mod)
        dist = spos[corr_idx[-1]] - spos[corr_idx[0]]
        dist -= MeasIDIntegral.DEFAULT_CORR_LEN
        iintx = kicky[:, 0] * dist * mod.brho
        iinty = kickx[:, 0] * dist * mod.brho

        self.analysis['phase'] = phs
        self.analysis['kickx'] = kickx
        self.analysis['kicky'] = kicky
        self.analysis['Ix'] = intx
        self.analysis['Iy'] = inty
        self.analysis['IIx'] = iintx
        self.analysis['IIy'] = iinty

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

    @staticmethod
    def _adjust_tune(mod, tunex, tuney):
        tunecorr = TuneCorr(
            model=mod, acc='SI',
            method=TuneCorr.METHODS.Proportional,
            grouping=TuneCorr.GROUPING.TwoKnobs)
        tunemat = tunecorr.calc_jacobian_matrix()
        tunecorr.correct_parameters(
            model=mod,
            goal_parameters=_np.array([tunex, tuney]),
            jacobian_matrix=tunemat)
