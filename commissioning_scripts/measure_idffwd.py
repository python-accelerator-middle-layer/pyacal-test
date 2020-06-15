"""Measuremt classes for ID feedforward tables."""

import time as _time
import numpy as _np

from siriuspy.devices import SOFB as _SOFB
from siriuspy.devices import APU as _APU
from siriuspy.devices import IDCorrectors as _IDCorrectors
from siriuspy.devices import APUFeedForward as _APUFeedForward

from .base import BaseClass as _BaseClass


class APUFFWDParams:
    """."""

    def __init__(self):
        """."""
        self.phase_parking = 11.0  # [mm]
        self.phases = _np.linspace(0, 11, 23)
        self.phase_speed = 1.5  # [mm/s]
        self.verbose = 0
        self.wait_corr = 0.5  # [s]
        self.wait_sofb = 2.0  # [s]
        self.sofb_buffersize = 10
        self.sofb_overwait = 100  # [%]
        self.corr_delta = 0.1  # [A]

    def __str__(self):
        """."""
        rst = ''
        rst += 'phase_parking [mm]: {}'.format(self.phase_parking)
        rst += 'phases [mm]: {}'.format(self.phases)
        rst += 'verbose: {}'.format(self.verbose)
        rst += 'wait_corr [s]: {}'.format(self.wait_corr)
        rst += 'wait_sofb [s]: {}'.format(self.wait_sofb)
        rst += 'sofb_buffersize: {}'.format(self.sofb_buffersize)
        rst += 'sofb_overwait [%]: {}'.format(self.sofb_overwait)
        rst += 'corr_delta [A]: {}'.format(self.corr_delta)
        return rst


class MeasAPUFFWD(_BaseClass):
    """."""

    # TODO: Enhance class to allow for offline FFWD construction from
    # dynamic measurements (BPM data taken as APU phase is varying)

    DEVICES = _APUFeedForward.DEVICES

    _SOFB_SLOWORB = 1  # SOFB SlowOrb Mode
    _MOVE_SLEEP = 0.2  # [s]
    _SOFB_FREQ = 10  # [Hz]

    def __init__(self, idname):
        """."""
        super().__init__()
        self.params = APUFFWDParams()
        self._idname = idname
        self.devices['sofb'], self.devices['apu'], self.devices['corr'] = \
            self._create_devices()
        self._nr_corrs = len(self.devices['corr'].orbitcorr_psnames)
        self.data['ffwd'] = self._init_ffwd_table()

    def __str__(self):
        """Print FFWD table in cs-constants format."""
        rst = ''
        rst += '# HEADER\n'
        rst += '# ======\n'
        rst += '# label             {}\n'.format(self._idname)
        rst += '# harmonics         {}\n'.format(
            ' '.join(str(i) for i in range(self._nr_corrs // 2)))
        rst += '# main_harmonic     0 normal\n'
        rst += '# rescaling_factor  1.0\n'
        rst += '# units             mm  {}\n'.format(
            '  '.join('A A' for i in range(self._nr_corrs // 2)))
        rst += '\n'
        rst += '# CORRECTOR FEEDFORWARD TABLE\n'
        rst += '# ===========================\n'
        line = '       '.join(
            ['CH-{}[A]       CV-{}[A]'.format(2*i+1, 2*i+2) for i
                in range(self._nr_corrs // 2)])
        rst += '# Phase[mm] ' + line + '\n'
        for i, values in enumerate(self.data['ffwd']):
            # values = ffwd[i, :]
            line = ' '.join(['{:+.6e}'.format(value) for value in values])
            rst += '{:+08.2f}    {}\n'.format(self.params.phases[i], line)
        rst += '\n'
        rst += '# COMMENTS\n'
        rst += '# ========\n'
        rst += '#\n'
        rst += ('# 1. This table contains correctors feedforward'
                ' current values that depend on the APU phase\n')
        rst += ('# 2. This file was generated automatically '
                'with MeasAPUFFWD class\n')

        return rst

    def measure_at_phase(self, phase):
        """."""
        # move APU to parked phase
        self._print('- move APU to parking phase...')
        self._static_move(self.params.phase_parking)

        # correct orbit with SOFB
        self._static_correct_orbit()

        # get initial trajectory
        self._print('- measure init traj...')
        traj0 = self._static_get_orbit()

        # move APU to measurement phase
        self._print('- move APU to measurement phase...')
        self._static_move(phase)

        # get trajectory at phase
        self._print('- measure traj at phase...')
        traj1 = self._static_get_orbit()

        # measure response matrix
        curr = self.devices['corr'].orbitcorr_current_sp
        respm = _np.zeros((len(traj0), len(curr)))
        for corr_idx, val0 in enumerate(curr):
            # set negative
            curr[corr_idx] = val0 - self.params.corr_delta/2
            self.devices['corr'].orbitcorr_current = curr
            _time.sleep(self.params.wait_corr)
            trajn = self._static_get_orbit()

            # set positive
            curr[corr_idx] = val0 + self.params.corr_delta/2
            self.devices['corr'].orbitcorr_current = curr
            _time.sleep(self.params.wait_corr)
            trajp = self._static_get_orbit()

            # return current to init and register matrix column
            curr[corr_idx] = val0
            self.devices['corr'].orbitcorr_current = curr
            respm[:, corr_idx] = (trajp - trajn) / self.params.corr_delta
        _time.sleep(self.params.wait_corr)

        return {'respm': respm, 'traj0': traj0, 'traj1': traj1}

    def measure(self):
        """."""
        self.data['meas'] = dict()
        meas = self.data['meas']
        # ffwd = self.data['ffwd']
        self._print('Measurements begin...')

        for phase in self.params.phases:
            self._print(
                'Measuring FFWD table for phase {} mm...'.format(phase))
            meas_datum = self.measure_at_phase(phase)
            meas[phase] = meas_datum
            # ffwd[i, :] += currs_delta
            self._print('')
        self._static_move(self.params.phase_parking)
        self._print('Measurements end.')

    def calculate_at_phase(self, respm, traj0, traj1):
        """Calculate correctors current from orbit data."""
        dtraj = traj1 - traj0
        umat, smat, vhmat = _np.linalg.svd(respm, full_matrices=False)
        inv_s = 1/smat
        inv_respm = _np.dot(_np.dot(vhmat.T, inv_s), umat.T)
        curr_deltas = - _np.dot(inv_respm, dtraj)
        return curr_deltas, smat, umat, vhmat

    def calculate(self):
        """Calculate corrector currents from measurements."""
        if 'meas' not in self.data:
            return

        phases = sorted(self.data['meas'].keys())
        nr_corrs = len(self.devices['corr'].orbitcorr_psnames)

        # init ffwd array
        ffwd = _np.zeros((len(phases), nr_corrs))
        self.data['ffwd'] = ffwd

        # loop through different phases
        for i, phase in enumerate(phases):
            mdatum = self.data['meas'][phase]
            curr_deltas, *_ = \
                self.calculate_at_phase(**mdatum)
            ffwd[i, :] = curr_deltas

    # --- private methods ---

    def _create_devices(self):
        """."""
        sofb = _SOFB(_SOFB.DEVICES.SI)
        apu = _APU(self._idname)
        correctors = _IDCorrectors(self._idname)
        sofb.wait_for_connection()
        apu.wait_for_connection()
        correctors.wait_for_connection()
        return sofb, apu, correctors

    def _static_correct_orbit(self):
        """."""
        self._print('correct initial orbit with SOFB...')

        # turn SOFB correction off
        self._print('initialize SOFB...')
        self.devices['sofb'].opmode = MeasAPUFFWD._SOFB_SLOWORB

        # correct orbit
        # TODO: to be implemented!

        # prepare soft for orbit distortion aquisitions.
        self.devices['sofb'].nr_points = self.params.sofb_buffersize
        self.devices['sofb'].cmd_turn_off_autocorr()
        _time.sleep(self.params.wait_sofb)

    def _init_ffwd_table(self):
        """."""
        if self.params.verbose:
            print('initialize APU ffwd table...')
        nr_corrs = len(self.devices['corr'].orbitcorr_psnames)
        ffwd = _np.zeros((len(self.params.phases), nr_corrs))
        return ffwd

    def _static_get_orbit(self):
        """."""
        sofb = self.devices['sofb']

        # reset SOFB buffer and wait for filling
        sofb.cmd_reset()
        sofb.wait_buffer()

        # get trajectory
        traj = _np.vstack((sofb.orbx, sofb.orby))

        return traj

    def _static_move(self, phase):
        """."""
        self._print('- moving to phase {} ... '.format(phase), end='')
        self.devices['apu'].phase_speed = self.params.phase_speed
        self.devices['apu'].phase = phase
        self.devices['apu'].cmd_move()
        self.devices['apu'].wait_move()
        self._print('ok')

    def _print(self, message, end=None):
        if self.params.verbose:
            print(message, end=end)
