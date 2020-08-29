#!/usr/bin/env python-sirius
"""."""

import time as _time
from threading import Thread as _Thread, Event as _Event

import numpy as np

from siriuspy.devices import SOFB, Tune, CurrInfo, RFGen
from siriuspy.clientconfigdb import ConfigDBClient

from .base import BaseClass


class RespMatParams:
    """."""

    def __init__(self):
        """."""
        self.nr_points_smooth = 50
        self.corr_nr_iters = 10
        self.respmat_name = ''

    def __str__(self):
        """."""
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stg = dtmp('nr_points_smooth', self.nr_points_smooth, '')
        stg += dtmp('corr_nr_iters', self.corr_nr_iters, '')
        stg += f"respmat_name = '{self.respmat_name:s}'\n"
        return stg


class MeasureRespMat(BaseClass):
    """."""
    _DEF_TIMEOUT = 60 * 60  # [s]

    def __init__(self):
        """."""
        super().__init__()
        sofb, tune, curr, rfgen = self._create_devices()
        self.devices['sofb'] = sofb
        self.devices['tune'] = tune
        self.devices['curr'] = curr
        self.devices['rfgen'] = rfgen
        self.params = RespMatParams()
        self.confdb = ConfigDBClient(config_type='si_orbcorr_respm')
        self._stopevt = _Event()
        self._thread = _Thread(target=self.run_respmat, daemon=True)

    def start(self):
        """."""
        if self._thread.is_alive():
            return
        self._stopevt.clear()
        self._thread = _Thread(target=self.run_respmat, daemon=True)
        self._thread.start()

    def stop(self):
        """."""
        self._stopevt.set()

    @property
    def ismeasuring(self):
        """."""
        return self._thread.is_alive()

    def wait(self, timeout=None):
        """."""
        timeout = timeout or MeasureRespMat._DEF_TIMEOUT
        interval = 1  # [s]
        ntrials = int(timeout/interval)
        for _ in range(ntrials):
            if not self.ismeasuring:
                break
            _time.sleep(interval)
        else:
            print('WARN: Timed out waiting respmat measurement.')

    def run_respmat(self):
        """."""
        if self._stopevt.is_set():
            return
        sofb = self.devices['sofb']
        sofb.nr_points = self.params.nr_points_smooth

        sofb.correct_orbit_manually(self.params.corr_nr_iters)
        self.get_loco_setup(self.params.respmat_name)
        if self._stopevt.is_set():
            return
        respmat = self.measure_respm()
        respmat = respmat.reshape((-1, self.devices['sofb'].data.nr_corrs))
        self.data['respmat'] = respmat
        self.confdb.insert_config(self.params.respmat_name, respmat)

    def measure_respm(self):
        """."""
        sofb = self.devices['sofb']
        # Start the respm measurement
        _time.sleep(1)
        sofb.cmd_measrespmat_start()

        # Wait measurement to be finished
        _time.sleep(1)
        sofb.wait_respm_meas()
        _time.sleep(1)
        return sofb.respmat

    def get_loco_setup(self, orbmat_name):
        """."""
        sofb = self.devices['sofb']
        tune = self.devices['tune']
        curr = self.devices['curr']
        rfgen = self.devices['rfgen']

        bpm_var = self.get_bpm_variation(period=10)
        self.data['timestamp'] = _time.time()
        self.data['rf_frequency'] = rfgen.frequency
        self.data['tunex'] = tune.tunex
        self.data['tuney'] = tune.tuney
        self.data['stored_current'] = curr.si.current
        self.data['sofb_nr_points'] = sofb.nr_points
        self.data['bpm_variation'] = bpm_var
        self.data['orbmat_name'] = orbmat_name

    def get_bpm_variation(self, period=10):
        """."""
        sofb = self.devices['sofb']
        orbx_pv = sofb.pv_object('SlowOrbX-Mon')
        orby_pv = sofb.pv_object('SlowOrbY-Mon')
        init_autox = orbx_pv.auto_monitor
        init_autoy = orby_pv.auto_monitor
        orbx_pv.auto_monitor = True
        orby_pv.auto_monitor = True
        orbx_mon, orby_mon = [], []

        def orbx_cb(**kwargs):
            nonlocal orbx_mon
            orbx_mon.append(kwargs['value'])

        def orby_cb(**kwargs):
            nonlocal orby_mon
            orby_mon.append(kwargs['value'])

        orbx_pv.add_callback(orbx_cb)
        orby_pv.add_callback(orby_cb)
        _time.sleep(period)
        orbx_pv.auto_monitor = init_autox
        orby_pv.auto_monitor = init_autoy
        orbx_pv.clear_callbacks()
        orby_pv.clear_callbacks()

        orbx = np.array(orbx_mon)
        orby = np.array(orby_mon)
        orb_var = np.array([np.std(orbx, axis=0), np.std(orby, axis=0)])
        return orb_var

    @staticmethod
    def _create_devices():
        """."""
        sofb = SOFB(SOFB.DEVICES.SI)
        tune = Tune(Tune.DEVICES.SI)
        curr = CurrInfo(CurrInfo.DEVICES.SI)
        rfgen = RFGen(RFGen.DEVICES.AS)
        return sofb, tune, curr, rfgen
