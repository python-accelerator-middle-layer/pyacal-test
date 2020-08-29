#!/usr/bin/env python-sirius
"""."""

import time
import pickle

import numpy as np

from pymodels import si
from siriuspy.devices import SOFB, Tune, CurrInfo, RFGen
from siriuspy.clientconfigdb import ConfigDBClient

from .base import BaseClass
from .measure_beta import MeasBeta


class RespMatBetaParams:
    """."""

    def __init__(self):
        """."""
        self.nmeas_respmat = 10
        self.nmeas_beta = 10

    def __str__(self):
        """."""
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stg = dtmp('nmeas_beta', self.nmeas_beta, '')
        stg += dtmp('nmeas_respmat', self.nmeas_respmat, '')
        return stg


class MeasureRespMatBeta(BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        sofb, tune, curr, rfgen = self.create_devices()
        self.devices['sofb'] = sofb
        self.devices['tune'] = tune
        self.devices['curr'] = curr
        self.devices['rfgen'] = rfgen
        self.params = RespMatBetaParams()
        self.measbeta = self._get_measbeta_object()
        self.confdb = ConfigDBClient(config_type='si_orbcorr_respm')

    @staticmethod
    def create_devices():
        """."""
        sofb = SOFB(SOFB.DEVICES.SI)
        tune = Tune(Tune.DEVICES.SI)
        curr = CurrInfo(CurrInfo.DEVICES.SI)
        rfgen = RFGen(RFGen.DEVICES.AS)
        return sofb, tune, curr, rfgen

    def get_bpm_variation(self, period=10):
        """."""
        sofb = self.devices['sofb']
        orbx_pv = sofb.pv_object('SlowOrbX-Mon')
        orby_pv = sofb.pv_object('SlowOrbY-Mon')
        orbx_mon, orby_mon = [], []

        def orbx_cb(**kwargs):
            nonlocal orbx_mon
            orbx_mon.append(kwargs['value'])

        def orby_cb(**kwargs):
            nonlocal orby_mon
            orby_mon.append(kwargs['value'])

        orbx_pv.add_callback(orbx_cb)
        orby_pv.add_callback(orby_cb)
        time.sleep(period)
        orbx_pv.clear_callbacks()
        orby_pv.clear_callbacks()

        orbx = np.array(orbx_mon)
        orby = np.array(orby_mon)
        orb_var = np.array([np.std(orbx, axis=0), np.std(orby, axis=0)])
        return orb_var

    def save_loco_setup(self, orbmat_name, fname):
        """."""
        sofb = self.devices['sofb']
        tune = self.devices['tune']
        curr = self.devices['curr']
        rfgen = self.devices['rfgen']

        loco_setup = dict()
        bpm_var = self.get_bpm_variation(period=10)
        loco_setup['timestamp'] = time.time()
        loco_setup['rf_frequency'] = rfgen.frequency
        loco_setup['tunex'] = tune.tunex
        loco_setup['tuney'] = tune.tuney
        loco_setup['stored_current'] = curr.si.current
        loco_setup['sofb_nr_points'] = sofb.nr_points
        loco_setup['bpm_variation'] = bpm_var
        loco_setup['orbmat_name'] = orbmat_name

        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'wb') as fil:
            pickle.dump(loco_setup, fil)

    def correct_orbit(self):
        """."""
        sofb = self.devices['sofb']
        wait_autocorr = 30

        sofb.cmd_turn_on_autocorr()
        time.sleep(wait_autocorr)
        sofb.cmd_turn_off_autocorr()

    def measure_respm(self):
        """."""
        sofb = self.devices['sofb']
        # Start the respm measurement
        time.sleep(1)
        sofb.cmd_measrespmat()

        # Wait measurement to be finished
        time.sleep(1)
        sofb.wait_respm_meas()
        time.sleep(1)
        return sofb.respmat

    def run_respmat(self):
        """."""
        for nms in range(self.params.nmeas_respmat):
            print('Respmat Measurement Number {:d}'.format(nms+1))
            orbmat_name = f'respmat_variation_n{nms+1:02d}'
            setup_name = f'loco_setup_n{nms+1:02d}'
            self.correct_orbit()
            self.save_loco_setup(orbmat_name, setup_name)
            respmat = self.measure_respm()
            self.confdb.insert_config(orbmat_name, respmat)

    def run_beta(self):
        """."""
        for nms in range(self.params.nmeas_beta):
            print('Beta Measurement Number {:d}'.format(nms+1))
            beta_name = f'beta_variation_n{nms+1:02d}'
            self.correct_orbit()
            self.measbeta = self._get_measbeta_object()
            self.measbeta.start()
            self.measbeta.wait()
            self.measbeta.process_data(mode='pos')
            self.measbeta.save_data(beta_name)

    def run_all(self):
        """."""
        self.run_respmat()
        self.run_beta()

    def _get_measbeta_object(self):
        """."""
        model = si.create_accelerator()
        famdata = si.get_family_data(model)
        measbeta = MeasBeta(model, famdata)
        measbeta.params.nr_measures = 1
        measbeta.params.quad_deltakl = 0.01/2  # [1/m]
        measbeta.params.wait_quadrupole = 1  # [s]
        measbeta.params.wait_tune = 1  # [s]
        measbeta.params.timeout_quad_turnon = 10  # [s]
        measbeta.params.recover_tune = True
        measbeta.params.recover_tune_tol = 1e-4
        measbeta.params.recover_tune_maxiter = 5

        measbeta.params.quad_nrcycles = 0
        measbeta.params.time_wait_quad_cycle = 0.3  # [s]
        measbeta.quads2meas = list(measbeta.data['quadnames'])
        return measbeta
