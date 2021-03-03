"""."""
import time as _time

import numpy as np

from siriuspy.devices import SOFB, Tune, CurrInfo, RFGen
from siriuspy.clientconfigdb import ConfigDBClient

from ..utils import ThreadedMeasBaseClass as _BaseClass


class RespMatParams:
    """."""

    def __init__(self):
        """."""
        self.nr_points_smooth = 50
        self.corr_nr_iters = 10
        self.delta_kickx = 15  # [urad]
        self.delta_kicky = 15  # [urad]
        self.delta_rf = 80     # [Hz]
        self.respmat_name = ''

    def __str__(self):
        """."""
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        ftmp = '{0:26s} = {1:9.2f}  {2:s}\n'.format
        stg = dtmp('nr_points_smooth', self.nr_points_smooth, '')
        stg += dtmp('corr_nr_iters', self.corr_nr_iters, '')
        stg += ftmp('delta_kickx', self.delta_kickx, '[urad]')
        stg += ftmp('delta_kicky', self.delta_kicky, '[urad]')
        stg += ftmp('delta_rf', self.delta_rf, '[Hz]')
        stg += f"respmat_name = '{self.respmat_name:s}'\n"
        return stg


class MeasureRespMat(_BaseClass):
    """."""

    _DEF_TIMEOUT = 60 * 60  # [s]

    def __init__(self):
        """."""
        super().__init__(params=RespMatParams(), target=self._run_respmat)
        sofb, tune, curr, rfgen = self._create_devices()
        self.devices['sofb'] = sofb
        self.devices['tune'] = tune
        self.devices['curr'] = curr
        self.devices['rfgen'] = rfgen
        self.confdb = ConfigDBClient(config_type='si_orbcorr_respm')

    def _run_respmat(self):
        """."""
        if self._stopevt.is_set():
            return
        sofb = self.devices['sofb']
        init_nr = sofb.nr_points
        init_kickx = sofb.measrespmat_kickch
        init_kicky = sofb.measrespmat_kickcv
        init_rf = sofb.measrespmat_kickrf

        sofb.nr_points = self.params.nr_points_smooth
        sofb.measrespmat_kickch = self.params.delta_kickx
        sofb.measrespmat_kickcv = self.params.delta_kicky
        sofb.measrespmat_kickrf = self.params.delta_rf

        sofb.correct_orbit_manually(self.params.corr_nr_iters)
        self._get_loco_setup(self.params.respmat_name)
        if self._stopevt.is_set():
            return
        respmat = self._measure_respm()
        respmat = respmat.reshape((-1, self.devices['sofb'].data.nr_corrs))
        self.data['respmat'] = respmat
        self.confdb.insert_config(self.params.respmat_name, respmat)

        sofb.nr_points = init_nr
        sofb.measrespmat_kickch = init_kickx
        sofb.measrespmat_kickcv = init_kicky
        sofb.measrespmat_kickrf = init_rf

    def _measure_respm(self):
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

    def _get_loco_setup(self, orbmat_name):
        """."""
        sofb = self.devices['sofb']
        tune = self.devices['tune']
        curr = self.devices['curr']
        rfgen = self.devices['rfgen']

        bpm_var = self._get_bpm_variation(period=10)
        self.data['timestamp'] = _time.time()
        self.data['rf_frequency'] = rfgen.frequency
        self.data['tunex'] = tune.tunex
        self.data['tuney'] = tune.tuney
        self.data['stored_current'] = curr.si.current
        self.data['sofb_nr_points'] = sofb.nr_points
        self.data['bpm_variation'] = bpm_var
        self.data['orbmat_name'] = orbmat_name

    def _get_bpm_variation(self, period=10):
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
