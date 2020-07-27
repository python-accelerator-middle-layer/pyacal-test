"""."""
import time as _time
import numpy as _np
from epics import PV
from apsuite.optimization import SimulAnneal
from siriuspy.devices import Tune, TuneCorr, CurrInfoSI


class InjSIParams:
    """."""

    def __init__(self):
        """."""
        self.nr_iter = 10
        self.nr_pulses = 5
        self.max_delta_tunex = 1e-2
        self.max_delta_tuney = 1e-2
        self.wait_tunecorr = 1  # [s]
        self.pulse_freq = 2  # [Hz]

    def __str__(self):
        """."""
        ftmp = '{0:15s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:15s} = {1:9d}  {2:s}\n'.format
        stg = dtmp('niter', self.niter, '')
        stg += dtmp('nr_pulses', self.nr_pulses, '')
        stg += ftmp('max_delta_tunex', self.max_delta_tunex, '')
        stg += ftmp('max_delta_tuney', self.max_delta_tuney, '')
        stg += ftmp('wait_tunecorr', self.wait_tunecorr, '')
        return stg


class TuneScanInjSI(SimulAnneal):
    """."""

    PV_INJECTION = 'AS-RaMO:TI-EVG:InjectionEvt-Sel'

    def __init__(self, save=False):
        """."""
        super().__init__(save=save)
        self.devices = dict()
        self.params = InjSIParams()
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['tunecorr'] = TuneCorr(TuneCorr.DEVICES.SI)
        self.devices['currinfo'] = CurrInfoSI(CurrInfoSI.DEVICES.SI)
        self.devices['injection'] = PV(TuneScanInjSI.PV_INJECTION)
        self.devices['tunecorr'].cmd_update_reference()
        self.hist_injeff = []

    def _inject(self):
        self.devices[['injection']].value = 1

    def _apply_variation(self):
        tunecorr = self.devices['tunecorr']
        dnux, dnuy = self.position[0], self.position[1]
        tunecorr.delta_tunex = dnux
        tunecorr.delta_tuney = dnuy
        tunecorr.cmd_apply_delta()
        _time.sleep(self.params.wait_tunecorr)

    def calc_obj_fun(self):
        """."""
        self._apply_variation()
        injeff = []
        for nrp in range(self.params.nr_pulses):
            self._inject()
            injeff.append(self.devices['currinfo'].injeff)
            _time.sleep(1/self.params.pulse_freq)
        return - _np.mean(injeff)

    def initialization(self):
        """."""
        self.niter = self.params.nr_iter
        self.position = _np.array([0, 0])
        self.limits_upper = np.array(
            [self.params.max_delta_tunex, self.params.max_delta_tuney])
        self.limits_lower = - self.limits_upper
        self.deltas = self.limits_upper.copy()
