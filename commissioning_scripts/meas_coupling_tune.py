"""Coupling Measurement from Minimal Tune Separation."""

import time as _time
from threading import Thread as _Thread, Event as _Event
import numpy as _np
import matplotlib.pyplot as _plt

from .base import BaseClass
from ..optimization import SimulAnneal as _SimulAnneal
from siriuspy.devices import PowerSupply, Tune


class CoupParams():
    """."""

    def __init__(self):
        """."""
        self.nr_points = 1
        self.time_wait = 5  # s
        self.neg_percent = 0.0/100
        self.pos_percent = 0.0/100

    def __str__(self):
        """."""
        ftmp = '{0:26s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stg = dtmp('nr_points', self.nr_points, '')
        stg += ftmp('time_wait [s]', self.time_wait, '')
        stg += ftmp('neg_percent', self.neg_percent, '')
        stg += ftmp('pos_percent', self.pos_percent, '')
        return stg


class MeasCoupling(BaseClass):
    """."""

    QUAD_PVNAME = 'SI-FAM:PS-QFB'

    def __init__(self, is_online=True):
        """."""
        super().__init__()
        self.params = CoupParams()
        if is_online:
            self.devices['quad'] = PowerSupply(MeasCoupling.QUAD_PVNAME)
            self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.data = dict()
        self._stopevt = _Event()
        self._thread = _Thread(target=self._do_meas, daemon=True)

    def start(self):
        """."""
        if self.ismeasuring:
            return
        self._stopevt.clear()
        self._thread = _Thread(target=self._do_meas, daemon=True)
        self._thread.start()

    def stop(self):
        """."""
        self._stopevt.set()

    @property
    def ismeasuring(self):
        """."""
        return self._thread.is_alive()

    def _do_meas(self):
        quad = self.devices['quad']
        tunes = self.devices['tune']

        curr0 = quad.current
        curr_vec = curr0 * _np.linspace(
            1-self.params.neg_percent,
            1+self.params.pos_percent,
            self.params.nr_points)
        tunes_vec = _np.zeros((len(curr_vec), 2))

        print('QFB Current:')
        for idx, curr in enumerate(curr_vec):
            quad.current = curr
            _time.sleep(self.params.time_wait)
            tunes_vec[idx, :] = tunes.tunex, tunes.tuney
            print('   {:.6f} A ({:+.3f} %), nux = {:.4f}, nuy = {:.4f}'.format(
                curr, (curr/curr0-1)*100),
                tunes_vec[idx, 0], tunes_vec[idx, 1])
        print('Finished!')
        quad.current = curr0
        self.data['qname'] = quad.devname
        self.data['current'] = curr_vec
        self.data['tunes'] = tunes_vec

    def process_and_plot_data(self, coup_resolution=None, niter=2000):
        """."""
        _np.random.seed(seed=13101971)
        tune1, tune2 = self.data['tunes'].T
        curr = self.data['current']

        if coup_resolution:
            FitTunes.COUPLING_RESOLUTION = coup_resolution
        fittune = FitTunes(param=curr, tune1=tune1, tune2=tune2)
        fittune.niter = niter
        fittune.start(print_flag=False)
        fittune.fitting_plot(oversampling=10, xlabel='QFB Variation [%]')


class FitTunes(_SimulAnneal):
    """Tune Fit.

    tunex = coeff1 * quad_parameter + offset1
    tuney = coeff2 * quad_parameter + offset2

    tune1, tune2 = Eigenvalues([[tunex, coupling/2], [coupling/2, tuney]])

    fit parameters: coeff1, offset1, coeff2, offset2, coupling

    Ex.:
            fittune = FitTunes(
                param=quad_strengths, tune1=meas_tune1, tune2=meas_tune2)
            fittune.fitting_plot()
            fittune.niter = 1000
            fittune.start()
            fittune.fitting_plot()

    NOTE: It maybe necessary to add a quadratic quadrupole strength
          dependency for tunes!
    """

    COUPLING_RESOLUTION = 0.02/100

    def __init__(self, param, tune1, tune2):
        """."""
        self._param = _np.asarray(param)
        self._tune1 = _np.asarray(tune1)
        self._tune2 = _np.asarray(tune2)

        # sort tune1 > tune2 at each point
        sel = self._tune1 <= self._tune2
        if sel.any():
            self._param = _np.array(param)
            self._tune1 = _np.array(tune1)
            self._tune2 = _np.array(tune2)
            self._tune1[sel], self._tune2[sel] = \
                self._tune2[sel], self._tune1[sel]

        # remove nearly degenerate measurement points
        dtunes = _np.abs(self._tune1 - self._tune2)
        sel = _np.where(dtunes < FitTunes.COUPLING_RESOLUTION)[0]
        self._param = _np.delete(self._param, sel)
        self._tune1 = _np.delete(self._tune1, sel)
        self._tune2 = _np.delete(self._tune2, sel)

        super().__init__(use_thread=False)

    # --- fitting parameters ---

    @property
    def param(self):
        """."""
        return self._param

    @property
    def tune1(self):
        """."""
        return self._tune1

    @property
    def tune2(self):
        """."""
        return self._tune2

    @property
    def coeff1(self):
        """."""
        return self.position[0]

    @coeff1.setter
    def coeff1(self, value):
        """."""
        self.position[0] = value

    @property
    def offset1(self):
        """."""
        return self.position[1]

    @offset1.setter
    def offset1(self, value):
        """."""
        self.position[1] = value

    @property
    def coeff2(self):
        """."""
        return self.position[2]

    @coeff2.setter
    def coeff2(self, value):
        """."""
        self.position[2] = value

    @property
    def offset2(self):
        """."""
        return self.position[3]

    @offset2.setter
    def offset2(self, value):
        """."""
        self.position[3] = value

    @property
    def coupling(self):
        """."""
        return self.position[4]

    @coupling.setter
    def coupling(self, value):
        """."""
        self.position[4] = value

    def calc_obj_fun(self):
        """."""
        # print(self.position)
        coeff1, offset1, coeff2, offset2, coupling = self.position
        _, tune1, tune2 = \
            FitTunes.calc_tunes(
                self._param, coeff1, offset1, coeff2, offset2, coupling)
        diff = ((tune1 - self._tune1)**2 + (tune2 - self._tune2)**2)**0.5
        residue = _np.mean(diff)
        # print(self.position, residue)
        return residue

    def initialization(self):
        """."""
        self._calc_init_parms()
        coeff1, offset1, coeff2, offset2, _ = self.position
        self.deltas = _np.array([
            0.01 * 0.002 * (coeff1 - coeff2),
            0.01 * 0.002 * (offset1 - offset2),
            0.01 * 0.002 * (coeff1 - coeff2),
            0.01 * 0.002 * (offset1 - offset2),
            0.5 * 0.1/100])

    @staticmethod
    def calc_tunes(
            param=None,
            coeff1=None, offset1=None,
            coeff2=None, offset2=None, coupling=None):
        """."""
        fx_ = coeff1 * param + offset1
        fy_ = coeff2 * param + offset2
        tune1, tune2 = _np.zeros(param.shape), _np.zeros(param.shape)
        for i in range(len(param)):
            mat = _np.array([[fx_[i], coupling/2], [coupling/2, fy_[i]]])
            eigvalues, _ = _np.linalg.eig(mat)
            tune1[i], tune2[i] = eigvalues
            if tune1[i] <= tune2[i]:
                tune1[i], tune2[i] = tune2[i], tune1[i]
        return param, tune1, tune2

    def fitting_plot(
            self, oversampling=1, title=None, xlabel=None, fig=None):
        """."""
        position = self.position
        param = _np.interp(
            _np.arange(0, len(self._param), 1/oversampling),
            _np.arange(0, len(self._param)),
            self._param)

        # NOTE: do error estimate properly!
        coup_error = self.calc_obj_fun()

        if xlabel is None:
            xlabel = 'Quadrupole Integrated Strength [1/m]'
        if title is None:
            title = 'Transverse Linear Coupling: {:.2f} ± {:.2f}%'.format(
                position[-1] * 100, coup_error * 100)

        prop_cycle = _plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        if fig is None:
            fig = _plt.figure()
        param, fittune1, fittune2 = FitTunes.calc_tunes(param, *position)

        # plot meas data
        _plt.plot(
            self._param, self._tune1, 'o', color=colors[0],
            label='measurement')
        _plt.plot(self._param, self._tune2, 'o', color=colors[0])

        # plot fitting
        _plt.plot(param, fittune1, color=colors[1], label='fitting')
        _plt.plot(param, fittune2, color=colors[1])
        _plt.legend()
        _plt.xlabel(xlabel)
        _plt.ylabel('Transverse Tunes')
        _plt.title(title)
        _plt.grid()
        _plt.show()

    def _calc_init_parms(self):
        """."""
        parm = self._param
        vyb = self._tune1[0], self._tune2[0]
        vye = self._tune1[-1], self._tune2[-1]
        dparm = parm[-1] - parm[0]
        coeff1 = (min(vye) - max(vyb)) / dparm
        offset1 = max(vyb) - coeff1 * parm[0]
        coeff2 = (max(vye) - min(vyb)) / dparm
        offset2 = min(vyb) - coeff2 * parm[0]
        coupling = min(_np.abs(self._tune1 - self._tune2))
        position = [coeff1, offset1, coeff2, offset2, coupling]
        self.position = position
