"""Coupling Measurement from Minimal Tune Separation."""

import time as _time
from threading import Thread as _Thread, Event as _Event
import numpy as _np
from scipy.optimize import least_squares
import matplotlib.pyplot as _plt

from siriuspy.devices import PowerSupply, Tune
from .base import BaseClass


class CouplingParams():
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
    """Coupling measurement and fitting.

    tunex = coeff1 * quad_parameter + offset1
    tuney = coeff2 * quad_parameter + offset2

    tune1, tune2 = Eigenvalues([[tunex, coupling/2], [coupling/2, tuney]])

    fit parameters: coeff1, offset1, coeff2, offset2, coupling

    NOTE: It maybe necessary to add a quadratic quadrupole strength
          dependency for tunes!
    """

    QUAD_PVNAME = 'SI-FAM:PS-QFB'
    COUPLING_RESOLUTION = 0.02/100

    def __init__(self, is_online=True):
        """."""
        super().__init__()
        self.params = CouplingParams()
        if is_online:
            self.devices['quad'] = PowerSupply(MeasCoupling.QUAD_PVNAME)
            self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.analysis = dict()
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

    def process_data(self, coup_resolution=None):
        """."""
        qcurr, tune1, tune2 = self._filter_data(coup_resolution)
        ini_param = self._calc_init_parms(qcurr, tune1, tune2)
        # least squares using Levenberg-Marquardt minimization algorithm
        fit_param = least_squares(
            fun=MeasCoupling._err_func, x0=ini_param,
            args=(qcurr, tune1, tune2), method='lm')

        self.analysis['qcurr'] = qcurr
        self.analysis['tune1'] = tune1
        self.analysis['tune2'] = tune2
        self.analysis['initial_param'] = ini_param
        self.analysis['fitted_param'] = fit_param
        fit_error = self._calc_fitting_error()
        self.analysis['fitting_error'] = fit_error

    def plot_fitting(
            self, oversampling=1, title=None, xlabel=None, fig=None,
            save=False, fname=None):
        """."""
        anl = self.analysis
        fit_vec = anl['fitted_param']['x']
        qcurr, tune1, tune2 = anl['qcurr'], anl['tune1'], anl['tune2']

        qcurr_interp = _np.interp(
            _np.arange(0, len(qcurr), 1/oversampling),
            _np.arange(0, len(qcurr)), qcurr)

        fittune1, fittune2 = MeasCoupling._get_normal_modes(
            params=fit_vec, curr=qcurr_interp)

        if fig is None:
            fig = _plt.figure(figsize=(8, 6))
        if xlabel is None:
            xlabel = 'Quadrupole Integrated Strength [1/m]'
        if title is None:
            title = 'Transverse Linear Coupling: ({:.2f} ± {:.2f}) %'.format(
                fit_vec[-1]*100, anl['fitting_error'][-1] * 100)

        # plot meas data
        _plt.plot(qcurr, tune1, 'o', color='C0', label=r'$\nu_1$')
        _plt.plot(qcurr, tune2, 'o', color='C1', label=r'$\nu_2$')

        # plot fitting
        _plt.plot(qcurr_interp, fittune1, color='tab:gray', label='fitting')
        _plt.plot(qcurr_interp, fittune2, color='tab:gray')
        _plt.legend()
        _plt.xlabel(xlabel)
        _plt.ylabel('Transverse Tunes')
        _plt.title(title)
        _plt.grid(ls='--', alpha=0.5)
        _plt.tight_layout(True)
        if save:
            if fname is None:
                date_string = _time.strftime("%Y-%m-%d-%H:%M")
                fname = 'coupling_fitting_{}.png'.format(date_string)
            _plt.savefig(fname, format='png', dpi=300)
        _plt.show()

    def _filter_data(self, coup_resolution=None):
        qcurr = _np.asarray(self.data['current'])
        tune1, tune2 = self.data['tunes'].T
        tune1 = _np.asarray(tune1)
        tune2 = _np.asarray(tune2)

        # sort tune1 > tune2 at each point
        sel = tune1 <= tune2
        if sel.any():
            tune1[sel], tune2[sel] = tune2[sel], tune1[sel]

        if coup_resolution:
            MeasCoupling.COUPLING_RESOLUTION = coup_resolution

        # remove nearly degenerate measurement points
        dtunes = _np.abs(tune1 - tune2)
        sel = _np.where(dtunes < MeasCoupling.COUPLING_RESOLUTION)[0]
        qcurr = _np.delete(qcurr, sel)
        tune1 = _np.delete(tune1, sel)
        tune2 = _np.delete(tune2, sel)
        return qcurr, tune1, tune2

    def _calc_fitting_error(self):
        # based on fitting error calculation of scipy.optimization.curve_fit
        # do Moore-Penrose inverse discarding zero singular values.
        fit_params = self.analysis['fitted_param']
        _, smat, vhmat = _np.linalg.svd(
            fit_params['jac'], full_matrices=False)
        thre = _np.finfo(float).eps * max(fit_params['jac'].shape)
        thre *= smat[0]
        smat = smat[smat > thre]
        vhmat = vhmat[:smat.size]
        pcov = _np.dot(vhmat.T / smat*smat, vhmat)

        # multiply covariance matrix by residue 2-norm
        ysize = len(fit_params['fun'])
        cost = 2 * fit_params['cost']  # res.cost is half sum of squares!
        popt = fit_params['x']
        if ysize > popt.size:
            # normalized by degrees of freedom
            s_sq = cost / (ysize - popt.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(_np.nan)
            print('# of fitting parameters larger than # of data points!')
        return _np.sqrt(_np.diag(pcov))

    # static methods
    @staticmethod
    def _get_normal_modes(params, curr):
        coeff1, offset1, coeff2, offset2, coupling = params
        fx_ = coeff1 * curr + offset1
        fy_ = coeff2 * curr + offset2
        tune1, tune2 = _np.zeros(curr.shape), _np.zeros(curr.shape)
        for idx, _ in enumerate(curr):
            mat = _np.array([[fx_[idx], coupling/2], [coupling/2, fy_[idx]]])
            eigvalues, _ = _np.linalg.eig(mat)
            tune1[idx], tune2[idx] = eigvalues
            if tune1[idx] <= tune2[idx]:
                tune1[idx], tune2[idx] = tune2[idx], tune1[idx]
        return tune1, tune2

    @staticmethod
    def _err_func(params, qcurr, tune1, tune2):
        tune1f, tune2f = MeasCoupling._get_normal_modes(params, qcurr)
        return _np.sqrt((tune1f - tune1)**2 + (tune2f - tune2)**2)

    @staticmethod
    def _calc_init_parms(curr, tune1, tune2):
        nu_beg = tune1[0], tune2[0]
        nu_end = tune1[-1], tune2[-1]
        dcurr = curr[-1] - curr[0]
        # estimative based on
        # nux = coeff1 * curr + offset1
        # nuy = coeff2 * curr + offset2
        coeff1 = (min(nu_end) - max(nu_beg)) / dcurr
        offset1 = max(nu_beg) - coeff1 * curr[0]
        coeff2 = (max(nu_end) - min(nu_beg)) / dcurr
        offset2 = min(nu_beg) - coeff2 * curr[0]
        coupling = min(_np.abs(tune1 - tune2))
        return [coeff1, offset1, coeff2, offset2, coupling]
