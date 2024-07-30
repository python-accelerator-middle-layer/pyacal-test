"""."""

import time as _time
from math import floor as _floor, log10 as _log10

import matplotlib.gridspec as _mpl_gs
import matplotlib.pyplot as _plt
import numpy as _np

from .. import _get_facility, _get_simulator

from ..devices import SOFB, Tune
from .base import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _BaseClass


class DispChromParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.meas_nrsteps = 5
        self.max_delta_freq = +200  # [Hz]
        self.min_delta_freq = -200  # [Hz]
        self.rf_timeout = 10        # [s]
        self.rffreq_tol = 1         # [%]
        self.wait_tune = 5          # [s]
        self.sofb_nrpoints = 10

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stg = ftmp('max_delta_freq [Hz]', self.max_delta_freq, '')
        stg += ftmp('min_delta_freq [Hz]', self.min_delta_freq, '')
        stg += dtmp('meas_nrsteps', self.meas_nrsteps, '')
        stg += ftmp('wait_tune [s]', self.wait_tune, '')
        stg += dtmp('sofb_nrpoints', self.sofb_nrpoints, '')
        return stg


class DispChrom(_BaseClass):
    """."""

    def __init__(self, accelerator, isonline=True, mom_compact=None):
        """."""
        super().__init__(
            params=DispChromParams(), target=self._do_meas, isonline=isonline
        )

        self.mom_compact = mom_compact

        self.accelerator = accelerator or _get_facility().default_accelerator
        if self.isonline:
            self.devices['sofb'] = SOFB(self.accelerator)
            self.devices['rf'] = self.devices["sofb"].rfgen
            self.devices['tune'] = Tune(accelerator=self.accelerator)

    def __str__(self):
        """."""
        stn = 'Params\n'
        stp = self.params.__str__()
        stp = '    ' + stp.replace('\n', '\n    ')
        stn += stp + '\n'
        stn += 'Connected?  ' + str(self.connected) + '\n\n'
        return stn

    def _do_meas(self):
        sofb = self.devices['sofb']
        rfgen = self.devices['rf']
        tune = self.devices['tune']
        nr_bpms = sofb.nr_bpms

        meas_nrsteps = self.params.meas_nrsteps
        sofb.orb_nrpoints = self.params.sofb_nrpoints
        freq0 = rfgen.frequency
        tunex0, tuney0 = tune.tunex, tune.tuney
        concat_orb = sofb.get_orbit()
        orbx0, orby0 = concat_orb[:nr_bpms], concat_orb[nr_bpms:]

        min_df = self.params.min_delta_freq
        max_df = self.params.max_delta_freq
        span = freq0 + _np.linspace(min_df, max_df, meas_nrsteps)

        freq = []
        tunex, tuney = [], []
        orbx, orby = [], []
        for frq in span:
            if self._stopevt.is_set():
                print('   exiting...')
                break
            rfgen.set_frequency(
                frq, tol=self.params.rffreq_tol,
                timeout=self.params.rf_timeout
            )
            t0 = _time.time()
            concat_orb = sofb.get_orbit()
            et = _time.time() - t0
            wait_tune = self.params.wait_tune - et
            _time.sleep(max(0, wait_tune))

            freq.append(rfgen.frequency)
            orbx.append(concat_orb[:nr_bpms])
            orby.append(concat_orb[nr_bpms:])
            tunex.append(tune.tunex)
            tuney.append(tune.tuney)

            print('delta frequency: {} Hz'.format((rfgen.frequency-freq0)))
            dtunex = tunex[-1] - tunex0
            dtuney = tuney[-1] - tuney0
            print(f'(Spec. Analy.) dtune x: {dtunex:} y: {dtuney:}')
            print('')

        print('Restoring RF frequency...')
        rfgen.set_frequency(
                freq0, tol=self.params.rffreq_tol,
                timeout=self.params.rf_timeout
        )
        self.data['freq0'] = freq0
        self.data['tunex0'] = tunex0
        self.data['tuney0'] = tuney0
        self.data['orbx0'] = _np.array(orbx0)
        self.data['orby0'] = _np.array(orby0)
        self.data['freq'] = _np.array(freq)
        self.data['tunex'] = _np.array(tunex)
        self.data['tuney'] = _np.array(tuney)
        self.data['orbx'] = _np.array(orbx)
        self.data['orby'] = _np.array(orby)

        print('Finished!')

    def process_data(self, fitorder=1, discardpoints=None):
        """."""
        data = self.data

        usepts = set(range(data['tunex'].shape[0]))
        if discardpoints is not None:
            usepts = set(usepts) - set(discardpoints)
        usepts = sorted(usepts)

        freq0 = data['freq0']
        den = -(data['freq'] - freq0)/freq0/self.mom_compact
        den = den[usepts]
        suffix = ''
        tunex = data['tunex' + suffix][usepts]
        tuney = data['tuney' + suffix][usepts]
        orbx = data['orbx'][usepts, :]
        orby = data['orby'][usepts, :]

        if tunex.size > fitorder + 1:
            chromx, chromxcov = _np.polyfit(den, tunex, deg=fitorder, cov=True)
            chromy, chromycov = _np.polyfit(den, tuney, deg=fitorder, cov=True)
            dispx, dispxcov = _np.polyfit(den, orbx, deg=fitorder, cov=True)
            dispy, dispycov = _np.polyfit(den, orby, deg=fitorder, cov=True)
        else:
            chromx = _np.polyfit(den, tunex, deg=fitorder, cov=False)
            chromy = _np.polyfit(den, tuney, deg=fitorder, cov=False)
            dispx = _np.polyfit(den, orbx, deg=fitorder, cov=False)
            dispy = _np.polyfit(den, orby, deg=fitorder, cov=False)
            chromxcov = chromycov = _np.zeros(
                (fitorder+1, fitorder+1), dtype=float)
            dispxcov = dispycov = _np.zeros(
                (fitorder+1, fitorder+1, orbx.shape[1]), dtype=float)

        um2m = 1e-6
        self.analysis['delta'] = den
        self.analysis['orbx'] = orbx
        self.analysis['orby'] = orby
        self.analysis['dispx'] = dispx * um2m
        self.analysis['dispy'] = dispy * um2m
        self.analysis['dispx_err'] = _np.sqrt(_np.diagonal(dispxcov)) * um2m
        self.analysis['dispy_err'] = _np.sqrt(_np.diagonal(dispycov)) * um2m

        self.analysis['tunex'] = tunex
        self.analysis['tuney'] = tuney
        self.analysis['chromx'] = chromx
        self.analysis['chromy'] = chromy
        self.analysis['chromx_err'] = _np.sqrt(_np.diagonal(chromxcov))
        self.analysis['chromy_err'] = _np.sqrt(_np.diagonal(chromycov))

    def make_figure_chrom(self, analysis=None, title='', fname=''):
        """."""
        fig = _plt.figure(figsize=(10, 5))
        grid = _mpl_gs.GridSpec(1, 1)
        grid.update(
            left=0.12, right=0.95, bottom=0.15, top=0.9,
            hspace=0.5, wspace=0.35)

        if title:
            fig.suptitle(title)

        if analysis is None:
            analysis = self.analysis

        den = self.analysis['delta']
        tunex = self.analysis['tunex']
        tuney = self.analysis['tuney']
        chromx = self.analysis['chromx']
        chromx_err = self.analysis['chromx_err']
        chromy = self.analysis['chromy']
        chromy_err = self.analysis['chromy_err']
        dtunex = tunex - chromx[-1]
        dtuney = tuney - chromy[-1]
        dtunex_fit = _np.polyval(chromx, den) - chromx[-1]
        dtuney_fit = _np.polyval(chromy, den) - chromy[-1]

        axx = _plt.subplot(grid[0, 0])
        axx.plot(den*100, dtunex*1000, '.b', label='horizontal')
        axx.plot(den*100, dtunex_fit*1000, '-b')
        axx.plot(den*100, dtuney*1000, '.r', label='vertical')
        axx.plot(den*100, dtuney_fit*1000, '-r')
        axx.set_xlabel(r'$\delta$ [%]')
        axx.set_ylabel(r'$\Delta \nu \times 1000$')

        chromx = _np.flip(chromx)
        chromx_err = _np.flip(chromx_err)
        chromy = _np.flip(chromy)
        chromy_err = _np.flip(chromy_err)

        stx = DispChrom.polynomial_to_latex(chromx, chromx_err)
        sty = DispChrom.polynomial_to_latex(chromy, chromy_err)

        stg = r'$\Delta\nu_x = $' + stx + '\n'
        stg += r'$\Delta\nu_y = $' + sty
        axx.text(
            0.4, 0.05, stg, horizontalalignment='left',
            verticalalignment='bottom', transform=axx.transAxes,
            bbox=dict(edgecolor='k', facecolor='w', alpha=1.0))
        axx.legend()
        axx.grid(True)

        if fname:
            fig.savefig(fname+'.svg')
            _plt.close()
        else:
            fig.show()

    def make_figure_disp(self, analysis=None, disporder=1, title='', fname=''):
        """."""
        fig = _plt.figure(figsize=(10, 5))
        grid = _mpl_gs.GridSpec(1, 1)
        grid.update(
            left=0.12, right=0.95, bottom=0.15, top=0.9,
            hspace=0.5, wspace=0.35)

        if title:
            fig.suptitle(title)

        if analysis is None:
            analysis = self.analysis

        simulator = _get_simulator()
        fac = _get_facility()
        bpmnames = fac.find_aliases_from_accelerator(self.accelerator)
        bpmnames = fac.find_aliases_from_cs_devtype(
            {fac.CSDevTypes.BPM, fac.CSDevTypes.SOFB}, bpmnames)
        sposbpm = []
        for bpmn in bpmnames:
            bpmidx = fac.get_indices_from_alias(bpmn)
            sposbpm.append(simulator.get_positions(bpmidx, self.accelerator))
        sposbpm = _np.array(sposbpm).ravel()

        fitorder_anlys = analysis['dispx'].shape[0] - 1
        if disporder > fitorder_anlys:
            raise ValueError('Cannot plot a fit order higher than analysis.')
        fitidx = fitorder_anlys - disporder
        dispx = analysis['dispx'][fitidx, :]
        dispy = analysis['dispy'][fitidx, :]
        dispx_err = analysis['dispx_err'][:, fitidx]
        dispy_err = analysis['dispy_err'][:, fitidx]

        m2cm = 100
        axx = _plt.subplot(grid[0, 0])
        axx.errorbar(
            sposbpm, dispx*m2cm, dispx_err*m2cm, None, '.-b',
            label='horizontal')
        axx.errorbar(
            sposbpm, dispy*m2cm, dispy_err*m2cm, None, '.-r', label='vertical')

        axx.set_xlabel('s [m]')
        ylabel = r'$\eta_{:d}$ [cm]'.format(disporder)
        axx.set_ylabel(ylabel)
        axx.legend()
        axx.grid(True)

        if fname:
            fig.savefig(fname+'.svg')
            _plt.close()
        else:
            fig.show()

    # Adapted from:
    # https://perso.crans.org/besson/publis/notebooks/
    # Demonstration%20of%20numpy.polynomial.
    # Polynomial%20and%20nice%20display%20with%20LaTeX%20and%20MathJax%20
    # (python3).html
    @staticmethod
    def polynomial_to_latex(poly, error):
        """ Small function to print nicely the polynomial p as we write it in
            maths, in LaTeX code."""
        poly = _np.poly1d(poly)
        coefs = poly.coef  # List of coefficient, sorted by increasing degrees
        res = ''  # The resulting string
        for idx, coef_idx in enumerate(coefs):
            err = error[idx]
            sig_fig = int(_floor(_log10(abs(err))))
            err = round(err, -sig_fig)
            coef_idx = round(coef_idx, -sig_fig)
            if int(coef_idx) == coef_idx:  # Remove the trailing .0
                coef_idx = int(coef_idx)
            if idx == 0:  # First coefficient, no need for X
                continue
            elif idx == 1:  # Second coefficient, only X and not X**i
                if coef_idx == 1:  # coef_idx = 1 does not need to be displayed
                    res += "\delta + "
                elif coef_idx > 0:
                    res += "({a} \pm {b}) \;\delta + ".format(
                        a="{%g}" % coef_idx, b="{%g}" % err)
                elif coef_idx < 0:
                    res += "({a} \pm {b}) \;\delta + ".format(
                        a="{%g}" % coef_idx, b="{%g}" % err)
            else:
                if coef_idx == 1:
                    # A special care needs to be addressed to put the exponent
                    # in {..} in LaTeX
                    res += "\delta^{i} + ".format(i="{%d}" % idx)
                elif coef_idx > 0:
                    res += "({a} \pm {b}) \;\delta^{i} + ".format(
                        a="{%g}" % coef_idx, b="{%g}" % err, i="{%d}" % idx)
                elif coef_idx < 0:
                    res += "({a} \pm {b}) \;\delta^{i} + ".format(
                        a="{%g}" % coef_idx, b="{%g}" % err, i="{%d}" % idx)
        return "$" + res[:-3] + "$" if res else ""
