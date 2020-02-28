"""Main module."""
import time as _time
from threading import Thread as _Thread, Event as _Event

from copy import deepcopy as _dcopy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gs
import matplotlib.cm as cm

from siriuspy.devices import SOFB, RF, SITune

import pyaccel as _pyacc
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat
from .base import BaseClass


class MeasParams:

    MOM_COMPACT = 1.68e-4

    def __init__(self):
        self.delta_freq = 200  # [Hz]
        self.meas_nrsteps = 8
        self.npoints = 5
        self.wait_rf = 1  # [s]
        self.wait_tune = 5  # [s]
        self.timeout_wait_sofb = 3  # [s]
        self.sofb_nrpoints = 10

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        st = ftmp('delta_freq [Hz]', self.delta_freq, '')
        st += dtmp('meas_nrsteps', self.meas_nrsteps, '')
        st += ftmp('wait_rf [s]', self.wait_rf, '')
        st += ftmp('wait_tune [s]', self.wait_tune, '')
        st += ftmp(
            'timeout_wait_sofb [s]', self.timeout_wait_sofb, '(get orbit)')
        st += dtmp('sofb_nrpoints', self.sofb_nrpoints, '')
        return st


class MeasDispChrom(BaseClass):

    def __init__(self):
        super().__init__()
        self.params = MeasParams()
        self.devices['sofb'] = SOFB('SI')
        self.devices['tune'] = SITune()
        self.devices['rf'] = RF('SI')
        self.analysis = dict()
        self._stopevt = _Event()
        self._thread = _Thread(target=self._do_bba, daemon=True)

    def __str__(self):
        stn = 'Params\n'
        stp = self.params.__str__()
        stp = '    ' + stp.replace('\n', '\n    ')
        stn += stp + '\n'
        stn += 'Connected?  ' + str(self.connected) + '\n\n'
        return stn

    def start(self):
        if self.ismeasuring:
            return
        self._stopevt.clear()
        self._thread = _Thread(target=self._do_meas, daemon=True)
        self._thread.start()

    def stop(self):
        self._stopevt.set()

    @property
    def ismeasuring(self):
        return self._thread.is_alive()

    def _do_meas(self):
        sofb = self.devices['sofb']
        rf = self.devices['rf']
        tune = self.devices['tune']

        delta_freq = self.params.delta_freq
        npoints = self.params.meas_nrsteps
        sofb.nr_points = self.params.sofb_nrpoints
        freq0 = rf.frequency
        tunex0 = tune.tunex
        tuney0 = tune.tuney
        orbx0 = sofb.orbx
        orby0 = sofb.orby
        span = np.linspace(freq0-delta_freq/2, freq0+delta_freq/2, npoints)

        freq = []
        tunex, tuney = [], []
        orbx, orby = [], []
        for f in span:
            if self._stopevt.is_set():
                print('   exiting...')
                break
            rf.frequency = f
            _time.sleep(self.params.wait_rf)
            sofb.reset()
            _time.sleep(self.params.wait_tune)
            sofb.wait(self.params.timeout_wait_sofb)
            freq.append(rf.frequency)
            orbx.append(sofb.orbx)
            orby.append(sofb.orby)
            tunex.append(tune.tunex)
            tuney.append(tune.tuney)
            print('delta frequency: {} Hz'.format((rf.frequency-freq0)))
            print('dtune x: {}'.format((tunex[-1] - tunex0)))
            print('dtune y: {}'.format((tuney[-1] - tuney0)))
            print('')
        print('Restoring RF frequency...')
        rf.frequency = freq0
        self.data['freq'] = np.array(freq)
        self.data['tunex'] = np.array(tunex)
        self.data['tuney'] = np.array(tuney)
        self.data['tunex0'] = tunex0
        self.data['tuney0'] = tuney0
        self.data['orbx0'] = np.array(orbx0)
        self.data['orby0'] = np.array(orby0)
        self.data['orbx'] = np.array(orbx)
        self.data['orby'] = np.array(orby)
        self.data['freq0'] = freq0
        print('Finished!')

    def process_data(self, fitorder=1, discardpoints=None):
        data = self.data

        usepts = set(range(data.tunex.shape[0]))
        if discardpoints is not None:
            usepts = set(usepts) - set(discardpoints)
        usepts = sorted(usepts)

        freq0 = data['freq0']
        den = -(data['freq'] - freq0)/freq0/self.params.MOM_COMPACT
        den = den[usepts]
        tunex = data['tunex'][usepts]
        tuney = data['tuney'][usepts]
        orbx = data['orbx'][usepts, :]
        orby = data['orby'][usepts, :]

        if tunex.size > fitorder + 1:
            chromx, chromxcov = np.polyfit(den, tunex, deg=fitorder, cov=True)
            chromy, chromycov = np.polyfit(den, tuney, deg=fitorder, cov=True)
            dispx, dispxcov = np.polyfit(den, orbx, deg=fitorder, cov=True)
            dispy, dispycov = np.polyfit(den, orby, deg=fitorder, cov=True)
        else:
            chromx = np.polyfit(den, tunex, deg=fitorder, cov=False)
            chromy = np.polyfit(den, tuney, deg=fitorder, cov=False)
            dispx = np.polyfit(den, orbx, deg=fitorder, cov=False)
            dispy = np.polyfit(den, orby, deg=fitorder, cov=False)
            chromxcov = chromycov = np.zeros(
                (fitorder+1, fitorder+1), dtype=float)
            dispxcov = dispycov = np.zeros(
                (fitorder+1, fitorder+1, orbx.shape[1]), dtype=float)

        self.analysis['delta'] = den
        self.analysis['orbx'] = orbx
        self.analysis['orby'] = orby
        self.analysis['dispx'] = dispx
        self.analysis['dispy'] = dispy
        self.analysis['dispx_err'] = np.sqrt(np.diagonal(dispxcov))
        self.analysis['dispy_err'] = np.sqrt(np.diagonal(dispycov))

        self.analysis['tunex'] = tunex
        self.analysis['tuney'] = tuney
        self.analysis['chromx'] = chromx
        self.analysis['chromy'] = chromy
        self.analysis['chromx_err'] = np.sqrt(np.diagonal(chromxcov))
        self.analysis['chromy_err'] = np.sqrt(np.diagonal(chromycov))

    # Copied from:
    # https://perso.crans.org/besson/publis/notebooks/
    # Demonstration%20of%20numpy.polynomial.
    # Polynomial%20and%20nice%20display%20with%20LaTeX%20and%20MathJax%20
    # (python3).html
    @staticmethod
    def polynomial_to_latex(p):
        """ Small function to print nicely the polynomial p as we write it in
        maths, in LaTeX code."""
        p = np.poly1d(p)
        coefs = p.coef  # List of coefficient, sorted by increasing degrees
        res = ''  # The resulting string
        for i, a in enumerate(coefs):
            if int(a) == a:  # Remove the trailing .0
                a = int(a)
            if i == 0:  # First coefficient, no need for X
                if a > 0:
                    res += "{a} + ".format(a=a)
                elif a < 0:  # Negative a is printed like (a)
                    res += "({a}) + ".format(a=a)
                # a = 0 is not displayed
            elif i == 1:  # Second coefficient, only X and not X**i
                if a == 1:  # a = 1 does not need to be displayed
                    res += "X + "
                elif a > 0:
                    res += "{a} \;X + ".format(a=a)
                elif a < 0:
                    res += "({a}) \;X + ".format(a=a)
            else:
                if a == 1:
                    # A special care needs to be addressed to put the exponent
                    # in {..} in LaTeX
                    res += "X^{i} + ".format(i="{%d}" % i)
                elif a > 0:
                    res += "{a} \;X^{i} + ".format(a=a, i="{%d}" % i)
                elif a < 0:
                    res += "({a}) \;X^{i} + ".format(a=a, i="{%d}" % i)
        return "$" + res[:-3] + "$" if res else ""

    def make_figure_chrom(self, title='', fname=''):
        f = plt.figure(figsize=(10, 5))
        gs = mpl_gs.GridSpec(1, 1)
        gs.update(
            left=0.12, right=0.95, bottom=0.15, top=0.9,
            hspace=0.5, wspace=0.35)

        if title:
            f.suptitle(title)

        den = self.analysis['delta']
        tunex = self.analysis['tunex']
        tuney = self.analysis['tuney']
        chromx = self.analysis['chromx']
        chromy = self.analysis['chromy']
        dtunex = tunex - chromx[-1]
        dtuney = tuney - chromy[-1]
        dtunex_fit = np.polyval(chromx, den) - chromx[-1]
        dtuney_fit = np.polyval(chromy, den) - chromy[-1]

        axx = plt.subplot(gs[0, 0])
        axx.plot(den*100, dtunex*1000, 'bo', label='horizontal')
        axx.plot(den*100, dtunex_fit*1000, '-b')
        axx.plot(den*100, dtuney*1000, 'ro', label='vertical')
        axx.plot(den*100, dtuney_fit*1000, '-r')
        axx.set_xlabel(r'$\delta$ [%]')
        axx.set_ylabel(r'$\Delta \nu \times 1000$')

        chromx = px[0]
        chromy = py[0]
        st = r'$\xi_x$: {:.2f}'.format(chromx) + '\n'
        st += r'$\xi_y$: {:.2f}'.format(chromy)
        axx.text(
            0.7, 0.05, st, horizontalalignment='left',
            verticalalignment='bottom', transform=axx.transAxes,
            bbox=dict(edgecolor='k', facecolor='w', alpha=1.0))
        axx.legend()
        axx.grid(True)

        if fname:
            f.savefig(fname+'.svg')
            plt.close()
        else:
            f.show()
