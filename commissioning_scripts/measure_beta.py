"""Main module."""
import time as _time
from threading import Thread as _Thread, Event as _Event
import math

from copy import deepcopy as _dcopy
import numpy as np
from collections import namedtuple as _namedtuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gs

from siriuspy.devices import PowerSupply, Tune, SOFB

import pyaccel
from pymodels import si
from .base import BaseClass


class BetaParams:
    """."""

    def __init__(self):
        """."""
        self.nr_measures = 1
        self.quad_deltakl = 0.01  # [1/m]
        self.wait_quadrupole = 1  # [s]
        self.wait_tune = 3  # [s]
        self.timeout_quad_turnon = 10  # [s]
        self.recover_tune = True
        self.recover_tune_tol = 1e-4
        self.recover_tune_maxiter = 3
        self.quad_nrcycles = 0
        self.time_wait_quad_cycle = 0.5  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:26s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stg = dtmp('nr_measures', self.nr_measures, '')
        stg += ftmp('quad_deltakl [1/m]', self.quad_deltakl, '')
        stg += ftmp('wait_quadrupole [s]', self.wait_quadrupole, '')
        stg += ftmp('wait_tune [s]', self.wait_tune, '')
        stg += ftmp('timeout_quad_turnon [s]', self.timeout_quad_turnon, '')
        stg += ftmp('recover_tune', self.recover_tune, '')
        stg += ftmp('recover_tune_tol', self.recover_tune_tol, '')
        stg += dtmp(
            'recover_tune_maxiter', self.recover_tune_maxiter, '')
        stg += ftmp('quad_nrcycles', self.quad_nrcycles, '')
        stg += ftmp(
            'time_wait_quad_cycle [s]', self.time_wait_quad_cycle, '')
        return stg


class MeasBeta(BaseClass):
    """."""

    METHODS = _namedtuple('Methods', ['Analytic', 'Numeric'])(0, 1)

    def __init__(
            self, model, famdata=None,
            is_online=True, calc_method=None):
        """."""
        super().__init__()
        self.is_online = is_online
        self.quads_betax = []
        self.quads_betay = []
        self.params = BetaParams()
        self._calc_method = MeasBeta.METHODS.Numeric
        if self.is_online:
            self.devices['tune'] = Tune(Tune.DEVICES.SI)
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.data['quadnames'] = list()
        self.data['cycling'] = dict()
        self.data['betax_int'] = dict()
        self.data['betay_int'] = dict()
        self.data['measure'] = dict()
        self.data['tunex_ref'] = 0
        self.data['tuney_ref'] = 0
        self.analysis = dict()
        self._quads2meas = list()
        self._stopevt = _Event()
        self._thread = _Thread(target=self._meas_beta, daemon=True)
        self.model = model
        self.calc_method = calc_method
        self.famdata = famdata or si.get_family_data(model)
        self._initialize_data()
        if self.is_online:
            self._connect_to_objects()

    @property
    def calc_method(self):
        """."""
        return self._calc_method

    @calc_method.setter
    def calc_method(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._calc_method = int(value in MeasBeta.METHODS._fields[1])
        elif int(value) in MeasBeta.METHODS:
            self._calc_method = int(value)

    @property
    def calc_method_str(self):
        """."""
        return MeasBeta.METHODS._fields[self._calc_method]

    def start(self):
        """."""
        if self._thread.is_alive():
            return
        self._stopevt.clear()
        self._thread = _Thread(target=self._meas_beta, daemon=True)
        self._thread.start()

    def stop(self):
        """."""
        self._stopevt.set()

    @property
    def ismeasuring(self):
        """."""
        return self._thread.is_alive()

    @property
    def measuredquads(self):
        """."""
        return sorted(self.data['measure'])

    @property
    def quads2meas(self):
        """."""
        if self._quads2meas:
            return self._quads2meas
        return sorted(
            set(self.data['quadnames']) - self.data['measure'].keys())

    @quads2meas.setter
    def quads2meas(self, quadslist):
        """."""
        self._quads2meas = _dcopy(quadslist)

    def _connect_to_objects(self):
        """."""
        for qname in self.data['quadnames']:
            if qname not in self.devices:
                self.devices[qname] = PowerSupply(qname)

    def _initialize_data(self):
        """."""
        quadnames, quadsidx = MeasBeta.get_quads(self.model, self.famdata)
        self.data['quadnames'] = quadnames
        twi, *_ = pyaccel.optics.calc_twiss(self.model, indices='open')

        if self.calc_method == MeasBeta.METHODS.Analytic:
            for idx, qname in zip(quadsidx, quadnames):
                L = self.model[idx].length
                K = self.model[idx].K
                Kr = np.sqrt(abs(K))
                bxi = twi.betax[idx]
                byi = twi.betay[idx]
                axi = twi.alphax[idx]
                ayi = twi.alphay[idx]
                gxi = (1+axi*axi)/bxi
                gyi = (1+ayi*ayi)/byi
                self.data['betax_int'][qname] = self._calc_beta_integral(
                    K, L, bxi, axi, gxi)
                self.data['betay_int'][qname] = self._calc_beta_integral(
                    -K, L, bxi, axi, gxi)
        elif self.calc_method == MeasBeta.METHODS.Numeric:
            nux0, nuy0, *_ = pyaccel.optics.get_frac_tunes(self.model)
            dkl = self.params.quad_deltakl
            for idx, qname in zip(quadsidx, quadnames):
                korig = self.model[idx].KL
                if 'QF' in self.model[idx].fam_name:
                    deltakl = +dkl
                else:
                    deltakl = -dkl
                self.model[idx].KL = korig + deltakl
                nuxp, nuyp, *_ = pyaccel.optics.get_frac_tunes(self.model)
                dnux = nuxp - nux0
                dnuy = nuyp - nuy0
                self.data['betax_int'][qname] = +4 * np.pi * dnux/deltakl
                self.data['betay_int'][qname] = -4 * np.pi * dnuy/deltakl
                self.model[idx].KL = korig

    @staticmethod
    def _calc_beta_integral(K, L, beta0, alpha0, gamma0):
        Kr = np.sqrt(abs(K))
        fac1 = math.sin(2*Kr*L) if K > 0 else math.sinh(2*Kr*L)
        fac2 = (math.sin(Kr*L))**2 if K > 0 else \
            (math.sinh(Kr*L))**2
        beta_int = (beta0 + gamma0/K)/2
        beta_int += fac1/Kr/L/4 * (beta0 - gamma0/K)
        beta_int -= alpha0*fac2/abs(K)/L
        return beta_int

    def _meas_beta(self):
        """."""
        sofb = self.devices['sofb']
        loop_on_rf = False
        ti0 = _time.time()

        _time.sleep(self.params.wait_tune)
        self.data['tunex_ref'] = self.devices['tune'].tunex
        self.data['tuney_ref'] = self.devices['tune'].tuney

        if sofb.autocorrsts and sofb.rfenbl:
            loop_on_rf = True
            print('RF is enable in SOFB feedback, disabling it...')
            sofb.rfenbl = 0

        self._cycle_quads()

        for quadname in self.quads2meas:
            if self._stopevt.is_set():
                return
            print('\n  measuring quad: ' + quadname, end=' ')
            self._meas_beta_single_quad(quadname)

        if loop_on_rf:
            print(
                'RF was enable in SOFB feedback, restoring original state...')
            sofb.rfenbl = 1
        print('finished!')
        tif = _time.time()
        print('time elapsed: {:.2f} min'.format((tif-ti0)/60))

    @staticmethod
    def get_cycling_curve():
        """."""
        return [+1, 0]

    def _cycle_quads(self):
        tune = self.devices['tune']
        deltakl = self.params.quad_deltakl
        cycling_curve = MeasBeta.get_cycling_curve()
        tunex_cycle = []
        tuney_cycle = []
        print('\n preparing all quads: ', end='')
        for cynum in range(self.params.quad_nrcycles):
            print('\n   cycle: {0:02d}/{1:02d} --> '.format(
                cynum+1, self.params.quad_nrcycles), end='')
            for quadname in self.data['quadnames']:
                if self._stopevt.is_set():
                    print('exiting...')
                    break
                print('\n  cycling quad ' + quadname, end=' ')
                quad = self.devices[quadname]
                korig = quad.strength
                for fac in cycling_curve:
                    quad.strength = korig + deltakl*fac
                    if fac:
                        _time.sleep(self.params.time_wait_quad_cycle)
                tunex_cycle.append(tune.tunex)
                tuney_cycle.append(tune.tuney)

        self.data['cycling']['tunex'] = np.array(tunex_cycle)
        self.data['cycling']['tuney'] = np.array(tuney_cycle)
        print(' Ok!')

    def _recover_tune(self, meas, quadname):
        print('recovering tune...')
        tunex0 = meas['tunex_ini'][0]
        tuney0 = meas['tuney_ini'][0]
        deltakl = meas['delta_kl']

        dnux = meas['tunex_pos'][-1] - meas['tunex_ini'][-1]
        dnuy = meas['tuney_pos'][-1] - meas['tuney_ini'][-1]
        cxx = dnux/deltakl
        cyy = dnuy/deltakl

        _time.sleep(self.params.wait_tune)
        tunex_now = self.devices['tune'].tunex
        tuney_now = self.devices['tune'].tuney
        dtunex = tunex_now - tunex0
        dtuney = tuney_now - tuney0

        tol = self.params.recover_tune_tol
        niter = self.params.recover_tune_maxiter

        for _ in range(niter):
            dtune_mod = np.sqrt(dtunex*dtunex + dtuney*dtuney)
            print(
                '   delta tune x, y: {0:.6f}, {1:.6f} ({2:.6f})'.format(
                    dtunex, dtuney, dtune_mod))
            if dtune_mod < tol:
                return True

            # minimization of squares [cx cy]*dkl = [dtunex dtuney]
            dkl = (cxx * dtunex + cyy * dtuney)/(cxx*cxx + cyy*cyy)

            if np.abs(dkl) > np.abs(deltakl):
                print('   deltakl calculated is too big!')
                return False

            self.devices[quadname].strength -= dkl

            _time.sleep(self.params.wait_quadrupole)
            _time.sleep(self.params.wait_tune)

            tunex_now = self.devices['tune'].tunex
            tuney_now = self.devices['tune'].tuney
            dtunex = tunex_now - tunex0
            dtuney = tuney_now - tuney0
        return False

    def _meas_beta_single_quad(self, quadname):
        """."""
        quad = self.devices[quadname]
        tune = self.devices['tune']

        if not quad.pwrstate:
            print('turning quadrupole ' + quadname + ' On', end='')
            quad.cmd_turn_on(self.params.timeout_quad_turnon)

        if not quad.pwrstate:
            print('\n    error: quadrupole ' + quadname + ' is Off.')
            self._stopevt.set()
            print('    exiting...')
            return

        deltakl = self.params.quad_deltakl
        korig = quad.strength
        cycling_curve = MeasBeta.get_cycling_curve()

        # measurement always increases the power supply current
        if 'QD' in quadname:
            deltakl *= -1

        tunex_ini, tunex_neg, tunex_pos = [], [], []
        tuney_ini, tuney_neg, tuney_pos = [], [], []
        tunex_wfm_ini, tunex_wfm_neg, tunex_wfm_pos = [], [], []
        tuney_wfm_ini, tuney_wfm_neg, tuney_wfm_pos = [], [], []

        for nrmeas in range(self.params.nr_measures):

            if self._stopevt.is_set():
                print('exiting...')
                break
            print('   meas. {0:02d}/{1:02d} --> '.format(
                nrmeas+1, self.params.nr_measures), end='')

            tunex_ini.append(tune.tunex)
            tuney_ini.append(tune.tuney)
            tunex_wfm_ini.append(tune.tunex_wfm)
            tuney_wfm_ini.append(tune.tuney_wfm)
            for j, fac in enumerate(cycling_curve):
                quad.strength = korig + deltakl*fac
                _time.sleep(self.params.wait_quadrupole)
                if not j:
                    if 'QD' in quadname:
                        print(' -dk ', end='')
                    else:
                        print(' +dk ', end='')
                    _time.sleep(self.params.wait_tune)
                    tunex_pos.append(tune.tunex)
                    tuney_pos.append(tune.tuney)
                    tunex_wfm_pos.append(tune.tunex_wfm)
                    tuney_wfm_pos.append(tune.tuney_wfm)
            dnux = tunex_pos[-1] - tunex_ini[-1]
            dnuy = tuney_pos[-1] - tuney_ini[-1]
            print('--> dnux = {:.5f}, dnuy = {:.5f}'.format(dnux, dnuy))

        meas = dict()
        meas['tunex_ini'] = np.array(tunex_ini)
        meas['tuney_ini'] = np.array(tuney_ini)
        meas['tunex_neg'] = np.array(tunex_neg)
        meas['tuney_neg'] = np.array(tuney_neg)
        meas['tunex_pos'] = np.array(tunex_pos)
        meas['tuney_pos'] = np.array(tuney_pos)
        meas['tunex_wfm_ini'] = np.array(tunex_wfm_ini)
        meas['tuney_wfm_ini'] = np.array(tuney_wfm_ini)
        meas['tunex_wfm_neg'] = np.array(tunex_wfm_neg)
        meas['tuney_wfm_neg'] = np.array(tuney_wfm_neg)
        meas['tunex_wfm_pos'] = np.array(tunex_wfm_pos)
        meas['tuney_wfm_pos'] = np.array(tuney_wfm_pos)
        meas['delta_kl'] = deltakl

        if self.params.recover_tune:
            if self._recover_tune(meas, quadname):
                print('tune recovered!')
            else:
                print('cannot recover tune for :{:s}'.format(quadname))

        self.data['measure'][quadname] = meas

    def process_data(self, mode='pos', discardpoints=None):
        """."""
        for quad in self.data['measure']:
            self.analysis[quad] = self.calc_beta(
                quad, mode=mode, discardpoints=discardpoints)

    def calc_beta(self, quadname, mode='pos', discardpoints=None):
        """."""
        anl = dict()
        datameas = self.data['measure'][quadname]

        if mode.lower().startswith('symm'):
            dnux = datameas['tunex_pos'] - datameas['tunex_neg']
            dnuy = datameas['tuney_pos'] - datameas['tuney_neg']
        elif mode.lower().startswith('pos'):
            dnux = datameas['tunex_pos'] - datameas['tunex_ini']
            dnuy = datameas['tuney_pos'] - datameas['tuney_ini']
        else:
            dnux = datameas['tunex_ini'] - datameas['tunex_neg']
            dnuy = datameas['tuney_ini'] - datameas['tuney_neg']

        usepts = set(range(dnux.shape[0]))
        if discardpoints is not None:
            usepts = set(usepts) - set(discardpoints)
        usepts = sorted(usepts)

        dkl = datameas['delta_kl']
        anl['betasx'] = +4 * np.pi * dnux[usepts] / dkl
        anl['betasy'] = -4 * np.pi * dnuy[usepts] / dkl
        anl['betax_ave'] = np.mean(anl['betasx'])
        anl['betay_ave'] = np.mean(anl['betasy'])
        return anl

    @staticmethod
    def get_quads(model, fam_data):
        """."""
        quads_idx = _dcopy(fam_data['QN']['index'])
        quads_idx = np.array([idx[len(idx)//2] for idx in quads_idx])

        qnames = list()
        for qidx in quads_idx:
            name = model[qidx].fam_name
            idc = fam_data[name]['index'].index([qidx, ])
            sub = fam_data[name]['subsection'][idc]
            inst = fam_data[name]['instance'][idc]
            qname = 'SI-{0:s}:PS-{1:s}-{2:s}'.format(sub, name, inst)
            qnames.append(qname.strip('-'))
        return qnames, quads_idx

    def plot_results(self, quads=None, title=''):
        """."""
        fig = plt.figure(figsize=(9, 7))
        grids = mpl_gs.GridSpec(ncols=1, nrows=2, figure=fig)
        grids.update(
            left=0.1, right=0.8, bottom=0.15, top=0.9,
            hspace=0.0, wspace=0.35)

        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[1, 0], sharex=ax1)

        if title:
            fig.suptitle(title)

        quads = quads or self.data['quadnames']
        indcs, nom_bx, nom_by = [], [], []
        mes_bx, mes_by, bx_ave, by_ave = [], [], [], []
        for quad in quads:
            if quad not in self.analysis:
                continue
            indcs.append(self.data['quadnames'].index(quad))
            nom_bx.append(self.data['betax_int'][quad])
            nom_by.append(self.data['betay_int'][quad])
            mes_bx.append(self.analysis[quad]['betasx'])
            mes_by.append(self.analysis[quad]['betasy'])
            bx_ave.append(self.analysis[quad]['betax_ave'])
            by_ave.append(self.analysis[quad]['betay_ave'])

        ax1.plot(indcs, bx_ave, '--bx')
        ax1.plot(indcs, nom_bx, '-bo')
        ax1.plot(indcs, mes_bx, '.b')
        ax2.plot(indcs, by_ave, '--rx')
        ax2.plot(indcs, nom_by, '-ro')
        ax2.plot(indcs, mes_by, '.r')

        ax1.set_ylabel(r'$\beta_x$ [m]')
        ax2.set_ylabel(r'$\beta_y$ [m]')
        ax1.legend(
            ['measure', 'nominal'], loc='center left',
            bbox_to_anchor=(1, 0), fontsize='x-small')
        ax1.grid(True)
        ax2.grid(True)
        fig.show()
