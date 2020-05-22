"""Main module."""
import time as _time
from threading import Thread as _Thread, Event as _Event
import math

from copy import deepcopy as _dcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gs

from siriuspy.devices import PowerSupply, Tune, SOFB

import pyaccel
from pymodels import si
from .base import BaseClass


class BetaParams:
    """."""

    SYM = 0
    UNI = 1

    def __init__(self):
        """."""
        self.nr_measures = 1
        self.quad_deltakl = 0.01  # [1/m]
        self.quad_nrcycles = 0
        self.wait_quadrupole = 1  # [s]
        self.wait_tune = 3  # [s]
        self.timeout_quad_turnon = 10  # [s]
        self.time_waitquad_cycle = 0.5  # [s]
        self.recover_tune = True
        self.recover_tune_tol = 1e-4

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stg = dtmp('nr_measures', self.nr_measures, '')
        stg += ftmp('quad_deltakl [1/m]', self.quad_deltakl, '')
        stg += ftmp('quad_nrcycles', self.quad_nrcycles, '')
        stg += ftmp('wait_quadrupole [s]', self.wait_quadrupole, '')
        stg += ftmp('wait_tune [s]', self.wait_tune, '')
        stg += ftmp(
            'wait_quadrupole_cycle [s]', self.time_waitquad_cycle, '')
        stg += ftmp('timeout_quad_turnon [s]', self.timeout_quad_turnon, '')
        stg += ftmp('recover tune?', self.recover_tune, '')
        stg += ftmp('tolerance to recover tune [s]', self.recover_tune_tol, '')
        return stg


class MeasBeta(BaseClass):
    """."""

    def __init__(self, model, famdata=None):
        """."""
        super().__init__()
        self.quads_betax = []
        self.quads_betay = []
        self.params = BetaParams()
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.data['quadnames'] = list()
        self.data['cycling'] = dict()
        self.data['betax_in'] = dict()
        self.data['betay_in'] = dict()
        self.data['betax_mid'] = dict()
        self.data['betay_mid'] = dict()
        self.data['betax_out'] = dict()
        self.data['betay_out'] = dict()
        self.data['measure'] = dict()
        self.analysis = dict()
        self._quads2meas = list()
        self._stopevt = _Event()
        self._thread = _Thread(target=self._meas_beta, daemon=True)
        self.model = model
        self.famdata = famdata or si.get_family_data(model)
        self._initialize_data()
        self._connect_to_objects()

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
        twi, *_ = pyaccel.optics.calc_twiss(self.model, indices='open')
        for idx, qname in zip(quadsidx, quadnames):
            L = self.model[idx].length/2
            K = self.model[idx].K
            Kr = np.sqrt(abs(K))
            Cx = math.cos(Kr*L) if K > 0 else math.cosh(Kr*L)
            Sx = math.sin(Kr*L) if K > 0 else math.sinh(Kr*L)
            Sx = (Sx / Kr) if Kr > 0 else L
            Cy = math.cos(Kr*L) if K < 0 else math.cosh(Kr*L)
            Sy = math.sin(Kr*L) if K < 0 else math.sinh(Kr*L)
            Sy = (Sy / Kr) if Kr > 0 else L
            bxi = twi.betax[idx]
            byi = twi.betay[idx]
            axi = twi.alphax[idx]
            ayi = twi.alphay[idx]
            gxi = (1+axi*axi)/bxi
            gyi = (1+ayi*ayi)/byi
            bxm = Cx*Cx*bxi - 2*Cx*Sx*axi + Sx*Sx*gxi
            bym = Cy*Cy*byi - 2*Cy*Sy*ayi + Sy*Sy*gyi
            self.data['betax_in'][qname] = bxi
            self.data['betay_in'][qname] = byi
            self.data['betax_mid'][qname] = bxm
            self.data['betay_mid'][qname] = bym
            self.data['betax_out'][qname] = twi.betax[idx+1]
            self.data['betay_out'][qname] = twi.betay[idx+1]
        self.data['quadnames'] = quadnames

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
                        _time.sleep(self.params.time_waitquad_cycle)
                tunex_cycle.append(tune.tunex)
                tuney_cycle.append(tune.tuney)

        self.data['cycling']['tunex'] = np.array(tunex_cycle)
        self.data['cycling']['tuney'] = np.array(tuney_cycle)
        print(' Ok!')

    def _meas_beta(self):
        """."""
        sofb = self.devices['sofb']
        loop_on_rf = False
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

    @staticmethod
    def get_cycling_curve():
        """."""
        return [-1/2, 1/2, 0]

    def _recover_tune(self, meas, quadname):
        print('recovering tune...')
        tunex0 = meas['tunex_ini']
        tuney0 = meas['tuney_ini']
        deltakl = self.params.quad_deltakl

        dnux1 = meas['tunex_neg'] - tunex0
        dnuy1 = meas['tuney_neg'] - tuney0
        cx1 = -dnux1/deltakl/2
        cy1 = -dnuy1/deltakl/2
        dnux2 = meas['tunex_pos'] - meas['tunex_neg']
        dnuy2 = meas['tuney_pos'] - meas['tuney_neg']
        cx2 = dnux2/deltakl
        cy2 = dnuy2/deltakl
        cxx = (cx1 + cx2)/2
        cyy = (cy1 + cy2)/2

        _time.sleep(self.params.wait_tune)
        tunex_now = self.devices['tune'].tunex
        tuney_now = self.devices['tune'].tuney
        dtunex = tunex_now - tunex0
        dtuney = tuney_now - tuney0

        tol = self.params.recover_tune_tol
        niter = 3

        for _ in range(niter):
            if np.abs(dtunex) < tol and np.abs(dtuney) < tol:
                return True
            print('   delta tune x: {:.6f}'.format(dtunex))
            print('   delta tune y: {:.6f}'.format(dtuney))

            dkl = (cxx * dtunex + cyy * dtuney)/(cxx*cxx + cyy*cyy)

            if np.abs(dkl) > deltakl:
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
                    print(' -dk/2 ', end='')
                    _time.sleep(self.params.wait_tune)
                    tunex_neg.append(tune.tunex)
                    tuney_neg.append(tune.tuney)
                    tunex_wfm_neg.append(tune.tunex_wfm)
                    tuney_wfm_neg.append(tune.tuney_wfm)
                elif j == 1:
                    print(' +dk/2 ', end='')
                    _time.sleep(self.params.wait_tune)
                    tunex_pos.append(tune.tunex)
                    tuney_pos.append(tune.tuney)
                    tunex_wfm_pos.append(tune.tunex_wfm)
                    tuney_wfm_pos.append(tune.tuney_wfm)
            dnux = tunex_pos[-1] - tunex_neg[-1]
            dnuy = tuney_pos[-1] - tuney_neg[-1]
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

    def process_data(self, mode='symm', discardpoints=None):
        """."""
        for quad in self.data['measure']:
            self.analysis[quad] = self.calc_beta(
                quad, mode=mode, discardpoints=discardpoints)

    def calc_beta(self, quadname, mode='symm', discardpoints=None):
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
            nom_bx.append(self.data['betax_mid'][quad])
            nom_by.append(self.data['betay_mid'][quad])
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
