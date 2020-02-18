"""Main module."""
import time as _time
from threading import Thread as _Thread, Event as _Event
import math

from copy import deepcopy as _dcopy
import numpy as np

from pymodels.middlelayer.devices import Quadrupole, SITune
from pymodels import si
import pyaccel
from siriuspy.epics import PV
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
        self.wait_quadrupole = 2  # [s]
        self.wait_tune = 3  # [s]
        self.timeout_quad_turnon = 10  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        st = dtmp('nr_measures', self.nr_measures, '')
        st += ftmp('quad_deltakl [1/m]', self.quad_deltakl, '')
        st += ftmp('quad_nrcycles', self.quad_nrcycles, '')
        st += ftmp('wait_quadrupole [s]', self.wait_quadrupole, '')
        st += ftmp('wait_tune [s]', self.wait_tune, '')
        st += ftmp('timeout_quad_turnon [s]', self.timeout_quad_turnon, '')
        return st


class MeasBeta(BaseClass):
    """."""

    def __init__(self, model, famdata=None):
        """."""
        super().__init__()
        self.quads_betax = []
        self.quads_betay = []
        self.params = BetaParams()
        self.devices['tune'] = SITune()
        self.data['quadnames'] = list()
        self.data['betax_in'] = dict()
        self.data['betay_in'] = dict()
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
        return self._thread.is_alive()

    @property
    def measuredquads(self):
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
                self.devices[qname] = Quadrupole(qname)

    def _initialize_data(self):
        """."""
        quadnames, quadsidx = MeasBeta.get_quads(self.model, self.famdata)
        twi, *_ = pyaccel.optics.calc_twiss(self.model, indices='open')
        for idx, qname in enumerate(quadnames):
            L = self.model[idx].length/2
            K = self.model[idx].K
            Kr = np.sqrt(abs(K))
            Cx = math.cos(Kr*L) if K > 0 else math.cosh(Kr*L)
            Sx = (math.sin(Kr*L) if K > 0 else math.sinh(Kr*L)) / Kr
            Cy = math.cosh(Kr*L) if K > 0 else math.cos(Kr*L)
            Sy = (math.sinh(Kr*L) if K > 0 else math.sin(Kr*L)) / Kr
            bxi = twi.betax[quadsidx[idx]]
            byi = twi.betay[quadsidx[idx]]
            axi = twi.alphax[quadsidx[idx]]
            ayi = twi.alphay[quadsidx[idx]]
            gxi = (1+axi*axi)/bxi
            gyi = (1+ayi*ayi)/byi
            bxm = Cx*Cx*bxi - 2*Cx*Sx*axi + Sx*Sx*gxi
            bym = Cy*Cy*byi - 2*Cy*Sy*ayi + Sy*Sy*gyi
            self.data['betax_in'][qname] = bxi
            self.data['betay_in'][qname] = byi
            self.data['betax_mid'][qname] = bxm
            self.data['betay_mid'][qname] = bym
            self.data['betax_out'][qname] = twi.betax[quadsidx[idx]+1]
            self.data['betay_out'][qname] = twi.betay[quadsidx[idx]+1]
        self.data['quadnames'] = quadnames

    def _meas_beta(self):
        """."""
        for quadname in self.quads2meas:
            if self._stopevt.is_set():
                return
            self._meas_beta_single_quad(quadname)

    @staticmethod
    def get_cycling_curve():
        return [-1/2, 1/2, -1/8, 0]

    def _meas_beta_single_quad(self, quadname):
        """."""
        quad = self.devices[quadname]
        tune = self.devices['tune']

        print('    turning quadrupole ' + quadname + ' On', end='')
        quad.turnon(self.params.timeout_quad_turnon)
        if not quad.pwr_state:
            print('\n    error: quadrupole ' + quadname + ' is Off.')
            self._stopevt.set()
            print('    exiting...')
            return

        deltakl = self.params.quad_deltakl
        korig = quad.strength
        cycling_curve = MeasBeta.get_cycling_curve()

        print(' and cycling it: ', end='')
        for _ in range(self.params.quad_nrcycles):
            print('.', end='')
            for fac in cycling_curve:
                quad.strength = korig + deltakl*fac
                _time.sleep(self.params.wait_quadrupole)
        print(' Ok!')

        tunex_ini, tunex_neg, tunex_pos = [], [], []
        tuney_ini, tuney_neg, tuney_pos = [], [], []
        tunex_wfm_ini, tunex_wfm_neg, tunex_wfm_pos = [], [], []
        tuney_wfm_ini, tuney_wfm_neg, tuney_wfm_pos = [], [], []

        for i in range(self.params.nr_measures):
            if self._stopevt.is_set():
                print('   exiting...')
                break
            print('    {0:02d}/{1:02d} --> '.format(
                i+1, self.params.nr_measures), end='')

            tunex_ini.append(tune.tunex)
            tuney_ini.append(tune.tuney)
            tunex_wfm_ini.append(tune.tunex_wf)
            tuney_wfm_ini.append(tune.tuney_wf)
            for j, fac in range(cycling_curve):
                quad.strength = korig + deltakl*fac
                _time.sleep(self.params.wait_quadrupole)
                if not j:
                    print(' -dk/2 ', end='')
                    _time.sleep(self.params.wait_tune)
                    tunex_neg.append(tune.tunex)
                    tuney_neg.append(tune.tuney)
                    tunex_wfm_neg.append(tune.tunex_wf)
                    tuney_wfm_neg.append(tune.tuney_wf)
                elif j == 1:
                    print(' +dk/2 ', end='')
                    _time.sleep(self.params.wait_tune)
                    tunex_pos.append(tune.tunex)
                    tuney_pos.append(tune.tuney)
                    tunex_wfm_pos.append(tune.tunex_wf)
                    tuney_wfm_pos.append(tune.tuney_wf)
            print('--> dnux = {:.5f}, dnuy = {:.5f}'.format(
               tunex_pos[-1] - tunex_neg[-1], tuney_pos[-1] - tuney_neg[-1]))

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

        self.data['measure'][quadname] = meas

        print('    turning quadrupole ' + quadname + ' Off')
        quad.turnoff(self.params.timeout_quad_turnon)
        if quad.pwr_state:
            print('    error: quadrupole ' + quadname + ' is still On.')
            self._stopevt.set()
            print('    exiting...')

    def process_data(self):
        """."""
        for quad in self.data['measure']:
            self.analysis[quad] = self.calc_beta(quad)

    def calc_beta(self, quadname):
        """."""
        anl = dict()
        datameas = self.data['measure'][quadname]
        dnux = datameas['tunex_pos'] - datameas['tunex_neg']
        dnuy = datameas['tuney_pos'] - datameas['tuney_neg']
        dkl = datameas['delta_kl']
        anl['betax'] = +4 * np.pi * dnux / dkl
        anl['betay'] = -4 * np.pi * dnuy / dkl
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
