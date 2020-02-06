"""Main module."""
import time as _time
from threading import Thread as _Thread, Event as _Event

from copy import deepcopy as _dcopy
import numpy as np

from pymodels.middlelayer.devices import TrimQuad, SOFB
import pyaccel
from siriuspy.epics import PV
from .base import BaseClass


class TuneFrac:
    """."""

    def __init__(self):
        """."""
        pre = 'SI-Glob:DI-Tune'
        prop = 'TuneFrac-Mon'
        prop_wf = 'SpecArray-Mon'
        self._tunex = PV(pre + '-H:' + prop)
        self._tuney = PV(pre + '-V:' + prop)
        self._tunex_wf = PV(pre + 'Proc-H:' + prop_wf)
        self._tuney_wf = PV(pre + 'Proc-V:' + prop_wf)

    @property
    def tunex(self):
        """."""
        return self._tunex.value

    @property
    def tuney(self):
        """."""
        return self._tuney.value

    @property
    def tunex_wf(self):
        """."""
        return self._tunex_wf.value

    @property
    def tuney_wf(self):
        """."""
        return self._tuney_wf.value


class BetaParams:
    """."""

    SYM = 0
    UNI = 1

    def __init__(self):
        """."""
        self.beta_method = 0  # 0 - Symmetric 1 - Unidirectional
        self.quad_max_deltakl = 0.01  # [1/m]
        self.dtunex_max = 0.005
        self.dtuney_max = 0.005
        self.wait_quadrupole = 2  # [s]
        self.wait_tune = 10  # [s]
        self.timeout_quad_turnon = 10  # [s]
        self.timeout_wait_sofb = 3  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        st = dtmp(
            'beta_method', self.beta_method, '(0-Symmetric 1-Unidirectional)')
        st += ftmp('dtunex_max [um]', self.dtunex_max, '')
        st += ftmp('dtuney_max [um]', self.dtuney_max, '')
        st += ftmp('quad_max_deltakl [1/m]', self.quad_max_deltakl, '')
        st += ftmp('wait_quadrupole [s]', self.wait_quadrupole, '')
        st += ftmp('wait_tune [s]', self.wait_tune, '')
        st += ftmp('timeout_quad_turnon [s]', self.timeout_quad_turnon, '')
        return st


class MeasBeta(BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.quads_betax = []
        self.quads_betay = []
        self.params = BetaParams()
        self.tunefrac = TuneFrac()
        self.devices['sofb'] = SOFB('SI')
        self.data['quadnames'] = list()
        self.data['betax_in'] = dict()
        self.data['betay_in'] = dict()
        self.data['betax_out'] = dict()
        self.data['betay_out'] = dict()
        self._quads2meas = list()
        self._stopevt = _Event()
        self._thread = _Thread(target=self._meas_beta, daemon=True)

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
    def quads2meas(self):
        """."""
        return self._quads2meas or self.data['quadnames']

    @quads2meas.setter
    def quads2meas(self, quadslist):
        """."""
        self._quads2meas = _dcopy(quadslist)

    def connect_to_objects(self):
        """."""
        for qname in self.quads2meas:
            if qname and qname not in self.devices:
                self.devices[qname] = TrimQuad(qname)

    def initialize_data(self, model, famdata):
        """."""
        quadnames, quadsidx = MeasBeta.get_default_quads(model, famdata)
        twi, *_ = pyaccel.optics.calc_twiss(model, indices='open')
        for idx, qname in enumerate(quadnames):
            self.data['betax_in'][qname] = twi.betax[quadsidx[idx]]
            self.data['betay_in'][qname] = twi.betay[quadsidx[idx]]
            self.data['betax_out'][qname] = twi.betax[quadsidx[idx]+1]
            self.data['betay_out'][qname] = twi.betay[quadsidx[idx]+1]
        self.data['quadnames'] = quadnames
        self.connect_to_objects()

    def _meas_beta(self):
        """."""
        for quadname in self.quads2meas:
            if self._stopevt.is_set():
                return
            self._meas_beta_single_quad(quadname)

    def _meas_beta_single_quad(self, quadname):
        """."""
        quad = self.devices[quadname]

        print('    turning quadrupole ' + quadname + ' On')
        quad.turnon(self.params.timeout_quad_turnon)
        if not quad.pwr_state:
            print('    error: quadrupole ' + quadname + ' is Off.')
            self._stopevt.set()
            print('    exiting...')
            return

        sofb = self.devices['sofb']

        # betax = self.data['betax_in'][quadname]
        # betay = self.data['betay_in'][quadname]

        # deltaklx = self.params.dtunex_max/betax
        # deltakly = self.params.dtuney_max/betay
        # deltakl = 4 * np.pi * np.min([deltaklx, deltakly])

        # if self.params.beta_method == BetaParams.SYM:
        #     deltaklmin = deltakl/2
        # else:
        #     deltaklmin = deltakl

        # if 'QD' in quadname:
        #     deltakl *= -1

        # if np.abs(deltaklmin) > self.params.quad_max_deltakl:
        #     print(
        #         'warning: delta kl quadrupole ' + quadname + ' over max.')
        #     print('setting max kl')
        #     deltakl = np.sign(deltakl) * self.params.quad_max_deltakl

        deltakl = self.params.quad_max_deltakl

        korig = quad.strength
        tune_neg, tune_pos = [], []
        tunewf_neg, tunewf_pos = [], []
        orb_neg, orb_pos = [], []

        if self.params.beta_method == BetaParams.SYM:
            quad.strength = korig - deltakl/2
            print('setting -deltakl/2 = ' + str(-deltakl/2))
            _time.sleep(self.params.wait_quadrupole)

            sofb.reset()
            _time.sleep(self.params.wait_tune)
            tune_neg.append([self.tunefrac.tunex, self.tunefrac.tuney])
            tunewf_neg.append(
                np.hstack([self.tunefrac.tunex_wf, self.tunefrac.tuney_wf]))
            orb_neg.append(np.hstack([sofb.orbx, sofb.orby]))

            quad.strength = korig + deltakl/2
            print('setting +deltakl/2 = ' + str(deltakl/2))
            _time.sleep(self.params.wait_quadrupole)

            sofb.reset()
            _time.sleep(self.params.wait_tune)
            tune_pos.append([self.tunefrac.tunex, self.tunefrac.tuney])
            tunewf_pos.append(
                np.hstack([self.tunefrac.tunex_wf, self.tunefrac.tuney_wf]))
            orb_pos.append(np.hstack([sofb.orbx, sofb.orby]))

        if self.params.beta_method == BetaParams.UNI:
            sofb.reset()
            _time.sleep(self.params.wait_tune)
            tune_neg.append([self.tunefrac.tunex, self.tunefrac.tuney])
            tunewf_neg.append(
                np.hstack([self.tunefrac.tunex_wf, self.tunefrac.tuney_wf]))
            orb_neg.append(np.hstack([sofb.orbx, sofb.orby]))

            quad.strength = korig + deltakl
            print('setting deltakl = ' + str(deltakl))
            _time.sleep(self.params.wait_quadrupole)

            sofb.reset()
            _time.sleep(self.params.wait_tune)
            tune_pos.append([self.tunefrac.tunex, self.tunefrac.tuney])
            tunewf_pos.append(
                np.hstack([self.tunefrac.tunex_wf, self.tunefrac.tuney_wf]))
            orb_pos.append(np.hstack([sofb.orbx, sofb.orby]))

        quad.strength = korig

        if 'measure' not in self.data:
            self.data['measure'] = dict()
        if quadname not in self.data['measure']:
            self.data['measure'][quadname] = dict()

        self.data['measure'][quadname]['tune_neg'] = np.array(tune_neg)
        self.data['measure'][quadname]['tune_pos'] = np.array(tune_pos)
        self.data['measure'][quadname]['tune_wf_neg'] = np.array(tunewf_neg)
        self.data['measure'][quadname]['tune_wf_pos'] = np.array(tunewf_pos)
        self.data['measure'][quadname]['orb_neg'] = np.array(orb_neg)
        self.data['measure'][quadname]['orb_pos'] = np.array(orb_pos)
        self.data['measure'][quadname]['delta_kl'] = np.array(deltakl)

        print('    turning quadrupole ' + quadname + ' Off')
        quad.turnoff(self.params.timeout_quad_turnon)
        if quad.pwr_state:
            print('    error: quadrupole ' + quadname + ' is still On.')
            self._stopevt.set()
            print('    exiting...')

    def process_data(self):
        """."""
        if 'analysis' not in self.data:
            self.data['analysis'] = dict()
        for quad in list(self.data['measure'].keys()):
            analysis = self.calc_beta(quad)
            self.data['analysis'][quad] = analysis

    def calc_beta(self, quadname):
        """."""
        analysis = dict()
        datameas = self.data['measure'][quadname]
        dtune = datameas['tune_pos'] - datameas['tune_neg']
        dkl = datameas['delta_kl']
        beta = 4 * np.pi * dtune / dkl
        analysis['betax'] = beta[0][0]
        analysis['betay'] = -beta[0][1]
        return analysis

    @staticmethod
    def get_default_quads(model, fam_data):
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
