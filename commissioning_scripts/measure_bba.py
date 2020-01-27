"""Main module."""
import time as _time
from threading import Thread as _Thread, Event as _Event

from copy import deepcopy as _dcopy
import numpy as np

import pyaccel as _pyacc
from pymodels.middlelayer.devices import SOFB, TrimQuad, Corrector
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat
from .base import BaseClass


class BBAParams:
    SCAN = 0
    SEARCH = 1

    def __init__(self):
        self.bba_method = 0  # 0 - Scan    1 - Search
        self.dorbx_negative = 500  # [um]
        self.dorbx_positive = 500  # [um]
        self.dorby_negative = 500  # [um]
        self.dorby_positive = 500  # [um]
        self.max_corrstrength = 300  # [urad]
        self.meas_nrsteps = 10
        self.quad_deltakl = 0.02  # [1/m]
        self.wait_correctors = 2  # [s]
        self.wait_quadrupole = 2  # [s]
        self.timeout_quad_turnon = 10  # [s]
        self.wait_sofb = 3  # [s]
        self.sofb_nrpoints = 10


class DoBBA(BaseClass):

    def __init__(self):
        super().__init__()
        self.params = BBAParams()
        self.data['bpmnames'] = list()
        self.data['quadnames'] = list()
        self.data['chnames'] = list()
        self.data['cvnames'] = list()
        self._bpms2dobba = list()
        self.devices['sofb'] = SOFB('SI')
        self._stopevt = _Event()
        self._thread = _Thread(target=self._do_bba, daemon=True)

    def start(self):
        if self._thread.is_alive():
            return
        self._stopevt.clear()
        self._thread = _Thread(target=self._do_bba, daemon=True)
        self._thread.start()

    def stop(self):
        self._stopevt.set()

    @property
    def bpms2dobba(self):
        return self._bpms2dobba or self.data['bpmnames']

    @bpms2dobba.setter
    def bpms2dobba(self, bpmlist):
        self._bpms2dobba = _dcopy(bpmlist)

    def connect_to_objects(self):
        for bpm in self.bpms2dobba:
            idx = self.data['bpmnames'].index(bpm)
            qname = self.data['quadnames'][idx]
            chname = self.data['chnames'][idx]
            cvname = self.data['cvnames'][idx]
            if qname and qname not in self.devices:
                self.devices[qname] = TrimQuad(qname)
            if chname and chname not in self.devices:
                self.devices[chname] = Corrector(chname)
            if cvname and cvname not in self.devices:
                self.devices[cvname] = Corrector(cvname)

    def initialize_data(self, model, respmat=None):
        respmat_calc = OrbRespmat(model, 'SI')
        bpmnames, quadnames, quadsidx = DoBBA.get_default_quads(
            model, respmat_calc.fam_data)

        if respmat is None:
            respmat = respmat_calc.get_respm()
        self.data['bpmnames'] = bpmnames
        self.data['quadnames'] = quadnames
        self.data['respmat'] = respmat
        chnames = list()
        cvnames = list()
        indcs = np.flip(np.argsort(np.abs(respmat[:, :-1]), axis=1), axis=1)
        nbpms = len(self.data['bpmnames'])
        for i in range(nbpms):
            idxh = indcs[i, 0]
            idxv = indcs[i+nbpms, 0] - len(self.sofb.data.CH_NAMES)
            chnames.append(self.sofb.data.CH_NAMES[idxh])
            cvnames.append(self.sofb.data.CV_NAMES[idxv])
        self.data['chnames'] = chnames
        self.data['cvnames'] = cvnames

        respmat_calc.ch = quadsidx
        respmat_calc.cv = quadsidx
        quadrespmat = respmat_calc.get_respm()
        self.data['quadrespx'] = quadrespmat[:, 1:len(quadsidx)]
        self.data['quadrespy'] = quadrespmat[:, len(quadsidx):]
        self.connect_to_objects()

    def check_correctors_range(self):
        doxp = self.params.dorbx_positive
        doxn = self.params.dorbx_negative
        doyp = self.params.dorby_positive
        doyn = self.params.dorby_negative
        for bpm in self.bpms2dobba:
            dkx, dky = self.calc_orbcorr(bpm, 1, 1)
            idx = self.data['bpmnames'].index(bpm)
            ch = self.data['chnames'][idx]
            cv = self.data['cvnames'][idx]
            kickx = self.devices[ch].strength
            kicky = self.devices[cv].strength
            dkxp = self.params.max_corrstrength - kickx
            dkxn = self.params.max_corrstrength + kickx
            dkyp = self.params.max_corrstrength - kicky
            dkyn = self.params.max_corrstrength + kicky
            dorbxp = abs(dkxp/dkx)
            dorbxn = abs(dkxn/dkx)
            dorbyp = abs(dkyp/dky)
            dorbyn = abs(dkyn/dky)
            if dorbxp < doxp:
                print('{0:s}: {1:s} max pos range is {2:.2f} um'.format(
                    bpm, ch, dorbxp))
            if dorbxn < doxn:
                print('{0:s}: {1:s} max neg range is {2:.2f} um'.format(
                    bpm, ch, dorbxn))
            if dorbyp < doyp:
                print('{0:s}: {1:s} max pos range is {2:.2f} um'.format(
                    bpm, cv, dorbyp))
            if dorbyn < doyn:
                print('{0:s}: {1:s} max neg range is {2:.2f} um'.format(
                    bpm, cv, dorbyn))

    def get_correctors_candidates(self, bpmname, ncorrs=10):
        idx = self.data['bpmnames'].index(bpmname)
        nbpms = len(self.data['bpmnames'])
        respmat = self.data['respmat']

        indcs = np.flip(np.argsort(np.abs(respmat[:, :-1]), axis=1), axis=1)
        chnames = list()
        cvnames = list()
        chstreng = respmat[idx, indcs[idx, :ncorrs]]
        cvstreng = respmat[idx+nbpms, indcs[idx+nbpms, :ncorrs]]
        for i in range(ncorrs):
            idxh = indcs[idx, i]
            idxv = indcs[idx+nbpms, i] - len(self.sofb.data.CH_NAMES)
            chnames.append(self.sofb.data.CH_NAMES[idxh])
            cvnames.append(self.sofb.data.CV_NAMES[idxv])
        return chnames, cvnames, chstreng, cvstreng

    def _do_bba(self):
        self.devices['sofb'].nr_points = self.params.sofb_nrpoints
        for bpm in self._bpms2dobba:
            if self._stopevt.is_set():
                return
            self._dobba_single_bpm_scan(bpm)

    def _dobba_single_bpm(self, bpmname):
        idx = self.data['bpmnames'].index(bpmname)
        quadname = self.data['quadnames'][idx]
        chname = self.data['chnames'][idx]
        cvname = self.data['cvnames'][idx]
        quad = self.devices[quadname]
        corrx = self.devices[chname]
        corry = self.devices[cvname]

        print('Doing BBA for BPM : ' + bpmname)
        print('    turning quadrupole ' + quadname + ' On')
        quad.turnon(self.params.timeout_quad_turnon)
        if not quad.pwr_state:
            print('    error: quadrupole ' + quadname + ' is Off.')
            self._stopevt.set()
            print('    exiting...')
            return

        sofb = self.devices['sofb']

        nrsteps = self.params.meas_nrsteps
        dorbxneg = self.params.dorbx_negative
        dorbxpos = self.params.dorbx_positive
        dorbyneg = self.params.dorby_negative
        dorbypos = self.params.dorby_positive
        maxkick = self.params.max_corrstrength
        dorbsx = self._calc_dorb_scan(dorbxneg, dorbxpos, nrsteps//2)
        dorbsy = self._calc_dorb_scan(dorbyneg, dorbypos, nrsteps//2)
        deltakl = self.params.quad_deltakl

        corrx_kicks, corry_kicks = [], []
        orb_pos, orb_neg = [], []
        orig_kickx = corrx.strength
        orig_kicky = corrx.strength
        if self.params.bba_method == BBAParams.SCAN:
            npts = 2*(nrsteps//2) + 1
        else:
            npts = nrsteps
        for i in range(npts):
            if self._stopevt.is_set():
                print('   exiting...')
                break
            corrx_kicks.append(corrx.strength)
            corry_kicks.append(corry.strength)

            korig = quad.strength
            quad.strength = korig + deltakl/2
            _time.sleep(self.params.wait_quadrupole)

            sofb.reset()
            sofb.wait(self.params.wait_sofb)
            orb_pos.append(np.hstack([sofb.orbx, sofb.orby]))

            quad.strength = korig - deltakl/2
            _time.sleep(self.params.wait_quadrupole)

            sofb.reset()
            sofb.wait(self.params.wait_sofb)
            orb_neg.append(np.hstack([sofb.orbx, sofb.orby]))

            quad.strength = korig

            dorb = (orb_pos[-1]-orb_neg[-1])/deltakl
            xcen, ycen = self.calc_offset(bpmname, dorb)
            rms = np.dot(dorb, dorb)
            print('    Center ({0:.1f}, {1:.1f}), RMS {2:.2f}'.format(
                xcen, ycen, rms))
            if self.params.bba_method == BBAParams.SCAN:
                xcen, ycen = dorbsx[i], dorbsy[i]
            else:
                xcen, ycen = -xcen, -ycen
            dkx, dky = self.calc_orbcorr(bpmname, xcen, ycen)

            if i < npts-1:
                kickx = min(max(-maxkick, orig_kickx + dkx), maxkick)
                kicky = min(max(-maxkick, orig_kicky + dky), maxkick)
                corrx.strength = kickx
                corry.strength = kicky
                _time.sleep(self.params.wait_correctors)

        corrx.strength = orig_kickx
        corry.strength = orig_kicky

        self.data['measure'][bpmname]['corrxkicks'] = np.array(corrx_kicks)
        self.data['measure'][bpmname]['corrykicks'] = np.array(corry_kicks)
        self.data['measure'][bpmname]['orbpos'] = np.array(orb_pos)
        self.data['measure'][bpmname]['orbneg'] = np.array(orb_neg)

        print('    turning quadrupole ' + quadname + ' On')
        quad.turnoff(self.params.timeout_quad_turnon)
        if quad.pwr_state:
            print('    error: quadrupole ' + quadname + ' is still On.')
            self._stopevt.set()
            print('    exiting...')

    def calc_offset(self, bpmname, dorb):
        idx = self.data['bpmnames'].index(bpmname)
        respx = self.data['quadrespx'][idx]
        respy = self.data['quadrespy'][idx]
        isskew = '-QS' in self.data['quadnames'][idx]

        mat = np.array([respx, respy]).T
        x0, y0 = np.linalg.lstsq(mat, dorb)
        if isskew:
            return y0, x0
        return x0, y0

    def calc_orbcorr(self, bpmname, x0, y0):
        idxh = self.data['bpmnames'].index(bpmname)
        idxv = idxh + len(self.data['bpmnames'])
        chname = self.data['chnames'][idxh]
        cvname = self.data['cvnames'][idxh]
        chidx = self.sofb.data.CH_NAMES.index(chname)
        cvidx = self.sofb.data.CV_NAMES.index(cvname)
        cvidx += len(self.sofb.data.CH_NAMES)

        respx = self.data['respmat'][idxh, chidx]
        respy = self.data['respmat'][idxv, cvidx]

        dkx = x0/respx
        dky = y0/respy
        return dkx, dky

    def process_data(self):
        if 'analysis' not in self.data:
            self.data['analysis'] = dict()
        for bpm in self.data['measure']:
            analysis = self.process_data_single_bpm(bpm)
            self.data['analysis'][bpm] = analysis

    def process_data_single_bpm(self, bpm):
        analysis = dict()
        idx = self.data['bpmnames'].index(bpm)
        nbpms = len(self.data['bpmnames'])
        orbpos = self.data['measure'][bpm]['orbpos']
        orbneg = self.data['measure'][bpm]['orbneg']
        xpos = (orbpos[:, idx] + orbneg[:, idx])/2
        ypos = (orbpos[:, idx+nbpms] + orbneg[:, idx+nbpms])/2
        dorb = orbpos - orbneg
        dorbx = dorb[:, :nbpms]
        dorby = dorb[:, nbpms:]
        if '-QS' in self.data['quadnames'][idx]:
            dorbx, dorby = dorby, dorbx
        px = np.polyfit(xpos, dorbx, deg=1)
        py = np.polyfit(ypos, dorby, deg=1)
        x0 = -np.dot(px[0], px[1]) / np.dot(px[0], px[0])
        y0 = -np.dot(py[0], py[1]) / np.dot(py[0], py[0])
        analysis['xpos'] = xpos
        analysis['ypos'] = ypos
        analysis['linear_fitting'] = dict()
        analysis['linear_fitting']['coeffsx'] = px
        analysis['linear_fitting']['coeffsy'] = py
        analysis['linear_fitting']['x0'] = x0
        analysis['linear_fitting']['y0'] = y0

        px = np.polyfit(xpos, np.sum(dorbx*dorbx, axis=1), deg=2)
        py = np.polyfit(ypos, np.sum(dorby*dorby, axis=1), deg=2)
        x0 = -px[1] / px[0] / 2
        y0 = -py[1] / py[0] / 2
        analysis['quadratic_fitting'] = dict()
        analysis['quadratic_fitting']['coeffsx'] = px
        analysis['quadratic_fitting']['coeffsy'] = py
        analysis['quadratic_fitting']['x0'] = x0
        analysis['quadratic_fitting']['y0'] = y0
        return analysis

    @staticmethod
    def get_default_quads(model, fam_data):
        quads_idx = _dcopy(fam_data['QN']['index'])
        quads_idx.extend(fam_data['QS']['index'])
        quads_idx = np.array([idx[len(idx)//2] for idx in quads_idx])
        quads_pos = np.array(_pyacc.lattice.find_spos(model, quads_idx))

        bpms_idx = np.array([idx[0] for idx in fam_data['BPM']['index']])
        bpms_pos = np.array(_pyacc.lattice.find_spos(model, bpms_idx))

        diff = np.abs(bpms_pos[:, None] - quads_pos[None, :])
        bba_idx = np.argmin(diff, axis=1)
        quads_bba_idx = quads_idx[bba_idx]
        bpmnames = list()
        qnames = list()
        for i, qidx in enumerate(quads_bba_idx):
            name = model[qidx].fam_name
            idc = fam_data[name]['index'].index([qidx, ])
            sub = fam_data[name]['subsection'][idc]
            inst = fam_data[name]['instance'][idc]
            name = 'QS' if name.startswith('S') else name
            qname = 'SI-{0:s}:PS-{1:s}-{2:s}'.format(sub, name, inst)
            qnames.append(qname.strip('-'))

            sub = fam_data['BPM']['subsection'][i]
            inst = fam_data['BPM']['instance'][i]
            bname = 'SI-{0:s}:DI-BPM-{1:s}'.format(sub, inst)
            bname = bname.strip('-')
            bpmnames.append(bname.strip('-'))
        return bpmnames, qnames, quads_bba_idx

    @staticmethod
    def _calc_dorb_scan(dorbneg, dorbpos, nrpts):
        dorbspos = np.linspace(dorbpos, 0, nrpts)[:-1]
        dorbsneg = np.linspace(-dorbneg, 0, nrpts)[:-1]
        dorbs = np.array([dorbsneg, dorbspos]).T.flatten()
        dorbs = np.hstack([0, dorbs, 0])
        dorbs = -np.diff(dorbs)
        return dorbs
