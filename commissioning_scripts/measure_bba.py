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
        self.dorbx_stretch = 1.2
        self.dorby_stretch = 1.2
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
        self.timeout_wait_sofb = 3  # [s]
        self.sofb_nrpoints = 10

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        st = dtmp('bba_method', self.bba_method, '(0-Scan 1-Search)')
        st += ftmp(
            'dorbx_stretch', self.dorbx_stretch,
            '(apply to Search estimatives)')
        st += ftmp(
            'dorby_stretch', self.dorby_stretch,
            '(apply to Search estimatives)')
        st += ftmp('dorbx_negative [um]', self.dorbx_negative, '')
        st += ftmp('dorbx_positive [um]', self.dorbx_positive, '')
        st += ftmp('dorby_negative [um]', self.dorby_negative, '')
        st += ftmp('dorby_positive [um]', self.dorby_positive, '')
        st += ftmp('max_corrstrength [urad]', self.max_corrstrength, '')
        st += dtmp('meas_nrsteps', self.meas_nrsteps, '')
        st += ftmp('quad_deltakl [1/m]', self.quad_deltakl, '')
        st += ftmp('wait_correctors [s]', self.wait_correctors, '')
        st += ftmp('wait_quadrupole [s]', self.wait_quadrupole, '')
        st += ftmp('timeout_quad_turnon [s]', self.timeout_quad_turnon, '')
        st += dtmp('sofb_nrpoints', self.sofb_nrpoints, '')
        return st


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

    def __str__(self):
        stn = 'Params\n'
        stp = self.params.__str__()
        stp = '    ' + stp.replace('\n', '\n    ')
        stn += stp + '\n'
        stn += 'Connected?  ' + str(self.connected) + '\n\n'

        stn += '     {:^20s} {:^20s} {:^20s} {:^20s} {:6s} {:6s}\n'.format(
            'BPM', 'CH', 'CV', 'Quad', 'Xc [um]', 'Yc [um]')
        tmplt = '{:03d}: {:^20s} {:^20s} {:^20s} {:^20s} {:^6.1f} {:^6.1f}\n'
        dta = self.data
        for bpm in self.bpms2dobba:
            idx = dta['bpmnames'].index(bpm)
            stn += tmplt.format(
                idx, dta['bpmnames'][idx], dta['chnames'][idx],
                dta['cvnames'][idx], dta['quadnames'][idx],
                dta['scancenterx'][idx], dta['scancentery'][idx])
        return stn

    def start(self):
        if self._thread.is_alive():
            return
        self._stopevt.clear()
        self._thread = _Thread(target=self._do_bba, daemon=True)
        self._thread.start()

    def stop(self):
        self._stopevt.set()

    @property
    def ismeasuring(self):
        return self._thread.is_alive()

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
        self.respmat_calc = respmat_calc
        bpmnames, quadnames, quadsidx = DoBBA.get_default_quads(
            model, respmat_calc.fam_data)
        sofb = self.devices['sofb']

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
            idxv = indcs[i+nbpms, 0] - len(sofb.data.CH_NAMES)
            chnames.append(sofb.data.CH_NAMES[idxh])
            cvnames.append(sofb.data.CV_NAMES[idxv])
        self.data['chnames'] = chnames
        self.data['cvnames'] = cvnames

        respmat_calc.ch = quadsidx
        respmat_calc.cv = quadsidx
        quadrespmat = respmat_calc.get_respm()
        self.data['quadrespx'] = quadrespmat[:, :len(quadsidx)]
        self.data['quadrespy'] = quadrespmat[:, len(quadsidx):-1]
        self.data['scancenterx'] = np.zeros(len(bpmnames))
        self.data['scancentery'] = np.zeros(len(bpmnames))
        self.connect_to_objects()

    def check_correctors_range(self):
        doxp = self.params.dorbx_positive
        doxn = self.params.dorbx_negative
        doyp = self.params.dorby_positive
        doyn = self.params.dorby_negative
        orb = self._get_orbit()
        for bpm in self.bpms2dobba:
            dkx, dky = self.calc_orbcorr(bpm, 1, 1)
            idx = self.data['bpmnames'].index(bpm)

            xpos = orb[idx]
            xcen = self.data['scancenterx'][idx]
            doxpi = abs(max(xpos - (xcen+doxp), 0))
            doxni = abs(min(xpos - (xcen-doxn), 0))

            ypos = orb[idx+len(self.data['bpmnames'])]
            ycen = self.data['scancentery'][idx]
            doypi = abs(max(ypos - (ycen+doyp), 0))
            doyni = abs(min(ypos - (ycen-doyn), 0))

            ch = self.data['chnames'][idx]
            cv = self.data['cvnames'][idx]
            kickx = self.devices[ch].strength
            kicky = self.devices[cv].strength
            dkxp = self.params.max_corrstrength - kickx
            dkxn = self.params.max_corrstrength + kickx
            dkyp = self.params.max_corrstrength - kicky
            dkyn = self.params.max_corrstrength + kicky
            if dkx > 0:
                dorbxp = abs(dkxp/dkx)
                dorbxn = abs(dkxn/dkx)
            else:
                dorbxp = abs(dkxn/dkx)
                dorbxn = abs(dkxp/dkx)
            if dky > 0:
                dorbyp = abs(dkyp/dky)
                dorbyn = abs(dkyn/dky)
            else:
                dorbyp = abs(dkyn/dky)
                dorbyn = abs(dkyp/dky)
            if dorbxp < doxpi:
                print('{0:s}: {1:s} +dorb = {2:.2f}, need {3:.2f}'.format(
                    bpm, ch, dorbxp, doxpi))
            if dorbxn < doxni:
                print('{0:s}: {1:s} -dorb = {2:.2f}, need {3:.2f}'.format(
                    bpm, ch, dorbxn, doxni))
            if dorbyp < doypi:
                print('{0:s}: {1:s} +dorb = {2:.2f}, need {3:.2f}'.format(
                    bpm, cv, dorbyp, doypi))
            if dorbyn < doyni:
                print('{0:s}: {1:s} -dorb = {2:.2f}, need {3:.2f}'.format(
                    bpm, cv, dorbyn, doyni))

    def get_correctors_candidates(self, bpmname, ncorrs=10):
        idx = self.data['bpmnames'].index(bpmname)
        nbpms = len(self.data['bpmnames'])
        respmat = self.data['respmat']
        sofb = self.devices['sofb']

        indcs = np.flip(np.argsort(np.abs(respmat[:, :-1]), axis=1), axis=1)
        chnames = list()
        cvnames = list()
        chstreng = respmat[idx, indcs[idx, :ncorrs]]
        cvstreng = respmat[idx+nbpms, indcs[idx+nbpms, :ncorrs]]
        for i in range(ncorrs):
            idxh = indcs[idx, i]
            idxv = indcs[idx+nbpms, i] - len(sofb.data.CH_NAMES)
            chnames.append(sofb.data.CH_NAMES[idxh])
            cvnames.append(sofb.data.CV_NAMES[idxv])
        return chnames, cvnames, chstreng, cvstreng

    def _get_orbit(self):
        sofb = self.devices['sofb']
        sofb.reset()
        sofb.wait(self.params.timeout_wait_sofb)
        return np.hstack([sofb.orbx, sofb.orby])

    def _do_bba(self):
        self.devices['sofb'].nr_points = self.params.sofb_nrpoints
        for bpm in self._bpms2dobba:
            if self._stopevt.is_set():
                print('stopped!')
                return
            self._dobba_single_bpm(bpm)
        print('finished!')

    def _dobba_single_bpm(self, bpmname):
        idx = self.data['bpmnames'].index(bpmname)
        quadname = self.data['quadnames'][idx]
        chname = self.data['chnames'][idx]
        cvname = self.data['cvnames'][idx]
        quad = self.devices[quadname]
        corrx = self.devices[chname]
        corry = self.devices[cvname]

        print('Doing BBA for BPM {:03d}: {:s}'.format(idx, bpmname))
        print('    turning quadrupole ' + quadname + ' On')
        quad.turnon(self.params.timeout_quad_turnon)
        if not quad.pwr_state:
            print('    error: quadrupole ' + quadname + ' is Off.')
            self._stopevt.set()
            print('    exiting...')
            return

        orb = self._get_orbit()
        xcen = self.data['scancenterx'][idx]
        ycen = self.data['scancentery'][idx]
        xpos = orb[idx]
        ypos = orb[idx+len(self.data['bpmnames'])]
        dkx0, dky0 = self.calc_orbcorr(bpmname, xcen-xpos, ycen-ypos)

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
        orb_ini, orb_pos, orb_neg = [], [], []
        orig_kickx = corrx.strength
        orig_kicky = corry.strength

        corrx.strength = orig_kickx + dkx0
        corry.strength = orig_kicky + dky0
        _time.sleep(self.params.wait_correctors)

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

            orb_ini.append(self._get_orbit())

            korig = quad.strength
            quad.strength = korig + deltakl/2
            _time.sleep(self.params.wait_quadrupole)

            orb_pos.append(self._get_orbit())

            quad.strength = korig - deltakl/2
            _time.sleep(self.params.wait_quadrupole)

            orb_neg.append(self._get_orbit())

            quad.strength = korig

            dorb = (orb_pos[-1]-orb_neg[-1])/deltakl
            xcen, ycen = self.calc_offset(bpmname, dorb)
            rms = np.sqrt(np.sum(dorb*dorb) / dorb.shape[0])
            print('    {0:02d}/{1:02d}:  '.format(i+1, npts), end='')
            print('x0 = {0:6.1f}  y0 = {1:6.1f}, '.format(
                xcen[0], ycen[0]), end='')
            print('rms = {0:8.1f} um'.format(rms))
            if self.params.bba_method == BBAParams.SCAN:
                xcen, ycen = dorbsx[i], dorbsy[i]
                dkx, dky = self.calc_orbcorr(bpmname, xcen, ycen)
                kickx = orig_kickx + dkx0 + dkx
                kicky = orig_kicky + dky0 + dky
            else:
                dkx, dky = self.calc_orbcorr(
                    bpmname,
                    -xcen[0]*self.params.dorbx_stretch,
                    -ycen[0]*self.params.dorby_stretch)
                kickx = corrx.strength + dkx
                kicky = corry.strength + dky

            if i < npts-1:
                kickx = min(max(-maxkick, kickx), maxkick)
                kicky = min(max(-maxkick, kicky), maxkick)
                corrx.strength = kickx
                corry.strength = kicky
                _time.sleep(self.params.wait_correctors)

        corrx.strength = orig_kickx
        corry.strength = orig_kicky

        if 'measure' not in self.data:
            self.data['measure'] = dict()
        if bpmname not in self.data['measure']:
            self.data['measure'][bpmname] = dict()

        self.data['measure'][bpmname]['corrxkicks'] = np.array(corrx_kicks)
        self.data['measure'][bpmname]['corrykicks'] = np.array(corry_kicks)
        self.data['measure'][bpmname]['orbini'] = np.array(orb_ini)
        self.data['measure'][bpmname]['orbpos'] = np.array(orb_pos)
        self.data['measure'][bpmname]['orbneg'] = np.array(orb_neg)

        print('    turning quadrupole ' + quadname + ' Off')
        quad.turnoff(self.params.timeout_quad_turnon)
        if quad.pwr_state:
            print('    error: quadrupole ' + quadname + ' is still On.')
            self._stopevt.set()
            print('    exiting...')
        print('')

    def calc_offset(self, bpmname, dorb):
        idx = self.data['bpmnames'].index(bpmname)
        respx = self.data['quadrespx'][:, idx]
        respy = self.data['quadrespy'][:, idx]
        isskew = '-QS' in self.data['quadnames'][idx]

        if len(dorb.shape) < 2:
            dorb = dorb[:, None]
        mat = np.array([respx, respy]).T
        res, *_ = np.linalg.lstsq(mat, dorb, rcond=None)
        x0, y0 = res
        if isskew:
            return -y0, x0
        return -x0, y0

    def calc_orbcorr(self, bpmname, x0, y0):
        idxh = self.data['bpmnames'].index(bpmname)
        idxv = idxh + len(self.data['bpmnames'])
        chname = self.data['chnames'][idxh]
        cvname = self.data['cvnames'][idxh]
        chidx = self.devices['sofb'].data.CH_NAMES.index(chname)
        cvidx = self.devices['sofb'].data.CV_NAMES.index(cvname)
        cvidx += len(self.devices['sofb'].data.CH_NAMES)

        respx = self.data['respmat'][idxh, chidx]
        respy = self.data['respmat'][idxv, cvidx]

        dkx = x0/respx
        dky = y0/respy
        return dkx, dky

    def process_data(self):
        analysis = dict()
        if 'analysis' not in self.data:
            self.data['analysis'] = dict()
        for bpm in self.data['measure']:
            self.data['analysis'][bpm] = self.process_data_single_bpm(bpm)

    def process_data_single_bpm(self, bpm):
        analysis = dict()
        idx = self.data['bpmnames'].index(bpm)
        nbpms = len(self.data['bpmnames'])
        orbini = self.data['measure'][bpm]['orbini']
        orbpos = self.data['measure'][bpm]['orbpos']
        orbneg = self.data['measure'][bpm]['orbneg']
        xpos = orbini[:, idx]
        ypos = orbini[:, idx+nbpms]
        dorb = orbpos - orbneg
        dorbx = dorb[:, :nbpms]
        dorby = dorb[:, nbpms:]
        if '-QS' in self.data['quadnames'][idx]:
            dorbx, dorby = dorby, dorbx
        analysis['xpos'] = xpos
        analysis['ypos'] = ypos

        px = np.polyfit(xpos, dorbx, deg=1)
        py = np.polyfit(ypos, dorby, deg=1)
        x0s = -px[1]/px[0]
        y0s = -py[1]/py[0]
        x0 = -np.dot(px[0], px[1]) / np.dot(px[0], px[0])
        y0 = -np.dot(py[0], py[1]) / np.dot(py[0], py[0])
        stdx0 = np.sqrt(np.dot(px[1], px[1]) / np.dot(px[0], px[0]) - x0*x0)
        stdy0 = np.sqrt(np.dot(py[1], py[1]) / np.dot(py[0], py[0]) - y0*y0)
        extrapx = not min(xpos) <= x0 <= max(xpos)
        extrapy = not min(ypos) <= y0 <= max(ypos)
        analysis['linear_fitting'] = dict()
        analysis['linear_fitting']['dorbx'] = dorbx
        analysis['linear_fitting']['dorby'] = dorby
        analysis['linear_fitting']['coeffsx'] = px
        analysis['linear_fitting']['coeffsy'] = py
        analysis['linear_fitting']['x0s'] = x0s
        analysis['linear_fitting']['y0s'] = y0s
        analysis['linear_fitting']['extrapolatedx'] = extrapx
        analysis['linear_fitting']['extrapolatedy'] = extrapy
        analysis['linear_fitting']['x0'] = x0
        analysis['linear_fitting']['y0'] = y0
        analysis['linear_fitting']['stdx0'] = stdx0
        analysis['linear_fitting']['stdy0'] = stdy0

        rmsx = np.sum(dorbx*dorbx, axis=1) / dorbx.shape[0]
        rmsy = np.sum(dorby*dorby, axis=1) / dorby.shape[0]
        if xpos.size > 3:
            px, covx = np.polyfit(xpos, rmsx, deg=2, cov=True)
            py, covy = np.polyfit(ypos, rmsy, deg=2, cov=True)
        else:
            px = np.polyfit(xpos, rmsx, deg=2, cov=False)
            py = np.polyfit(ypos, rmsy, deg=2, cov=False)
            covx = covy = np.zeros((3, 3))
        x0 = -px[1] / px[0] / 2
        y0 = -py[1] / py[0] / 2
        stdx0 = x0*np.sqrt(np.sum(np.diag(covx)[:2]/px[:2]/px[:2]))
        stdy0 = y0*np.sqrt(np.sum(np.diag(covy)[:2]/py[:2]/py[:2]))
        extrapx = not min(xpos) <= x0 <= max(xpos)
        extrapy = not min(ypos) <= y0 <= max(ypos)
        analysis['quadratic_fitting'] = dict()
        analysis['quadratic_fitting']['meansqrx'] = rmsx
        analysis['quadratic_fitting']['meansqry'] = rmsy
        analysis['quadratic_fitting']['coeffsx'] = px
        analysis['quadratic_fitting']['coeffsy'] = py
        analysis['quadratic_fitting']['extrapolatedx'] = extrapx
        analysis['quadratic_fitting']['extrapolatedy'] = extrapy
        analysis['quadratic_fitting']['x0'] = x0
        analysis['quadratic_fitting']['y0'] = y0
        analysis['quadratic_fitting']['stdx0'] = stdx0
        analysis['quadratic_fitting']['stdy0'] = stdy0

        dorb = dorb.T/self.params.quad_deltakl
        x0, y0 = self.calc_offset(bpm, dorb)
        extrapx = not min(xpos) <= x0 <= max(xpos)
        extrapy = not min(ypos) <= y0 <= max(ypos)
        analysis['model_estimative'] = dict()
        analysis['model_estimative']['x0s'] = xpos-x0
        analysis['model_estimative']['y0s'] = ypos-y0
        analysis['model_estimative']['extrapolatedx'] = extrapx
        analysis['model_estimative']['extrapolatedy'] = extrapy
        analysis['model_estimative']['x0'] = np.mean(xpos-x0)
        analysis['model_estimative']['y0'] = np.mean(ypos-y0)
        analysis['model_estimative']['stdx0'] = np.std(xpos-x0)
        analysis['model_estimative']['stdy0'] = np.std(ypos-y0)
        return analysis

    def get_bba_results(self, method='linear_fitting', error=False):
        data = self.data
        anl = data['analysis']
        bpms = data['bpmnames']
        bbax = np.zeros(len(bpms))
        bbay = np.zeros(len(bpms))
        if error:
            bbaxerr = np.zeros(len(bpms))
            bbayerr = np.zeros(len(bpms))
        for idx, bpm in enumerate(bpms):
            if bpm not in anl:
                continue
            res = anl[bpm][method]
            bbax[idx] = res['x0']
            bbay[idx] = res['y0']
            if error and 'stdx0' in res:
                bbaxerr[idx] = res['stdx0']
                bbayerr[idx] = res['stdy0']
        if error:
            return bbax, bbay, bbaxerr, bbayerr
        return bbax, bbay

    @staticmethod
    def get_default_quads(model, fam_data):
        quads_idx = _dcopy(fam_data['QN']['index'])
        qs_idx = [idx for idx in fam_data['QS']['index'] \
                  if not model[idx[0]].fam_name.startswith('FC2')]
        quads_idx.extend(qs_idx)
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
        dorbspos = np.linspace(dorbpos, 0, nrpts+1)[:-1]
        dorbsneg = np.linspace(-dorbneg, 0, nrpts+1)[:-1]
        dorbs = np.array([dorbsneg, dorbspos]).T.flatten()
        dorbs = np.hstack([dorbs, 0])
        return dorbs
