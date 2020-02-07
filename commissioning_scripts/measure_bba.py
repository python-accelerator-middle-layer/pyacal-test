"""Main module."""
import time as _time
from threading import Thread as _Thread, Event as _Event

from copy import deepcopy as _dcopy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gs
import matplotlib.cm as cm

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

    def check_correctors_range(self, do_print=True):
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
            doxpi = abs(min(xpos - (xcen+doxp), 0))
            doxni = abs(max(xpos - (xcen-doxn), 0))

            ypos = orb[idx+len(self.data['bpmnames'])]
            ycen = self.data['scancentery'][idx]
            doypi = abs(min(ypos - (ycen+doyp), 0))
            doyni = abs(max(ypos - (ycen-doyn), 0))

            ch = self.data['chnames'][idx]
            cv = self.data['cvnames'][idx]
            kickx = self.devices[ch].strength
            kicky = self.devices[cv].strength
            dkxp = self.params.max_corrstrength - kickx
            dkxn = self.params.max_corrstrength + kickx
            dkyp = self.params.max_corrstrength - kicky
            dkyn = self.params.max_corrstrength + kicky

            dorbxp = abs(dkxp/dkx) if dkx > 0 else abs(dkxn/dkx)
            dorbxn = abs(dkxn/dkx) if dkx > 0 else abs(dkxp/dkx)
            dorbyp = abs(dkyp/dky) if dky > 0 else abs(dkyn/dky)
            dorbyn = abs(dkyn/dky) if dky > 0 else abs(dkyp/dky)
            ok = dorbxp < doxpi
            if ok:
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

        dkx0, dky0, kickx0, kicky0 = self._go_to_initial_position(
            bpmname, idx, corrx, corry)

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
            dorbx = dorb[:len(self.data['bpmnames'])]
            dorby = dorb[len(self.data['bpmnames']):]
            xcen, ycen = self.calc_offset(bpmname, dorb)
            rmsx = np.sqrt(np.sum(dorbx*dorbx) / dorbx.shape[0])
            rmsy = np.sqrt(np.sum(dorby*dorby) / dorby.shape[0])
            print('    {0:02d}/{1:02d}:  '.format(i+1, npts), end='')
            print('x0 = {0:6.1f}  y0 = {1:6.1f}, '.format(
                xcen[0], ycen[0]), end='')
            print('rmsx = {:8.1f} rmsy = {:8.1f} um'.format(rmsx, rmsy))
            if self.params.bba_method == BBAParams.SCAN:
                xcen, ycen = dorbsx[i], dorbsy[i]
                dkx, dky = self.calc_orbcorr(bpmname, xcen, ycen)
                kickx = kickx0 + dkx0 + dkx
                kicky = kicky0 + dky0 + dky
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

        corrx.strength = kickx0
        corry.strength = kicky0

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

    def _go_to_initial_position(self, bpmname, idx, corrx, corry):
        print('    sending orbit to initial position ...', end='')
        kickx0 = corrx.strength
        kicky0 = corry.strength
        xcen = self.data['scancenterx'][idx]
        ycen = self.data['scancentery'][idx]
        for i in range(5):
            orb = self._get_orbit()
            xpos = orb[idx]
            ypos = orb[idx+len(self.data['bpmnames'])]
            dorbx = xpos - xcen
            dorby = ypos - ycen
            fmet = max(abs(dorbx), abs(dorby))
            if fmet < 20:
                print( 'Ok! it took {:d} iterations'.format(i))
                break
            dkx, dky = self.calc_orbcorr(bpmname, -dorbx, -dorby)
            corrx.strength += dkx
            corry.strength += dky
            _time.sleep(self.params.wait_correctors)
        else:
            print('NOT Ok!: dorb is {:.1f} um'.format(fmet))

        dkx0 = corrx.strength - kickx0
        dky0 = corry.strength - kicky0
        return dkx0, dky0, kickx0, kicky0

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

    def process_data(self, nbpms_linfit=None):
        analysis = dict()
        if 'analysis' not in self.data:
            self.data['analysis'] = dict()
        for bpm in self.data['measure']:
            self.data['analysis'][bpm] = self.process_data_single_bpm(
                bpm, nbpms_linfit=nbpms_linfit)

    def process_data_single_bpm(self, bpm, nbpms_linfit=None):
        analysis = dict()
        idx = self.data['bpmnames'].index(bpm)
        nbpms = len(self.data['bpmnames'])
        orbini = self.data['measure'][bpm]['orbini']
        orbpos = self.data['measure'][bpm]['orbpos']
        orbneg = self.data['measure'][bpm]['orbneg']
        corrx = self.data['measure'][bpm]['corrxkicks']
        corry = self.data['measure'][bpm]['corrykicks']

        xpos = orbini[:, idx]
        ypos = orbini[:, idx+nbpms]
        dorb = orbpos - orbneg
        dorbx = dorb[:, :nbpms]
        dorby = dorb[:, nbpms:]
        if '-QS' in self.data['quadnames'][idx]:
            dorbx, dorby = dorby, dorbx
        analysis['xpos'] = xpos
        analysis['ypos'] = ypos

        respmx = np.diff(xpos) / np.diff(corrx)
        respmy = np.diff(ypos) / np.diff(corry)
        analysis['respmx'] = respmx
        analysis['respmy'] = respmy

        px = np.polyfit(xpos, dorbx, deg=1)
        py = np.polyfit(ypos, dorby, deg=1)
        sidx = np.argsort(np.abs(px[0]))
        sidy = np.argsort(np.abs(py[0]))
        # sidx = sidx[]
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
        stdx0 = np.abs(x0)*np.sqrt(np.sum(np.diag(covx)[:2]/px[:2]/px[:2]))
        stdy0 = np.abs(y0)*np.sqrt(np.sum(np.diag(covy)[:2]/py[:2]/py[:2]))
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
        x0s, y0s = self.calc_offset(bpm, dorb)
        extrapx = not min(xpos) <= np.mean(x0s) <= max(xpos)
        extrapy = not min(ypos) <= np.mean(y0s) <= max(ypos)
        analysis['model_estimative'] = dict()
        analysis['model_estimative']['x0s'] = xpos-x0s
        analysis['model_estimative']['y0s'] = ypos-y0s
        analysis['model_estimative']['extrapolatedx'] = extrapx
        analysis['model_estimative']['extrapolatedy'] = extrapy
        analysis['model_estimative']['x0'] = np.mean(xpos-x0s)
        analysis['model_estimative']['y0'] = np.mean(ypos-y0s)
        analysis['model_estimative']['stdx0'] = np.std(xpos-x0s)
        analysis['model_estimative']['stdy0'] = np.std(ypos-y0s)
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

    @staticmethod
    def list_bpm_subsections(bpms):
        subinst = [bpm[5:7]+bpm[14:] for bpm in bpms]
        sec = [bpm[2:4] for bpm in bpms]
        subsecs = {typ: [] for typ in subinst}
        for sub, sec, bpm in zip(subinst, sec, bpms):
            subsecs[sub].append(bpm)
        return subsecs

    def combine_bbas(self, bbalist):
        items = [
            'chnames', 'cvnames', 'quadnames', 'scancenterx', 'scancentery',
            'quadrespx', 'quadrespy']
        dobba = DoBBA()
        dobba.params = self.params
        dobba.data = _dcopy(self.data)
        for bba in bbalist:
            for bpm, data in bba.data['measure'].items():
                dobba.data['measure'][bpm] = _dcopy(data)
                idx = dobba.data['bpmnames'].index(bpm)
                for item in items:
                    dobba.data[item][idx] = bba.data[item][idx]
        return dobba

    def filter_problems(self, maxstd=100, maxorb=9, maxrms=100,
                        method='lin quad', probtype='std', pln='xy'):
        bpms = []
        islin = 'lin' in method
        isquad = 'quad' in method
        for bpm in self.data['bpmnames']:
            anal = self.data['analysis'].get(bpm)
            if not anal:
                continue
            concx = anal['quadratic_fitting']['coeffsx'][0]
            concy = anal['quadratic_fitting']['coeffsy'][0]
            probc = False
            if 'x' in pln:
                probc |= concx < 0
            if 'y' in pln:
                probc |= concy < 0

            rmsx = anal['quadratic_fitting']['meansqrx']
            rmsy = anal['quadratic_fitting']['meansqry']
            probmaxrms = False
            if 'x' in pln:
                probmaxrms |= np.max(rmsx) < maxrms
            if 'y' in pln:
                probmaxrms |= np.max(rmsy) < maxrms

            extqx = isquad and anal['quadratic_fitting']['extrapolatedx']
            extqy = isquad and anal['quadratic_fitting']['extrapolatedy']
            extlx = islin and anal['linear_fitting']['extrapolatedx']
            extly = islin and anal['linear_fitting']['extrapolatedy']
            probe = False
            if 'x' in pln:
                probe |= extqx or extlx
            if 'y' in pln:
                probe |= extqy or extly

            stdqx = isquad and anal['quadratic_fitting']['stdx0'] > maxstd
            stdqy = isquad and anal['quadratic_fitting']['stdy0'] > maxstd
            stdlx = islin and anal['linear_fitting']['stdx0'] > maxstd
            stdly = islin and anal['linear_fitting']['stdy0'] > maxstd
            probs = False
            if 'x' in pln:
                probs |= stdqx or stdlx
            if 'y' in pln:
                probs |= stdqy or stdly

            dorbx = anal['linear_fitting']['dorbx']
            dorby = anal['linear_fitting']['dorby']
            probmaxorb = False
            if 'x' in pln:
                probmaxorb |= np.max(np.abs(dorbx)) < maxorb
            if 'y' in pln:
                probmaxorb |= np.max(np.abs(dorby)) < maxorb

            prob = False
            if 'std'in probtype:
                prob |= probs
            if 'ext' in probtype:
                prob |= probe
            if 'conc'in probtype:
                prob |= probc
            if 'rms'in probtype:
                prob |= probmaxrms
            if 'orb'in probtype:
                prob |= probmaxorb
            if 'all' in probtype:
                prob = probs and probe and probc and probmaxrms and probmaxorb
            if 'any' in probtype:
                prob = probs or probe or probc or probmaxrms or probmaxorb
            if prob:
                bpms.append(bpm)
        return bpms

    def bpm_summary(self, bpm, save=False):
        f  = plt.figure(figsize=(9.5, 9))
        gs = mpl_gs.GridSpec(3, 2)
        gs.update(left=0.11, right=0.98, bottom=0.1, top=0.9, hspace=0.35, wspace=0.35)

        f.suptitle(bpm, fontsize=20)

        alx = plt.subplot(gs[0, 0])
        aly = plt.subplot(gs[0, 1])
        aqx = plt.subplot(gs[1, 0])
        aqy = plt.subplot(gs[1, 1])
        adt = plt.subplot(gs[2, 0])
        axy = plt.subplot(gs[2, 1])

        allax = [alx, aly, aqx, aqy, axy]

        for ax in allax:
            ax.grid(True)

        data = self.data['analysis'][bpm]
        xpos = data['xpos']
        ypos = data['ypos']
        sxpos = np.sort(xpos)
        sypos = np.sort(ypos)
        respmx = data['respmx']
        respmy = data['respmy']

        adt.set_frame_on(False)
        adt.axes.get_yaxis().set_visible(False)
        adt.axes.get_xaxis().set_visible(False)
        idx = self.data['bpmnames'].index(bpm)
        xini = self.data['scancenterx'][idx]
        yini = self.data['scancentery'][idx]
        tmp = '{:5s}: {:15s}'
        tmp2 = tmp + ' (dKL={:.4f} 1/m)'
        adt.text(0, 0, 'Initial Search values = ({:.2f}, {:.2f})'.format(xini, yini), fontsize=10)
        adt.text(0, 1, tmp2.format('Quad', self.data['quadnames'][idx], self.params.quad_deltakl), fontsize=10)

        tmp2 = '      RM0={:5.2f},  RM={:5.2f}+-{:5.2f}  m/rad'
        dkx, dky = self.calc_orbcorr(bpm, 1, 1)
        adt.text(0, 3, tmp.format('CH', self.data['chnames'][idx]), fontsize=10)
        adt.text(0, 2, tmp2.format(1/dkx, np.mean(respmx), np.std(respmx)), fontsize=10)

        adt.text(0, 5, tmp.format('CV', self.data['cvnames'][idx]), fontsize=10)
        adt.text(0, 4, tmp2.format(1/dky, np.mean(respmy), np.std(respmy)), fontsize=10)
        adt.set_xlim([0,8])
        adt.set_ylim([0,8])

        rmsx = data['quadratic_fitting']['meansqrx']
        rmsy = data['quadratic_fitting']['meansqry']
        px = data['quadratic_fitting']['coeffsx']
        py = data['quadratic_fitting']['coeffsy']
        x0 = data['quadratic_fitting']['x0']
        y0 = data['quadratic_fitting']['y0']
        stdx0 = data['quadratic_fitting']['stdx0']
        stdy0 = data['quadratic_fitting']['stdy0']
        fitx = np.polyval(px, sxpos)
        fity = np.polyval(py, sypos)
        fitx0 = np.polyval(px, x0)
        fity0 = np.polyval(py, y0)

        aqx.plot(xpos, rmsx, 'bo')
        aqx.plot(sxpos, fitx, 'b')
        aqx.errorbar(x0, fitx0, xerr=stdx0, fmt='kx', markersize=20)
        aqy.plot(ypos, rmsy, 'ro')
        aqy.plot(sypos, fity, 'r')
        aqy.errorbar(y0, fity0, xerr=stdy0, fmt='kx', markersize=20)
        axy.errorbar(x0, y0, xerr=stdx0, yerr=stdy0, fmt='gx', markersize=20, label='parabollic')

        dorbx = data['linear_fitting']['dorbx']
        dorby = data['linear_fitting']['dorby']
        x0 = data['linear_fitting']['x0']
        y0 = data['linear_fitting']['y0']
        stdx0 = data['linear_fitting']['stdx0']
        stdy0 = data['linear_fitting']['stdy0']
        x0s = data['linear_fitting']['x0s']
        y0s = data['linear_fitting']['y0s']
        px = data['linear_fitting']['coeffsx']
        py = data['linear_fitting']['coeffsy']
        sidx = np.argsort(np.abs(px[0]))
        sidy = np.argsort(np.abs(py[0]))
        pvx, pvy = [], []
        npts = 6
        for ii in range(npts):
            pvx.append(np.polyval(px[:, sidx[-ii-1]], sxpos))
            pvy.append(np.polyval(py[:, sidy[-ii-1]], sypos))
        pvx, pvy = np.array(pvx), np.array(pvy)
        alx.plot(xpos, dorbx[:, sidx[-npts:]], 'b.')
        alx.plot(sxpos, pvx.T, 'b', linewidth=1)
        alx.errorbar(x0, 0, xerr=stdx0, fmt='kx', markersize=20)
        aly.plot(ypos, dorby[:, sidy[-npts:]], 'r.')
        aly.plot(sypos, pvy.T, 'r', linewidth=1)
        aly.errorbar(y0, 0, xerr=stdy0, fmt='kx', markersize=20)
        axy.errorbar(x0, y0, xerr=stdx0, yerr=stdy0, fmt='mx', markersize=20, label='linear')

        axy.legend(loc='best', fontsize='x-small')
        axy.set_xlabel('X0 [$\mu$m]')
        axy.set_ylabel('Y0 [$\mu$m]')
        alx.set_xlabel('X [$\mu$m]')
        alx.set_ylabel('$\Delta$ COD [$\mu$m]')
        aly.set_xlabel('Y [$\mu$m]')
        aly.set_ylabel('$\Delta$ COD [$\mu$m]')
        aqx.set_xlabel('X [$\mu$m]')
        aqx.set_ylabel('RMS COD [$\mu$m$^2$]')
        aqy.set_xlabel('Y [$\mu$m]')
        aqy.set_ylabel('RMS COD [$\mu$m$^2$]')

        if save:
            f.savefig(bpm+'.svg')
            plt.close()
        else:
            f.show()

    def make_quadfit_figures(self, bpms=None, fname='', title=''):
        f  = plt.figure(figsize=(9.5, 9))
        gs = mpl_gs.GridSpec(2, 1)
        gs.update(left=0.1, right=0.78, bottom=0.15, top=0.9, hspace=0.5, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = plt.subplot(gs[0, 0])
        ayy = plt.subplot(gs[1, 0])

        bpms = bpms or self.data['bpmnames']
        indcs = np.array([self.data['bpmnames'].index(bpm) for bpm in bpms])
        colors = cm.brg(np.linspace(0, 1, len(bpms)))
        for i, bpm in enumerate(bpms):
            if bpm not in self.data['analysis']:
                print('Data not found for ', bpm)
                continue
            data = self.data['analysis'][bpm]
            rmsx = data['quadratic_fitting']['meansqrx']
            rmsy = data['quadratic_fitting']['meansqry']

            px = data['quadratic_fitting']['coeffsx']
            py = data['quadratic_fitting']['coeffsy']

            x0 = data['quadratic_fitting']['x0']
            y0 = data['quadratic_fitting']['y0']

            sxpos = np.sort(data['xpos'])
            sypos = np.sort(data['ypos'])
            fitx = np.polyval(px, sxpos)
            fity = np.polyval(py, sypos)

            axx.plot(data['xpos']-x0, rmsx, 'o', color=colors[i], label=bpm)
            axx.plot(sxpos-x0, fitx, color=colors[i])
            ayy.plot(data['ypos']-y0, rmsy, 'o', color=colors[i], label=bpm)
            ayy.plot(sypos-y0, fity, color=colors[i])

        axx.legend(bbox_to_anchor=(1.0, 1.1), fontsize='xx-small')
        axx.grid(True)
        ayy.grid(True)
        axx.set_xlabel('X - X0 [um]')
        axx.set_ylabel('$\Delta$ COD')
        ayy.set_xlabel('Y - Y0 [um]')
        ayy.set_ylabel('$\Delta$ COD')
        if fname:
            f.savefig(fname+'.svg')
            plt.close()
        else:
            f.show()

    def make_modelestimate_figures(self, bpms=None):
        f  = plt.figure(figsize=(9.5, 5))
        gs = mpl_gs.GridSpec(2, 1)
        gs.update(left=0.1, right=0.78, bottom=0.15, top=0.9, hspace=0.5, wspace=0.35)

        axx = plt.subplot(gs[0, 0])
        axy = plt.subplot(gs[1, 0])

        bpms = bpms or self.data['bpmnames']
        indcs = np.array([self.data['bpmnames'].index(bpm) for bpm in bpms])
        colors = cm.brg(np.linspace(0, 1, len(bpms)))
        for i, bpm in enumerate(bpms):
            if bpm not in self.data['analysis']:
                print('Data not found for ', bpm)
                continue
            data = self.data['analysis'][bpm]

            x0s = data['model_estimative']['x0s']
            y0s = data['model_estimative']['y0s']
            x0 = data['model_estimative']['x0']
            y0 = data['model_estimative']['y0']
            stdx0 = data['model_estimative']['stdx0']
            stdy0 = data['model_estimative']['stdy0']

            xpos = data['xpos']
            ypos = data['ypos']
            sxpos = np.sort(xpos)
            sypos = np.sort(ypos)

            axx.plot(xpos-x0, x0s, 'o', color=colors[i], label=bpm)
            axy.plot(ypos-y0, y0s, 'o', color=colors[i], label=bpm)

        axx.legend(bbox_to_anchor=(1.0, 1.1), fontsize='xx-small')
        axx.grid(True)
        axy.grid(True)
        f.show()

    def make_linfit_figures(self, bpms=None, fname='', title=''):
        f  = plt.figure(figsize=(9.5, 9))
        gs = mpl_gs.GridSpec(2, 1)
        gs.update(left=0.1, right=0.78, bottom=0.15, top=0.9, hspace=0.5, wspace=0.35)

        axx = plt.subplot(gs[0, 0])
        axy = plt.subplot(gs[1, 0])

        bpms = bpms or self.data['bpmnames']
        indcs = np.array([self.data['bpmnames'].index(bpm) for bpm in bpms])
        colors = cm.brg(np.linspace(0, 1, len(bpms)))
        for i, bpm in enumerate(bpms):
            if bpm not in self.data['analysis']:
                print('Data not found for ', bpm)
                continue
            data = self.data['analysis'][bpm]
            x0 = data['linear_fitting']['x0']
            y0 = data['linear_fitting']['y0']
            stdx0 = data['linear_fitting']['stdx0']
            stdy0 = data['linear_fitting']['stdy0']

            x0s = data['linear_fitting']['x0s']
            y0s = data['linear_fitting']['y0s']
            px = data['linear_fitting']['coeffsx']
            py = data['linear_fitting']['coeffsy']

            sidx = np.argsort(np.abs(px[0]))
            sidy = np.argsort(np.abs(py[0]))

            xpos = data['xpos']
            ypos = data['ypos']
            sxpos = np.sort(xpos)
            sypos = np.sort(ypos)

            pvx, pvy = [], []
            for ii in range(3):
                pvx.append(np.polyval(px[:, sidx[ii]], sxpos))
                pvy.append(np.polyval(py[:, sidy[ii]], sypos))
            pvx, pvy = np.array(pvx), np.array(pvy)

            axx.plot(sxpos, pvx.T, color=colors[i])
            axx.plot(x0, 0, 'x', markersize=20, color=colors[i], label=bpm)
            axy.plot(sypos, pvy.T, color=colors[i])
            axy.plot(y0, 0, 'x', markersize=20, color=colors[i], label=bpm)

        axx.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='xx-small')
        axx.grid(True)
        axy.grid(True)
        f.show()

    def make_figures_compare_methods(self, bpms_ok=None, bpms_nok=None,
                                     fname='', title=''):
        f  = plt.figure(figsize=(9.2, 9))
        gs = mpl_gs.GridSpec(3, 2)
        gs.update(left=0.15, right=0.98, bottom=0.19, top=0.9, hspace=0, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = plt.subplot(gs[0, :])
        ayy = plt.subplot(gs[1, :], sharex=axx)
        axy = plt.subplot(gs[2, 0])
        pos = list(axy.get_position().bounds)
        pos[1] += -0.05
        axy.set_position(pos)

        bpms_ok = bpms_ok or self.data['bpmnames']
        bpms_nok = bpms_nok or []
        idc_ok = np.array([self.data['bpmnames'].index(bpm) for bpm in bpms_ok])
        idc_nok = np.array([self.data['bpmnames'].index(bpm) for bpm in bpms_nok])

        labels = ['linear', 'quadratic']
        cors = cm.brg(np.linspace(0, 1, 3))

        x0l, y0l, stdx0l, stdy0l = self.get_bba_results(method='linear_fitting', error=True)
        x0q, y0q, stdx0q, stdy0q = self.get_bba_results(method='quadratic_fitting', error=True)
        minx = np.min([x0q, x0l])*1.1
        maxx = np.max([x0q, x0l])*1.1
        miny = np.min([y0q, y0l])*1.1
        maxy = np.max([y0q, y0l])*1.1

        axx.errorbar(idc_ok, x0l[idc_ok], yerr=stdx0l[idc_ok], fmt='o', color=cors[0])
        axx.errorbar(idc_ok, x0q[idc_ok], yerr=stdx0q[idc_ok], fmt='o', color=cors[1])
        ayy.errorbar(idc_ok, y0l[idc_ok], yerr=stdy0l[idc_ok], fmt='o', color=cors[0], label=labels[0])
        ayy.errorbar(idc_ok, y0q[idc_ok], yerr=stdy0q[idc_ok], fmt='o', color=cors[1], label=labels[1])
        axy.errorbar(
            x0l[idc_ok], y0l[idc_ok], xerr=stdx0l[idc_ok], yerr=stdy0l[idc_ok], fmt='o', color=cors[0],
            label='Reliable')
        axy.errorbar(
            x0q[idc_ok], y0q[idc_ok], xerr=stdx0q[idc_ok], yerr=stdy0q[idc_ok], fmt='o', color=cors[1],
            label='Reliable')

        axx.errorbar(idc_nok, x0l[idc_nok], yerr=stdx0l[idc_nok], fmt='x', color=cors[0])
        axx.errorbar(idc_nok, x0q[idc_nok], yerr=stdx0q[idc_nok], fmt='x', color=cors[1])
        ayy.errorbar(idc_nok, y0l[idc_nok], yerr=stdy0l[idc_nok], fmt='x', color=cors[0], label=labels[0])
        ayy.errorbar(idc_nok, y0q[idc_nok], yerr=stdy0q[idc_nok], fmt='x', color=cors[1], label=labels[1])
        axy.errorbar(
            x0l[idc_nok], y0l[idc_nok], xerr=stdx0l[idc_nok], yerr=stdy0l[idc_nok], fmt='x', color=cors[0],
            label='Not Reliable')
        axy.errorbar(
            x0q[idc_nok], y0q[idc_nok], xerr=stdx0q[idc_nok], yerr=stdy0q[idc_nok], fmt='x', color=cors[1],
            label='Not Reliable')

        ayy.legend(loc='upper right', bbox_to_anchor=(0.6, -0.4), fontsize='xx-small')
        axy.legend(loc='upper right', bbox_to_anchor=(1.8, 0.4), fontsize='xx-small')
        axx.grid(True)
        ayy.grid(True)
        axx.set_ylabel('X0 [um]')
        ayy.set_ylabel('Y0 [um]')
        axx.set_ylim([minx, maxx])
        ayy.set_ylim([miny, maxy])

        axy.grid(True)
        axy.set_xlabel('X0 [um]')
        axy.set_ylabel('Y0 [um]')
        axy.set_ylim([minx, maxx])
        axy.set_ylim([miny, maxy])

        if fname:
            f.savefig(fname+'.svg')
            plt.close()
        else:
            f.show()

    @staticmethod
    def make_figures_compare_bbas(bbalist, method='linear_fitting', labels=[],
                                  bpms_ok=None, bpms_nok=None, fname='',
                                  title=''):
        f  = plt.figure(figsize=(9.2, 9))
        gs = mpl_gs.GridSpec(3, 2)
        gs.update(left=0.12, right=0.98, bottom=0.13, top=0.9, hspace=0, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = plt.subplot(gs[0, :])
        ayy = plt.subplot(gs[1, :], sharex=axx)
        axy = plt.subplot(gs[2, 0])
        pos = list(axy.get_position().bounds)
        pos[1] += -0.05
        axy.set_position(pos)

        bpms_ok = bpms_ok or bbalist[0].data['bpmnames']
        bpms_nok = bpms_nok or []
        idc_ok = np.array(
            [bbalist[0].data['bpmnames'].index(bpm) for bpm in bpms_ok],
            dtype=int)
        idc_nok = np.array(
            [bbalist[0].data['bpmnames'].index(bpm) for bpm in bpms_nok],
            dtype=int)

        if not labels:
            labels = [str(i) for i in range(len(bbalist))]
        cors = cm.brg(np.linspace(0, 1, len(bbalist)))

        minx = miny = np.inf
        maxx = maxy = -np.inf
        for i, dobba in enumerate(bbalist):
            x0l, y0l, stdx0l, stdy0l = dobba.get_bba_results(method=method, error=True)
            minx = np.min(np.hstack([minx, x0l.flatten()]))*1.1
            maxx = np.max(np.hstack([maxx, x0l.flatten()]))*1.1
            miny = np.min(np.hstack([miny, y0l.flatten()]))*1.1
            maxy = np.max(np.hstack([maxy, y0l.flatten()]))*1.1

            axx.errorbar(idc_ok, x0l[idc_ok], yerr=stdx0l[idc_ok], fmt='o', color=cors[i])
            ayy.errorbar(idc_ok, y0l[idc_ok], yerr=stdy0l[idc_ok], fmt='o', color=cors[i], label=labels[i])
            axy.errorbar(
                x0l[idc_ok], y0l[idc_ok], xerr=stdx0l[idc_ok], yerr=stdy0l[idc_ok], fmt='o', color=cors[i],
                label='Reliable')

            axx.errorbar(idc_nok, x0l[idc_nok], yerr=stdx0l[idc_nok], fmt='x', color=cors[i])
            ayy.errorbar(idc_nok, y0l[idc_nok], yerr=stdy0l[idc_nok], fmt='x', color=cors[i], label=labels[i])
            axy.errorbar(
                x0l[idc_nok], y0l[idc_nok], xerr=stdx0l[idc_nok], yerr=stdy0l[idc_nok], fmt='x', color=cors[i],
                label='Not Reliable')


        ayy.legend(loc='upper right', bbox_to_anchor=(0.6, -0.4), fontsize='xx-small')
        axy.legend(loc='upper right', bbox_to_anchor=(1.8, 0.2), fontsize='xx-small')
        axx.grid(True)
        ayy.grid(True)
        axx.set_ylabel('X0 [um]')
        ayy.set_ylabel('Y0 [um]')
        axx.set_ylim([minx, maxx])
        ayy.set_ylim([miny, maxy])

        axy.grid(True)
        axy.set_xlabel('X0 [um]')
        axy.set_ylabel('Y0 [um]')
        axy.set_ylim([minx, maxx])
        axy.set_ylim([miny, maxy])

        if fname:
            f.savefig(fname+'.svg')
            plt.close()
        else:
            f.show()
