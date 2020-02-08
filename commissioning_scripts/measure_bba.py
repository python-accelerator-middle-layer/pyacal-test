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
        self._bpms2dobba = list()
        self.devices['sofb'] = SOFB('SI')
        for chname in self.devices['SOFB'].data.CH_NAMES:
            self.devices[chname] = Corrector(chname)
        for cvname in self.devices['SOFB'].data.CV_NAMES:
            self.devices[cvname] = Corrector(cvname)
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
            if qname and qname not in self.devices:
                self.devices[qname] = TrimQuad(qname)

    def initialize_data(self, model):
        respmat_calc = OrbRespmat(model, 'SI')
        bpmnames, quadnames, quadsidx = DoBBA.get_default_quads(
            model, respmat_calc.fam_data)
        sofb = self.devices['sofb']
        self.data['bpmnames'] = bpmnames
        self.data['quadnames'] = quadnames
        self.data['scancenterx'] = np.zeros(len(bpmnames))
        self.data['scancentery'] = np.zeros(len(bpmnames))
        self.connect_to_objects()

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
        quad = self.devices[quadname]

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

        orb_ini, orb_pos, orb_neg = [], [], []

        dkx0, dky0, kickx0, kicky0 = self._go_to_initial_position(
            bpmname, idx)

        if self.params.bba_method == BBAParams.SCAN:
            npts = 2*(nrsteps//2) + 1
        else:
            npts = nrsteps
        for i in range(npts):
            if self._stopevt.is_set():
                print('   exiting...')
                break
            orb_ini.append(self._get_orbit())

            korig = quad.strength
            quad.strength = korig + deltakl/2
            _time.sleep(self.params.wait_quadrupole)

            orb_pos.append(self._get_orbit())

            quad.strength = korig - deltakl/2
            _time.sleep(self.params.wait_quadrupole)

            orb_neg.append(self._get_orbit())

            quad.strength = korig

            dorb = (orb_pos[-1]-orb_neg[-1])
            dorbx = dorb[:len(self.data['bpmnames'])]
            dorby = dorb[len(self.data['bpmnames']):]
            rmsx = np.sqrt(np.sum(dorbx*dorbx) / dorbx.shape[0])
            rmsy = np.sqrt(np.sum(dorby*dorby) / dorby.shape[0])
            print('    {0:02d}/{1:02d}:  '.format(i+1, npts), end='')
            print('rmsx = {:8.1f} rmsy = {:8.1f} um'.format(rmsx, rmsy))
            if self.params.bba_method == BBAParams.SCAN:
                xcen, ycen = dorbsx[i], dorbsy[i]
                dkx, dky = self.calc_orbcorr(bpmname, xcen, ycen)
            else:
                dkx, dky = self.calc_orbcorr(
                    bpmname,
                    -xcen[0]*self.params.dorbx_stretch,
                    -ycen[0]*self.params.dorby_stretch)

                _time.sleep(self.params.wait_correctors)


        if 'measure' not in self.data:
            self.data['measure'] = dict()
        if bpmname not in self.data['measure']:
            self.data['measure'][bpmname] = dict()

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
            _time.sleep(self.params.wait_correctors)
        else:
            print('NOT Ok!: dorb is {:.1f} um'.format(fmet))

        return dkx0, dky0, kickx0, kicky0

    def calc_orbcorr(self, bpmname, x0, y0):
        idxh = self.data['bpmnames'].index(bpmname)
        idxv = idxh + len(self.data['bpmnames'])
        chidx = self.devices['sofb'].data.CH_NAMES.index(chname)
        cvidx = self.devices['sofb'].data.CV_NAMES.index(cvname)
        cvidx += len(self.devices['sofb'].data.CH_NAMES)

        return dkx, dky

    def process_data(self, nbpms_linfit=None, thres=None):
        analysis = dict()
        if 'analysis' not in self.data:
            self.data['analysis'] = dict()
        for bpm in self.data['measure']:
            self.data['analysis'][bpm] = self.process_data_single_bpm(
                bpm, nbpms_linfit=nbpms_linfit, thres=thres)

    def process_data_single_bpm(self, bpm, nbpms_linfit=None, thres=None):
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

        nbpms_linfit = nbpms_linfit or len(self.data['bpmnames'])
        sidx = np.argsort(np.abs(px[0]))
        sidy = np.argsort(np.abs(py[0]))
        sidx = sidx[-nbpms_linfit:][::-1]
        sidy = sidy[-nbpms_linfit:][::-1]
        pxc = px[:, sidx]
        pyc = py[:, sidy]
        if thres:
            ax2 = pxc[0]*pxc[0]
            ay2 = pyc[0]*pyc[0]
            ax2 /= ax2[0]
            ay2 /= ay2[0]
            nx = np.sum(ax2 > thres)
            ny = np.sum(ay2 > thres)
            pxc = pxc[:, :nx]
            pyc = pyc[:, :ny]

        x0s = -pxc[1]/pxc[0]
        y0s = -pyc[1]/pyc[0]
        x0 = -np.dot(pxc[0], pxc[1]) / np.dot(pxc[0], pxc[0])
        y0 = -np.dot(pyc[0], pyc[1]) / np.dot(pyc[0], pyc[0])
        stdx0 = np.sqrt(
            np.dot(pxc[1], pxc[1]) / np.dot(pxc[0], pxc[0]) - x0*x0)
        stdy0 = np.sqrt(
            np.dot(pyc[1], pyc[1]) / np.dot(pyc[0], pyc[0]) - y0*y0)
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

    def get_analysis_properties(self, propty, method='linear_fitting'):
        data = self.data
        anl = data['analysis']
        bpms = data['bpmnames']
        prop = [[], ] * len(bpms)
        for idx, bpm in enumerate(bpms):
            if bpm not in anl:
                continue
            res = anl[bpm][method]
            prop[idx] = res[propty]
        return prop

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
        items = ['quadnames', 'scancenterx', 'scancentery']
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

    # ##### Make Figures #####
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

    def make_figures_compare_methods(self, bpmsok=None, bpmsnok=None,
                                     xlim=0, ylim=0, fname='', title=''):
        f  = plt.figure(figsize=(9.2, 9))
        gs = mpl_gs.GridSpec(3, 3)
        gs.update(
            left=0.15, right=0.98, bottom=0.12, top=0.95,
            hspace=0.01, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = plt.subplot(gs[0, :])
        ayy = plt.subplot(gs[1, :], sharex=axx)
        axy = plt.subplot(gs[2, :2])
        pos = list(axy.get_position().bounds)
        pos[1] += -0.05
        axy.set_position(pos)

        bpmsok = bpmsok or self.data['bpmnames']
        bpmsnok = bpmsnok or []
        iok = np.array([self.data['bpmnames'].index(bpm) for bpm in bpmsok])
        inok = np.array([self.data['bpmnames'].index(bpm) for bpm in bpmsnok])

        labels = ['linear', 'quadratic']
        cors = cm.brg(np.linspace(0, 1, 3))

        x0l, y0l, stdx0l, stdy0l = self.get_bba_results(
            method='linear_fitting', error=True)
        x0q, y0q, stdx0q, stdy0q = self.get_bba_results(
            method='quadratic_fitting', error=True)
        minx = -xlim or np.min([x0q, x0l])*1.1
        maxx = xlim or np.max([x0q, x0l])*1.1
        miny = -ylim or np.min([y0q, y0l])*1.1
        maxy = ylim or np.max([y0q, y0l])*1.1

        axx.errorbar(
            iok, x0l[iok], yerr=stdx0l[iok], fmt='o', color=cors[0])
        axx.errorbar(
            iok, x0q[iok], yerr=stdx0q[iok], fmt='o', color=cors[1])
        ayy.errorbar(
            iok, y0l[iok], yerr=stdy0l[iok], fmt='o', color=cors[0],
            label=labels[0])
        ayy.errorbar(
            iok, y0q[iok], yerr=stdy0q[iok], fmt='o', color=cors[1],
            label=labels[1])
        axy.errorbar(
            x0l[iok], y0l[iok], xerr=stdx0l[iok], yerr=stdy0l[iok],
            fmt='o', color=cors[0], label='Reliable')
        axy.errorbar(
            x0q[iok], y0q[iok], xerr=stdx0q[iok], yerr=stdy0q[iok],
            fmt='o', color=cors[1])

        if inok:
            axx.errorbar(
                inok, x0l[inok], yerr=stdx0l[inok], fmt='x', color=cors[0])
            axx.errorbar(
                inok, x0q[inok], yerr=stdx0q[inok], fmt='x', color=cors[1])
            ayy.errorbar(
                inok, y0l[inok], yerr=stdy0l[inok], fmt='x', color=cors[0],
                label=labels[0])
            ayy.errorbar(
                inok, y0q[inok], yerr=stdy0q[inok], fmt='x', color=cors[1],
                label=labels[1])
            axy.errorbar(
                x0l[inok], y0l[inok], xerr=stdx0l[inok], yerr=stdy0l[inok],
                fmt='x', color=cors[0], label='Not Reliable')
            axy.errorbar(
                x0q[inok], y0q[inok], xerr=stdx0q[inok], yerr=stdy0q[inok],
                fmt='x', color=cors[1])
            axy.legend(
                loc='upper right', bbox_to_anchor=(1.8, 0.4),
                fontsize='xx-small')

        ayy.legend(
            loc='upper right', bbox_to_anchor=(1, -0.4), fontsize='small',
            title='Fitting method')
        axx.grid(True)
        ayy.grid(True)
        axx.set_ylabel('X0 [um]')
        ayy.set_ylabel('Y0 [um]')
        axx.set_ylim([minx, maxx])
        ayy.set_ylim([miny, maxy])

        axy.grid(True)
        axy.set_xlabel('X0 [um]')
        axy.set_ylabel('Y0 [um]')
        axy.set_xlim([minx, maxx])
        axy.set_ylim([miny, maxy])

        if fname:
            f.savefig(fname+'.svg')
            plt.close()
        else:
            f.show()

    @staticmethod
    def make_figures_compare_bbas(bbalist, method='linear_fitting', labels=[],
                                  bpmsok=None, bpmsnok=None, fname='',
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

        bpmsok = bpmsok or bbalist[0].data['bpmnames']
        bpmsnok = bpmsnok or []
        iok = np.array(
            [bbalist[0].data['bpmnames'].index(bpm) for bpm in bpmsok],
            dtype=int)
        inok = np.array(
            [bbalist[0].data['bpmnames'].index(bpm) for bpm in bpmsnok],
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

            axx.errorbar(iok, x0l[iok], yerr=stdx0l[iok], fmt='o', color=cors[i])
            ayy.errorbar(iok, y0l[iok], yerr=stdy0l[iok], fmt='o', color=cors[i], label=labels[i])
            axy.errorbar(
                x0l[iok], y0l[iok], xerr=stdx0l[iok], yerr=stdy0l[iok], fmt='o', color=cors[i],
                label='Reliable')

            axx.errorbar(inok, x0l[inok], yerr=stdx0l[inok], fmt='x', color=cors[i])
            ayy.errorbar(inok, y0l[inok], yerr=stdy0l[inok], fmt='x', color=cors[i], label=labels[i])
            axy.errorbar(
                x0l[inok], y0l[inok], xerr=stdx0l[inok], yerr=stdy0l[inok], fmt='x', color=cors[i],
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
