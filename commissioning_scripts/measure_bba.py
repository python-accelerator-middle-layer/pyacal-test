"""Main module."""
import time as _time
from threading import Thread as _Thread, Event as _Event

from copy import deepcopy as _dcopy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gs
import matplotlib.cm as _cmap

from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import SOFB, PowerSupply

import pyaccel as _pyacc

from .base import BaseClass


class BBAParams:
    """."""

    BPMNAMES = (
        'SI-01M2:DI-BPM', 'SI-01C1:DI-BPM-1',
        'SI-01C1:DI-BPM-2', 'SI-01C2:DI-BPM',
        'SI-01C3:DI-BPM-1', 'SI-01C3:DI-BPM-2',
        'SI-01C4:DI-BPM', 'SI-02M1:DI-BPM',
        'SI-02M2:DI-BPM', 'SI-02C1:DI-BPM-1',
        'SI-02C1:DI-BPM-2', 'SI-02C2:DI-BPM',
        'SI-02C3:DI-BPM-1', 'SI-02C3:DI-BPM-2',
        'SI-02C4:DI-BPM', 'SI-03M1:DI-BPM',
        'SI-03M2:DI-BPM', 'SI-03C1:DI-BPM-1',
        'SI-03C1:DI-BPM-2', 'SI-03C2:DI-BPM',
        'SI-03C3:DI-BPM-1', 'SI-03C3:DI-BPM-2',
        'SI-03C4:DI-BPM', 'SI-04M1:DI-BPM',
        'SI-04M2:DI-BPM', 'SI-04C1:DI-BPM-1',
        'SI-04C1:DI-BPM-2', 'SI-04C2:DI-BPM',
        'SI-04C3:DI-BPM-1', 'SI-04C3:DI-BPM-2',
        'SI-04C4:DI-BPM', 'SI-05M1:DI-BPM',
        'SI-05M2:DI-BPM', 'SI-05C1:DI-BPM-1',
        'SI-05C1:DI-BPM-2', 'SI-05C2:DI-BPM',
        'SI-05C3:DI-BPM-1', 'SI-05C3:DI-BPM-2',
        'SI-05C4:DI-BPM', 'SI-06M1:DI-BPM',
        'SI-06M2:DI-BPM', 'SI-06C1:DI-BPM-1',
        'SI-06C1:DI-BPM-2', 'SI-06C2:DI-BPM',
        'SI-06C3:DI-BPM-1', 'SI-06C3:DI-BPM-2',
        'SI-06C4:DI-BPM', 'SI-07M1:DI-BPM',
        'SI-07M2:DI-BPM', 'SI-07C1:DI-BPM-1',
        'SI-07C1:DI-BPM-2', 'SI-07C2:DI-BPM',
        'SI-07C3:DI-BPM-1', 'SI-07C3:DI-BPM-2',
        'SI-07C4:DI-BPM', 'SI-08M1:DI-BPM',
        'SI-08M2:DI-BPM', 'SI-08C1:DI-BPM-1',
        'SI-08C1:DI-BPM-2', 'SI-08C2:DI-BPM',
        'SI-08C3:DI-BPM-1', 'SI-08C3:DI-BPM-2',
        'SI-08C4:DI-BPM', 'SI-09M1:DI-BPM',
        'SI-09M2:DI-BPM', 'SI-09C1:DI-BPM-1',
        'SI-09C1:DI-BPM-2', 'SI-09C2:DI-BPM',
        'SI-09C3:DI-BPM-1', 'SI-09C3:DI-BPM-2',
        'SI-09C4:DI-BPM', 'SI-10M1:DI-BPM',
        'SI-10M2:DI-BPM', 'SI-10C1:DI-BPM-1',
        'SI-10C1:DI-BPM-2', 'SI-10C2:DI-BPM',
        'SI-10C3:DI-BPM-1', 'SI-10C3:DI-BPM-2',
        'SI-10C4:DI-BPM', 'SI-11M1:DI-BPM',
        'SI-11M2:DI-BPM', 'SI-11C1:DI-BPM-1',
        'SI-11C1:DI-BPM-2', 'SI-11C2:DI-BPM',
        'SI-11C3:DI-BPM-1', 'SI-11C3:DI-BPM-2',
        'SI-11C4:DI-BPM', 'SI-12M1:DI-BPM',
        'SI-12M2:DI-BPM', 'SI-12C1:DI-BPM-1',
        'SI-12C1:DI-BPM-2', 'SI-12C2:DI-BPM',
        'SI-12C3:DI-BPM-1', 'SI-12C3:DI-BPM-2',
        'SI-12C4:DI-BPM', 'SI-13M1:DI-BPM',
        'SI-13M2:DI-BPM', 'SI-13C1:DI-BPM-1',
        'SI-13C1:DI-BPM-2', 'SI-13C2:DI-BPM',
        'SI-13C3:DI-BPM-1', 'SI-13C3:DI-BPM-2',
        'SI-13C4:DI-BPM', 'SI-14M1:DI-BPM',
        'SI-14M2:DI-BPM', 'SI-14C1:DI-BPM-1',
        'SI-14C1:DI-BPM-2', 'SI-14C2:DI-BPM',
        'SI-14C3:DI-BPM-1', 'SI-14C3:DI-BPM-2',
        'SI-14C4:DI-BPM', 'SI-15M1:DI-BPM',
        'SI-15M2:DI-BPM', 'SI-15C1:DI-BPM-1',
        'SI-15C1:DI-BPM-2', 'SI-15C2:DI-BPM',
        'SI-15C3:DI-BPM-1', 'SI-15C3:DI-BPM-2',
        'SI-15C4:DI-BPM', 'SI-16M1:DI-BPM',
        'SI-16M2:DI-BPM', 'SI-16C1:DI-BPM-1',
        'SI-16C1:DI-BPM-2', 'SI-16C2:DI-BPM',
        'SI-16C3:DI-BPM-1', 'SI-16C3:DI-BPM-2',
        'SI-16C4:DI-BPM', 'SI-17M1:DI-BPM',
        'SI-17M2:DI-BPM', 'SI-17C1:DI-BPM-1',
        'SI-17C1:DI-BPM-2', 'SI-17C2:DI-BPM',
        'SI-17C3:DI-BPM-1', 'SI-17C3:DI-BPM-2',
        'SI-17C4:DI-BPM', 'SI-18M1:DI-BPM',
        'SI-18M2:DI-BPM', 'SI-18C1:DI-BPM-1',
        'SI-18C1:DI-BPM-2', 'SI-18C2:DI-BPM',
        'SI-18C3:DI-BPM-1', 'SI-18C3:DI-BPM-2',
        'SI-18C4:DI-BPM', 'SI-19M1:DI-BPM',
        'SI-19M2:DI-BPM', 'SI-19C1:DI-BPM-1',
        'SI-19C1:DI-BPM-2', 'SI-19C2:DI-BPM',
        'SI-19C3:DI-BPM-1', 'SI-19C3:DI-BPM-2',
        'SI-19C4:DI-BPM', 'SI-20M1:DI-BPM',
        'SI-20M2:DI-BPM', 'SI-20C1:DI-BPM-1',
        'SI-20C1:DI-BPM-2', 'SI-20C2:DI-BPM',
        'SI-20C3:DI-BPM-1', 'SI-20C3:DI-BPM-2',
        'SI-20C4:DI-BPM', 'SI-01M1:DI-BPM',
        )
    QUADNAMES = (
        'SI-01M2:PS-QS', 'SI-01C1:PS-Q1',
        'SI-01C1:PS-QS', 'SI-01C2:PS-Q4',
        'SI-01C3:PS-Q4', 'SI-01C3:PS-QS',
        'SI-01C4:PS-Q1', 'SI-02M1:PS-QDB2',
        'SI-02M2:PS-QDB2', 'SI-02C1:PS-Q1',
        'SI-02C1:PS-QS', 'SI-02C2:PS-Q4',
        'SI-02C3:PS-Q4', 'SI-02C3:PS-QS',
        'SI-02C4:PS-Q1', 'SI-03M1:PS-QDP2',
        'SI-03M2:PS-QDP2', 'SI-03C1:PS-Q1',
        'SI-03C1:PS-QS', 'SI-03C2:PS-Q4',
        'SI-03C3:PS-Q4', 'SI-03C3:PS-QS',
        'SI-03C4:PS-Q1', 'SI-04M1:PS-QDB2',
        'SI-04M2:PS-QDB2', 'SI-04C1:PS-Q1',
        'SI-04C1:PS-QS', 'SI-04C2:PS-Q4',
        'SI-04C3:PS-Q4', 'SI-04C3:PS-QS',
        'SI-04C4:PS-Q1', 'SI-05M1:PS-QS',
        'SI-05M2:PS-QS', 'SI-05C1:PS-Q1',
        'SI-05C1:PS-QS', 'SI-05C2:PS-Q4',
        'SI-05C3:PS-Q4', 'SI-05C3:PS-QS',
        'SI-05C4:PS-Q1', 'SI-06M1:PS-QDB2',
        'SI-06M2:PS-QDB2', 'SI-06C1:PS-Q1',
        'SI-06C1:PS-QS', 'SI-06C2:PS-Q4',
        'SI-06C3:PS-Q4', 'SI-06C3:PS-QS',
        'SI-06C4:PS-Q1', 'SI-07M1:PS-QDP2',
        'SI-07M2:PS-QDP2', 'SI-07C1:PS-Q1',
        'SI-07C1:PS-QS', 'SI-07C2:PS-Q4',
        'SI-07C3:PS-Q4', 'SI-07C3:PS-QS',
        'SI-07C4:PS-Q1', 'SI-08M1:PS-QDB2',
        'SI-08M2:PS-QDB2', 'SI-08C1:PS-Q1',
        'SI-08C1:PS-QS', 'SI-08C2:PS-Q4',
        'SI-08C3:PS-Q4', 'SI-08C3:PS-QS',
        'SI-08C4:PS-Q1', 'SI-09M1:PS-QS',
        'SI-09M2:PS-QS', 'SI-09C1:PS-Q1',
        'SI-09C1:PS-QS', 'SI-09C2:PS-Q4',
        'SI-09C3:PS-Q4', 'SI-09C3:PS-QS',
        'SI-09C4:PS-Q1', 'SI-10M1:PS-QDB2',
        'SI-10M2:PS-QDB2', 'SI-10C1:PS-Q1',
        'SI-10C1:PS-QS', 'SI-10C2:PS-Q4',
        'SI-10C3:PS-Q4', 'SI-10C3:PS-QS',
        'SI-10C4:PS-Q1', 'SI-11M1:PS-QDP2',
        'SI-11M2:PS-QDP2', 'SI-11C1:PS-Q1',
        'SI-11C1:PS-QS', 'SI-11C2:PS-Q4',
        'SI-11C3:PS-Q4', 'SI-11C3:PS-QS',
        'SI-11C4:PS-Q1', 'SI-12M1:PS-QDB2',
        'SI-12M2:PS-QDB2', 'SI-12C1:PS-Q1',
        'SI-12C1:PS-QS', 'SI-12C2:PS-Q4',
        'SI-12C3:PS-Q4', 'SI-12C3:PS-QS',
        'SI-12C4:PS-Q1', 'SI-13M1:PS-QS',
        'SI-13M2:PS-QS', 'SI-13C1:PS-Q1',
        'SI-13C1:PS-QS', 'SI-13C2:PS-Q4',
        'SI-13C3:PS-Q4', 'SI-13C3:PS-QS',
        'SI-13C4:PS-Q1', 'SI-14M1:PS-QDB2',
        'SI-14M2:PS-QDB2', 'SI-14C1:PS-Q1',
        'SI-14C1:PS-QS', 'SI-14C2:PS-Q4',
        'SI-14C3:PS-Q4', 'SI-14C3:PS-QS',
        'SI-14C4:PS-Q1', 'SI-15M1:PS-QDP2',
        'SI-15M2:PS-QDP2', 'SI-15C1:PS-Q1',
        'SI-15C1:PS-QS', 'SI-15C2:PS-Q4',
        'SI-15C3:PS-Q4', 'SI-15C3:PS-QS',
        'SI-15C4:PS-Q1', 'SI-16M1:PS-QDB2',
        'SI-16M2:PS-QDB2', 'SI-16C1:PS-Q1',
        'SI-16C1:PS-QS', 'SI-16C2:PS-Q4',
        'SI-16C3:PS-Q4', 'SI-16C3:PS-QS',
        'SI-16C4:PS-Q1', 'SI-17M1:PS-QS',
        'SI-17M2:PS-QS', 'SI-17C1:PS-Q1',
        'SI-17C1:PS-QS', 'SI-17C2:PS-Q4',
        'SI-17C3:PS-Q4', 'SI-17C3:PS-QS',
        'SI-17C4:PS-Q1', 'SI-18M1:PS-QDB2',
        'SI-18M2:PS-QDB2', 'SI-18C1:PS-Q1',
        'SI-18C1:PS-QS', 'SI-18C2:PS-Q4',
        'SI-18C3:PS-Q4', 'SI-18C3:PS-QS',
        'SI-18C4:PS-Q1', 'SI-19M1:PS-QDP2',
        'SI-19M2:PS-QDP2', 'SI-19C1:PS-Q1',
        'SI-19C1:PS-QS', 'SI-19C2:PS-Q4',
        'SI-19C3:PS-Q4', 'SI-19C3:PS-QS',
        'SI-19C4:PS-Q1', 'SI-20M1:PS-QDB2',
        'SI-20M2:PS-QDB2', 'SI-20C1:PS-Q1',
        'SI-20C1:PS-QS', 'SI-20C2:PS-Q4',
        'SI-20C3:PS-Q4', 'SI-20C3:PS-QS',
        'SI-20C4:PS-Q1', 'SI-01M1:PS-QS',
        )

    BPMNAMES = tuple([_PVName(bpm) for bpm in BPMNAMES])
    QUADNAMES = tuple([_PVName(quad) for quad in QUADNAMES])

    def __init__(self):
        """."""
        self.deltaorbx = 100  # [um]
        self.deltaorby = 100  # [um]
        self.meas_nrsteps = 8
        self.quad_deltakl = 0.01  # [1/m]
        self.quad_nrcycles = 1
        self.wait_sofb = 0.3  # [s]
        self.wait_correctors = 2  # [s]
        self.wait_quadrupole = 2  # [s]
        self.timeout_wait_sofb = 3  # [s]
        self.sofb_nrpoints = 10
        self.sofb_maxcorriter = 5
        self.sofb_maxorberr = 10  # [um]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        st = ftmp('deltaorbx [um]', self.deltaorbx, '')
        st += ftmp('deltaorby [um]', self.deltaorby, '')
        st += dtmp('meas_nrsteps', self.meas_nrsteps, '')
        st += ftmp('quad_deltakl [1/m]', self.quad_deltakl, '')
        st += ftmp('quad_nrcycles', self.quad_nrcycles, '')
        st += ftmp('wait_sofb [s]', self.wait_sofb, '(time to process calcs)')
        st += ftmp('wait_correctors [s]', self.wait_correctors, '')
        st += ftmp('wait_quadrupole [s]', self.wait_quadrupole, '')
        st += ftmp(
            'timeout_wait_sofb [s]', self.timeout_wait_sofb, '(get orbit)')
        st += dtmp('sofb_nrpoints', self.sofb_nrpoints, '')
        st += dtmp('sofb_maxcorriter', self.sofb_maxcorriter, '')
        st += ftmp('sofb_maxorberr [um]', self.sofb_maxorberr, '')
        return st


class DoBBA(BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.params = BBAParams()
        self._bpms2dobba = list()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.data['bpmnames'] = list(BBAParams.BPMNAMES)
        self.data['quadnames'] = list(BBAParams.QUADNAMES)
        self.data['scancenterx'] = np.zeros(len(BBAParams.BPMNAMES))
        self.data['scancentery'] = np.zeros(len(BBAParams.BPMNAMES))
        self.data['measure'] = dict()
        self.analysis = dict()
        self.connect_to_quadrupoles()

        self._stopevt = _Event()
        self._thread = _Thread(target=self._do_bba, daemon=True)

    def __str__(self):
        """."""
        stn = 'Params\n'
        stp = self.params.__str__()
        stp = '    ' + stp.replace('\n', '\n    ')
        stn += stp + '\n'
        stn += 'Connected?  ' + str(self.connected) + '\n\n'

        stn += '     {:^20s} {:^20s} {:6s} {:6s}\n'.format(
            'BPM', 'Quad', 'Xc [um]', 'Yc [um]')
        tmplt = '{:03d}: {:^20s} {:^20s} {:^6.1f} {:^6.1f}\n'
        dta = self.data
        for bpm in self.bpms2dobba:
            idx = dta['bpmnames'].index(bpm)
            stn += tmplt.format(
                idx, dta['bpmnames'][idx], dta['quadnames'][idx],
                dta['scancenterx'][idx], dta['scancentery'][idx])
        return stn

    def start(self):
        """."""
        if self.ismeasuring:
            return
        self._stopevt.clear()
        self._thread = _Thread(target=self._do_bba, daemon=True)
        self._thread.start()

    def stop(self):
        """."""
        self._stopevt.set()

    @property
    def ismeasuring(self):
        """."""
        return self._thread.is_alive()

    @property
    def measuredbpms(self):
        """."""
        return sorted(self.data['measure'])

    @property
    def bpms2dobba(self):
        """."""
        if self._bpms2dobba:
            return _dcopy(self._bpms2dobba)
        return sorted(
            set(self.data['bpmnames']) - self.data['measure'].keys())

    @bpms2dobba.setter
    def bpms2dobba(self, bpmlist):
        """."""
        self._bpms2dobba = sorted([_PVName(bpm) for bpm in bpmlist])

    def connect_to_quadrupoles(self):
        """."""
        for bpm in self.bpms2dobba:
            idx = self.data['bpmnames'].index(bpm)
            qname = self.data['quadnames'][idx]
            if qname and qname not in self.devices:
                self.devices[qname] = PowerSupply(qname)

    def get_orbit(self):
        """."""
        sofb = self.devices['sofb']
        sofb.cmd_reset()
        sofb.wait_buffer(self.params.timeout_wait_sofb)
        return np.hstack([sofb.orbx, sofb.orby])

    @staticmethod
    def get_cycling_curve():
        """."""
        return [1/2, -1/2, 0]

    def correct_orbit(self, bpmname, x0, y0):
        """."""
        sofb = self.devices['sofb']
        idxx = self.data['bpmnames'].index(bpmname)
        idxy = idxx + len(self.data['bpmnames'])
        refx, refy = sofb.refx, sofb.refy
        refx[idxx], refy[idxx] = x0, y0
        sofb.refx = refx
        sofb.refy = refy
        _time.sleep(self.params.wait_sofb)
        for i in range(self.params.sofb_maxcorriter+1):
            orb = self.get_orbit()
            fmet = max(abs(orb[idxx] - x0), abs(orb[idxy] - y0))
            if fmet < self.params.sofb_maxorberr:
                return i, fmet

            if i < self.params.sofb_maxcorriter:
                sofb.cmd_calccorr()
                _time.sleep(self.params.wait_sofb)
                sofb.cmd_applycorr()
                _time.sleep(self.params.wait_correctors)
        return -1, fmet

    def process_data(self, nbpms_linfit=None, thres=None, mode='symm',
                     discardpoints=None):
        """."""
        for bpm in self.data['measure']:
            self.analysis[bpm] = self.process_data_single_bpm(
                bpm, nbpms_linfit=nbpms_linfit, thres=thres, mode=mode,
                discardpoints=discardpoints)

    def process_data_single_bpm(self, bpm, nbpms_linfit=None, thres=None,
                                mode='symm', discardpoints=None):
        """."""
        anl = dict()
        idx = self.data['bpmnames'].index(bpm)
        nbpms = len(self.data['bpmnames'])
        orbini = self.data['measure'][bpm]['orbini']
        orbpos = self.data['measure'][bpm]['orbpos']
        orbneg = self.data['measure'][bpm]['orbneg']

        usepts = set(range(orbini.shape[0]))
        if discardpoints is not None:
            usepts = set(usepts) - set(discardpoints)
        usepts = sorted(usepts)

        xpos = orbini[usepts, idx]
        ypos = orbini[usepts, idx+nbpms]
        if mode.lower().startswith('symm'):
            dorb = orbpos - orbneg
        elif mode.lower().startswith('pos'):
            dorb = orbpos - orbini
        else:
            dorb = orbini - orbneg

        dorbx = dorb[usepts, :nbpms]
        dorby = dorb[usepts, nbpms:]
        if '-QS' in self.data['quadnames'][idx]:
            dorbx, dorby = dorby, dorbx
        anl['xpos'] = xpos
        anl['ypos'] = ypos

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
        x0 = np.dot(pxc[0], -pxc[1]) / np.dot(pxc[0], pxc[0])
        y0 = np.dot(pyc[0], -pyc[1]) / np.dot(pyc[0], pyc[0])
        stdx0 = np.sqrt(
            np.dot(pxc[1], pxc[1]) / np.dot(pxc[0], pxc[0]) - x0*x0)
        stdy0 = np.sqrt(
            np.dot(pyc[1], pyc[1]) / np.dot(pyc[0], pyc[0]) - y0*y0)
        extrapx = not min(xpos) <= x0 <= max(xpos)
        extrapy = not min(ypos) <= y0 <= max(ypos)
        anl['linear_fitting'] = dict()
        anl['linear_fitting']['dorbx'] = dorbx
        anl['linear_fitting']['dorby'] = dorby
        anl['linear_fitting']['coeffsx'] = px
        anl['linear_fitting']['coeffsy'] = py
        anl['linear_fitting']['x0s'] = x0s
        anl['linear_fitting']['y0s'] = y0s
        anl['linear_fitting']['extrapolatedx'] = extrapx
        anl['linear_fitting']['extrapolatedy'] = extrapy
        anl['linear_fitting']['x0'] = x0
        anl['linear_fitting']['y0'] = y0
        anl['linear_fitting']['stdx0'] = stdx0
        anl['linear_fitting']['stdy0'] = stdy0

        rmsx = np.sum(dorbx*dorbx, axis=1) / dorbx.shape[1]
        rmsy = np.sum(dorby*dorby, axis=1) / dorby.shape[1]
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
        anl['quadratic_fitting'] = dict()
        anl['quadratic_fitting']['meansqrx'] = rmsx
        anl['quadratic_fitting']['meansqry'] = rmsy
        anl['quadratic_fitting']['coeffsx'] = px
        anl['quadratic_fitting']['coeffsy'] = py
        anl['quadratic_fitting']['extrapolatedx'] = extrapx
        anl['quadratic_fitting']['extrapolatedy'] = extrapy
        anl['quadratic_fitting']['x0'] = x0
        anl['quadratic_fitting']['y0'] = y0
        anl['quadratic_fitting']['stdx0'] = stdx0
        anl['quadratic_fitting']['stdy0'] = stdy0
        return anl

    def get_bba_results(self, method='linear_fitting', error=False):
        """."""
        data = self.data
        bpms = data['bpmnames']
        bbax = np.zeros(len(bpms))
        bbay = np.zeros(len(bpms))
        if error:
            bbaxerr = np.zeros(len(bpms))
            bbayerr = np.zeros(len(bpms))
        for idx, bpm in enumerate(bpms):
            anl = self.analysis.get(bpm)
            if not anl:
                continue
            res = anl[method]
            bbax[idx] = res['x0']
            bbay[idx] = res['y0']
            if error and 'stdx0' in res:
                bbaxerr[idx] = res['stdx0']
                bbayerr[idx] = res['stdy0']
        if error:
            return bbax, bbay, bbaxerr, bbayerr
        return bbax, bbay

    def get_analysis_properties(self, propty, method='linear_fitting'):
        """."""
        data = self.data
        bpms = data['bpmnames']
        prop = [[], ] * len(bpms)
        for idx, bpm in enumerate(bpms):
            anl = self.analysis.get(bpm)
            if not anl:
                continue
            res = anl[method]
            prop[idx] = res[propty]
        return prop

    @staticmethod
    def get_default_quads(model, fam_data):
        """."""
        quads_idx = _dcopy(fam_data['QN']['index'])
        qs_idx = [idx for idx in fam_data['QS']['index']
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
    def _calc_dorb_scan(deltaorb, nrpts):
        dorbspos = np.linspace(deltaorb, 0, nrpts+1)[:-1]
        dorbsneg = np.linspace(-deltaorb, 0, nrpts+1)[:-1]
        dorbs = np.array([dorbsneg, dorbspos]).T.flatten()
        dorbs = np.hstack([0, dorbs])
        return dorbs

    @staticmethod
    def list_bpm_subsections(bpms):
        """."""
        subinst = [bpm[5:7]+bpm[14:] for bpm in bpms]
        sec = [bpm[2:4] for bpm in bpms]
        subsecs = {typ: [] for typ in subinst}
        for sub, sec, bpm in zip(subinst, sec, bpms):
            subsecs[sub].append(bpm)
        return subsecs

    def combine_bbas(self, bbalist):
        """."""
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

    def filter_problems(
            self, maxstd=100, maxorb=9, maxrms=100, method='lin quad',
            probtype='std', pln='xy'):
        """."""
        bpms = []
        islin = 'lin' in method
        isquad = 'quad' in method
        for bpm in self.data['bpmnames']:
            anl = self.analysis.get(bpm)
            if not anl:
                continue
            concx = anl['quadratic_fitting']['coeffsx'][0]
            concy = anl['quadratic_fitting']['coeffsy'][0]
            probc = False
            if 'x' in pln:
                probc |= concx < 0
            if 'y' in pln:
                probc |= concy < 0

            rmsx = anl['quadratic_fitting']['meansqrx']
            rmsy = anl['quadratic_fitting']['meansqry']
            probmaxrms = False
            if 'x' in pln:
                probmaxrms |= np.max(rmsx) < maxrms
            if 'y' in pln:
                probmaxrms |= np.max(rmsy) < maxrms

            extqx = isquad and anl['quadratic_fitting']['extrapolatedx']
            extqy = isquad and anl['quadratic_fitting']['extrapolatedy']
            extlx = islin and anl['linear_fitting']['extrapolatedx']
            extly = islin and anl['linear_fitting']['extrapolatedy']
            probe = False
            if 'x' in pln:
                probe |= extqx or extlx
            if 'y' in pln:
                probe |= extqy or extly

            stdqx = isquad and anl['quadratic_fitting']['stdx0'] > maxstd
            stdqy = isquad and anl['quadratic_fitting']['stdy0'] > maxstd
            stdlx = islin and anl['linear_fitting']['stdx0'] > maxstd
            stdly = islin and anl['linear_fitting']['stdy0'] > maxstd
            probs = False
            if 'x' in pln:
                probs |= stdqx or stdlx
            if 'y' in pln:
                probs |= stdqy or stdly

            dorbx = anl['linear_fitting']['dorbx']
            dorby = anl['linear_fitting']['dorby']
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

    # ##### Make Figures #####
    def make_figure_bpm_summary(self, bpm, save=False):
        """."""
        f = plt.figure(figsize=(9.5, 9))
        gs = mpl_gs.GridSpec(3, 2)
        gs.update(
            left=0.11, right=0.98, bottom=0.1, top=0.9,
            hspace=0.35, wspace=0.35)

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

        anl = self.analysis.get(bpm)
        if not anl:
            print('no dada found for ' + bpm)
            return
        xpos = anl['xpos']
        ypos = anl['ypos']
        sxpos = np.sort(xpos)
        sypos = np.sort(ypos)

        xq0 = anl['quadratic_fitting']['x0']
        yq0 = anl['quadratic_fitting']['y0']
        stdxq0 = anl['quadratic_fitting']['stdx0']
        stdyq0 = anl['quadratic_fitting']['stdy0']
        xl0 = anl['linear_fitting']['x0']
        yl0 = anl['linear_fitting']['y0']
        stdxl0 = anl['linear_fitting']['stdx0']
        stdyl0 = anl['linear_fitting']['stdy0']

        adt.set_frame_on(False)
        adt.axes.get_yaxis().set_visible(False)
        adt.axes.get_xaxis().set_visible(False)
        idx = self.data['bpmnames'].index(bpm)
        xini = self.data['scancenterx'][idx]
        yini = self.data['scancentery'][idx]
        deltakl = self.data['measure'][bpm]['deltakl']
        qname = self.data['quadnames'][idx]
        tmp = '{:6.1f} ' + r'$\pm$' + ' {:<6.1f}'
        st = 'Quad: {:15s} (dKL={:.4f} 1/m)\n'.format(qname, deltakl)
        st += '\nInitial Search values = ({:.2f}, {:.2f})\n'.format(xini, yini)
        st += 'BBA Results:\n'
        x0s = tmp.format(xl0, stdxl0)
        y0s = tmp.format(yl0, stdyl0)
        st += '  Linear: X = {:s}  Y = {:s}\n'.format(x0s, y0s)
        x0s = tmp.format(xq0, stdxq0)
        y0s = tmp.format(yq0, stdyq0)
        st += '  Parab.: X = {:s}  Y = {:s}'.format(x0s, y0s)
        adt.text(
            0.5, 0.5, st, fontsize=10, horizontalalignment='center',
            verticalalignment='center', transform=adt.transAxes,
            bbox=dict(edgecolor='k', facecolor='w', alpha=1.0))
        adt.set_xlim([0, 8])
        adt.set_ylim([0, 8])

        rmsx = anl['quadratic_fitting']['meansqrx']
        rmsy = anl['quadratic_fitting']['meansqry']
        px = anl['quadratic_fitting']['coeffsx']
        py = anl['quadratic_fitting']['coeffsy']
        fitx = np.polyval(px, sxpos)
        fity = np.polyval(py, sypos)
        fitx0 = np.polyval(px, xq0)
        fity0 = np.polyval(py, yq0)

        aqx.plot(xpos, rmsx, 'bo')
        aqx.plot(sxpos, fitx, 'b')
        aqx.errorbar(xq0, fitx0, xerr=stdxq0, fmt='kx', markersize=20)
        aqy.plot(ypos, rmsy, 'ro')
        aqy.plot(sypos, fity, 'r')
        aqy.errorbar(yq0, fity0, xerr=stdyq0, fmt='kx', markersize=20)
        axy.errorbar(
            xq0, yq0, xerr=stdxq0, yerr=stdyq0, fmt='gx', markersize=20,
            label='parabollic')

        dorbx = anl['linear_fitting']['dorbx']
        dorby = anl['linear_fitting']['dorby']
        x0s = anl['linear_fitting']['x0s']
        y0s = anl['linear_fitting']['y0s']
        px = anl['linear_fitting']['coeffsx']
        py = anl['linear_fitting']['coeffsy']
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
        alx.errorbar(xl0, 0, xerr=stdxl0, fmt='kx', markersize=20)
        aly.plot(ypos, dorby[:, sidy[-npts:]], 'r.')
        aly.plot(sypos, pvy.T, 'r', linewidth=1)
        aly.errorbar(yl0, 0, xerr=stdyl0, fmt='kx', markersize=20)
        axy.errorbar(
            xl0, yl0, xerr=stdxl0, yerr=stdyl0, fmt='mx', markersize=20,
            label='linear')

        axy.legend(loc='best', fontsize='x-small')
        axy.set_xlabel(r'$X_0$ [$\mu$m]')
        axy.set_ylabel(r'$Y_0$ [$\mu$m]')
        alx.set_xlabel(r'X [$\mu$m]')
        alx.set_ylabel(r'$\Delta$ COD [$\mu$m]')
        aly.set_xlabel(r'Y [$\mu$m]')
        aly.set_ylabel(r'$\Delta$ COD [$\mu$m]')
        aqx.set_xlabel(r'X [$\mu$m]')
        aqx.set_ylabel(r'COD$^2$ [$\mu$m$^2$]')
        aqy.set_xlabel(r'Y [$\mu$m]')
        aqy.set_ylabel(r'COD$^2$ [$\mu$m$^2$]')

        if save:
            f.savefig(bpm+'.svg')
            plt.close()
        else:
            f.show()

    def make_figure_quadfit(self, bpms=None, fname='', title=''):
        """."""
        f = plt.figure(figsize=(9.5, 9))
        gs = mpl_gs.GridSpec(2, 1)
        gs.update(
            left=0.1, right=0.78, bottom=0.15, top=0.9,
            hspace=0.5, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = plt.subplot(gs[0, 0])
        ayy = plt.subplot(gs[1, 0])

        bpms = bpms or self.data['bpmnames']
        colors = _cmap.brg(np.linspace(0, 1, len(bpms)))
        for i, bpm in enumerate(bpms):
            anl = self.analysis.get(bpm)
            if not anl:
                print('Data not found for ' + bpm)
                continue
            rmsx = anl['quadratic_fitting']['meansqrx']
            rmsy = anl['quadratic_fitting']['meansqry']

            px = anl['quadratic_fitting']['coeffsx']
            py = anl['quadratic_fitting']['coeffsy']

            x0 = anl['quadratic_fitting']['x0']
            y0 = anl['quadratic_fitting']['y0']

            sxpos = np.sort(anl['xpos'])
            sypos = np.sort(anl['ypos'])
            fitx = np.polyval(px, sxpos)
            fity = np.polyval(py, sypos)

            axx.plot(anl['xpos']-x0, rmsx, 'o', color=colors[i], label=bpm)
            axx.plot(sxpos-x0, fitx, color=colors[i])
            ayy.plot(anl['ypos']-y0, rmsy, 'o', color=colors[i], label=bpm)
            ayy.plot(sypos-y0, fity, color=colors[i])

        axx.legend(bbox_to_anchor=(1.0, 1.1), fontsize='xx-small')
        axx.grid(True)
        ayy.grid(True)
        axx.set_xlabel(r'$X - X_0$ [$\mu$m]')
        axx.set_ylabel(r'$\Delta$ COD$^2$')
        ayy.set_xlabel(r'$Y - Y_0$ [$\mu$m]')
        ayy.set_ylabel(r'$\Delta$ COD$^2$')
        if fname:
            f.savefig(fname+'.svg')
            plt.close()
        else:
            f.show()

    def make_figure_linfit(self, bpms=None, fname='', title=''):
        """."""
        f = plt.figure(figsize=(9.5, 9))
        gs = mpl_gs.GridSpec(2, 1)
        gs.update(
            left=0.1, right=0.78, bottom=0.15, top=0.9,
            hspace=0.5, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = plt.subplot(gs[0, 0])
        axy = plt.subplot(gs[1, 0])

        bpms = bpms or self.data['bpmnames']
        colors = _cmap.brg(np.linspace(0, 1, len(bpms)))
        for i, bpm in enumerate(bpms):
            anl = self.analysis.get(bpm)
            if not anl:
                print('Data not found for ' + bpm)
                continue
            x0 = anl['linear_fitting']['x0']
            y0 = anl['linear_fitting']['y0']
            px = anl['linear_fitting']['coeffsx']
            py = anl['linear_fitting']['coeffsy']

            sidx = np.argsort(np.abs(px[0]))
            sidy = np.argsort(np.abs(py[0]))

            xpos = anl['xpos']
            ypos = anl['ypos']
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

        axx.legend(
            loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='xx-small')
        axx.grid(True)
        axy.grid(True)

        if fname:
            f.savefig(fname+'.svg')
            plt.close()
        else:
            f.show()

    def make_figure_compare_with_initial(
            self, method='linear_fitting', bpmsok=None, bpmsnok=None,
            xlim=None, ylim=None, fname='', title='', plotdiff=True):
        """."""
        f = plt.figure(figsize=(9.2, 9))
        gs = mpl_gs.GridSpec(2, 1)
        gs.update(
            left=0.1, right=0.98, bottom=0.08, top=0.9,
            hspace=0.01, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = plt.subplot(gs[0, 0])
        ayy = plt.subplot(gs[1, 0], sharex=axx)

        bpmsok = bpmsok or self.data['bpmnames']
        bpmsnok = bpmsnok or []
        iok = np.array(
            [self.data['bpmnames'].index(bpm) for bpm in bpmsok], dtype=int)
        inok = np.array(
            [self.data['bpmnames'].index(bpm) for bpm in bpmsnok], dtype=int)

        labels = ['initial', method]
        cors = _cmap.brg(np.linspace(0, 1, 3))

        x0c = np.array(self.data['scancenterx'])
        y0c = np.array(self.data['scancentery'])
        x0q, y0q, stdx0q, stdy0q = self.get_bba_results(
            method=method, error=True)
        if plotdiff:
            x0q -= x0c
            y0q -= y0c
            x0c -= x0c
            y0c -= y0c

        minx = np.min(
            np.hstack([x0q[iok], x0c[iok], x0q[inok], x0c[inok]]))*1.1
        maxx = np.max(
            np.hstack([x0q[iok], x0c[iok], x0q[inok], x0c[inok]]))*1.1
        miny = np.min(
            np.hstack([y0q[iok], y0c[iok], y0q[inok], y0c[inok]]))*1.1
        maxy = np.max(
            np.hstack([y0q[iok], y0c[iok], y0q[inok], y0c[inok]]))*1.1
        minx = -1*xlim if xlim is not None else minx
        maxx = xlim if xlim is not None else maxx
        miny = -1*ylim if ylim is not None else miny
        maxy = ylim if ylim is not None else maxy

        axx.errorbar(iok, x0c[iok], fmt='o', color=cors[0], label=labels[0])
        axx.errorbar(
            iok, x0q[iok], yerr=stdx0q[iok], fmt='o', color=cors[1],
            label=labels[1], elinewidth=1)
        ayy.errorbar(iok, y0c[iok], fmt='o', color=cors[0])
        ayy.errorbar(
            iok, y0q[iok], yerr=stdy0q[iok], fmt='o', color=cors[1],
            elinewidth=1)

        if inok.size:
            axx.errorbar(inok, x0c[inok], fmt='x', color=cors[0])
            axx.errorbar(
                inok, x0q[inok], yerr=stdx0q[inok], fmt='x', color=cors[1],
                elinewidth=1)
            ayy.errorbar(
                inok, y0c[inok], fmt='x', color=cors[0], label=labels[0])
            ayy.errorbar(
                inok, y0q[inok], yerr=stdy0q[inok], fmt='x', color=cors[1],
                elinewidth=1, label=labels[1])

        axx.legend(
            loc='lower right', bbox_to_anchor=(1, 1), fontsize='small', ncol=2)
        axx.grid(True)
        ayy.grid(True)
        axx.set_ylim([minx, maxx])
        ayy.set_ylim([miny, maxy])

        if plotdiff:
            axx.set_ylabel(r'$\Delta X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$\Delta Y_0$ [$\mu$m]')
        else:
            axx.set_ylabel(r'$X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$Y_0$ [$\mu$m]')
        ayy.set_xlabel('BPM Index')

        if fname:
            f.savefig(fname+'.svg')
            plt.close()
        else:
            f.show()

    def make_figure_compare_methods(
            self, bpmsok=None, bpmsnok=None, xlim=None, ylim=None, fname='',
            title='', plotdiff=True):
        """."""
        f = plt.figure(figsize=(9.2, 9))
        gs = mpl_gs.GridSpec(2, 1)
        gs.update(
            left=0.1, right=0.98, bottom=0.08, top=0.9,
            hspace=0.01, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = plt.subplot(gs[0, 0])
        ayy = plt.subplot(gs[1, 0], sharex=axx)

        bpmsok = bpmsok or self.data['bpmnames']
        bpmsnok = bpmsnok or []
        iok = np.array(
            [self.data['bpmnames'].index(bpm) for bpm in bpmsok], dtype=int)
        inok = np.array(
            [self.data['bpmnames'].index(bpm) for bpm in bpmsnok], dtype=int)

        labels = ['linear', 'quadratic']
        cors = _cmap.brg(np.linspace(0, 1, 3))

        x0l, y0l, stdx0l, stdy0l = self.get_bba_results(
            method='linear_fitting', error=True)
        x0q, y0q, stdx0q, stdy0q = self.get_bba_results(
            method='quadratic_fitting', error=True)
        if plotdiff:
            x0q -= x0l
            y0q -= y0l
            x0l -= x0l
            y0l -= y0l

        minx = np.min(
            np.hstack([x0q[iok], x0l[iok], x0q[inok], x0l[inok]]))*1.1
        maxx = np.max(
            np.hstack([x0q[iok], x0l[iok], x0q[inok], x0l[inok]]))*1.1
        miny = np.min(
            np.hstack([y0q[iok], y0l[iok], y0q[inok], y0l[inok]]))*1.1
        maxy = np.max(
            np.hstack([y0q[iok], y0l[iok], y0q[inok], y0l[inok]]))*1.1
        minx = -1*xlim if xlim is not None else minx
        maxx = xlim if xlim is not None else maxx
        miny = -1*ylim if ylim is not None else miny
        maxy = ylim if ylim is not None else maxy

        axx.errorbar(
            iok, x0l[iok], yerr=stdx0l[iok], fmt='o', color=cors[0],
            label=labels[0])
        axx.errorbar(
            iok, x0q[iok], yerr=stdx0q[iok], fmt='o', color=cors[1],
            label=labels[1], elinewidth=1)
        ayy.errorbar(
            iok, y0l[iok], yerr=stdy0l[iok], fmt='o', color=cors[0])
        ayy.errorbar(
            iok, y0q[iok], yerr=stdy0q[iok], fmt='o', color=cors[1],
            elinewidth=1)

        if inok.size:
            axx.errorbar(
                inok, x0l[inok], yerr=stdx0l[inok], fmt='x', color=cors[0])
            axx.errorbar(
                inok, x0q[inok], yerr=stdx0q[inok], fmt='x', color=cors[1],
                elinewidth=1,)
            ayy.errorbar(
                inok, y0l[inok], yerr=stdy0l[inok], fmt='x', color=cors[0],
                label=labels[0])
            ayy.errorbar(
                inok, y0q[inok], yerr=stdy0q[inok], fmt='x', color=cors[1],
                elinewidth=1, label=labels[1])

        axx.legend(
            loc='lower right', bbox_to_anchor=(1, 1), fontsize='small',
            ncol=2, title='Fitting method')
        axx.grid(True)
        ayy.grid(True)
        axx.set_ylim([minx, maxx])
        ayy.set_ylim([miny, maxy])

        if plotdiff:
            axx.set_ylabel(r'$\Delta X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$\Delta Y_0$ [$\mu$m]')
        else:
            axx.set_ylabel(r'$X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$Y_0$ [$\mu$m]')
        ayy.set_xlabel('BPM Index')

        if fname:
            f.savefig(fname+'.svg')
            plt.close()
        else:
            f.show()

    @staticmethod
    def make_figure_compare_bbas(
            bbalist, method='linear_fitting', labels=[], bpmsok=None,
            bpmsnok=None, fname='', xlim=None, ylim=None, title='',
            plotdiff=True):
        """."""
        f = plt.figure(figsize=(9.2, 9))
        gs = mpl_gs.GridSpec(2, 1)
        gs.update(
            left=0.12, right=0.98, bottom=0.13, top=0.9, hspace=0, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = plt.subplot(gs[0, 0])
        ayy = plt.subplot(gs[1, 0], sharex=axx)

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
        cors = _cmap.brg(np.linspace(0, 1, len(bbalist)))

        minx = miny = np.inf
        maxx = maxy = -np.inf
        x0li, y0li, = bbalist[0].get_bba_results(
            method=method, error=False)
        for i, dobba in enumerate(bbalist):
            x0l, y0l, stdx0l, stdy0l = dobba.get_bba_results(
                method=method, error=True)
            if plotdiff:
                x0l -= x0li
                y0l -= y0li

            minx = np.min(np.hstack([minx, x0l[iok], x0l[inok]]))
            maxx = np.max(np.hstack([maxx, x0l[iok], x0l[inok]]))
            miny = np.min(np.hstack([miny, y0l[iok], y0l[inok]]))
            maxy = np.max(np.hstack([maxy, y0l[iok], y0l[inok]]))

            axx.errorbar(
                iok, x0l[iok], yerr=stdx0l[iok], fmt='o', color=cors[i],
                label=labels[i])
            ayy.errorbar(
                iok, y0l[iok], yerr=stdy0l[iok], fmt='o', color=cors[i],
                elinewidth=1)

            if not inok.size:
                continue

            axx.errorbar(
                inok, x0l[inok], yerr=stdx0l[inok], fmt='x', color=cors[i])
            ayy.errorbar(
                inok, y0l[inok], yerr=stdy0l[inok], fmt='x', color=cors[i],
                elinewidth=1, label=labels[i])

        if inok.size:
            ayy.legend(
                loc='upper right', bbox_to_anchor=(1.8, 0.2),
                fontsize='xx-small')

        axx.legend(
            loc='lower right', bbox_to_anchor=(1, 1), fontsize='xx-small')
        axx.grid(True)
        ayy.grid(True)

        minx = -1*xlim if xlim else minx*1.1
        maxx = xlim if xlim else maxx*1.1
        miny = -1*ylim if ylim else miny*1.1
        maxy = ylim if ylim else maxy*1.1

        axx.set_ylim([minx, maxx])
        ayy.set_ylim([miny, maxy])
        ayy.set_xlabel('BPM Index')

        if plotdiff:
            axx.set_ylabel(r'$\Delta X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$\Delta Y_0$ [$\mu$m]')
        else:
            axx.set_ylabel(r'$X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$Y_0$ [$\mu$m]')

        if fname:
            f.savefig(fname+'.svg')
            plt.close()
        else:
            f.show()

    # #### private methods ####
    def _do_bba(self):
        sofb = self.devices['sofb']
        sofb.nr_points = self.params.sofb_nrpoints
        loop_on = False
        if sofb.autocorrsts:
            loop_on = True
            print('SOFB feedback is enable, disabling it...')
            sofb.cmd_autocorr_turn_off()

        for i, bpm in enumerate(self._bpms2dobba):
            if self._stopevt.is_set():
                print('stopped!')
                return
            print('\n{0:03d}/{1:03d}'.format(i+1, len(self._bpms2dobba)))
            self._dobba_single_bpm(bpm)

        if loop_on:
            print('SOFB feedback was enable, restoring original state...')
            sofb.cmd_autocorr_turn_on()
        print('finished!')

    def _dobba_single_bpm(self, bpmname):
        """."""
        idx = self.data['bpmnames'].index(bpmname)
        quadname = self.data['quadnames'][idx]
        x0 = self.data['scancenterx'][idx]
        y0 = self.data['scancentery'][idx]
        quad = self.devices[quadname]
        sofb = self.devices['sofb']

        print('Doing BBA for BPM {:03d}: {:s}'.format(idx, bpmname))
        if not quad.pwrstate:
            print('\n    error: quadrupole ' + quadname + ' is Off.')
            self._stopevt.set()
            print('    exiting...')
            return

        korig = quad.strength
        deltakl = self.params.quad_deltakl
        cycling_curve = DoBBA.get_cycling_curve()

        print('cycling ' + quadname + ': ', end='')
        for _ in range(self.params.quad_nrcycles):
            print('.', end='')
            for fac in cycling_curve:
                quad.strength = korig + deltakl*fac
                _time.sleep(self.params.wait_quadrupole)

        print(' Ok!')

        nrsteps = self.params.meas_nrsteps
        dorbsx = self._calc_dorb_scan(self.params.deltaorbx, nrsteps//2)
        dorbsy = self._calc_dorb_scan(self.params.deltaorby, nrsteps//2)

        refx0, refy0 = sofb.refx, sofb.refy
        enblx0, enbly0 = sofb.bpmxenbl, sofb.bpmyenbl
        ch0, cv0 = sofb.kickch, sofb.kickcv

        enblx, enbly = 0*enblx0, 0*enbly0
        enblx[idx], enbly[idx] = 1, 1
        sofb.bpmxenbl, sofb.bpmyenbl = enblx, enbly
        _time.sleep(self.params.wait_sofb)

        orbini, orbpos, orbneg = [], [], []
        npts = 2*(nrsteps//2) + 1
        tmpl = '{:25s}'.format
        for i in range(npts):
            if self._stopevt.is_set():
                print('   exiting...')
                break
            print('    {0:02d}/{1:02d} --> '.format(i+1, npts), end='')

            print('orbit corr: ', end='')
            ret, fmet = self.correct_orbit(bpmname, x0+dorbsx[i], y0+dorbsy[i])
            if ret >= 0:
                txt = tmpl('Ok! in {:02d} iters'.format(ret))
            else:
                txt = tmpl('NOT Ok! dorb={:5.1f} um'.format(fmet))
            print(txt, end='')

            orbini.append(self.get_orbit())

            for j, fac in enumerate(cycling_curve):
                quad.strength = korig + deltakl*fac
                _time.sleep(self.params.wait_quadrupole)
                if not j:
                    orbpos.append(self.get_orbit())
                elif j == 1:
                    orbneg.append(self.get_orbit())

            dorb = orbpos[-1] - orbneg[-1]
            dorbx = dorb[:len(self.data['bpmnames'])]
            dorby = dorb[len(self.data['bpmnames']):]
            rmsx = np.sqrt(np.sum(dorbx*dorbx) / dorbx.shape[0])
            rmsy = np.sqrt(np.sum(dorby*dorby) / dorby.shape[0])
            print('rmsx = {:8.1f} rmsy = {:8.1f} um'.format(rmsx, rmsy))

        self.data['measure'][bpmname] = {
            'orbini': np.array(orbini), 'orbpos': np.array(orbpos),
            'orbneg': np.array(orbneg), 'deltakl': deltakl}

        print('    restoring initial conditions.')
        sofb.refx, sofb.refy = refx0, refy0
        sofb.bpmxenbl, sofb.bpmyenbl = enblx0, enbly0

        # restore correctors gently to do not kill the beam.
        factch, factcv = sofb.deltafactorch, sofb.deltafactorcv
        chn, cvn = sofb.kickch, sofb.kickcv
        dch, dcv = ch0 - chn, cv0 - cvn
        sofb.deltakickch, sofb.deltakickcv = dch, dcv
        nrsteps = np.ceil(max(np.abs(dch).max(), np.abs(dcv).max()) / 1.0)
        for i in range(int(nrsteps)):
            sofb.deltafactorch = (i+1)/nrsteps * 100
            sofb.deltafactorcv = (i+1)/nrsteps * 100
            _time.sleep(self.params.wait_sofb)
            sofb.cmd_applycorr()
            _time.sleep(self.params.wait_correctors)
        sofb.deltakickch, sofb.deltakickcv = dch*0, dcv*0
        sofb.deltafactorch, sofb.deltafactorcv = factch, factcv
        print('')
