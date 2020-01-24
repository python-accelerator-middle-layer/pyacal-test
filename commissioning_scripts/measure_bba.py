"""Main module."""
import time as _time
from threading import Thread as _Thread, Event as _Event

from copy import deepcopy as _dcopy
import numpy as np

from siriuspy.csdevice.orbitcorr import SOFBFactory
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.search import BPMSearch, PSSearch

import pyaccel as _pyacc
from pymodels.middlelayer.devices import SOFB
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat
from .base import BaseClass


def get_default_correctors(model, fam_data):
    return list()


class BBAParams:
    SCAN = 0
    SEARCH = 1

    def __init__(self):
        self.bba_method = 0  # 0 - Scan    1 - Search
        self.corrx_dkick_negative = 50  # [urad]
        self.corrx_dkick_positive = 50  # [urad]
        self.corry_dkick_negative = 50  # [urad]
        self.corry_dkick_positive = 50  # [urad]
        self.corr_nrsteps = 10
        self.quad_deltakl = 0.02  # [1/m]
        self.wait_correctors = 2  # [s]
        self.wait_quadrupole = 2  # [s]
        self.timeout_quad_turnon = 10  # [s]
        self.wait_sofb = 3  # [s]
        self.sofb_nrpoints = 10
        self.bpms_info = dict()

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
        bpm_info = dict()
        for i, qidx in enumerate(quads_bba_idx):
            name = model[qidx].fam_name
            idc = fam_data[name]['index'].index([qidx, ])
            sub = fam_data[name]['subsection'][idc]
            inst = fam_data[name]['instance'][idc]
            name = 'QS' if name.startswith('S') else name
            qname = 'SI-{0:s}:PS-{1:s}-{2:s}'.format(sub, name, inst)
            qname = qname.strip('-')

            sub = fam_data['BPM']['subsection'][i]
            inst = fam_data['BPM']['instance'][i]
            bname = 'SI-{0:s}:DI-BPM-{1:s}'.format(sub, inst)
            bname = bname.strip('-')
            bpm_info[bname] = {'quadname': qname, 'bpmindex': i}
        return quads_bba_idx, bpm_info

    def fill_bpms_info(self, model, respmat=None):
        respmat_calc = OrbRespmat(model, 'SI')
        quads_idx, bpm_info = BBAParams.get_default_quads(
            model, respmat_calc.fam_data)

        respmat = respmat_calc.get_respm()
        respmat_calc.ch = quads_idx
        respmat_calc.cv = quads_idx
        quadrespmat = respmat_calc.get_respm()

        for bpm, info in bpm_info.items():
            idx = info['bpmindex']
            info['quadrespx'] = quadrespmat[:, idx]
            info['quadrespy'] = quadrespmat[:, idx + len(quads_idx)]
            self.bpms_info[bpm] = info


class DoBBA(BaseClass):

    def __init__(self):
        super().__init__()
        self.params = BBAParams()
        self.devices['sofb'] = SOFB('SI')
        self.connect_to_objects()
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

    def connect_to_objects(self):
        for data in self.params.bpms_info.values():
            qname = data.get('quadname')
            cxname = data.get('corrxname')
            cyname = data.get('corryname')
            if qname and qname not in self.devices:
                self.devices[qname] = Quad(qname)
            if cxname and cxname not in self.devices:
                self.devices[cxname] = Corr(cxname)
            if cyname and cyname not in self.devices:
                self.devices[cyname] = Corr(cyname)

    def _do_bba(self):
        self.devices['sofb'].nr_points = self.params.sofb_nrpoints
        for bpm in self.params.bpms_info:
            if self._stopevt.is_set():
                return
            self._dobba_single_bpm_scan(bpm)

    def _calc_dkicks_scan(self, dkickneg, dkickpos, nrpts):
        dkickspos = np.linspace(dkickpos, 0, nrpts+1)[:-1]
        dkicksneg = np.linspace(-dkickneg, 0, nrpts+1)[:-1]
        dkicks = np.array([dkicksneg, dkickspos]).T.flatten()
        dkicks = np.hstack([dkicks, 0])
        return dkicks

    def _dobba_single_bpm(self, bpmname):
        quad_name = self.params.bpms_info[bpmname]['quadname']
        corrx_name = self.params.bpms_info[bpmname]['corrxname']
        corry_name = self.params.bpms_info[bpmname]['corryname']
        quad = self.devices[quad_name]
        corrx = self.devices[corrx_name]
        corry = self.devices[corry_name]

        quad.turn_on(self.params.timeout_quad_turnon)
        if not quad.pwr_state:
            print('error: quadrupole ' + quad.name + ' is Off.')
            self._stopevt.set()
            return

        sofb = self.devices['sofb']

        nrsteps = self.params.corr_nrsteps
        dkickxneg = self.params.corrx_dkick_negative
        dkickxpos = self.params.corrx_dkick_positive
        dkickyneg = self.params.corry_dkick_negative
        dkickypos = self.params.corry_dkick_positive
        nrsteps = self.params.corr_nrsteps
        dkicksx = self._calc_dkicks_scan(dkickxneg, dkickxpos, nrsteps)
        dkicksy = self._calc_dkicks_scan(dkickyneg, dkickypos, nrsteps)
        deltakl = self.params.quad_deltakl

        corrx_kicks, corry_kicks = [], []
        orb_pos, orb_neg = [], []
        orig_kickx = corrx.value
        orig_kicky = corrx.value
        if self.params.bba_method == BBAParams.SCAN:
            npts = 2*nrsteps + 1
        else:
            npts = nrsteps
        for i in range(npts):
            if self._stopevt.is_set():
                break
            corrx_kicks.append(corrx.value)
            corry_kicks.append(corry.value)

            korig = quad.value
            quad.value = korig + deltakl/2
            _time.sleep(self.params.wait_quadrupole)

            sofb.reset()
            sofb.wait(self.params.wait_sofb)
            orb_pos.append(np.hstack([sofb.orbx, sofb.orby]))

            quad.value = korig - deltakl/2
            _time.sleep(self.params.wait_quadrupole)

            sofb.reset()
            sofb.wait(self.params.wait_sofb)
            orb_neg.append(np.hstack([sofb.orbx, sofb.orby]))

            quad.value = korig

            if self.params.bba_method == BBAParams.SCAN:
                dkx = dkicksx[i]
                dky = dkicksy[i]
            else:
                dorb = (orb_pos[-1]-orb_neg[-1])/deltakl
                xcen, ycen = self.calc_offset(bpmname, dorb)
                dkx, dky = self.calc_orbcorr(bpmname, xcen, ycen)

            if i < npts-1:
                corrx.value = orig_kickx + dkx
                corry.value = orig_kicky + dky
                _time.sleep(self.params.wait_correctors)

        corrx.value = orig_kickx
        corry.value = orig_kicky

        self.data[bpmname]['corrxkicks'] = np.array(corrx_kicks)
        self.data[bpmname]['corrykicks'] = np.array(corry_kicks)
        self.data[bpmname]['orbpos'] = np.array(orb_pos)
        self.data[bpmname]['orbneg'] = np.array(orb_neg)

        quad.turn_off(self.params.timeout_quad_turnon)
        if quad.pwr_state:
            print('error: quadrupole ' + quad.name + ' is still On.')
            self._stopevt.set()

    def calc_offset(self, bpmname, dorb):
        respx = self.params.bpms_info[bpmname]['quadrespx']
        respy = self.params.bpms_info[bpmname]['quadrespy']
        isskew = '-QS' in self.params.bpms_info[bpmname]['quadname']

        mat = np.array([respx, respy]).T
        x0, y0 = np.linalg.lstsq(mat, dorb)
        if isskew:
            return y0, x0
        return x0, y0

    def calc_orbcorr(self, bpmname, x0, y0):
        respx = self.params.bpms_info[bpmname]['corrxresp']
        respy = self.params.bpms_info[bpmname]['corryresp']

        mat = -np.array([respx, respy]).T
        kicks, *_ = np.linalg.lstsq(mat, [x0, y0])
        dkx, dky = kicks
        return dkx, dky
