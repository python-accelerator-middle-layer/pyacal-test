"""Main module."""

from collections import namedtuple
from copy import deepcopy as _dcopy
import numpy as np

from qtpy.QtCore import QObject, QThread

from siriuspy.csdevice.orbitcorr import SOFBFactory
from siriuspy.namesys import SiriusPVName as _PVName

import pyaccel as _pyacc


def get_default_quads(model, fam_data):
    quads_idx = _dcopy(fam_data['QN']['index'])
    quads_idx.extend(fam_data['QS']['index'])
    quads_idx = [idx[len(idx)//2] for idx in quads_idx]
    quads_pos = np.array(_pyacc.lattice.find_spos(model, quads_idx))

    bpms_idx = [idx[0] for idx in fam_data['BPM']['index']]
    bpms_pos = np.array(_pyacc.lattice.find_spos(model, bpms_idx))

    diff = np.abs(bpms_idx[:, None] - quads_idx[None, :])
    idx = np.argmin(diff, axis=1)
    bba_idx = np.array(quads_idx)[idx]



class DoBBA(QObject):
    Methods = namedtuple('Methods', ['Orbit', 'Trajectory'])(0, 1)

    def __init__(self, method=None):
        super().__init__()
        self._bpms_params = dict()
        self._method = method if method is not None else 0
        self._default_corrs = get_default_correctors()
        self._default_quads = get_detault_quads()

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, val):
        if val in self.Methods:
            self._method = val

    @property
    def method_name(self):
        return self.Methods._fields[self._method]

    @method_name.setter
    def method_name(self, val):
        if val in self.Methods._fields:
            self._method = self.Methods._fields.index(val)

    @property
    def bpms_dict(self):
        return _dcopy(self.bpms_dict)

    @bpms_dict.setter
    def bpms_dict(self, value):
        value = self._check_consistency(value)
        if value:
            self.bpms_dict = value
            self.bpms_order = sorted(value.keys())

    def _check_consistency(self, value):
        val = dict()
        for name, data in value.items():
            name = _PVName(name)
            dta = _dcopy(data)
            if {'quad', 'corr'} - data.keys():
                return False
            dta['quad'] = _PVName(dta['quad'])
            dta['corr'] = _PVName(dta['corr'])
            val[name] = dta
        return val

    def start(self):
        self.connect_to_objects()
        self.do_bba()

    def connect_to_objects(self):
        self._bpms = dict()
        self._quads = dict()
        self._corrs = dict()
        for bpm, data in self.bpms_dict.items():
            self._bpms[bpm] = BPM(bpm)
            self._quads[data['quad']] = Quad(data['quad'])
            self._corrs[data['corr']] = Corr(data['corr'])
        self._orbit = Orbit()

    def do_bba(self):
        for bpm, data in self.bpms_dict.items():
            quad = data['quad']
            corr = data['corr']
            plane = data['plane']
            self._dobba_single_bpm(bpm, quad, corr, plane)

    def _dobba_single_bpm(self, bpm, quad, corr, plane):
        quad = self._quads[quad]
        bpm = self._bpms[bpm]
        corr = self._corrs[corr]
