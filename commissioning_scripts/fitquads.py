#!/usr/bin/env python-sirius
"""."""

import numpy as np
import pymodels
import pyaccel
import time
from apsuite.commissioning_scripts.calcrespm import CalcRespm


class FitQuads():

    def __init__(self):
        bomod = pymodels.bo.create_accelerator()
        sept = pyaccel.lattice.find_indices(bomod, 'fam_name', 'InjSept')
        bomod = pyaccel.lattice.shift(bomod, sept[0])
        t = time.time()
        self.respm = CalcRespm(model=bomod)
        ft = time.time() - t
        print(ft)
        self.vector = self._respm2vector(matrix=self.respm.respm)
        self.quads_qf = self.respm.fam_data['QF']['index']
        self.quads_qd = self.respm.fam_data['QD']['index']
        self.Krespm = self._calcKrespm()

    def _respm2vector(self, matrix):
        return matrix.flatten()

    def _vector2respm(self, vector):
        row = self.respm.respm.shape[0]
        col = self.respm.respm.shape[1]
        return np.reshape(vector, (row, col))

    def _calcKrespm(self):
        self.kqfs = pyaccel.lattice.get_attribute(
            self.respm.model, 'K', self.quads_qf)
        self.kqds = pyaccel.lattice.get_attribute(
            self.respm.model, 'K', self.quads_qd)
        delta = 0.05
        deltakqfs = [k*(1+delta) for k in self.kqfs]
        deltakqds = [k*(1+delta) for k in self.kqds]
        nquads = len(self.quads_qd) + len(self.quads_qf)
        Krespm = np.zeros((len(self.vector), nquads))
        for kqfidx in range(len(deltakqfs)):
            pyaccel.lattice.set_attribute(
                self.respm.model, 'K', self.quads_qf[kqfidx], deltakqfs[kqfidx]
                )
            new_respm = self.respm.get_respm(model=self.respm.model)
            Krespm[:, kqfidx] = self._respm2vector(
                (new_respm - self.respm.respm)/delta)
            pyaccel.lattice.set_attribute(
                self.respm.model, 'K', self.quads_qf[kqfidx], self.kqfs[kqfidx]
                )
        for kqdidx in range(len(deltakqds)):
            pyaccel.lattice.set_attribute(
                self.respm.model, 'K', self.quads_qd[kqdidx], deltakqfs[kqdidx]
                )
            new_respm = self.respm.get_respm(model=self.respm.model)
            Krespm[:, len(kqfidx) + kqdidx] = self._respm2vector(
                (new_respm - self.respm.respm)/delta)
            pyaccel.lattice.set_attribute(
                self.respm.model, 'K', self.quads_qd[kqdidx], self.kqfs[kqdidx]
                )
        return Krespm
