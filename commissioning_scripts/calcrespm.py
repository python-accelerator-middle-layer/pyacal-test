#!/usr/bin/env python-sirius
"""."""

import numpy as np
import pymodels
import pyaccel


class CalcRespm():

    def __init__(self, model, dim='4d'):
        self.model = model
        if self.model.harmonic_number == 828:
            self.fam_data = pymodels.bo.families.get_family_data(self.model)
        elif self.model.harmonic_number == 864:
            self.fam_data = pymodels.si.families.get_family_data(self.model)
        else:
            raise Exception('Set models: BO or SI')
        self.dim = dim
        self.bpms = self.fam_data['BPM']['index']
        self.ch = self.fam_data['CH']['index']
        self.cv = self.fam_data['CV']['index']
        self.nbpm = len(self.bpms)
        self.nch = len(self.ch)
        self.ncv = len(self.cv)
        self.mxx = np.zeros((self.nbpm, self.nch))
        self.mxy = np.zeros((self.nbpm, self.ncv))
        self.myx = np.zeros((self.nbpm, self.nch))
        self.myy = np.zeros((self.nbpm, self.ncv))
        self.respm = self.get_respm()

    def get_respm(self, model=None):
        if model is None:
            model = self.model

        if self.dim == '6d':
            M, T = pyaccel.tracking.find_m66(
                self.model, indices='open', closed_orbit=[0, 0, 0, 0, 0, 0])
        else:
            M, T = pyaccel.tracking.find_m44(
                self.model, indices='open', energy_offset=0.0,
                closed_orbit=[0, 0, 0, 0])

        T = np.array(T)
        total_len_ch = np.zeros(self.nch)
        total_len_cv = np.zeros(self.ncv)
        for jx in range(self.nch):
            total_len_ch[jx] = self.model[self.ch[jx][0]].length
        for jy in range(self.ncv):
            total_len_cv[jy] = self.model[self.cv[jy][0]].length
        D = np.eye(M.shape[0])
        for i in range(self.nbpm):
            Ri = T[self.bpms[i][0], :, :]
            DMi = D - np.dot(np.dot(Ri, M), np.linalg.inv(Ri))
            for jx in range(self.nch):
                self.mxx[i, jx], self.myx[i, jx], _, _ = self._getC(
                    T, DMi, Ri, self.bpms[i][0], self.ch[jx][0],
                    total_len_ch[jx]
                )
            for jy in range(self.ncv):
                _, _, self.mxy[i, jy], self.myy[i, jy] = self._getC(
                    T, DMi, Ri, self.bpms[i][0], self.cv[jy][0],
                    total_len_cv[jy]
                )
        Mx = np.concatenate((self.mxx, self.myx))
        My = np.concatenate((self.mxy, self.myy))
        return np.concatenate((Mx, My), axis=1)

    def _getC(self, T, DM, R, row, col, total_len):
        if row > col:
            Rij = np.dot(R, np.linalg.inv(T[col, :, :]))
        else:
            Rij = np.dot(
                R, np.dot(
                    T[-1, :, :], np.linalg.inv(T[col, :, :])))

        c = np.dot(np.linalg.inv(DM), Rij)
        cxx = c[0, 1] - total_len * c[0, 0] / 2
        cyx = c[2, 1] - total_len * c[2, 0] / 2
        cxy = c[0, 3] - total_len * c[0, 2] / 2
        cyy = c[2, 3] - total_len * c[2, 2] / 2
        return cxx, cyx, cxy, cyy
