#!/usr/bin/env python-sirius
"""."""

import numpy as np
from pymodels import tb, bo, ts, si
import pyaccel


class Respmat():
    """."""

    def __init__(self, model, dim='4d'):
        """."""
        self.model = model
        if self.model.harmonic_number == 828:
            self.fam_data = bo.families.get_family_data(self.model)
        elif self.model.harmonic_number == 864:
            self.fam_data = si.families.get_family_data(self.model)
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

    def get_respm(self, model=None):
        """."""
        if model is None:
            model = self.model
        if self.dim == '6d':
            M, T = pyaccel.tracking.find_m66(
                model, indices='open', closed_orbit=[0, 0, 0, 0, 0, 0])
        else:
            M, T = pyaccel.tracking.find_m44(
                model, indices='open', energy_offset=0.0,
                closed_orbit=[0, 0, 0, 0])
        T = np.array(T)
        total_len_ch = np.zeros(self.nch)
        total_len_cv = np.zeros(self.ncv)
        for jx in range(self.nch):
            total_len_ch[jx] = model[self.ch[jx][0]].length
        for jy in range(self.ncv):
            total_len_cv[jy] = model[self.cv[jy][0]].length
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


class Trajmat():
    """."""

    def __init__(self, model, acc):
        """."""
        self.model = model
        self.acc = acc
        self.bpms = []
        self.corrh = []
        self.corrv = []
        self.corrs = []
        self.fam = []

    def model_trajmat(self, model=None, acc=None, meth='middle'):
        """."""
        if model is None:
            model = self.model
        if acc is None:
            acc = self.acc

        ncorrs = 0
        if acc == 'TB':
            self.fam = tb.get_family_data(model)
        elif acc == 'BO':
            self.fam = bo.get_family_data(model)
        elif acc == 'TS':
            self.fam = ts.get_family_data(model)
        elif acc == 'SI':
            self.fam = si.get_family_data(model)
            ncorrs += 1  # RF Frequency as corrector

        self.bpms = self.fam['BPM']['index']
        self.bpms = np.array([b[0] for b in self.bpms])
        self.corrh = self.fam['CH']['index']

        if acc == 'TS':
            ejesept = pyaccel.lattice.find_indices(
                model, 'fam_name', 'EjeSeptG')
            segs = len(ejesept)
            self.corrh.append([ejesept[segs//2]])
            self.corrh = sorted(self.corrh)

        self.corrv = self.fam['CV']['index']
        self.corrs = self.corrh + self.corrv
        ncorrs += len(self.corrs)

        _, cumulmat = pyaccel.tracking.find_m44(
            model, indices='open', closed_orbit=[0, 0, 0, 0])

        trajmat = np.zeros((2*len(self.bpms), ncorrs))
        corrtype = 'horizontal'
        for idx, corr in enumerate(self.corrs):
            if corr not in self.corrh:
                corrtype = 'vertical'
            corr_len = pyaccel.lattice.get_attribute(
                model, 'length', corr[0])[0]
            trajmat[:, idx] = self._get_respmat_line(
                cumulmat, corr, length=corr_len,
                kxl=0, kyl=0, ksxl=0, ksyl=0,
                cortype=corrtype, meth=meth)
        return trajmat

    def _get_respmat_line(self, cumul_mat, indcs, length,
                          kxl=0, kyl=0, ksxl=0, ksyl=0,
                          cortype='vertical', meth='middle'):
        idx = 3 if cortype.startswith('vertical') else 1
        cor = indcs[0]
        if meth.lower().startswith('end'):
            cor = indcs[-1]+1
        elif meth.lower().startswith('mid'):
            # create a symplectic integrator of second order
            # for the last half of the element:
            drift = np.eye(4, dtype=float)
            drift[0, 1] = length/2 / 2
            drift[2, 3] = length/2 / 2
            quad = np.eye(4, dtype=float)
            quad[1, 0] = -kxl/2
            quad[3, 2] = -kyl/2
            quad[1, 2] = -ksxl/2
            quad[3, 0] = -ksyl/2
            half_cor = np.dot(np.dot(drift, quad), drift)

        m0c = cumul_mat[cor]
        if meth.lower().startswith('mid'):
            m0c = np.linalg.solve(half_cor, m0c)
        mat = np.linalg.solve(m0c.T, cumul_mat[self.bpms].transpose((0, 2, 1)))
        mat = mat.transpose(0, 2, 1)
        # if meth.lower().startswith('mid'):
        #     mat = np.dot(mat, half_cor)
        respx = mat[:, 0, idx]
        respy = mat[:, 2, idx]
        respx[self.bpms < indcs[0]] = 0
        respy[self.bpms < indcs[0]] = 0
        return np.hstack([respx, respy])
