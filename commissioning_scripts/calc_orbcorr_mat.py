#!/usr/bin/env python-sirius
"""."""

import numpy as np
from pymodels import tb, bo, ts, si
import pyaccel


class OrbRespmat():
    """."""

    def __init__(self, model, acc, dim='4d'):
        """."""
        self.model = model
        self.acc = acc
        if self.acc == 'BO':
            self.fam_data = bo.families.get_family_data(self.model)
        elif self.acc == 'SI':
            self.fam_data = si.families.get_family_data(self.model)
        else:
            raise Exception('Set models: BO or SI')
        self.dim = dim
        self.bpms = self._get_idx(self.fam_data['BPM']['index'])
        self.ch = self._get_idx(self.fam_data['CH']['index'])
        self.cv = self._get_idx(self.fam_data['CV']['index'])

    @staticmethod
    def _get_idx(indcs):
        return np.array([idx[0] for idx in indcs])

    def get_respm(self):
        """."""
        if self.dim == '6d':
            M, T = pyaccel.tracking.find_m66(
                self.model, indices='open', closed_orbit=[0, 0, 0, 0, 0, 0])
        else:
            M, T = pyaccel.tracking.find_m44(
                self.model, indices='open', closed_orbit=[0, 0, 0, 0])

        nch = len(self.ch)
        respmat = []
        corrs = np.hstack([self.ch, self.cv])
        for i, corr in enumerate(corrs):
            Rc = T[corr, :, :]
            Rb = T[self.bpms, :, :]
            corr_len = self.model[corr].length
            KL = self.model[corr].KL
            KsL = self.model[corr].KsL
            respx, respy = self._get_respmat_line(
                Rc, Rb, M, corr, corr_len,
                kxl=KL, kyl=-KL, ksxl=KsL, ksyl=KsL)
            if i < nch:
                respmat.append(respx)
            else:
                respmat.append(respy)
        return np.array(respmat).T

    def _get_respmat_line(self, Rc, Rb, M, corr, length,
                          kxl=0, kyl=0, ksxl=0, ksyl=0):
        # create a symplectic integrator of second order
        # for the last half of the element:
        drift = np.eye(Rc.shape[0], dtype=float)
        drift[0, 1] = length/2 / 2
        drift[2, 3] = length/2 / 2
        quad = np.eye(Rc.shape[0], dtype=float)
        quad[1, 0] = -kxl/2
        quad[3, 2] = -kyl/2
        quad[1, 2] = -ksxl/2
        quad[3, 0] = -ksyl/2
        half_cor = drift @ quad @ drift
        Rc = half_cor @ Rc

        Mc = np.linalg.solve(Rc.T, (Rc @ M).T).T  # Mc = Rc M Rc^-1
        Mci = np.eye(Mc.shape[0], dtype=float) - Mc

        small = self.bpms < corr
        large = np.logical_not(small)

        RcbL = np.linalg.solve(Rc.T, Rb.transpose((0, 2, 1)))
        RcbL = RcbL.transpose((0, 2, 1))
        RcbS = RcbL[small] @ Mc
        RcbL = RcbL[large]

        RcbL = np.linalg.solve(Mci.T, RcbL.transpose((0, 2, 1)))
        RcbL = RcbL.transpose((0, 2, 1))
        RcbS = np.linalg.solve(Mci.T, RcbS.transpose((0, 2, 1)))
        RcbS = RcbS.transpose((0, 2, 1))

        respxx = np.zeros(len(self.bpms))
        respyx = np.zeros(len(self.bpms))
        respxy = np.zeros(len(self.bpms))
        respyy = np.zeros(len(self.bpms))

        respxx[large] = RcbL[:, 0, 1]
        respyx[large] = RcbL[:, 2, 1]
        respxx[small] = RcbS[:, 0, 1]
        respyx[small] = RcbS[:, 2, 1]
        respx = np.hstack([respxx, respyx])

        respxy[large] = RcbL[:, 0, 3]
        respyy[large] = RcbL[:, 2, 3]
        respxy[small] = RcbS[:, 0, 3]
        respyy[small] = RcbS[:, 2, 3]
        respy = np.hstack([respxy, respyy])
        return respx, respy


class TrajRespmat():
    """."""

    def __init__(self, model, acc):
        """."""
        self.model = model
        self.acc = acc
        if acc == 'TB':
            self.fam_data = tb.get_family_data(model)
        elif acc == 'BO':
            self.fam_data = bo.get_family_data(model)
        elif acc == 'TS':
            self.fam_data = ts.get_family_data(model)
        elif acc == 'SI':
            self.fam_data = si.get_family_data(model)

        self.bpms = self._get_idx(self.fam_data['BPM']['index'])
        self.ch = self.fam_data['CH']['index']

        if acc == 'TS':
            ejesept = pyaccel.lattice.find_indices(
                model, 'fam_name', 'EjeSeptG')
            segs = len(ejesept)
            self.ch.append([ejesept[segs//2]])
            self.ch = sorted(self.ch)

        self.ch = self._get_idx(self.ch)
        self.cv = self._get_idx(self.fam_data['CV']['index'])

    @staticmethod
    def _get_idx(indcs):
        return np.array([idx[0] for idx in indcs])

    def get_respm(self):
        """."""
        _, cumulmat = pyaccel.tracking.find_m44(
            self.model, indices='open', closed_orbit=[0, 0, 0, 0])

        trajmat = []
        corrs = np.hstack([self.ch, self.cv])
        for idx, corr in enumerate(corrs):
            Rc = cumulmat[corr]
            Rb = cumulmat[self.bpms]
            corr_len = self.model[corr].length
            KL = self.model[corr].KL
            KsL = self.model[corr].KsL
            respx, respy = self._get_respmat_line(
                Rc, Rb, corr, length=corr_len,
                kxl=KL, kyl=-KL, ksxl=KsL, ksyl=KsL)
            if idx < len(self.ch):
                trajmat.append(respx)
            else:
                trajmat.append(respy)
        return np.array(trajmat).T

    def _get_respmat_line(self, Rc, Rb, corr, length,
                          kxl=0, kyl=0, ksxl=0, ksyl=0):
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
        half_cor = drift @ quad @ drift

        Rc = half_cor @ Rc

        large = self.bpms > corr

        Rb = Rb[large, :, :]
        Rcb = np.linalg.solve(Rc.T, Rb.transpose((0, 2, 1)))
        Rcb = Rcb.transpose(0, 2, 1)

        respxx = np.zeros(len(self.bpms), dtype=float)
        respyx = np.zeros(len(self.bpms), dtype=float)
        respxy = np.zeros(len(self.bpms), dtype=float)
        respyy = np.zeros(len(self.bpms), dtype=float)

        respxx[large] = Rcb[:, 0, 1]
        respyx[large] = Rcb[:, 2, 1]
        respx = np.hstack([respxx, respyx])

        respxy[large] = Rcb[:, 0, 3]
        respyy[large] = Rcb[:, 2, 3]
        respy = np.hstack([respxy, respyy])
        return respx, respy
