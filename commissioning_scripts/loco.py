#!/usr/bin/env python-sirius
"""."""

from copy import deepcopy as _dcopy
import numpy as np
import pymodels
import pyaccel
from apsuite.commissioning_scripts.calc_orbcorr_mat import Respmat


class FitQuads():

    QUAD_FAM = [
        'QFA', 'QDA', 'QDB2', 'QFB', 'QDB1', 'QDP2', 'QFP', 'QDP1',
        'Q1', 'Q2', 'Q3', 'Q4']

    def __init__(self, model, dim='6d', use_families=False):
        self.model = model
        self.dim = dim
        self.respm = Respmat(model=model, dim=dim)
        self.matrix = self.respm.get_respm()
        self.vector = self.matrix.flatten()
        self.use_families = use_families
        self.quadsidx = []
        for fam_name in self.QUAD_FAM:
            self.quadsidx.append(self.respm.fam_data[fam_name]['index'])
        self.kquads = []
        # self.kmatrix = self._calc_kmatrix()

    def _vector2respm(self, vector):
        row = self.matrix.shape[0]
        col = self.matrix.shape[1]
        return np.reshape(vector, (row, col))

    def _calc_kmatrix(self, deltak=1e-6):
        for qi in self.quadsidx:
            self.kquads.append(pyaccel.lattice.get_attribute(
                self.model, 'K', qi))
        if self.use_families:
            nquads = len(self.kquads)
        else:
            nquads = len([q for sub in self.kquads for q in sub])
        kmatrix = np.zeros((len(self.vector), nquads))

        count = 0
        for i1, fam in enumerate(self.quadsidx):
            for i2, idx in enumerate(fam):
                for i3, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        self.model, 'K', idx_seg, self.kquads[i1][i2][i3] +
                        deltak)
                if not self.use_families:
                    new_respm = Respmat(model=self.model, dim='6d').get_respm()
                    dmdk = (new_respm - self.matrix)/deltak
                    kmatrix[:, count] = dmdk.flatten()
                    count += 1

            if self.use_families:
                new_respm = Respmat(model=self.model, dim='6d').get_respm()
                dmdk = (new_respm - self.matrix)/deltak
                kmatrix[:, i1] = dmdk.flatten()

            for i2, idx in enumerate(fam):
                for i3, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        self.model, 'K', idx_seg, self.kquads[i1][i2][i3])
        return kmatrix

    def chi2(self, M1, M2):
        return np.sqrt(np.mean((M1-M2)**2))

    def fit_matrices(self, model=None, measmat=None, kmatrix=None, niter=10,                nsv=None):
        if model is None:
            model = self.model
            modelmat = self.matrix
            mod = _dcopy(model)
        else:
            mod = _dcopy(model)
            respmmodel = Respmat(model=mod, dim='6d')
            modelmat = respmmodel.get_respm()

        diffmat = measmat - modelmat
        chi2_old = self.chi2(measmat, modelmat)
        qnidx = self.respm.fam_data['QN']['index']
        print('Initial Matrix Deviation: {:.16e}'.format(chi2_old))
        is_fam = False
        if kmatrix.shape[1] < len(qnidx):
            is_fam = True
        u, s, v = np.linalg.svd(kmatrix, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        if nsv is not None:
            inv_s[nsv:] = 0
        inv_s = np.diag(inv_s)
        inv_kmatrix = np.dot(np.dot(v.T, inv_s), u.T)

        if is_fam:
            grad_quads = []
            for qi in self.quadsidx:
                grad_quads.append(pyaccel.lattice.get_attribute(
                    self.model, 'K', qi))
        else:
            grad_quads = np.array(
                pyaccel.lattice.get_attribute(mod, 'K', qnidx))

        grad_delta = np.zeros(inv_kmatrix.shape[0])
        tol = 1e-16

        for n in range(niter):
            grad_delta += np.dot(inv_kmatrix, diffmat.flatten())

            if is_fam:
                for i1, fam in enumerate(self.quadsidx):
                    for i2, idx in enumerate(fam):
                        for i3, idx_seg in enumerate(idx):
                            pyaccel.lattice.set_attribute(
                                mod, 'K', idx_seg, grad_quads[i1][i2][i3] +
                                grad_delta[i1])
            else:
                for i1, idx in enumerate(qnidx):
                    for i2, idx_seg in enumerate(idx):
                        pyaccel.lattice.set_attribute(
                            mod, 'K', idx_seg, grad_quads[i1][i2] +
                            grad_delta[i1])

            fitmat = Respmat(model=mod, dim='6d').get_respm()
            diffmat = measmat - fitmat
            print('Iter {:.1f}'.format(n+1))
            chi2_new = self.chi2(measmat, fitmat)
            print('Matrix Deviation: {:.16e}'.format(chi2_new))

            if np.isnan(chi2_new):
                print('Penalty Function is NaN!')
                break

            if (chi2_old - chi2_new) < tol:
                print('Limit Reached!')
                grad_delta -= np.dot(inv_kmatrix, diffmat.flatten())

                if is_fam:
                    for i1, fam in enumerate(self.quadsidx):
                        for i2, idx in enumerate(fam):
                            for i3, idx_seg in enumerate(idx):
                                pyaccel.lattice.set_attribute(
                                    mod, 'K', idx_seg, grad_quads[i1][i2][i3] +
                                    grad_delta[i1])
                else:
                    for i1, idx in enumerate(qnidx):
                        for i2, idx_seg in enumerate(idx):
                            pyaccel.lattice.set_attribute(
                                mod, 'K', idx_seg, grad_quads[i1][i2] +
                                grad_delta[i1])
                fitmat = Respmat(model=mod, dim='6d').get_respm()
                break
            else:
                chi2_old = chi2_new
        return fitmat, mod
