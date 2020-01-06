#!/usr/bin/env python-sirius
"""."""

from copy import deepcopy as _dcopy
import numpy as np
import pymodels
import pyaccel
from apsuite.commissioning_scripts.calc_orbcorr_mat import Respmat


class FitQuads():

    def __init__(self, model):
        self.model = model
        self.respm = Respmat(model=model, dim='6d')
        self.matrix = self.respm.get_respm()
        self.vector = self.matrix.flatten()
        self.quadsidx = self.respm.fam_data['QN']['index']
        # self.kmatrix = self._calc_kmatrix()

    def _vector2respm(self, vector):
        row = self.matrix.shape[0]
        col = self.matrix.shape[1]
        return np.reshape(vector, (row, col))

    def _calc_kmatrix(self, deltak=1e-6):
        self.kquads = pyaccel.lattice.get_attribute(
            self.model, 'K', self.quadsidx)

        nquads = len(self.kquads)
        kmatrix = np.zeros((len(self.vector), nquads))

        for i, idx in enumerate(self.quadsidx):
            for ii, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    self.model, 'K', idx_seg, self.kquads[i][ii] +
                    deltak)
            new_respm = Respmat(model=self.model, dim='6d').get_respm()
            dmdk = (new_respm - self.matrix)/deltak
            kmatrix[:, i] = dmdk.flatten()
            for ii, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    self.model, 'K', idx_seg, self.kquads[i][ii])
        return kmatrix

    def chi2(self, M1, M2):
        return np.sqrt(np.mean((M1-M2)**2))

    def fit_matrices(self, model=None, measmat=None, niter=10, nsv=None):
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
        quadidx = self.respm.fam_data['QN']['index']
        print('Initial Matrix Deviation: {:.16e}'.format(chi2_old))
        u, s, v = np.linalg.svd(self.kmatrix, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        if nsv is not None:
            inv_s[nsv:] = 0
        inv_s = np.diag(inv_s)
        inv_kmatrix = np.dot(np.dot(v.T, inv_s), u.T)

        grad_quads = np.array(
            pyaccel.lattice.get_attribute(mod, 'K', quadidx))
        grad_delta = np.zeros(inv_kmatrix.shape[0])
        tol = 1e-16

        for n in range(niter):
            grad_delta += np.dot(inv_kmatrix, diffmat.flatten())

            for i, idx in enumerate(quadidx):
                for ii, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        mod, 'K', idx_seg, grad_quads[i][ii] + grad_delta[i])

            fitmat = Respmat(model=mod, dim='6d').get_respm()
            diffmat = measmat - fitmat
            print('Iter {:.1f}'.format(n+1))
            chi2_new = self.chi2(measmat, fitmat)
            print('Matrix Deviation: {:.16e}'.format(chi2_new))

            if (chi2_old - chi2_new) < tol:
                print('Limit Reached!')
                grad_delta -= np.dot(inv_kmatrix, diffmat.flatten())

                for i, idx in enumerate(quadidx):
                    for ii, idx_seg in enumerate(idx):
                        pyaccel.lattice.set_attribute(
                            mod, 'K', idx_seg,
                            grad_quads[i][ii] + grad_delta[i])
                fitmat = Respmat(model=mod, dim='6d').get_respm()
                break
            else:
                chi2_old = chi2_new
        fit_grad_quads = grad_quads[:, 0] + grad_delta
        return fitmat, fit_grad_quads, mod
