#!/usr/bin/env python-sirius
"""."""

from copy import deepcopy as _dcopy
import numpy as np
import pyaccel
from apsuite.commissioning_scripts.calc_orbcorr_mat import Respmat


class FitQuads():
    """."""

    QUAD_FAM = [
        'QFA', 'QDA', 'QDB2', 'QFB', 'QDB1', 'QDP2', 'QFP', 'QDP1',
        'Q1', 'Q2', 'Q3', 'Q4']

    def __init__(self, model, dim='6d', use_families=False, gain_bpm=None,              gain_corr=None):
        """."""
        self.model = model
        self.dim = dim
        self.respm = Respmat(model=model, dim=dim)
        self.matrix = self.respm.get_respm()
        self.gain_bpm = gain_bpm
        self.gain_corr = gain_corr
        self.matrix = self.apply_gains(matrix=self.matrix)
        self.vector = self.matrix.flatten()
        self.use_families = use_families
        self.quadsidx = []
        for fam_name in self.QUAD_FAM:
            self.quadsidx.append(self.respm.fam_data[fam_name]['index'])
        self.qnidx = self.respm.fam_data['QN']['index']
        self.kquads = []
        # self.kmatrix = self._calc_kmatrix()

    def apply_bpm_gain(self, matrix_in, gain):
        matrix_out = np.zeros(matrix_in.shape)
        for bpm in range(matrix_in.shape[0]):
            matrix_out[bpm, :] = matrix_in[bpm, :]/gain[bpm][0]
        return matrix_out

    def apply_corr_gain(self, matrix_in, gain):
        matrix_out = np.zeros(matrix_in.shape)
        for corr in range(matrix_in.shape[1]):
            matrix_out[:, corr] = matrix_in[:, corr]/gain[corr][0]
        return matrix_out

    def apply_gains(self, matrix):
        if self.gain_bpm is not None:
            matrix = self.apply_bpm_gain(
                matrix_in=matrix, gain=self.gain_bpm)
        if self.gain_corr is not None:
            self.gain_corr = self.gain_corr
            matrix = self.apply_corr_gain(
                matrix_in=matrix, gain=self.gain_corr)
        return matrix

    def _vector2respm(self, vector):
        row = self.matrix.shape[0]
        col = self.matrix.shape[1]
        return np.reshape(vector, (row, col))

    def _calc_kmatrix(self, deltak=1e-6):
        self.kquads = self._get_quads_strengths()

        if self.use_families:
            nquads = len(self.quadsidx)
            self.kquads = self._get_quads_strengths()
        else:
            nquads = len(self.qnidx)
            self.kquads = np.array(
                pyaccel.lattice.get_attribute(self.model, 'K', self.qnidx))

        kmatrix = np.zeros((len(self.vector), nquads))
        mod = _dcopy(self.model)

        if self.use_families:
            for i1, fam in enumerate(self.quadsidx):
                for i2, idx in enumerate(fam):
                    for i3, idx_seg in enumerate(idx):
                        pyaccel.lattice.set_attribute(
                            mod, 'K', idx_seg, self.kquads[i1][i2][i3] +
                            deltak)
                new_respm = Respmat(model=mod, dim='6d').get_respm()
                dmdk = (new_respm - self.matrix)/deltak
                kmatrix[:, i1] = dmdk.flatten()
                mod = _dcopy(self.model)
        else:
            for i1, idx in enumerate(self.qnidx):
                for i2, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        mod, 'K', idx_seg, self.kquads[i1][i2] +
                        deltak)
                new_respm = Respmat(model=mod, dim='6d').get_respm()
                dmdk = (new_respm - self.matrix)/deltak
                kmatrix[:, i1] = dmdk.flatten()
                mod = _dcopy(self.model)
        return kmatrix

    def _get_quads_strengths(self, model=None, quadsidx=None):
        if model is None:
            model = self.model
        if quadsidx is None:
            quadsidx = self.quadsidx
        kquads = []
        for qi in quadsidx:
            kquads.append(pyaccel.lattice.get_attribute(
                model, 'K', qi))
        return kquads

    def chi2(self, matrix1, matrix2):
        """."""
        return np.sqrt(np.mean((matrix1-matrix2)**2))

    def fit_matrices(self, model=None, measmat=None, kmatrix=None, niter=10,                nsv=None):
        """."""
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
        chi2_init = chi2_old
        print('Initial Error: {:.6e}'.format(chi2_old))

        if not chi2_init:
            print('The initial error is zero! Model = Measurement')

        qnidx = self.respm.fam_data['QN']['index']
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
            grad_quads = self._get_quads_strengths()
        else:
            grad_quads = np.array(
                pyaccel.lattice.get_attribute(mod, 'K', qnidx))

        grad_delta = np.zeros(inv_kmatrix.shape[0])
        tol = 1e-16

        for n in range(niter):
            grad_delta += np.dot(inv_kmatrix, diffmat.flatten())
            if is_fam:
                mod = self._set_deltas_fams(
                    model=mod,
                    quadidx=self.quadsidx,
                    ref=grad_quads,
                    delta=grad_delta)
            else:
                mod = self._set_deltas_quads(
                    model=mod,
                    quadidx=qnidx,
                    ref=grad_quads,
                    delta=grad_delta)
            fitmat = Respmat(model=mod, dim='6d').get_respm()
            fitmat = self.apply_gains(matrix=fitmat)
            diffmat = measmat - fitmat
            print('Iter {0:d}'.format(n+1))
            chi2_new = self.chi2(measmat, fitmat)
            perc = (chi2_new - chi2_init)/chi2_init * 100
            print('Error: {0:.6e} ({1:.2f}%)'.format(chi2_new, perc))

            if np.isnan(chi2_new):
                print('Matrix deviation is NaN!')
                break

            if (chi2_old - chi2_new) < tol:
                print('Limit Reached!')
                grad_delta -= np.dot(inv_kmatrix, diffmat.flatten())
                if is_fam:
                    mod = self._set_deltas_fams(
                        model=mod,
                        quadidx=self.quadsidx,
                        ref=grad_quads,
                        delta=grad_delta)
                else:
                    mod = self._set_deltas_quads(
                        model=mod,
                        quadidx=qnidx,
                        ref=grad_quads,
                        delta=grad_delta)
                fitmat = Respmat(model=mod, dim='6d').get_respm()
                fitmat = self.apply_gains(matrix=fitmat)
                break
            else:
                chi2_old = chi2_new
        return fitmat, mod

    @staticmethod
    def _set_deltas_fams(model, quadidx, ref, delta):
        for i1, fam in enumerate(quadidx):
            for i2, idx in enumerate(fam):
                for i3, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        model, 'K', idx_seg, ref[i1][i2][i3] +
                        delta[i1])
        return model

    @staticmethod
    def _set_deltas_quads(model, quadidx, ref, delta):
        for i1, idx in enumerate(quadidx):
            for i2, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    model, 'K', idx_seg, ref[i1][i2] +
                    delta[i1])
        return model
