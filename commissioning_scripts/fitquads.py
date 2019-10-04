#!/usr/bin/env python-sirius
"""."""

import numpy as np
import pymodels
import pyaccel
from apsuite.commissioning_scripts.calcrespm import CalcRespm
from apsuite.optimization.simulated_annealing import SimulAnneal
from apsuite.commissioning_scripts.measure_respmat_tbbo import MeasureRespMatTBBO, calc_model_respmatTBBO


class FitQuads():

    def __init__(self):
        bomod = pymodels.bo.create_accelerator()
        sept = pyaccel.lattice.find_indices(bomod, 'fam_name', 'InjSept')
        bomod = pyaccel.lattice.shift(bomod, sept[0])
        self.respm = CalcRespm(model=bomod)
        self.vector = self._respm2vector(matrix=self.respm.respm)
        self.qfidx = self.respm.fam_data['QF']['index']
        self.qdidx = self.respm.fam_data['QD']['index']
        self.Krespm = self._calcKrespm()

    def _respm2vector(self, matrix):
        return matrix.flatten()

    def _vector2respm(self, vector):
        row = self.respm.respm.shape[0]
        col = self.respm.respm.shape[1]
        return np.reshape(vector, (row, col))

    def _calcKrespm(self):
        self.kqfs = pyaccel.lattice.get_attribute(
            self.respm.model, 'K', self.qfidx)
        self.kqds = pyaccel.lattice.get_attribute(
            self.respm.model, 'K', self.qdidx)

        delta = 1e-3
        nquads = len(self.qfidx) + len(self.qdidx)
        Krespm = np.zeros((len(self.vector), nquads))

        for i, idx in enumerate(self.qfidx):
            for ii, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    self.respm.model, 'K', idx_seg, self.kqfs[i][ii]*(1+delta))
            new_respm = self.respm.get_respm(model=self.respm.model)
            Krespm[:, i] = self._respm2vector(
                (new_respm - self.respm.respm)/delta)
            for ii, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    self.respm.model, 'K', idx_seg, self.kqfs[i][ii])

        for i, idx in enumerate(self.qdidx):
            for ii, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    self.respm.model, 'K', idx_seg, self.kqds[i][ii]*(1+delta))
            new_respm = self.respm.get_respm(model=self.respm.model)
            Krespm[:, i] = self._respm2vector(
                (new_respm - self.respm.respm)/delta)
            for ii, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    self.respm.model, 'K', idx_seg, self.kqds[i][ii])

        np.savetxt('KMatrix.txt', Krespm)
        return Krespm

    def chi2(self, M1, M2):
        return np.sum(np.sum((M1-M2)**2))

    def fit_matrices(self, Mmeas=None, niter=10, nsv=None):
        bomod = pymodels.bo.create_accelerator()
        sept = pyaccel.lattice.find_indices(bomod, 'fam_name', 'InjSept')
        bomod = pyaccel.lattice.shift(bomod, sept[0])
        respm_model = CalcRespm(bomod)
        Mmodel = respm_model.respm
        qfidx = respm_model.fam_data['QF']['index']
        qdidx = respm_model.fam_data['QD']['index']
        dM = Mmeas - Mmodel
        chi2_old = self.chi2(Mmeas, Mmodel)
        print('Initial Matrix Deviation: {:.16e}'.format(chi2_old))
        u, s, v = np.linalg.svd(self.Krespm, full_matrices=False)
        inv_s = 1/s
        isnan = np.isnan(inv_s)
        isinf = np.isinf(inv_s)
        inv_s[np.invert(isinf)] = 0
        inv_s[np.invert(isnan)] = 0
        if nsv is not None:
            inv_s[nsv:] = 0
        Inv_S = np.diag(inv_s)
        Mkinv = np.dot(np.dot(v.T, Inv_S), u.T)

        KQF = pyaccel.lattice.get_attribute(bomod, 'K', qfidx)
        KQD = pyaccel.lattice.get_attribute(bomod, 'K', qdidx)
        dK = np.zeros(Mkinv.shape[0])

        for n in range(niter):
            dK += np.dot(Mkinv, dM.flatten())

            for i, idx in enumerate(qfidx):
                for ii, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        bomod, 'K', idx_seg, KQF[i][ii] + dK[i])

            for i, idx in enumerate(qdidx):
                for ii, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        bomod, 'K', idx_seg, KQD[i][ii] + dK[len(qfidx) + i])

            new_respm = CalcRespm(bomod)
            dM = Mmeas - new_respm.respm
            print('Iter {:.1f}'.format(n+1))
            chi2_new = self.chi2(Mmeas, new_respm.respm)
            print('Matrix Deviation: {:.16e}'.format(chi2_old))
            if chi2_new > chi2_old:
                print('Limit Reached!')
                dK -= np.dot(Mkinv, dM.flatten())
                for i, idx in enumerate(qfidx):
                    for ii, idx_seg in enumerate(idx):
                        pyaccel.lattice.set_attribute(
                            bomod, 'K', idx_seg, KQF[i][ii] + dK[i])

                for i, idx in enumerate(qdidx):
                    for ii, idx_seg in enumerate(idx):
                        pyaccel.lattice.set_attribute(
                            bomod, 'K', idx_seg, KQD[i][ii] + dK[len(qfidx) + i])
                new_respm = CalcRespm(bomod)
                break
            else:
                chi2_old = chi2_new
        return new_respm.respm


class FindSeptQuad(SimulAnneal):

    def __init__(self, tb_model, bo_model, corr_names, elems,
                 respmat, nturns=1, save=False, in_sept=True):
        super().__init__(save=save)
        self.tb_model = tb_model
        self.bo_model = bo_model
        self.corr_names = corr_names
        self.elems = elems
        self.nturns = nturns
        self.respmat = respmat
        self.in_sept = in_sept

    def initialization(self):
        return

    def calc_obj_fun(self):
        if self.in_sept:
            sept_idx = pyaccel.lattice.find_indices(
                self.tb_model, 'fam_name', 'InjSept')
        else:
            sept_idx = self.elems['TB-04:MA-CV-2'].model_indices
        kxl, kyl, ksxl, ksyl = self._position
        self.set_k_septum(kxl=kxl, kyl=kyl, ksxl=ksxl, ksyl=ksyl)
        respmat = calc_model_respmatTBBO(
            self.tb_model, self.tb_model + self.bo_model, self.corr_names,self.elems, meth='middle', ishor=True)
        respmat -= self.respmat
        return np.sqrt(np.mean(respmat*respmat))

    def set_k_septum(self, kxl, kyl, ksxl, ksyl):
        sept_idx = pyaccel.lattice.find_indices(
                self.tb_model, 'fam_name', 'InjSept')
        pyaccel.lattice.set_attribute(self.tb_model, 'KxL', sept_idx, kxl)
        pyaccel.lattice.set_attribute(self.tb_model, 'KyL', sept_idx, kyl)
        pyaccel.lattice.set_attribute(self.tb_model, 'KsxL', sept_idx, ksxl)
        pyaccel.lattice.set_attribute(self.tb_model, 'KsyL', sept_idx, ksyl)
