#!/usr/bin/env python-sirius
"""."""

import numpy as np
import pymodels
import pyaccel
from copy import deepcopy as _dcopy
from apsuite.commissioning_scripts.calcrespm import CalcRespm
from apsuite.optimization.simulated_annealing import SimulAnneal
from apsuite.commissioning_scripts.measure_respmat_tbbo import calc_model_respmatTBBO
from apsuite.commissioning_scripts.measure_disp_tbbo import calc_model_dispersionTBBO


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
                 respmat, disp, nturns=1, save=False, in_sept=True):
        super().__init__(save=save)
        self.tb_model = tb_model
        self.bo_model = bo_model
        self.corr_names = corr_names
        self.elems = elems
        self.nturns = nturns
        self.respmat = self.merge_disp_respm(respmat, disp)
        self.in_sept = in_sept

    def initialization(self):
        return

    def calc_obj_fun(self):
        ksqs, kxl, kyl, ksxl, ksyl = self._position
        tbmod = self.set_ks_qs(self.tb_model, ksqs)
        tbmod = self.set_k_septum(
            tbmod, kxl, kyl, ksxl, ksyl)
        respmat = calc_model_respmatTBBO(
            tbmod, tbmod + self.bo_model, self.corr_names, self.elems,
            meth='middle', ishor=True)
        disp = self.calc_disp(tbmod, self.bo_model)
        respmat = self.merge_disp_respm(respmat, disp)
        respmat -= self.respmat
        return np.sqrt(np.mean(respmat*respmat))

    def set_k_septum(self, tbmodel, kxl, kyl, ksxl, ksyl):
        tbmod = _dcopy(tbmodel)
        sept_idx = pyaccel.lattice.find_indices(
                tbmod, 'fam_name', 'InjSeptM66')
        nmat = len(sept_idx)
        for idx in sept_idx:
            tbmod[idx].KxL = kxl / nmat
            tbmod[idx].KyL = kyl / nmat
            tbmod[idx].KsxL = ksxl / nmat
            tbmod[idx].KsyL = ksyl / nmat
        return tbmod

    def get_k_septum(self, tbmodel):
        tbmod = _dcopy(tbmodel)
        sept_idx = pyaccel.lattice.find_indices(
                tbmod, 'fam_name', 'InjSeptM66')
        kxl, kyl, ksxl, ksyl = 0, 0, 0, 0
        for idx in sept_idx:
            kxl += tbmod[idx].KxL
            kyl += tbmod[idx].KyL
            ksxl += tbmod[idx].KsxL
            ksyl += tbmod[idx].KsyL
        return kxl, kyl, ksxl, ksyl

    def set_ks_qs(self, tbmodel, ksqs):
        tbmod = _dcopy(tbmodel)
        qs_idx = pyaccel.lattice.find_indices(
                tbmod, 'fam_name', 'QS')
        segqs = len(qs_idx)
        for idx in qs_idx:
            tbmod[idx].Ks = ksqs / segqs
        return tbmod

    def set_knorm_septum(self, tbmod, kl, ksl):
        sept_idx = pyaccel.lattice.find_indices(
                tbmod, 'fam_name', 'InjSept')
        tbmod[sept_idx[0]].KL = kl
        tbmod[sept_idx[0]].KsL = ksl
        return tbmod

    def calc_disp(self, tb_mod, bo_mod):
        mod = tb_mod + bo_mod
        ind = pyaccel.lattice.find_indices(mod, 'fam_name', 'InjKckr')[0]
        pyaccel.lattice.set_attribute(mod, 'angle', ind, 22e-3)
        bpms = pyaccel.lattice.find_indices(mod, 'fam_name', 'BPM')[1:]
        disp = calc_model_dispersionTBBO(mod, bpms)
        return disp

    def merge_disp_respm(self, respm, disp):
        newrespm = np.zeros((respm.shape[0]+1, respm.shape[1]))
        newrespm[:-1, :] = respm
        newrespm[-1, :] = disp
        return newrespm

    def chi2(self, M1, M2):
        return np.sqrt(np.mean((M1-M2)**2))

    def calc_Ksept_matrix(self):
        tb_mod = _dcopy(self.tb_model)
        delta = 1e-3
        deltak = np.eye(4) * delta
        K = np.zeros(4)
        K[0], K[1], K[2], K[3] = self.get_k_septum(tb_mod)
        respmat0 = calc_model_respmatTBBO(
                tb_mod, tb_mod + self.bo_model, self.corr_names, self.elems,
                meth='middle', ishor=True)
        disp0 = self.calc_disp(tb_mod, self.bo_model)
        respm0 = self.merge_disp_respm(respmat0, disp0)
        v0 = respm0.flatten()
        Krespm = np.zeros((len(v0), 4))

        for k in range(4):
            kxl, kyl, ksxl, ksyl = deltak[k, :]
            tb_mod = self.set_k_septum(
                tb_mod, K[0] + kxl, K[1] + kyl, K[2] + ksxl, K[3] + ksyl)
            respmat_delta = calc_model_respmatTBBO(
                tb_mod, tb_mod + self.bo_model, self.corr_names, self.elems,
                meth='middle', ishor=True)
            disp_delta = self.calc_disp(tb_mod, self.bo_model)
            respm_delta = self.merge_disp_respm(respmat_delta, disp_delta)
            vdelta = respm_delta.flatten()
            Krespm[:, k] = (vdelta - v0)/delta
        return Krespm

    def calc_dispsept_matrix(self):
        tb_mod = _dcopy(self.tb_model)
        delta = 1e-3
        deltak = np.eye(4) * delta
        K = np.zeros(4)
        K[0], K[1], K[2], K[3] = self.get_k_septum(tb_mod)
        disp0 = self.calc_disp(tb_mod, self.bo_model)
        DispM = np.zeros((len(disp0), 4))

        for k in range(4):
            kxl, kyl, ksxl, ksyl = deltak[k, :]
            tb_mod = self.set_k_septum(
                tb_mod, K[0] + kxl, K[1] + kyl, K[2] + ksxl, K[3] + ksyl)
            disp_delta = self.calc_disp(tb_mod, self.bo_model)
            DispM[:, k] = (disp_delta - disp0)/delta
        return DispM

    def fit_matrices(self, tb_mod, Krespm, Mmeas, Mmodel=None, niter=10, nsv=None):
        tbmodel = _dcopy(tb_mod)
        if Mmodel is None:
            Mmodel = calc_model_respmatTBBO(
                    tbmodel, tbmodel + self.bo_model, self.corr_names,  self.elems, meth='middle', ishor=True)
            DispModel = self.calc_disp(tbmodel, self.bo_model)
            Mmodel = self.merge_disp_respm(Mmodel, DispModel)
        dM = Mmeas - Mmodel
        chi2_old = self.chi2(Mmeas, Mmodel)
        print('Initial Matrix Deviation: {:.6f}'.format(chi2_old))
        U, s, V = np.linalg.svd(Krespm, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        if nsv is not None:
            inv_s[nsv:] = 0
        Inv_S = np.diag(inv_s)
        Mkinv = np.dot(np.dot(V.T, Inv_S), U.T)
        K = np.zeros(4)
        K[0], K[1], K[2], K[3] = self.get_k_septum(tbmodel)
        dK = np.zeros(4)
        tol = 1e-8

        for n in range(niter):
            dK += np.dot(Mkinv, dM.flatten())

            newmod = self.set_k_septum(
                tbmodel, (K+dK)[0], (K+dK)[1], (K+dK)[2], (K+dK)[3])

            new_respm = calc_model_respmatTBBO(
                newmod, newmod + self.bo_model, self.corr_names, self.elems,
                meth='middle', ishor=True)
            new_disp = self.calc_disp(newmod, self.bo_model)
            new_respm = self.merge_disp_respm(new_respm, new_disp)
            dM = Mmeas - new_respm
            print('Iter {:.1f}'.format(n+1))
            chi2_new = self.chi2(Mmeas, new_respm)
            print('Matrix Deviation: {:.6f}'.format(chi2_old))
            if abs(chi2_new - chi2_old) < tol:
                print('Limit Reached!')
                dK -= np.dot(Mkinv, dM.flatten())
                newmod = self.set_k_septum(
                    tbmodel, (K+dK)[0], (K+dK)[1], (K+dK)[2], (K+dK)[3])

                new_respm = calc_model_respmatTBBO(
                    newmod, newmod + self.bo_model, self.corr_names,
                    self.elems, meth='middle', ishor=True)
                new_disp = self.calc_disp(newmod, self.bo_model)
                new_respm = self.merge_disp_respm(new_respm, new_disp)
                break
            else:
                chi2_old = chi2_new
        return new_respm, K+dK


    def fit_matrices2(self, tb_mod, DispM, dispmeas, dispmodel, niter=10,                    nsv=None):
        tbmodel = _dcopy(tb_mod)
        if dispmodel is None:
            dispmodel = self.calc_disp(tbmodel, self.bo_model)
        deta = dispmeas - dispmodel
        chi2_old = self.chi2(dispmeas, dispmodel)
        print('Initial Matrix Deviation: {:.6f}'.format(chi2_old))
        U, s, V = np.linalg.svd(DispM, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        if nsv is not None:
            inv_s[nsv:] = 0
        Inv_S = np.diag(inv_s)
        Mdispinv = np.dot(np.dot(V.T, Inv_S), U.T)
        K = np.zeros(4)
        K[0], K[1], K[2], K[3] = self.get_k_septum(tbmodel)
        dK = np.zeros(4)
        tol = 1e-16

        for n in range(niter):
            dK += np.dot(Mdispinv, deta)

            newmod = self.set_k_septum(
                tbmodel, (K+dK)[0], (K+dK)[1], (K+dK)[2], (K+dK)[3])

            newdisp = self.calc_disp(newmod, self.bo_model)
            deta = dispmeas - newdisp
            print('Iter {:.1f}'.format(n+1))
            chi2_new = self.chi2(dispmeas, newdisp)
            print('Matrix Deviation: {:.6f}'.format(chi2_new))
            if abs(chi2_new - chi2_old) < tol:
                print('Limit Reached!')
                dK -= np.dot(Mdispinv, deta)
                newmod = self.set_k_septum(
                    tbmodel, (K+dK)[0], (K+dK)[1], (K+dK)[2], (K+dK)[3])
                newdisp = self.calc_disp(newmod, self.bo_model)
                break
            else:
                chi2_old = chi2_new
        return newdisp, K+dK
