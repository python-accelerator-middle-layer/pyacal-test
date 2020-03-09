"""."""

from copy import deepcopy as _dcopy

import numpy as np
import pyaccel
from apsuite.commissioning_scripts.calcrespm import CalcRespm
from apsuite.optimization.simulated_annealing import SimulAnneal
from apsuite.commissioning_scripts.measure_respmat_tbbo import \
    calc_model_respmatTBBO
from apsuite.commissioning_scripts.measure_disp_tbbo import \
    calc_model_dispersionTBBO


class FitQuads():

    def __init__(self, model):
        self.model = model
        self.respm = CalcRespm(model=model)
        self.vector = self.respm.respm.flatten()
        self.qfidx = self.respm.fam_data['QF']['index']
        self.qdidx = self.respm.fam_data['QD']['index']
        self.kmatrix = self._calc_kmatrix()

    def _vector2respm(self, vector):
        row = self.respm.respm.shape[0]
        col = self.respm.respm.shape[1]
        return np.reshape(vector, (row, col))

    def _calc_kmatrix(self, deltak=1e-6):
        self.kqfs = pyaccel.lattice.get_attribute(
            self.respm.model, 'K', self.qfidx)
        self.kqds = pyaccel.lattice.get_attribute(
            self.respm.model, 'K', self.qdidx)

        nquads = len(self.qfidx) + len(self.qdidx)
        kmatrix = np.zeros((len(self.vector), nquads))

        for i, idx in enumerate(self.qfidx):
            for ii, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    self.respm.model, 'K', idx_seg, self.kqfs[i][ii] + deltak)
            new_respm = self.respm.get_respm(model=self.respm.model)
            dmdk = (new_respm - self.respm.respm)/deltak
            kmatrix[:, i] = dmdk.flatten()
            for ii, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    self.respm.model, 'K', idx_seg, self.kqfs[i][ii])

        for i, idx in enumerate(self.qdidx):
            for ii, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    self.respm.model, 'K', idx_seg, self.kqds[i][ii] + deltak)
            new_respm = self.respm.get_respm(model=self.respm.model)
            dmdk = (new_respm - self.respm.respm)/deltak
            kmatrix[:, i+len(self.qfidx)] = dmdk.flatten()
            for ii, idx_seg in enumerate(idx):
                pyaccel.lattice.set_attribute(
                    self.respm.model, 'K', idx_seg, self.kqds[i][ii])
        return kmatrix

    @staticmethod
    def chi2(self, M1, M2):
        return np.sqrt(np.mean((M1-M2)**2))

    def fit_matrices(self, model, measmat, niter=10, nsv=None):
        bomod = _dcopy(model)
        respm_model = CalcRespm(bomod)
        modelmat = respm_model.respm
        diffmat = measmat - modelmat
        chi2_old = FitQuads.chi2(measmat, modelmat)
        qfidx = respm_model.fam_data['QF']['index']
        qdidx = respm_model.fam_data['QD']['index']
        print('Initial Matrix Deviation: {:.16e}'.format(chi2_old))
        u, s, v = np.linalg.svd(self.kmatrix, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        if nsv is not None:
            inv_s[nsv:] = 0
        inv_s = np.diag(inv_s)
        inv_kmatrix = np.dot(np.dot(v.T, inv_s), u.T)

        grad_qf = np.array(pyaccel.lattice.get_attribute(bomod, 'K', qfidx))
        grad_qd = np.array(pyaccel.lattice.get_attribute(bomod, 'K', qdidx))
        grad_delta = np.zeros(inv_kmatrix.shape[0])
        tol = 1e-16

        for n in range(niter):
            grad_delta += np.dot(inv_kmatrix, diffmat.flatten())

            for i, idx in enumerate(qfidx):
                for ii, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        bomod, 'K', idx_seg, grad_qf[i][ii] + grad_delta[i])

            for i, idx in enumerate(qdidx):
                for ii, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        bomod, 'K', idx_seg,
                        grad_qd[i][ii] + grad_delta[len(qfidx) + i])

            fitmat = CalcRespm(bomod)
            diffmat = measmat - fitmat.respm
            print('Iter {:.1f}'.format(n+1))
            chi2_new = FitQuads.chi2(measmat, fitmat.respm)
            print('Matrix Deviation: {:.16e}'.format(chi2_new))

            if (chi2_old - chi2_new) < tol:
                print('Limit Reached!')
                grad_delta -= np.dot(inv_kmatrix, diffmat.flatten())

                for i, idx in enumerate(qfidx):
                    for ii, idx_seg in enumerate(idx):
                        pyaccel.lattice.set_attribute(
                            bomod, 'K', idx_seg,
                            grad_qf[i][ii] + grad_delta[i])

                for i, idx in enumerate(qdidx):
                    for ii, idx_seg in enumerate(idx):
                        pyaccel.lattice.set_attribute(
                            bomod, 'K', idx_seg,
                            grad_qd[i][ii] + grad_delta[len(qfidx) + i])
                fitmat = CalcRespm(bomod)
                break
            else:
                chi2_old = chi2_new
        fit_grad_qf = grad_qf[:, 0] + grad_delta[:len(qfidx)]
        fit_grad_qd = grad_qd[:, 0] + grad_delta[len(qfidx):]
        return fitmat.respm, fit_grad_qf, fit_grad_qd


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
        self.modelmat = self.calc_model_mat()
        self.in_sept = in_sept

    def initialization(self):
        return

    def calc_model_mat(self):
        modelmat = calc_model_respmatTBBO(
            self.tb_model, self.tb_model + self.bo_model, self.corr_names,
            self.elems, meth='middle', ishor=True)
        modeldisp = self.calc_disp(self.tb_model, self.bo_model)
        return self.merge_disp_respm(modelmat, modeldisp)

    def calc_obj_fun(self):
        tbmod = _dcopy(self.tb_model)
        ksqs, kxl, kyl, ksxl, ksyl = self._position
        self.set_ks_qs(tbmod, ksqs)
        self.set_k_septum(
            tbmod, kxl, kyl, ksxl, ksyl)
        respmat = calc_model_respmatTBBO(
            tbmod, tbmod + self.bo_model, self.corr_names, self.elems,
            meth='middle', ishor=True)
        disp = self.calc_disp(tbmod, self.bo_model)
        respmat = self.merge_disp_respm(respmat, disp)
        respmat -= self.respmat
        return np.sqrt(np.mean(respmat*respmat))

    def set_k_septum(self, tbmod, kxl, kyl, ksxl, ksyl):
        sept_idx = pyaccel.lattice.find_indices(
            tbmod, 'fam_name', 'InjSeptM66')
        nsegs = len(sept_idx)
        for idx in sept_idx:
            tbmod[idx].KxL = kxl / nsegs
            tbmod[idx].KyL = kyl / nsegs
            tbmod[idx].KsxL = ksxl / nsegs
            tbmod[idx].KsyL = ksyl / nsegs

    def get_k_septum(self, tbmod):
        sept_idx = pyaccel.lattice.find_indices(
            tbmod, 'fam_name', 'InjSeptM66')
        grad_sept = np.zeros(4)
        for idx in sept_idx:
            grad_sept[0] += tbmod[idx].KxL
            grad_sept[1] += tbmod[idx].KyL
            grad_sept[2] += tbmod[idx].KsxL
            grad_sept[3] += tbmod[idx].KsyL
        return grad_sept

    def set_ks_qs(self, tbmod, ksqs):
        qs_idx = pyaccel.lattice.find_indices(
                tbmod, 'fam_name', 'QS')
        segqs = len(qs_idx)
        for idx in qs_idx:
            tbmod[idx].Ks = ksqs / segqs

    def get_ks_qs(self, tbmod):
        qs_idx = pyaccel.lattice.find_indices(
                tbmod, 'fam_name', 'QS')
        grad_qs = 0
        for idx in qs_idx:
            grad_qs += tbmod[idx].Ks
        return grad_qs

    def calc_disp(self, tb_mod, bo_mod):
        tbbo_mod = tb_mod + bo_mod
        ind = pyaccel.lattice.find_indices(tbbo_mod, 'fam_name', 'InjKckr')[0]
        pyaccel.lattice.set_attribute(tbbo_mod, 'angle', ind, 22e-3)
        bpms = pyaccel.lattice.find_indices(tbbo_mod, 'fam_name', 'BPM')[1:]
        disp = calc_model_dispersionTBBO(tbbo_mod, bpms)
        return disp

    def merge_disp_respm(self, respm, disp):
        newrespm = np.zeros((respm.shape[0]+1, respm.shape[1]))
        newrespm[:-1, :] = respm
        newrespm[-1, :] = disp
        return newrespm

    def chi2(self, M1, M2):
        return np.sqrt(np.mean((M1-M2)**2))

    def calc_sept_kmatrix(self):
        tbmod = _dcopy(self.tb_model)
        delta = 1e-3
        deltak = np.eye(4) * delta
        grad_sept = self.get_k_septum(tbmod)
        kmatrix = np.zeros((len(self.modelmat.flatten()), 4))

        for k in range(4):
            kxl, kyl, ksxl, ksyl = deltak[k, :]
            self.set_k_septum(
                tbmod, grad_sept[0] + kxl, grad_sept[1] + kyl, grad_sept[2] +
                ksxl, grad_sept[3] + ksyl)
            respmat_delta = calc_model_respmatTBBO(
                tbmod, tbmod + self.bo_model, self.corr_names, self.elems,
                meth='middle', ishor=True)
            disp_delta = self.calc_disp(tbmod, self.bo_model)
            respmat_delta = self.merge_disp_respm(respmat_delta, disp_delta)
            diffmat = (respmat_delta - self.modelmat).flatten()
            kmatrix[:, k] = diffmat/delta
        return kmatrix

    def fit_matrices(self, tbmod, kmatrix, measmat, niter=10, nsv=None):
        diffmat = measmat - self.modelmat
        chi2_old = self.chi2(measmat, self.modelmat)
        print('Initial Matrix Deviation: {:.6f}'.format(chi2_old))
        u, s, v = np.linalg.svd(kmatrix, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        if nsv is not None:
            inv_s[nsv:] = 0
        inv_s = np.diag(inv_s)
        inv_kmatrix = np.dot(np.dot(v.T, inv_s), u.T)
        grad_sept = self.get_k_septum(tbmod)
        grad_delta = np.zeros(4)
        tol = 1e-16

        for n in range(niter):
            grad_delta += np.dot(inv_kmatrix, diffmat.flatten())
            new_grad_sept = grad_sept + grad_delta
            kxl = new_grad_sept[0]
            kyl = new_grad_sept[1]
            ksxl = new_grad_sept[2]
            ksyl = new_grad_sept[3]

            self.set_k_septum(tbmod, kxl, kyl, ksxl, ksyl)
            fitmat = calc_model_respmatTBBO(
                tbmod, tbmod + self.bo_model, self.corr_names, self.elems,
                meth='middle', ishor=True)
            fitdisp = self.calc_disp(tbmod, self.bo_model)
            fitmat = self.merge_disp_respm(fitmat, fitdisp)
            diffmat = measmat - fitmat
            print('Iter {:.1f}'.format(n+1))
            chi2_new = self.chi2(measmat, fitmat)
            print('Matrix Deviation: {:.6f}'.format(chi2_new))
            if chi2_old - chi2_new < tol:
                print('Limit Reached!')
                grad_delta -= np.dot(inv_kmatrix, diffmat.flatten())
                new_grad_sept = grad_sept + grad_delta
                kxl = new_grad_sept[0]
                kyl = new_grad_sept[1]
                ksxl = new_grad_sept[2]
                ksyl = new_grad_sept[3]

                self.set_k_septum(tbmod, kxl, kyl, ksxl, ksyl)
                fitmat = calc_model_respmatTBBO(
                    tbmod, tbmod + self.bo_model, self.corr_names, self.elems,
                    meth='middle', ishor=True)
                fitdisp = self.calc_disp(tbmod, self.bo_model)
                fitmat = self.merge_disp_respm(fitmat, fitdisp)
                break
            else:
                chi2_old = chi2_new
        return fitmat, new_grad_sept
