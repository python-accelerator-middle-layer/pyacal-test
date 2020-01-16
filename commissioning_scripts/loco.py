#!/usr/bin/env python-sirius
"""."""

from copy import deepcopy as _dcopy
import numpy as np
import pyaccel
from apsuite.commissioning_scripts.calc_orbcorr_mat import Respmat


class LOCO():
    """."""

    QUAD_FAM = [
        'QFA', 'QDA', 'QDB2', 'QFB', 'QDB1', 'QDP2', 'QFP', 'QDP1',
        'Q1', 'Q2', 'Q3', 'Q4']

    def __init__(self, loco_input):
        """."""
        model = loco_input['model']
        dim = loco_input['dim']
        use_families = loco_input['use_families']
        gain_bpm = loco_input['gain_bpm']
        roll_bpm = loco_input['roll_bpm']
        gain_corr = loco_input['gain_corr']
        use_disp = loco_input['use_dispersion']
        self.kmatrix = loco_input['kmatrix']
        self.measmat = loco_input['measured_matrix']
        self.niter = loco_input['number_of_iterations']
        self.nsv = loco_input['singular_values']
        self.use_coupling = loco_input['use_coupling']
        self.fit_gains_bpm = loco_input['fit_gains_bpm']
        self.fit_gains_corr = loco_input['fit_gains_corr']
        self.fit_quadrupoles = loco_input['fit_quadrupoles']

        if not model.cavity_on and dim == '6d':
            model.cavity_on = True
        if not model.radiation_on:
            model.radiation_on = True
        self.model = model
        self.dim = dim
        self.respm = Respmat(model=model, dim=dim)
        self.matrix = self.respm.get_respm()
        self.use_disp = use_disp
        self.bpmidx = self.respm.fam_data['BPM']['index']
        if self.use_disp:
            self.cavidx = pyaccel.lattice.find_indices(
                self.model, 'fam_name', 'SRFCav')[0]
            self.rfline = self.calc_rf_line(self.model)
        else:
            self.rfline = np.zeros((self.matrix.shape[0], 1))
            self.cavidx = None

        if not self.use_coupling:
            self.measmat = self.remove_coupling(self.measmat)

        self.matrix = np.hstack(
                [self.matrix, self.rfline])
        self.nbpm = self.matrix.shape[0]//2
        self.ncorr = self.matrix.shape[1]
        if self.use_disp:
            self.ncorr -= 1

        if gain_bpm is None:
            self.gain_bpm = np.ones(2*self.nbpm)
        if roll_bpm is None:
            self.roll_bpm = np.zeros(self.nbpm)
        if gain_corr is None:
            self.gain_corr = np.ones(self.ncorr)
        self.matrix = self.apply_all_gains(
            matrix=self.matrix,
            gain_bpm=self.gain_bpm,
            roll_bpm=self.roll_bpm,
            gain_corr=self.gain_corr)
        self.vector = self.matrix.flatten()
        self.use_families = use_families
        self.quadsidx = []
        for fam_name in self.QUAD_FAM:
            self.quadsidx.append(self.respm.fam_data[fam_name]['index'])
        self.qnidx = self.respm.fam_data['QN']['index']
        self.kquads = []

    def remove_x(self, matrix):
        matrix[:self.nbpm, :] *= 0
        return matrix

    def remove_y(self, matrix):
        matrix[self.nbpm:, :] *= 0
        return matrix

    def calc_rf_line(self, model, delta=100):
        if self.cavidx is None:
            self.cavidx = pyaccel.lattice.find_indices(
                model, 'fam_name', 'SRFCav')[0]
        f0 = model[self.cavidx].frequency
        model[self.cavidx].frequency = f0 + delta/2
        orbp = pyaccel.tracking.find_orbit6(model, indices='open')
        model[self.cavidx].frequency = f0 - delta/2
        orbm = pyaccel.tracking.find_orbit6(model, indices='open')
        model[self.cavidx].frequency = f0
        dorb = orbp - orbm
        return np.vstack(
            [dorb[0, self.bpmidx], dorb[2, self.bpmidx]])/delta

    def apply_bpm_gain(self, matrix, gain):
        return np.dot(np.diag(gain), matrix)

    def apply_bpm_roll(self, matrix, roll):
        nbpm = len(roll)
        cos_mat = np.diag(np.cos(roll))
        sin_mat = np.diag(np.sin(roll))
        R_alpha = np.zeros((2*nbpm, 2*nbpm))
        R_alpha[:nbpm, :nbpm] = cos_mat
        R_alpha[:nbpm, nbpm:] = sin_mat
        R_alpha[nbpm:, :nbpm] = -sin_mat
        R_alpha[nbpm:, nbpm:] = cos_mat
        return np.dot(R_alpha, matrix)

    def apply_corr_gain(self, matrix, gain):
        matrix[:, :-1] = np.dot(matrix[:, :-1], np.diag(gain))
        # matrix = np.dot(matrix, np.diag(gain))
        return matrix

    def apply_all_gains(self, matrix, gain_bpm, roll_bpm, gain_corr):
        matrix = self.apply_bpm_gain(matrix, gain_bpm)
        matrix = self.apply_bpm_roll(matrix, roll_bpm)
        matrix = self.apply_corr_gain(matrix, gain_corr)
        return matrix

    def _vector2respm(self, vector):
        row = self.matrix.shape[0]
        col = self.matrix.shape[1]
        return np.reshape(vector, (row, col))

    def _calc_kmatrix(self, deltak=1e-6):
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
                new_respm = self.respm.get_respm(model=mod)
                if self.use_disp:
                    new_respm = np.hstack(
                        [new_respm, self.calc_rf_line(mod)])
                else:
                    new_respm = np.hstack(
                        [new_respm, np.zeros((new_respm.shape[0], 1))])
                dmdk = (new_respm - self.matrix)/deltak
                kmatrix[:, i1] = dmdk.flatten()
                mod = _dcopy(self.model)
        else:
            for i1, idx in enumerate(self.qnidx):
                for i2, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        mod, 'K', idx_seg, self.kquads[i1][i2] +
                        deltak)
                new_respm = self.respm.get_respm(model=mod)
                if self.use_disp:
                    new_respm = np.hstack(
                        [new_respm, self.calc_rf_line(mod)])
                else:
                    new_respm = np.hstack(
                        [new_respm, np.zeros((new_respm.shape[0], 1))])
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

    def remove_coupling(self, matrix_in):
        nbpm = 160
        nch = 120
        ncv = 160
        matrix_out = np.zeros(matrix_in.shape)
        matrix_out[:nbpm, :nch] = matrix_in[:nbpm, :nch]
        matrix_out[nbpm:, nch:nch+ncv] = matrix_in[nbpm:, nch:nch+ncv]
        matrix_out[:nbpm, -1] = matrix_in[:nbpm, -1]
        return matrix_out

    def calc_linear_part(self, matrix):
        nbpm = 160
        nch = 120
        ncv = 160
        shape0 = matrix.shape[0]
        shape1 = matrix.shape[1]

        if shape0 != 2*nbpm:
            raise Exception('Problem with BPM number in matrix')
        if shape1 not in (nch + ncv, nch + ncv + 1):
            raise Exception('Problem with correctors number in matrix')

        if shape1 < nch + ncv + 1 and self.use_disp:
            raise Exception('There is no dispersion line in the matrix')

        g_bpm = np.ones(2*nbpm)
        alpha_bpm = np.zeros(nbpm)
        cos_mat = np.diag(np.cos(alpha_bpm))
        sin_mat = np.diag(np.sin(alpha_bpm))
        G_bpm = np.diag(g_bpm)

        R_alpha = np.hstack((cos_mat, sin_mat))
        R_alpha = np.vstack((R_alpha, np.hstack((-sin_mat, cos_mat))))

        dR_alpha = np.hstack((-sin_mat, cos_mat))
        dR_alpha = np.vstack((dR_alpha, np.hstack((-cos_mat, sin_mat))))

        dmdg_bpm = np.zeros((shape0*shape1, 2*self.nbpm))
        for n in range(shape0):
            kron = self.kronecker(n, n, shape0)
            dB = np.dot(R_alpha, kron)
            dmdg_bpm[:, n] = np.dot(dB, matrix).flatten()

        dmdalpha_bpm = np.zeros((shape0*shape1, self.nbpm))
        for n in range(shape0//2):
            kron = self.kronecker(n, n, shape0//2)
            kron = np.tile(kron, (2, 2))
            dR = np.dot(kron, dR_alpha)
            dB = np.dot(dR, G_bpm)
            dmdalpha_bpm[:, n] = np.dot(dB, matrix).flatten()

        dmdg_corr = np.zeros((shape0*shape1, self.ncorr))
        for c in range(self.ncorr):
            kron = self.kronecker(c, c, shape1)
            dmdg_corr[:, c] = np.dot(matrix, kron).flatten()
        return dmdg_bpm, dmdalpha_bpm, dmdg_corr

    def kronecker(self, i, j, size):
        kron = np.zeros((size, size))
        if i != j:
            kron[i, j] = 1
            kron[j, i] = 1
        else:
            kron[i, i] = 1
        return kron

    def merge_kmatrix_linear(self, kmatrix, dmdg_bpm, dmdalpha_bpm, dmdg_corr):
        nbpm = 160
        nch = 120
        ncv = 160
        nfam = kmatrix.shape[1]
        J_loco = np.zeros((kmatrix.shape[0], nfam + 3*nbpm + nch + ncv))
        J_loco[:, :nfam] = kmatrix
        J_loco[:, nfam:nfam+2*nbpm] = dmdg_bpm
        J_loco[:, nfam+2*nbpm:nfam+3*nbpm] = dmdalpha_bpm
        J_loco[:, nfam+3*nbpm:] = dmdg_corr
        return J_loco

    def fit_matrices(self, model=None):
        """."""
        if model is None:
            model = self.model
            modelmat = self.matrix
            mod = _dcopy(model)
        else:
            mod = _dcopy(model)
            modelmat = self.respm.get_respm(model=mod)
            if self.use_disp:
                modelmat = np.hstack(
                     [modelmat, self.calc_rf_line(mod)])
            else:
                modelmat = np.hstack(
                     [modelmat, np.zeros((modelmat.shape[0], 1))])

        if not self.use_disp:
            self.measmat[:, -1] *= 0

        dmdg_bpm, dmdalpha_bpm, dmdg_corr = self.calc_linear_part(modelmat)
        self.Jloco = self.merge_kmatrix_linear(
            self.kmatrix, dmdg_bpm, dmdalpha_bpm, dmdg_corr)

        self.delta_gains_bpms = np.zeros(2*self.nbpm)
        self.delta_rolls_bpms = np.zeros(self.nbpm)
        self.delta_gains_corrs = np.zeros(self.ncorr)

        qnidx = self.respm.fam_data['QN']['index']

        if self.use_families:
            self.idx_grads = len(self.quadsidx)
        else:
            self.idx_grads = len(qnidx)

        self.Jloco = self.filter_Jloco(self.Jloco)

        diffmat = self.measmat - modelmat
        chi2_old = self.chi2(self.measmat, modelmat)
        chi2_init = chi2_old
        print('Initial Error: {:.6e}'.format(chi2_old))

        if not chi2_init:
            print('The initial error is zero! Model = Measurement')

        u, s, v = np.linalg.svd(self.Jloco, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        if self.nsv is not None:
            inv_s[self.nsv:] = 0
        inv_s = np.diag(inv_s)
        self.invJloco = np.dot(np.dot(v.T, inv_s), u.T)

        if self.use_families:
            self.grad_quads = self._get_quads_strengths(model=mod)
            self.grad_delta = np.zeros(len(self.quadsidx))
        else:
            self.grad_quads = np.array(
                pyaccel.lattice.get_attribute(mod, 'K', qnidx))
            self.grad_delta = np.zeros(len(qnidx))

        tol = 1e-16
        loco_out = dict()
        for n in range(self.niter):
            new_pars = np.dot(self.invJloco, diffmat.flatten())
            fitmat, mod = self.get_fitmat(mod, new_pars)
            diffmat = self.measmat - fitmat
            print('Iter {0:d}'.format(n+1))
            chi2_new = self.chi2(self.measmat, fitmat)
            perc = (chi2_new - chi2_init)/chi2_init * 100
            print('Error: {0:.6e} ({1:.2f}%)'.format(chi2_new, perc))

            if np.isnan(chi2_new):
                print('Matrix deviation is NaN!')
                break

            if (chi2_old - chi2_new) < tol:
                print('Limit Reached!')
                fitmat, mod = self.get_fitmat(mod, -1*new_pars)
                break
            else:
                chi2_old = chi2_new
        print('Finished!')
        gain_bpm = self.gain_bpm + self.delta_gains_bpms
        roll_bpm = self.roll_bpm + self.delta_rolls_bpms
        gain_corr = self.gain_corr + self.delta_gains_corrs
        loco_out['fit_matrix'] = fitmat
        loco_out['fit_model'] = mod
        loco_out['gain_bpm'] = gain_bpm
        loco_out['roll_bpm'] = roll_bpm
        loco_out['gain_corr'] = gain_corr
        return loco_out

    def filter_Jloco(self, J_loco):
        if not self.fit_quadrupoles:
            J_loco[:, :self.idx_grads] = None
        if not self.fit_gains_bpm:
            J_loco[:, self.idx_grads:self.idx_grads+2*self.nbpm] = None
        if not self.fit_gains_corr:
            J_loco[:, self.idx_grads+3*self.nbpm] = None
        if not self.use_coupling:
            J_loco[:,
            self.idx_grads+2*self.nbpm:self.idx_grads+3*self.nbpm] = None

        J_loco = J_loco[~np.isnan(J_loco)].reshape(J_loco.shape[0], -1)
        return J_loco

    def get_param(self, param):
        parameters = dict()
        parameters['k'] = None
        parameters['bpm_gain'] = None
        parameters['bpm_roll'] = None
        parameters['corr_gain'] = None
        size = len(param)
        one = False

        if size == len(self.grad_quads) + 3*self.nbpm + self.ncorr:
            parameters['k'] = param[:self.idx_grads]
            parameters['bpm_gain'] = param[
                self.idx_grads:self.idx_grads+2*self.nbpm]
            parameters['bpm_roll'] = param[
                self.idx_grads+2*self.nbpm:self.idx_grads+3*self.nbpm]
            parameters['corr_gain'] = param[
                self.idx_grads+3*self.nbpm:]
        elif size == len(self.grad_quads) + 2*self.nbpm + self.ncorr:
            parameters['k'] = param[:self.idx_grads]
            parameters['bpm_gain'] = param[
                self.idx_grads:self.idx_grads+2*self.nbpm]
            parameters['corr_gain'] = param[self.idx_grads+2*self.nbpm:]
        elif size == len(self.grad_quads) + 1*self.nbpm + self.ncorr:
            parameters['k'] = param[:self.idx_grads]
            parameters['bpm_roll'] = param[
                self.idx_grads:self.idx_grads+1*self.nbpm]
            parameters['corr_gain'] = param[self.idx_grads+1*self.nbpm:]
        elif size == len(self.grad_quads) + 2*self.nbpm:
            parameters['k'] = param[:self.idx_grads]
            parameters['bpm_gain'] = param[self.idx_grads:]
        elif size == len(self.grad_quads) + 3*self.nbpm:
            parameters['k'] = param[:self.idx_grads]
            parameters['bpm_gain'] = param[
                self.idx_grads:self.idx_grads+2*self.nbpm]
            parameters['bpm_roll'] = param[self.idx_grads+2*self.nbpm:]
        elif size == len(self.grad_quads) + 1*self.nbpm:
            parameters['k'] = param[:self.idx_grads]
            parameters['bpm_roll'] = param[self.idx_grads:]
        elif size == len(self.grad_quads) + self.ncorr:
            parameters['k'] = param[:self.idx_grads]
            parameters['corr_gain'] = param[self.idx_grads:]
        elif size == 2*self.nbpm + self.ncorr:
            parameters['bpm_gain'] = param[:2*self.nbpm]
            parameters['corr_gain'] = param[2*self.nbpm:]
        elif size == 3*self.nbpm + self.ncorr:
            parameters['bpm_gain'] = param[:2*self.nbpm]
            parameters['bpm_roll'] = param[2*self.nbpm:3*self.nbpm]
            parameters['corr_gain'] = param[3*self.nbpm:]
        elif size == self.nbpm + self.ncorr:
            parameters['bpm_roll'] = param[:self.nbpm]
            parameters['corr_gain'] = param[self.nbpm:]
        elif size == 3*self.nbpm:
            parameters['bpm_gain'] = param[:2*self.nbpm]
            parameters['bpm_roll'] = param[2*self.nbpm:]

        if size == len(self.grad_quads):
            name = 'k'
            one = True
        elif size == 2*self.nbpm:
            name = 'bpm_gain'
            one = True
        elif size == self.nbpm:
            name = 'bpm_roll'
            one = True
        elif size == self.ncorr:
            name = 'corr_gain'
            one = True

        if one:
            parameters[name] = param
        return parameters

    def get_fitmat(self, mod, new_pars):
        modc = mod
        parameters = self.get_param(new_pars)
        if parameters['k'] is not None:
            self.grad_delta += parameters['k']

        if parameters['bpm_gain'] is not None:
            self.delta_gains_bpms += parameters['bpm_gain']

        if parameters['bpm_roll'] is not None:
            self.delta_rolls_bpms += parameters['bpm_roll']

        if parameters['corr_gain'] is not None:
            self.delta_gains_corrs += parameters['corr_gain']

        if self.use_families:
            modc = self._set_deltas_fams(
                model=modc,
                quadidx=self.quadsidx,
                ref=self.grad_quads,
                delta=self.grad_delta)
        else:
            modc = self._set_deltas_quads(
                model=modc,
                quadidx=self.qnidx,
                ref=self.grad_quads,
                delta=self.grad_delta)

        fitmat = self.respm.get_respm(model=modc)
        if self.use_disp:
            fitmat = np.hstack(
                [fitmat, self.calc_rf_line(mod)])
        else:
            fitmat = np.hstack(
                [fitmat, np.zeros((fitmat.shape[0], 1))])
        fitmat = self.apply_all_gains(
            matrix=fitmat,
            gain_bpm=self.gain_bpm+self.delta_gains_bpms,
            roll_bpm=self.roll_bpm+self.delta_rolls_bpms,
            gain_corr=self.gain_corr+self.delta_gains_corrs
            )
        return fitmat, modc

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
