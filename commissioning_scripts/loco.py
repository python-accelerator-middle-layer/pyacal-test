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
        self.kmatrix = loco_input['kmatrix']
        self.measmat = loco_input['measured_matrix']
        self.niter = loco_input['number_of_iterations']
        self.nsv = loco_input['singular_values']
        self.use_coupling = loco_input['use_coupling']
        self.fit_gains_bpm = loco_input['fit_gains_bpm']
        self.fit_gains_corr = loco_input['fit_gains_corr']
        self.fit_quadrupoles = loco_input['fit_quadrupoles']
        self.model = loco_input['model']
        self.dim = loco_input['dim']
        if not self.model.cavity_on and self.dim == '6d':
            self.model.cavity_on = True
        if not self.model.radiation_on:
            self.model.radiation_on = True
        self.respm = Respmat(model=self.model, dim=self.dim)
        self.matrix = self.respm.get_respm()
        self.use_disp = loco_input['use_dispersion']
        self.bpmidx = self.respm.fam_data['BPM']['index']
        if self.use_disp:
            self.rfline = self.calc_rf_line(self.model)
        else:
            self.rfline = np.zeros((self.matrix.shape[0], 1))
            self.cavidx = None

        self.cavidx = pyaccel.lattice.find_indices(
                self.model, 'fam_name', 'SRFCav')[0]
        if 'rf_frequency' not in loco_input.keys():
            self.rf_freq = self.model[self.cavidx].frequency
        else:
            self.rf_freq = loco_input['rf_frequency']
        self.alpha = pyaccel.optics.get_mcf(self.model)
        self.meas_disp = self.get_meas_disp()

        if not self.use_coupling:
            self.measmat = self.remove_coupling(self.measmat)

        self.matrix = np.hstack(
                [self.matrix, self.rfline])
        self.nbpm = self.matrix.shape[0]//2
        self.ncorr = self.matrix.shape[1] - 1

        if loco_input['gain_bpm'] is None:
            self.gain_bpm = np.ones(2*self.nbpm)
        else:
            self.gain_bpm = loco_input['gain_bpm']
        if loco_input['roll_bpm'] is None:
            self.roll_bpm = np.zeros(self.nbpm)
        else:
            self.roll_bpm = loco_input['roll_bpm']
        if loco_input['gain_corr'] is None:
            self.gain_corr = np.ones(self.ncorr)
        else:
            self.gain_corr = loco_input['gain_corr']
        self.matrix = self.apply_all_gains(
            matrix=self.matrix,
            gain_bpm=self.gain_bpm,
            roll_bpm=self.roll_bpm,
            gain_corr=self.gain_corr)
        self.vector = self.matrix.flatten()
        self.use_families = loco_input['use_families']
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

    def get_meas_disp(self):
        return self.alpha * self.rf_freq * self.measmat[:, -1]

    def disp_corrs(self):
        twi, *_ = pyaccel.optics.calc_twiss(self.model, indices='open')
        chidx = self.respm.fam_data['CH']['index']
        cvidx = self.respm.fam_data['CV']['index']
        corrsidx = chidx + cvidx
        disp = np.zeros((self.ncorr, 2))
        disp[:, 0] = twi.etax[corrsidx]
        disp[:, 1] = twi.etay[corrsidx]
        return disp

    def calc_energy_shift(self):
        disp_meas = self.get_meas_disp()
        dmde = np.zeros(
            (self.matrix.shape[0]*self.matrix.shape[1], self.ncorr))
        delta = 1e-3

        for j in range(self.ncorr):
            energy_shift = np.zeros((self.ncorr, 1))
            energy_shift[j] = delta
            diff = np.dot(disp_meas, energy_shift.T)/delta
            diff = np.hstack([diff, np.zeros((diff.shape[0], 1))])
            dmde[:, j] = diff.flatten()
        return dmde

    @staticmethod
    def get_indices(model):
        """."""
        indices = dict()
        # cavity
        indices['cavidx'] = pyaccel.lattice.find_indices(
            model, 'fam_name', 'SRFCav')[0]
        # bpm
        indices['bpmidx'] = pyaccel.lattice.find_indices(
            model, 'fam_name', 'BPM')
        return indices

    @staticmethod
    def calc_rf_line(model, delta=100):
        """."""
        ind = LOCO.get_indices(model)
        f0 = model[ind['cavidx']].frequency
        model[ind['cavidx']].frequency = f0 + delta/2
        orbp = pyaccel.tracking.find_orbit6(model, indices='open')
        model[ind['cavidx']].frequency = f0 - delta/2
        orbm = pyaccel.tracking.find_orbit6(model, indices='open')
        model[ind['cavidx']].frequency = f0
        dorb = orbp - orbm
        dorbx = dorb[0, ind['bpmidx']]
        dorby = dorb[2, ind['bpmidx']]
        data = np.zeros((len(dorbx) + len(dorby), 1))
        data[:, 0] = np.hstack([dorbx, dorby])/delta
        return data

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
            self.kquads = LOCO._get_quads_strengths(self.model, self.quadsidx)
        else:
            nquads = len(self.qnidx)
            self.kquads = np.array(
                pyaccel.lattice.get_attribute(self.model, 'K', self.qnidx))

        kmatrix = np.zeros((len(self.vector), nquads))
        mod = _dcopy(self.model)

        if self.use_families:
            for idx1, fam in enumerate(self.quadsidx):
                for idx2, idx in enumerate(fam):
                    for idx3, idx_seg in enumerate(idx):
                        pyaccel.lattice.set_attribute(
                            mod, 'K', idx_seg, self.kquads[idx1][idx2][idx3] +
                            deltak)
                new_respm = self.respm.get_respm(model=mod)
                if self.use_disp:
                    new_respm = np.hstack(
                        [new_respm, LOCO.calc_rf_line(mod)])
                else:
                    new_respm = np.hstack(
                        [new_respm, np.zeros((new_respm.shape[0], 1))])
                dmdk = (new_respm - self.matrix)/deltak
                kmatrix[:, idx1] = dmdk.flatten()
                mod = _dcopy(self.model)
        else:
            for idx1, idx in enumerate(self.qnidx):
                for idx2, idx_seg in enumerate(idx):
                    pyaccel.lattice.set_attribute(
                        mod, 'K', idx_seg, self.kquads[idx1][idx2] +
                        deltak)
                new_respm = self.respm.get_respm(model=mod)
                if self.use_disp:
                    new_respm = np.hstack(
                        [new_respm, LOCO.calc_rf_line(mod)])
                else:
                    new_respm = np.hstack(
                        [new_respm, np.zeros((new_respm.shape[0], 1))])
                dmdk = (new_respm - self.matrix)/deltak
                kmatrix[:, idx1] = dmdk.flatten()
                mod = _dcopy(self.model)
        return kmatrix

    @staticmethod
    def _get_quads_strengths(model, quadsidx):
        kquads = []
        for qidx in quadsidx:
            kquads.append(pyaccel.lattice.get_attribute(
                model, 'K', qidx))
        return kquads

    def chidx2(self, matrix1, matrix2):
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
        J_loco[:, nfam+3*nbpm:nfam+3*nbpm+nch+ncv] = dmdg_corr
        # J_loco[:, nfam+3*nbpm+nch+ncv:] = self.calc_energy_shift()
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
                     [modelmat, LOCO.calc_rf_line(mod)])
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
        chidx2_old = self.chidx2(self.measmat, modelmat)
        chidx2_init = chidx2_old
        print('Initial Error: {:.6e}'.format(chidx2_old))

        if not chidx2_init:
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
            self.grad_quads = LOCO._get_quads_strengths(
                model=mod, quadsidx=self.quadsidx)
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
            chidx2_new = self.chidx2(self.measmat, fitmat)
            perc = (chidx2_new - chidx2_init)/chidx2_init * 100
            print('Error: {0:.6e} ({1:.2f}%)'.format(chidx2_new, perc))

            if np.isnan(chidx2_new):
                print('Matrix deviation is NaN!')
                break

            if (chidx2_old - chidx2_new) < tol:
                print('Limit Reached!')
                fitmat, mod = self.get_fitmat(mod, -1*new_pars)
                break
            else:
                chidx2_old = chidx2_new
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
            idx1 = self.idx_grads
            idx2 = idx1 + 2*self.nbpm
            J_loco[:, idx1:idx2] = None
        if not self.fit_gains_corr:
            idx1 = self.idx_grads+3*self.nbpm
            idx2 = idx1 + self.ncorr
            J_loco[:, idx1:idx2] = None
        if not self.use_coupling:
            idx1 = self.idx_grads+2*self.nbpm
            idx2 = idx1 + self.nbpm
            J_loco[:, idx1:idx2] = None

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
            idx1 = self.idx_grads
            idx2 = idx1 + 2*self.nbpm
            idx3 = idx2 + self.nbpm
            parameters['k'] = param[:idx1]
            parameters['bpm_gain'] = param[idx1:idx2]
            parameters['bpm_roll'] = param[idx2:idx3]
            parameters['corr_gain'] = param[idx3:]
        elif size == len(self.grad_quads) + 2*self.nbpm + self.ncorr:
            idx1 = self.idx_grads
            idx2 = idx1 + 2*self.nbpm
            parameters['k'] = param[:idx1]
            parameters['bpm_gain'] = param[idx1:idx2]
            parameters['corr_gain'] = param[idx2:]
        elif size == len(self.grad_quads) + 1*self.nbpm + self.ncorr:
            idx1 = self.idx_grads
            idx2 = idx1 + self.nbpm
            parameters['k'] = param[:idx1]
            parameters['bpm_roll'] = param[idx1:idx2]
            parameters['corr_gain'] = param[idx2:]
        elif size == len(self.grad_quads) + 2*self.nbpm:
            idx1 = self.idx_grads
            parameters['k'] = param[:idx1]
            parameters['bpm_gain'] = param[idx1:]
        elif size == len(self.grad_quads) + 3*self.nbpm:
            idx1 = self.idx_grads
            idx2 = idx1 + 2*self.nbpm
            parameters['k'] = param[:idx1]
            parameters['bpm_gain'] = param[idx1:idx2]
            parameters['bpm_roll'] = param[idx2:]
        elif size == len(self.grad_quads) + 1*self.nbpm:
            idx1 = self.idx_grads
            parameters['k'] = param[:idx1]
            parameters['bpm_roll'] = param[idx1:]
        elif size == len(self.grad_quads) + self.ncorr:
            idx1 = self.idx_grads
            parameters['k'] = param[:idx1]
            parameters['corr_gain'] = param[idx1:]
        elif size == 2*self.nbpm + self.ncorr:
            idx1 = 2*self.nbpm
            parameters['bpm_gain'] = param[:idx1]
            parameters['corr_gain'] = param[idx1:]
        elif size == 3*self.nbpm + self.ncorr:
            idx1 = 2*self.nbpm
            idx2 = idx1 + self.nbpm
            parameters['bpm_gain'] = param[:idx1]
            parameters['bpm_roll'] = param[idx1:idx2]
            parameters['corr_gain'] = param[idx2:]
        elif size == self.nbpm + self.ncorr:
            idx1 = self.nbpm
            parameters['bpm_roll'] = param[:idx1]
            parameters['corr_gain'] = param[idx1:]
        elif size == 3*self.nbpm:
            idx1 = 2*self.nbpm
            parameters['bpm_gain'] = param[:idx1]
            parameters['bpm_roll'] = param[idx1:]

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
            for idx1, idx_fam in enumerate(self.quadsidx):
                kvalues = self.grad_quads[idx1]
                kdelta = self.grad_delta[idx1]
                LOCO.set_quadfam_deltak(modc, idx_fam, kvalues, kdelta)
        else:
            idx_mag = self.qnidx
            kvalues = self.grad_quads
            kdelta = self.grad_delta
            LOCO.set_quadmag_deltak(modc, idx_mag, kvalues, kdelta)

        fitmat = self.respm.get_respm(model=modc)
        if self.use_disp:
            fitmat = np.hstack(
                [fitmat, LOCO.calc_rf_line(mod)])
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

    # @staticmethod
    # def _set_deltas_fams(model, quadidx, ref, delta):
    #     for idx1, fam in enumerate(quadidx):
    #         for idx2, idx in enumerate(fam):
    #             for idx3, idx_seg in enumerate(idx):
    #                 pyaccel.lattice.set_attribute(
    #                     model, 'K', idx_seg, ref[idx1][idx2][idx3] +
    #                     delta[idx1])
    #     return model

    # @staticmethod
    # def _set_deltas_quads(model, quadidx, ref, delta):
    #     for idx1, idx in enumerate(quadidx):
    #         for idx2, idx_seg in enumerate(idx):
    #             pyaccel.lattice.set_attribute(
    #                 model, 'K', idx_seg, ref[idx1][idx2] +
    #                 delta[idx1])
    #     return model

    @staticmethod
    def add_rf_response(model, matrix, use_disp):
        """."""
        if use_disp:
            rfline = LOCO.calc_rf_line(model)
        else:
            rfline = np.zeros((matrix.shape[0], 1))
        matrix = np.hstack([matrix, rfline])
        return matrix

    @staticmethod
    def calc_matrix_rf(model, respm, use_disp):
        matrix = respm.get_respm(model=model)
        matrix = LOCO.add_rf_response(model, matrix, use_disp)
        return matrix

    @staticmethod
    def set_quadmag_deltak(model, idx_mag, kvalues, kdelta):
        """."""
        for idx, idx_seg in enumerate(idx_mag):
            pyaccel.lattice.set_attribute(
                model, 'K', idx_seg, kvalues[idx] +
                kdelta)

    @staticmethod
    def set_quadset_kdelta(model, idx_set, kvalues, kdelta):
        """."""
        for idx, idx_mag in enumerate(idx_set):
            LOCO.set_quadmag_deltak(model, idx_mag, kvalues[idx], kdelta)

    @staticmethod
    def calc_kmatrix(respm,
                     kdelta=1e-6,
                     use_disp=True,
                     use_families=False):
        """."""
        if use_families:
            kindices = []
            for fam_name in LOCO.QUAD_FAM:
                kindices.append(respm.fam_data[fam_name]['index'])
            kvalues = LOCO._get_quads_strengths(respm.model, kindices)
            func = LOCO.set_quadset_kdelta
        else:
            kindices = respm.fam_data['QN']['index']
            kvalues = np.array(
                pyaccel.lattice.get_attribute(respm.model, 'K', kindices))
            func = LOCO.set_quadmag_deltak
        matrix_nominal = LOCO.calc_matrix_rf(respm.model, respm, use_disp)

        kmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(kindices)))
        model = _dcopy(respm.model)

        for idx, idx_set in enumerate(kindices):
            func(model, idx_set, kvalues[idx], kdelta)
            matrix = LOCO.calc_matrix_rf(model, respm, use_disp)
            dmatrix = (matrix - matrix_nominal)/kdelta
            kmatrix[:, idx] = dmatrix.flatten()
            func(model, idx_set, kvalues[idx], 0)
        return kmatrix
