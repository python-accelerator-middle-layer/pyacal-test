#!/usr/bin/env python-sirius
"""."""

from copy import deepcopy as _dcopy
import numpy as np
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat
import pyaccel


class Coupling():
    """."""

    def __init__(self, model, acc, dim='4d'):
        """."""
        self.model = model
        self.acc = acc
        self.dim = dim
        self.coup_matrix = []
        self.respm = OrbRespmat(model=self.model, acc=self.acc, dim=self.dim)
        self.skew_idx = self.respm.fam_data['QS']['index']
        self.bpm_idx = self.respm.fam_data['BPM']['index']
        self.ch_idx = self.respm.fam_data['CH']['index']
        self.cv_idx = self.respm.fam_data['CV']['index']
        self.nbpm = len(self.bpm_idx)
        self.nch = len(self.ch_idx)
        self.ncv = len(self.cv_idx)
        self.nskew = len(self.skew_idx)

    def calc_coupling_matrix(self, model=None):
        """."""
        if model is None:
            model = self.model

        nvec = self.nbpm*(self.nch+self.ncv + 1)
        self.coup_matrix = np.zeros((nvec, len(self.skew_idx)))
        delta = 1e-6

        for idx, nmag in enumerate(self.skew_idx):
            modcopy = _dcopy(model)
            for seg in nmag:
                modcopy[seg].KsL += delta/len(nmag)
                elem = self.get_coupling_residue(modcopy) / (delta/len(nmag))
            self.coup_matrix[:, idx] = elem
        return self.coup_matrix

    def get_coupling_residue(self, model):
        """."""
        self.respm.model = model
        orbmat = self.respm.get_respm()
        twi, *_ = pyaccel.optics.calc_twiss(model)
        dispy = twi.etay[self.bpm_idx]
        mxy = orbmat[:self.nbpm, self.nch:-1]
        myx = orbmat[self.nbpm:, :self.nch]
        res = mxy.flatten()
        res = np.hstack((res, myx.flatten()))
        res = np.hstack((res, dispy.flatten()))
        return res

    def get_ksl(self, model=None, skewidx=None):
        """."""
        if model is None:
            model = self.model
        if skewidx is None:
            skewidx = self.skew_idx
        ksl = []
        for mag in skewidx:
            ksl.append(model[mag[0]].KsL)
        return np.array(ksl)

    def set_ksl(self, model=None, skewidx=None, ksl=None):
        """."""
        if model is None:
            model = self.model
        if skewidx is None:
            skewidx = self.skew_idx
        if ksl is None:
            raise Exception('Missing KsL values')
        newmod = _dcopy(model)
        for idx, mag in enumerate(skewidx):
            newmod[mag[0]].KsL = ksl[idx]
        return newmod

    def correct_coupling(self,
                         model,
                         matrix=None,
                         nsv=None,
                         niter=10,
                         tol=1e-6,
                         res0=None):
        """."""
        if matrix is None:
            matrix = self.calc_coupling_matrix(model)
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        if nsv is not None:
            inv_s[nsv:] = 0
        inv_s = np.diag(inv_s)
        inv_matrix = np.dot(np.dot(v.T, inv_s), u.T)
        if res0 is None:
            res = self.get_coupling_residue(model)
        else:
            res = res0
        bestfm = np.sum(np.abs(res)**2)/res.size
        ksl0 = self.get_ksl(model)
        ksl = ksl0

        for i in range(niter):
            dksl = np.dot(inv_matrix, -res)
            ksl += dksl
            model = self.set_ksl(model=model, ksl=ksl)
            res = self.get_coupling_residue(model)
            fm = np.sum(np.abs(res)**2)/res.size
            diff_fm = np.abs(bestfm - fm)
            print(i, bestfm)
            if fm < bestfm:
                bestfm = fm
            if diff_fm < tol:
                break
        print('done!')
        return model

    @staticmethod
    def calc_emittance_coupling(model, xlist=None, nr_turns=100):
        """."""
        if xlist is None:
            xlist = 1e-3 * np.linspace(-0.1, 0.1, 10)

        factor = 1e-5

        model.cavity_on = False
        model.radiation_on = False

        cod = pyaccel.tracking.find_orbit4(model)
        cod = np.vstack((cod, np.array([[0], [0]])))
        twiss, *_ = pyaccel.optics.calc_twiss(model)

        coupling = []
        for x in xlist:
            dcod = np.array([x, 0, x*factor, 0, 0, 0])
            dcod = np.reshape(dcod, (-1, 1))
            p0 = list((cod + dcod).T)
            traj, *_ = pyaccel.tracking.ring_pass(
                model, particles=p0, nr_turns=nr_turns, turn_by_turn='closed')
            dtraj = traj - cod
            emitx, emity = Coupling.calc_emittances(dtraj, twiss, 0)
            coupling.append(emity/emitx)
        return coupling

    @staticmethod
    def calc_emittances(orbit, twiss, idx):
        """."""
        alphax = twiss.alphax[idx]
        betax = twiss.betax[idx]
        gammax = (1 + alphax*alphax)/betax

        alphay = twiss.alphay[idx]
        betay = twiss.betay[idx]
        gammay = (1 + alphay*alphay)/betay

        x, xl, y, yl = orbit[0, :], orbit[1, :], orbit[2, :], orbit[3, :]

        emitx = gammax * x**2 + 2 * alphax * x * xl + betax * xl**2
        emity = gammay * y**2 + 2 * alphay * y * yl + betay * yl**2
        return emitx, emity
