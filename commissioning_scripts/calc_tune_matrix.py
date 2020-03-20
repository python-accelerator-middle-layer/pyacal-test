#!/usr/bin/env python-sirius
"""."""

from copy import deepcopy as _dcopy
import numpy as np
from pymodels import bo, si
import pyaccel


class Tune():
    """."""

    SIQUADS = ['QFA', 'QFB', 'QFP', 'QDA', 'QDB1', 'QDB2', 'QDP1', 'QDP2']
    BOQUADS = ['QF', 'QD']

    def __init__(self, model, acc):
        """."""
        self.model = model
        self.acc = acc
        if acc == 'BO':
            self.quads = self.BOQUADS
            self.fam = bo.get_family_data(model)
        elif acc == 'SI':
            self.quads = self.SIQUADS
            self.fam = si.get_family_data(model)
        self.matrix = []
        self.tunex, self.tuney = self.get_tunes()

    def get_tunes(self, model=None):
        """."""
        if model is None:
            model = self.model
        twinom, *_ = pyaccel.optics.calc_twiss(
            accelerator=model, indices='open')
        tunex = twinom.mux[-1]/2/np.pi
        tuney = twinom.muy[-1]/2/np.pi
        return tunex, tuney

    def calctunematrix(self, model=None, acc=None):
        """."""
        if model is None:
            model = self.model
        if acc is None:
            acc = self.acc

        self.matrix = np.zeros((2, len(self.quads)))

        delta = 1e-2
        for idx, q in enumerate(self.quads):
            modcopy = _dcopy(model)
            for nmag in self.fam[q]['index']:
                dlt = delta/len(nmag)
                for seg in nmag:
                    modcopy[seg].KL += dlt
            tunex, tuney = self.get_tunes(model=modcopy)
            self.matrix[:, idx] = [
                (tunex - self.tunex)/dlt, (tuney - self.tuney)/dlt]
        return self.matrix

    def get_kl(self, model=None, quads=None):
        """."""
        if model is None:
            model = self.model
        if quads is None:
            quads = self.quads
        kl = []
        for q in quads:
            kl.append(model[self.fam[q]['index'][0][0]].KL)
        return kl

    def set_kl(self, model=None, quads=None, kl=None):
        """."""
        if model is None:
            model = self.model
        if quads is None:
            quads = self.quads
        if kl is None:
            raise Exception('Missing KL values')
        newmod = _dcopy(model)
        for idx, q in enumerate(quads):
            newmod[self.fam[q]['index'][0][0]].KL = kl[idx]
        return newmod

    def change_tunes(self, model, tunex, tuney, matrix=None):
        """."""
        if matrix is None:
            matrix = self.calctunematrix(model)
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        inv_s = np.diag(inv_s)
        inv_matrix = np.dot(np.dot(v.T, inv_s), u.T)
        tunex0, tuney0 = self.get_tunes(model)
        print(tunex0, tuney0)
        tunex_new, tuney_new = tunex0, tuney0
        kl = self.get_kl(model)
        tol = 1e-6

        while abs(tunex_new - tunex) > tol or abs(tuney_new - tuney) > tol:
            dtune = [tunex-tunex_new, tuney-tuney_new]
            dkl = np.dot(inv_matrix, dtune)
            kl += dkl
            model = self.set_kl(model=model, kl=kl)
            tunex_new, tuney_new = self.get_tunes(model)
            print(tunex_new, tuney_new)
        print('done!')
        return model
