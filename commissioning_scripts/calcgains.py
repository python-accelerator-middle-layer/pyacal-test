#!/usr/bin/env python-sirius
"""."""

import numpy as np


class CalcGain():

    def __init__(self, meas, model, coupling=False):
        self.meas = meas
        self.model = model
        self.nbpm = self.meas.shape[0] // 2
        self.ncorr = self.meas.shape[1]
        self.coupling = coupling
        if not coupling:
            self.meas = self.remove_coupling(matrix_in=meas)

    def corr_gain_old(self, m_in=None):
        bx = np.zeros((2, 2))
        by = np.zeros((2, 2))
        br = np.zeros((2, 2))
        A = np.zeros((3, 3))
        B = np.zeros((3, 1))
        bx[0, 0] = 1
        by[1, 1] = 1
        br[0, 1] = 1
        br[1, 0] = -1

        Rc = np.zeros((2, 2, self.ncorr))

        if m_in is None:
            m_in = self.meas

        for j in range(self.ncorr):
            for i in range(self.nbpm):
                Mij = np.zeros((2, 1))
                Mij[0] = self.meas[i, j]
                Mij[1] = self.meas[i + self.nbpm, j]

                Tij = np.zeros((2, 1))
                Tij[0] = self.model[i, j]
                Tij[1] = self.model[i + self.nbpm, j]

                mx = np.dot(bx, Mij)[:, 0]
                my = np.dot(by, Mij)[:, 0]
                mr = np.dot(br, Mij)[:, 0]
                Axx = np.dot(mx, mx)
                Ayy = np.dot(my, my)
                Arr = np.dot(mr, mr)
                Axy = np.dot(mx, my)
                Axr = np.dot(mx, mr)
                Ayr = np.dot(my, mr)

                A += np.array([
                    [Axx, Axy, Axr],
                    [Axy, Ayy, Ayr],
                    [Axr, Ayr, Arr]
                ])

                B[0] += np.dot(mx, Tij)
                B[1] += np.dot(my, Tij)
                B[2] += np.dot(mr, Tij)

            if not np.linalg.det(A):
                if np.any(A[:, 0]):
                    rc_elem = B / A[:, 0]
                    rc_elem[np.isnan(rc_elem)] = 0
                else:
                    rc_elem = B / A[:, 1]
                    rc_elem[np.isnan(rc_elem)] = 0
            else:
                rc_elem = np.linalg.solve(A, B)

            if not rc_elem[0] or np.isinf(rc_elem[0]):
                rc_elem[0] = 1

            if not rc_elem[1] or np.isinf(rc_elem[1]):
                rc_elem[1] = 1

            Rc[0, 0, j] = rc_elem[0]
            Rc[0, 1, j] = rc_elem[2]
            Rc[1, 0, j] = -rc_elem[2]
            Rc[1, 1, j] = rc_elem[1]

            A = np.zeros((3, 3))
            B = np.zeros((3, 1))
        return Rc

    def bpm_gain(self, m_in=None):
        bx = np.zeros((2, 2))
        by = np.zeros((2, 2))
        brx = np.zeros((2, 2))
        bry = np.zeros((2, 2))
        A = np.zeros((4, 4))
        B = np.zeros((4, 1))
        bx[0, 0] = 1
        by[1, 1] = 1
        brx[0, 1] = 1
        bry[1, 0] = 1

        Rb = np.zeros((2, 2, self.nbpm))

        if m_in is None:
            m_in = self.model

        for i in range(self.nbpm):
            for j in range(self.ncorr):
                Mij = np.zeros((2, 1))
                Mij[0] = m_in[i, j]
                Mij[1] = m_in[i + self.nbpm, j]

                Tij = np.zeros((2, 1))
                Tij[0] = self.meas[i, j]
                Tij[1] = self.meas[i + self.nbpm, j]

                mx = np.dot(bx, Mij)[:, 0]
                my = np.dot(by, Mij)[:, 0]
                mrx = np.dot(brx, Mij)[:, 0]
                mry = np.dot(bry, Mij)[:, 0]
                Axx = np.dot(mx, mx)
                Ayy = np.dot(my, my)
                Arxrx = np.dot(mrx, mrx)
                Aryry = np.dot(mry, mry)
                Axy = np.dot(mx, my)
                Axrx = np.dot(mx, mrx)
                Axry = np.dot(mx, mry)
                Ayrx = np.dot(my, mrx)
                Ayry = np.dot(my, mry)
                Arxry = np.dot(mrx, mry)

                A += np.array([
                    [Axx, Axy, Axrx, Axry],
                    [Axy, Ayy, Ayrx, Ayry],
                    [Axrx, Ayrx, Arxrx, Arxry],
                    [Axry, Ayry, Arxry, Aryry]
                ])

                B[0] += np.dot(mx, Tij)
                B[1] += np.dot(my, Tij)
                B[2] += np.dot(mrx, Tij)
                B[3] += np.dot(mry, Tij)

            if not self.coupling:
                A = A[0:2, 0:2]
                B = B[0:2]

            if not np.linalg.det(A):
                rc_elem = B.T / A[:, 0]
                rc_elem[np.isnan(rc_elem)] = 0
            else:
                rc_elem = np.linalg.solve(A, B)

            if not rc_elem[0]:
                rc_elem[0] = 1
            if not rc_elem[1]:
                rc_elem[1] = 1

            Rb[0, 0, i] = rc_elem[0]
            Rb[1, 1, i] = rc_elem[1]

            if self.coupling:
                Rb[0, 1, i] = rc_elem[2]
                Rb[1, 0, i] = rc_elem[3]

            A = np.zeros((4, 4))
            B = np.zeros((4, 1))
        return self.rearrange_bpm_gain(Rb=Rb), Rb

    def rearrange_bpm_gain(self, Rb):
        Rbmod = np.zeros((2*self.nbpm, 2*self.nbpm))
        for b in range(self.nbpm):
            idx1 = 2*b
            idx2 = 2*b + 2
            Rbmod[idx1:idx2, idx1:idx2] = Rb[:, :, b]
        return Rbmod

    def corr_gain(self, m_in=None):
        Rc = np.zeros(self.ncorr)

        if m_in is None:
            m_in = self.model

        for j in range(self.ncorr):
            for i in range(self.nbpm):
                if j < self.ncorr//2:
                    if m_in[i, j]:
                        Rc[j] += self.meas[i, j]/m_in[i, j]
                else:
                    if m_in[i+self.nbpm, j]:
                        Rc[j] += self.meas[i+self.nbpm, j] / \
                            m_in[i+self.nbpm, j]
            if not Rc[j]:
                Rc[j] = self.nbpm
        return Rc/self.nbpm

    def get_gains(self):
        Rb, Rbtensor = self.bpm_gain()
        Mb = self.apply_gains(matrix_in=self.model, bpm_gain=Rb)
        Rc = self.corr_gain(m_in=Mb)
        return Rc, Rb, Rbtensor

    def calc_chi2(self, meas, model):
        # for i in range(self.nbpm):
        #     for j in range(self.ncorr):
        #         chi2 += (meas[i, j] - model[i, j])**2
        diff = meas - model
        return np.sqrt(np.sum(diff*diff))

    def apply_gains(self, matrix_in, bpm_gain=None, corr_gain=None):
        rows = matrix_in.shape[0]
        cols = matrix_in.shape[1]
        m_out = np.zeros((rows, cols))
        m_aux = np.zeros((rows, cols))
        if bpm_gain is not None:
            for b in range(rows//2):
                m_aux[2*b, :] = matrix_in[b, :]
                m_aux[2*b+1, :] = matrix_in[b+self.nbpm, :]

            m_out1 = bpm_gain @ m_aux
            for b in range(rows//2):
                m_out[b, :] = m_out1[2*b, :]
                m_out[b+self.nbpm, :] = m_out1[2*b+1, :]
            if corr_gain is not None:
                for c in range(cols):
                    m_out[:, c] *= corr_gain[c]
        if corr_gain is not None and bpm_gain is None:
            for c in range(cols):
                m_out[:, c] = matrix_in[:, c] * corr_gain[c]
        return m_out

    def remove_coupling(self, matrix_in):
        rows = matrix_in.shape[0]
        cols = matrix_in.shape[1]
        matrix_out = np.zeros((rows, cols))
        matrix_out[:rows//2, :cols//2] = matrix_in[:rows//2, :cols//2]
        matrix_out[rows//2:, cols//2:] = matrix_in[rows//2:, cols//2:]
        return matrix_out

    def apply_gains_old(self, matrix_in, bpm_gain=None, corr_gain=None):
        matrix_out = np.zeros((2*self.nbpm, self.ncorr))
        if corr_gain is not None:
            for i in range(self.ncorr):
                for j in range(self.nbpm):
                    Mij = np.zeros((2, 1))
                    Mij[0] = matrix_in[j, i]
                    Mij[1] = matrix_in[j + self.nbpm, i]
                    Mgij = np.dot(corr_gain[:, :, i], Mij)[:, 0]
                    matrix_out[j, i] = Mgij[0]
                    matrix_out[j + self.nbpm, i] = Mgij[1]

        if bpm_gain is not None:
            for i in range(self.nbpm):
                for j in range(self.ncorr):
                    Mij = np.zeros((2, 1))
                    Mij[0] = matrix_out[i, j]
                    Mij[1] = matrix_out[i + self.nbpm, j]
                    Mgij = np.dot(bpm_gain[:, :, i], Mij)[:, 0]
                    matrix_out[i, j] = Mgij[0]
                    matrix_out[i + self.nbpm, j] = Mgij[1]
        return matrix_out
