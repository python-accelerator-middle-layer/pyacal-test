"""."""

import numpy as np
from pymodels import tb, bo, ts, si
import pyaccel


class OrbRespmat():
    """."""

    _FREQ_DELTA = 10
    _ENERGY_DELTA = 1e-5

    def __init__(self, model, acc, dim='4d'):
        """."""
        self.model = model
        self.acc = acc
        if self.acc == 'BO':
            self.fam_data = bo.families.get_family_data(self.model)
        elif self.acc == 'SI':
            self.fam_data = si.families.get_family_data(self.model)
        else:
            raise Exception('Set models: BO or SI')
        self.dim = dim
        self.bpms = self._get_idx(self.fam_data['BPM']['index'])
        self.ch_idx = self._get_idx(self.fam_data['CH']['index'])
        self.cv_idx = self._get_idx(self.fam_data['CV']['index'])

    @staticmethod
    def _get_idx(indcs):
        return np.array([idx[0] for idx in indcs])

    def _calc_rfline(self):
        idx = pyaccel.lattice.find_indices(
            self.model, 'pass_method', 'cavity_pass')[0]
        rffreq = self.model[idx].frequency
        if self.dim == '6d':
            dfreq = OrbRespmat._FREQ_DELTA
            self.model[idx].frequency = rffreq + dfreq
            orbp = pyaccel.tracking.find_orbit6(self.model, indices='open')
            self.model[idx].frequency = rffreq - dfreq
            orbn = pyaccel.tracking.find_orbit6(self.model, indices='open')
            self.model[idx].frequency = rffreq
            rfline = (orbp[[0, 2], :] - orbn[[0, 2], :])/2/dfreq
            rfline = rfline[:, self.bpms].flatten()
        else:
            denergy = OrbRespmat._ENERGY_DELTA
            orbp = pyaccel.tracking.find_orbit4(
                self.model, energy_offset=denergy, indices='open')
            orbn = pyaccel.tracking.find_orbit4(
                self.model, energy_offset=-denergy, indices='open')
            dispbpm = (orbp[[0, 2], :] - orbn[[0, 2], :])/2/denergy
            dispbpm = dispbpm[:, self.bpms].flatten()

            rin = np.zeros((6, 2))
            rin[4, :] = [denergy, -denergy]
            rout, *_ = pyaccel.tracking.ring_pass(self.model, rin)
            leng = self.model.length
            alpha = -1 * np.diff(rout[5, :]) / 2 / denergy / leng
            # Convert dispersion to deltax/deltaf:
            rfline = - dispbpm / rffreq / alpha
        return rfline

    def get_respm(self):
        """."""
        cav = self.model.cavity_on
        self.model.cavity_on = self.dim == '6d'
        if self.dim == '6d':
            m_mat, t_mat = pyaccel.tracking.find_m66(
                self.model, indices='open')
        else:
            m_mat, t_mat = pyaccel.tracking.find_m44(
                self.model, indices='open')

        nch = len(self.ch_idx)
        respmat = []
        corrs = np.hstack([self.ch_idx, self.cv_idx])
        for idx, corr in enumerate(corrs):
            rc_mat = t_mat[corr, :, :]
            rb_mat = t_mat[self.bpms, :, :]
            corr_len = self.model[corr].length
            kl_stren = self.model[corr].KL
            ksl_stren = self.model[corr].KsL
            respx, respy = self._get_respmat_line(
                rc_mat, rb_mat, m_mat, corr, corr_len,
                kxl=kl_stren, kyl=-kl_stren, ksxl=ksl_stren, ksyl=ksl_stren)
            if idx < nch:
                respmat.append(respx)
            else:
                respmat.append(respy)

        rfline = self._calc_rfline()
        respmat.append(rfline)
        respmat = np.array(respmat).T

        self.model.cavity_on = cav
        return respmat

    def _get_respmat_line(self, rc_mat, rb_mat, m_mat, corr, length,
                          kxl=0, kyl=0, ksxl=0, ksyl=0):
        # create a symplectic integrator of second order
        # for the last half of the element:
        drift = np.eye(rc_mat.shape[0], dtype=float)
        drift[0, 1] = length/2 / 2
        drift[2, 3] = length/2 / 2
        quad = np.eye(rc_mat.shape[0], dtype=float)
        quad[1, 0] = -kxl/2
        quad[3, 2] = -kyl/2
        quad[1, 2] = -ksxl/2
        quad[3, 0] = -ksyl/2
        half_cor = drift @ quad @ drift
        rc_mat = half_cor @ rc_mat

        mc_mat = np.linalg.solve(
            rc_mat.T, (rc_mat @ m_mat).T).T  # Mc = Rc M Rc^-1
        mci_mat = np.eye(mc_mat.shape[0], dtype=float) - mc_mat

        small = self.bpms < corr
        large = np.logical_not(small)

        rcbl_mat = np.linalg.solve(rc_mat.T, rb_mat.transpose((0, 2, 1)))
        rcbl_mat = rcbl_mat.transpose((0, 2, 1))
        rcbs_mat = rcbl_mat[small] @ mc_mat
        rcbl_mat = rcbl_mat[large]

        rcbl_mat = np.linalg.solve(mci_mat.T, rcbl_mat.transpose((0, 2, 1)))
        rcbl_mat = rcbl_mat.transpose((0, 2, 1))
        rcbs_mat = np.linalg.solve(mci_mat.T, rcbs_mat.transpose((0, 2, 1)))
        rcbs_mat = rcbs_mat.transpose((0, 2, 1))

        respxx = np.zeros(len(self.bpms))
        respyx = np.zeros(len(self.bpms))
        respxy = np.zeros(len(self.bpms))
        respyy = np.zeros(len(self.bpms))

        respxx[large] = rcbl_mat[:, 0, 1]
        respyx[large] = rcbl_mat[:, 2, 1]
        respxx[small] = rcbs_mat[:, 0, 1]
        respyx[small] = rcbs_mat[:, 2, 1]
        respx = np.hstack([respxx, respyx])

        respxy[large] = rcbl_mat[:, 0, 3]
        respyy[large] = rcbl_mat[:, 2, 3]
        respxy[small] = rcbs_mat[:, 0, 3]
        respyy[small] = rcbs_mat[:, 2, 3]
        respy = np.hstack([respxy, respyy])
        return respx, respy


class TrajRespmat():
    """."""

    def __init__(self, model, acc):
        """."""
        self.model = model
        self.acc = acc
        if acc == 'TB':
            self.fam_data = tb.get_family_data(model)
        elif acc == 'BO':
            self.fam_data = bo.get_family_data(model)
        elif acc == 'TS':
            self.fam_data = ts.get_family_data(model)
        elif acc == 'SI':
            self.fam_data = si.get_family_data(model)

        self.bpms = self._get_idx(self.fam_data['BPM']['index'])
        self.ch_idx = self.fam_data['CH']['index']

        if acc == 'TS':
            ejesept = pyaccel.lattice.find_indices(
                model, 'fam_name', 'EjeSeptG')
            segs = len(ejesept)
            self.ch_idx.append([ejesept[segs//2]])
            self.ch_idx = sorted(self.ch_idx)

        self.ch_idx = self._get_idx(self.ch_idx)
        self.cv_idx = self._get_idx(self.fam_data['CV']['index'])

    @staticmethod
    def _get_idx(indcs):
        return np.array([idx[0] for idx in indcs])

    def get_respm(self):
        """."""
        _, cumulmat = pyaccel.tracking.find_m44(
            self.model, indices='open', closed_orbit=[0, 0, 0, 0])

        respmat = []
        corrs = np.hstack([self.ch_idx, self.cv_idx])
        for idx, corr in enumerate(corrs):
            rc_mat = cumulmat[corr]
            rb_mat = cumulmat[self.bpms]
            corr_len = self.model[corr].length
            kl_stren = self.model[corr].KL
            ksl_stren = self.model[corr].KsL
            respx, respy = self._get_respmat_line(
                rc_mat, rb_mat, corr, length=corr_len,
                kxl=kl_stren, kyl=-kl_stren, ksxl=ksl_stren, ksyl=ksl_stren)
            if idx < len(self.ch_idx):
                respmat.append(respx)
            else:
                respmat.append(respy)

        respmat.append(np.zeros(2*len(self.bpms)))
        respmat = np.array(respmat).T
        return respmat

    def _get_respmat_line(self, rc_mat, rb_mat, corr, length,
                          kxl=0, kyl=0, ksxl=0, ksyl=0):
        # create a symplectic integrator of second order
        # for the last half of the element:
        drift = np.eye(4, dtype=float)
        drift[0, 1] = length/2 / 2
        drift[2, 3] = length/2 / 2
        quad = np.eye(4, dtype=float)
        quad[1, 0] = -kxl/2
        quad[3, 2] = -kyl/2
        quad[1, 2] = -ksxl/2
        quad[3, 0] = -ksyl/2
        half_cor = drift @ quad @ drift

        rc_mat = half_cor @ rc_mat

        large = self.bpms > corr

        rb_mat = rb_mat[large, :, :]
        rcb_mat = np.linalg.solve(rc_mat.T, rb_mat.transpose((0, 2, 1)))
        rcb_mat = rcb_mat.transpose(0, 2, 1)

        respxx = np.zeros(len(self.bpms), dtype=float)
        respyx = np.zeros(len(self.bpms), dtype=float)
        respxy = np.zeros(len(self.bpms), dtype=float)
        respyy = np.zeros(len(self.bpms), dtype=float)

        respxx[large] = rcb_mat[:, 0, 1]
        respyx[large] = rcb_mat[:, 2, 1]
        respx = np.hstack([respxx, respyx])

        respxy[large] = rcb_mat[:, 0, 3]
        respyy[large] = rcb_mat[:, 2, 3]
        respy = np.hstack([respxy, respyy])
        return respx, respy
