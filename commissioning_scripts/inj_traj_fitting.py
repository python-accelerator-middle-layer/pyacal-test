"""."""

import numpy as np

from siriuspy.devices import SOFB
import pyaccel
from pymodels import si, bo

from .base import BaseClass


class Params:
    """."""

    def __init__(self):
        """."""
        self.simul_emitx = 3.5e-9
        self.simul_emity = 3.5e-11
        self.simul_espread = 1e-3
        self.simul_bunlen = 5e-3
        self.simul_npart = 1000
        self.simul_cutoff = 6
        self.count_rel_thres = 1/3
        self.count_init_ref = 3


class _FitInjTrajBase(BaseClass):
    """Base class for fitting injection trajectories."""

    CHAMBER_RADIUS = 0.0
    ANT_ANGLE = 0.0
    POLYNOM = 1e-9 * np.zeros(15, dtype=float)
    NONLINEAR = True

    def __init__(self):
        """."""
        super().__init__(Params())
        self.devices['sofb'] = None
        self.model = None
        self.simul_model = None
        self.famdata = None
        self.bpm_idx = None
        self.twiss = None

    def calc_traj(self, x0, xl0=0, y0=0, yl0=0, delta=0, size=160):
        """."""
        rin = np.array([x0, xl0, y0, yl0, delta, 0])
        rout, *_ = pyaccel.tracking.linepass(
            self.model, rin, self.bpm_idx[:size])
        return rout[0, :], rout[2, :]

    def calc_residue(self, vec, tx_meas, ty_meas):
        """."""
        tx_mod, ty_mod = self.calc_traj(*vec, size=tx_meas.size)
        nanidcs = np.logical_or(np.isnan(tx_mod), np.isnan(tx_mod))
        tx_mod[nanidcs] = 10.0
        ty_mod[nanidcs] = 10.0
        return np.hstack([tx_mod-tx_meas, ty_mod-ty_meas])

    def calc_chisqr(self, residue):
        """."""
        return np.sum(residue*residue)/residue.size

    def calc_jacobian(self, vec, size=160):
        """."""
        mat = np.zeros((2*size, 5))
        zer = np.zeros(size)
        dlt = 1e-5
        for i in range(vec.size):
            dvec = np.array(vec)
            dvec[i] += dlt/2
            res_pos = self.calc_residue(dvec, zer, zer)
            dvec[i] -= dlt
            res_neg = self.calc_residue(dvec, zer, zer)
            mat[:, i] = (res_pos - res_neg)/dlt
        return mat

    def calc_init_vals(self, trajx, trajy):
        """."""
        etax_ave = np.mean(self.twiss.etax[self.bpm_idx])
        x_ini, y_ini, xl_ini, yl_ini = trajx[0], trajy[0], 0, 0
        de_ini = np.mean(trajx) / etax_ave
        return np.array([x_ini, xl_ini, y_ini, yl_ini, de_ini])

    def do_fitting(
            self, trajx, trajy, vec0=None, max_iter=5, tol=1e-6, full=False):
        """."""
        vec0 = vec0 if vec0 is not None else self.calc_init_vals(trajx, trajy)
        res0 = self.calc_residue(vec0, trajx, trajy)
        chi0 = self.calc_chisqr(res0)
        vecs = [vec0, ]
        residues = [res0, ]
        chis = [chi0, ]

        vec, res = vec0.copy(), res0.copy()
        factor = 1
        for _ in range(max_iter):
            mat = self.calc_jacobian(vec0, size=trajx.size)
            u_mat, s_mat, vh_mat = np.linalg.svd(mat, full_matrices=False)
            imat = vh_mat.T @ np.diag(1/s_mat) @ u_mat.T

            dpos = imat @ res
            vec -= dpos * factor
            res = self.calc_residue(vec, trajx, trajy)
            chi = self.calc_chisqr(res)
            if chi >= chi0:
                vec = vec0.copy()
                res = res0.copy()
                factor = max(factor/2, 1e-2)
            else:
                vec0 = vec.copy()
                res0 = res.copy()
                chi0 = chi
                vecs.append(vec0)
                residues.append(res0)
                chis.append(chi0)
                factor = min(factor*2, 1)
            if chi0 < tol:
                break
        if full:
            return vecs, residues, chis
        else:
            return vecs

    def get_traj_from_sofb(self):
        """."""
        trajx = self.devices['sofb'].trajx.copy()
        trajy = self.devices['sofb'].trajy.copy()
        summ = self.devices['sofb'].sum.copy()

        ini = np.mean(summ[:self.params.count_init_ref])
        indcs = summ > ini * self.params.count_rel_thres
        maxidx = np.sum(indcs)
        trajx = trajx[:maxidx]
        trajy = trajy[:maxidx]
        trajx *= 1e-6  # from um to m
        trajy *= 1e-6  # from um to m
        return trajx, trajy, summ

    def simulate_sofb(
            self, x0, xl0, y0=0, yl0=0, delta=0, twi=None, errx=1e-3,
            erry=1e-3):
        """."""
        twi = twi if twi is not None else self.twiss[0]
        bun = pyaccel.tracking.generate_bunch(
            self.params.simul_emitx, self.params.simul_emity,
            self.params.simul_espread, self.params.simul_bunlen, twi,
            self.params.simul_npart, cutoff=self.params.simul_cutoff)
        bun += np.array([x0, xl0, y0, yl0, delta, 0])[:, None]

        rout, *_ = pyaccel.tracking.linepass(
            self.simul_model, bun, indices=self.bpm_idx)

        trajx, trajy = rout[0], rout[2]
        x_uncal, y_uncal = self._calc_bpm_uncal_pos(trajx, trajy)

        nanx = np.isnan(x_uncal)
        snan = np.sum(~nanx, axis=0)
        indcs = snan > self.params.simul_npart * self.params.count_rel_thres
        x_uncal = np.nanmean(x_uncal[:, indcs], axis=0)
        y_uncal = np.nanmean(y_uncal[:, indcs], axis=0)

        if self.NONLINEAR:
            trajx, trajy = self._apply_polyxy(x_uncal, y_uncal)
        else:
            trajx, trajx = self._apply_linearxy(x_uncal, y_uncal)

        trajx += errx * (np.random.rand(trajx.size)-0.5)*2
        trajy += erry * (np.random.rand(trajy.size)-0.5)*2
        return trajx, trajy, snan

    # ##### private methods #####
    @classmethod
    def _calc_bpm_uncal_pos(cls, x_pos, y_pos):
        """."""
        phi = np.array([1, 3, 5, 7])*np.pi/4
        phi = np.expand_dims(phi, axis=tuple([i+1 for i in range(x_pos.ndim)]))

        theta = np.arctan2(y_pos, x_pos)[None, ...]
        dist = np.sqrt(x_pos*x_pos + y_pos*y_pos)[None, ...]
        dist /= cls.CHAMBER_RADIUS
        dist2 = dist*dist

        sup_esq, sup_dir, inf_dir, inf_esq = np.arctan2(
            (1-dist2)*np.sin(cls.ANT_ANGLE/2),
            (1+dist2)*np.cos(cls.ANT_ANGLE/2) - 2*dist*np.cos(theta - phi))

        sum1 = sup_esq + inf_dir
        sum2 = inf_esq + sup_dir
        dif1 = sup_esq - inf_dir
        dif2 = inf_esq - sup_dir

        x_uncal = (dif1/sum1 + dif2/sum2)/2
        y_uncal = (dif1/sum1 - dif2/sum2)/2

        return x_uncal, y_uncal

    @classmethod
    def _apply_linearxy(cls, x_uncal, y_uncal):
        """."""
        gain = cls.CHAMBER_RADIUS/np.sqrt(2)
        gain *= cls.ANT_ANGLE / np.sin(cls.ANT_ANGLE)
        x_cal = x_uncal * gain
        y_cal = y_uncal * gain
        return x_cal, y_cal

    @classmethod
    def _apply_polyxy(cls, x_uncal, y_uncal):
        """."""
        x_cal = cls._calc_poly(x_uncal, y_uncal)
        y_cal = cls._calc_poly(y_uncal, x_uncal)
        return x_cal, y_cal

    @classmethod
    def _calc_poly(cls, th1, ot1):
        """."""
        ot2 = ot1*ot1
        ot4 = ot2*ot2
        ot6 = ot4*ot2
        ot8 = ot4*ot4
        th2 = th1*th1
        th3 = th1*th2
        th5 = th3*th2
        th7 = th5*th2
        th9 = th7*th2
        pol = cls.POLYNOM
        return (
            th1*(pol[0] + ot2*pol[1] + ot4*pol[2] + ot6*pol[3] + ot8*pol[4]) +
            th3*(pol[5] + ot2*pol[6] + ot4*pol[7] + ot6*pol[8]) +
            th5*(pol[9] + ot2*pol[10] + ot4*pol[11]) +
            th7*(pol[12] + ot2*pol[13]) +
            th9*pol[14])


class SIFitInjTraj(_FitInjTrajBase):
    """Fit injection trajectories in the Sirius storage ring.

    Examples:
    ---------
    >>> import numpy as np
    >>> from apsuite.commissioning_scripts.inj_traj_fitting import SIFitInjTraj
    >>> np.random.seed(42)
    >>> fit_traj = SIFitInjTraj()
    >>> x0, xl0, y0, yl0, de0 = -9.0e-3, 0.0e-3, 0.0e-3, 0.0, 0.01
    >>> trajx, trajy, trajsum = fit_traj.simulate_sofb(
            x0, xl0, y0, yl0, de0)
    >>> # trajx, trajy, trajsum = fit_traj.get_traj_from_sofb()
    >>> vecs = fit_traj.do_fitting(trajx, trajy, tol=1e-8)

    """

    CHAMBER_RADIUS = 12e-3
    ANT_ANGLE = 6 / CHAMBER_RADIUS
    POLYNOM = 1e-9 * np.array([
        8.57433100e6, 4.72784700e6, 4.03599000e6, 2.81406000e6,
        9.67341100e6, 4.01543800e6, 1.05648850e7, 9.85821200e6,
        8.68409560e7, 3.94657800e6, 5.27686400e6, 2.28461777e8,
        -1.13979600e6, 9.54919660e7, 2.43619500e7])
    NONLINEAR = True

    def __init__(self, ring=None, sim_mod=None):
        """."""
        super().__init__()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.model = ring if ring is not None else si.create_accelerator()
        self.simul_model = sim_mod if sim_mod is not None else self.model[:]

        injp = pyaccel.lattice.find_indices(
            self.model, 'fam_name', 'InjNLKckr')
        self.model = pyaccel.lattice.shift(self.model, injp[0]+1)
        self.simul_model = pyaccel.lattice.shift(self.simul_model, injp[0]+1)

        self.famdata = si.get_family_data(self.model)
        self.bpm_idx = np.array(self.famdata['BPM']['index']).flatten()
        self.twiss, *_ = pyaccel.optics.calc_twiss(self.model)

        self.model.vchamber_on = True
        self.simul_model.vchamber_on = True


class BOFitInjTraj(_FitInjTrajBase):
    """Fit injection trajectories in the Sirius booster.

    Examples:
    ---------
    >>> import numpy as np
    >>> from apsuite.commissioning_scripts.inj_traj_fitting import BOFitInjTraj
    >>> np.random.seed(42)
    >>> fit_traj = BOFitInjTraj()
    >>> x0, xl0, y0, yl0, de0 = -9.0e-3, 0.0e-3, 0.0e-3, 0.0, 0.01
    >>> trajx, trajy, trajsum = fit_traj.simulate_sofb(
            x0, xl0, y0, yl0, de0)
    >>> # trajx, trajy, trajsum = fit_traj.get_traj_from_sofb()
    >>> vecs = fit_traj.do_fitting(trajx, trajy, tol=1e-8)

    """

    CHAMBER_RADIUS = 17.5e-3
    ANT_ANGLE = 8.5 / CHAMBER_RADIUS  # 6/12*17.5
    POLYNOM = 1e-9 * np.array([
        1.30014e7, 7.67435e6, 5.80753e6, 4.2811e6, 3.02077e6, 5.78664e6,
        1.74211e7, 2.30335e7, 1.12014e8, 4.90624e6, 1.79161e7, 3.52371e8,
        1.54782e6, 9.90632e7, 2.06262e7, 0, 0, 0, 0, 0])
    NONLINEAR = False

    def __init__(self, ring=None, sim_mod=None):
        """."""
        super().__init__()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.BO)
        self.model = ring if ring is not None else bo.create_accelerator()
        self.simul_model = sim_mod if sim_mod is not None else self.model[:]

        injp = pyaccel.lattice.find_indices(self.model, 'fam_name', 'InjKckr')
        self.model = pyaccel.lattice.shift(self.model, injp[0]+1)
        self.simul_model = pyaccel.lattice.shift(self.simul_model, injp[0]+1)

        self.famdata = si.get_family_data(self.model)
        self.bpm_idx = np.array(self.famdata['BPM']['index']).flatten()
        self.twiss, *_ = pyaccel.optics.calc_twiss(self.model)

        self.model.vchamber_on = True
        self.simul_model.vchamber_on = True
