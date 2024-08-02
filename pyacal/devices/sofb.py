"""."""

import time as _time

import numpy as _np

from .. import _get_facility
from ..utils import get_namedtuple as _get_namedtuple
from .base import DeviceSet
from .fambpms import FamBPMs as _FamBPMs
from .famcms import FamCMs as _FamCMs
from .power_supply import PowerSupply as _PowerSupply
from .rfgen import RFGen as _RFGen


class SOFB(DeviceSet):
    """."""

    BPM_UPDATE_RATE = 10.4  # [Hz]
    RespMatMode = _get_namedtuple(
        "RespMatMode", ("Mxx", "Myy", "NoCoup", "Full")
    )

    def __init__(self, accelerator=None):
        """."""
        fac = _get_facility()
        self.accelerator = accelerator or fac.default_accelerator
        self.fambpms = _FamBPMs(accelerator=self.accelerator)
        self.famcms = _FamCMs(accelerator=self.accelerator)
        self.rfgen = _RFGen()
        super().__init__([self.fambpms, self.famcms, self.rfgen])

        self.nr_bpms = len(self.fambpms.bpm_names)
        self.nr_hcms = self.famcms.nr_hcms
        self.nr_vcms = self.famcms.nr_vcms
        self.nr_cors = self.nr_hcms + self.nr_vcms + 1

        self._bpmx_enbl = _np.ones(self.nr_bpms, dtype=bool)
        self._bpmy_enbl = _np.ones(self.nr_bpms, dtype=bool)
        self._hcm_enbl = _np.ones(self.nr_hcms, dtype=bool)
        self._vcm_enbl = _np.ones(self.nr_vcms, dtype=bool)
        self._rfg_enbl = True

        self.orb_nrpoints = 10
        self.orb_method = "average"

        self.corr_gain_hcm = 100  # [%]
        self.corr_gain_vcm = 100  # [%]
        self.corr_gain_rfg = 100  # [%]
        self._delta_strengths_hcm = _np.zeros(self.nr_hcms, dtype=float)
        self._delta_strengths_vcm = _np.zeros(self.nr_vcms, dtype=float)
        self._delta_frequency_rfg = 0.0

        self.ref_orbx = _np.zeros(self.nr_bpms, dtype=float)
        self.ref_orby = _np.zeros(self.nr_bpms, dtype=float)

        self._respmat = _np.zeros(
            (self.nr_bpms * 2, self.nr_cors), dtype=float
        )
        self._respmat_mode = SOFB.RespMatMode.Full
        self.min_relative_singval = 1e-9
        self.tikhonov_reg_const = 0
        self._sing_vals_raw = None
        self._sing_vals_proc = None
        self._sing_vals_nr = None
        self._inv_respmat = None
        self._respmat_processed = None

    @property
    def strengths_hcm(self):
        """."""
        return self.famcms.strengths_hcm

    @property
    def strengths_vcm(self):
        """."""
        return self.famcms.strengths_vcm

    @property
    def delta_strengths_hcm(self):
        """."""
        return self._delta_strengths_hcm

    @delta_strengths_hcm.setter
    def delta_strengths_hcm(self, value):
        value = _np.asarray(value)
        if value.size != self._delta_strengths_hcm.size:
            raise ValueError('Wrong shape for value.')
        self._delta_strengths_hcm = value.copy()

    @property
    def delta_strengths_vcm(self):
        """."""
        return self._delta_strengths_vcm

    @delta_strengths_vcm.setter
    def delta_strengths_vcm(self, value):
        value = _np.asarray(value)
        if value.size != self._delta_strengths_vcm.size:
            raise ValueError('Wrong shape for value.')
        self._delta_strengths_vcm = value.copy()

    @property
    def delta_frequency_rfg(self):
        """."""
        return self._delta_frequency_rfg

    @delta_frequency_rfg.setter
    def delta_frequency_rfg(self, value):
        self._delta_frequency_rfg = float(value)

    @property
    def bpmx_enbl(self):
        """."""
        return self._bpmx_enbl

    @bpmx_enbl.setter
    def bpmx_enbl(self, val):
        val = _np.asarray(val, dtype=bool)
        if val.size != self._bpmx_enbl.size:
            raise ValueError('Wrong shape.')
        self._bpmx_enbl = val

    @property
    def bpmy_enbl(self):
        """."""
        return self._bpmy_enbl

    @bpmy_enbl.setter
    def bpmy_enbl(self, val):
        val = _np.asarray(val, dtype=bool)
        if val.size != self._bpmy_enbl.size:
            raise ValueError('Wrong shape.')
        self._bpmy_enbl = val

    @property
    def hcm_enbl(self):
        """."""
        return self._hcm_enbl

    @hcm_enbl.setter
    def hcm_enbl(self, val):
        val = _np.asarray(val, dtype=bool)
        if val.size != self._hcm_enbl.size:
            raise ValueError('Wrong shape.')
        self._hcm_enbl = val

    @property
    def vcm_enbl(self):
        """."""
        return self._vcm_enbl

    @vcm_enbl.setter
    def vcm_enbl(self, val):
        val = _np.asarray(val, dtype=bool)
        if val.size != self._vcm_enbl.size:
            raise ValueError('Wrong shape.')
        self._vcm_enbl = val

    @property
    def rfg_enbl(self):
        """."""
        return self._rfg_enbl

    @rfg_enbl.setter
    def rfg_enbl(self, val):
        val = _np.asarray(val, dtype=bool)
        if val.size != self._rfg_enbl.size:
            raise ValueError('Wrong shape.')
        self._rfg_enbl = val

    @property
    def respmat(self):
        """."""
        return self._respmat

    @respmat.setter
    def respmat(self, value):
        value = _np.asarray(value)
        if value.shape != (2 * self.nr_bpms, self.nr_cors):
            raise ValueError("Wrong shape for response matrix.")
        self._calc_inv_respmat(value)
        self._respmat = value.copy()

    @property
    def respmat_mode_str(self):
        """."""
        return SOFB.RespMatMode._fields[self._respmat_mode]

    @property
    def respmat_mode(self):
        """."""
        return self._respmat_mode

    @respmat_mode.setter
    def respmat_mode(self, value):
        self._respmat_mode = self._enum_selector(value, SOFB.RespMatMode)

    @property
    def sing_vals_raw(self):
        """."""
        return self._sing_vals_raw

    @property
    def sing_vals_proc(self):
        """."""
        return self._sing_vals_proc

    @property
    def sing_vals_nr(self):
        """."""
        return self._sing_vals_nr

    @property
    def inv_respmat(self):
        """."""
        return self._inv_respmat

    @property
    def respmat_processed(self):
        """."""
        return self._respmat_processed

    def get_orbit(self):
        """Return concatenated orbit with desired statistics.

        This method will use the object properties `orb_method` and
        `orb_nrpoints` to define the statistic and number of acquisitions to
        calculate statistics. Each orbit will be acquired with an interval
        defined by SOFB.BPM_UPDATE_RATE.

        Returns:
            _np.ndarray, (2*NR_BPMs, ): Concatenation of horizontal and
                vertical orbits.

        """
        orbs = []
        for _ in range(self.orb_nrpoints):
            orbs.append(_np.r_[self.fambpms.orbx, self.fambpms.orby])
            _time.sleep(1 / SOFB.BPM_UPDATE_RATE)

        func = _np.mean
        if self.orb_method.lower().startswith("median"):
            func = _np.median
        return func(orbs, axis=0)

    def correct_orbit(self, nr_iters=5, residue=5):
        """Correct orbit until max iteration or convergence.

        Args:
            nr_iters (int, optional): Maximum number of iterations of the
                correction. Defaults to 5.
            residue (int, optional): Convergence criterion for STD in units of
                control system orbit. Defaults to 5.

        """
        for _ in range(nr_iters):
            orb = self.get_orbit()
            dorb = orb - _np.r_[self.ref_orbx, self.ref_orby]
            if dorb.std() < residue:
                break
            self.calculate_correction(dorb)
            self.apply_correction()

    def calculate_correction(self, dorb):
        """Calculate kicks variation to correct orbit.

        Args:
            dorb (_numpy.ndarray, (2*NBPMs, )): orbit to be corrected.

        Returns:
            _numpy.ndarray, (NCORS, ): kicks to correct the orbit.

        """
        inv_mat = self._calc_inv_respmat()
        dkicks = inv_mat @ dorb
        dkicks *= -1
        self._delta_strengths_hcm = dkicks[: self.nr_hcms]
        self._delta_strengths_vcm = dkicks[self.nr_hcms :][: self.nr_vcms]
        self._delta_frequency_rfg = dkicks[-1]

    def apply_correction(self, timeout=10):
        """Apply correction to machine.

        The properties `delta_strengths_hcm`, `delta_strengths_vcm` and
        `delta_frequency_rfg` will be multiplied by the respective correction
        gains (`corr_gain_hcm`, `corr_gain_vcm`, `corr_gain_rfg`) and applied
        to the correctors.

        Args:
            timeout (int, optional): Time to wait for correctors to get to
                desired strength. Defaults to 10.

        Returns:
            bool: whether or not all correctors reached desired strength.

        """
        dhcm = self._delta_strengths_hcm * (self.corr_gain_hcm / 100)
        dvcm = self._delta_strengths_vcm * (self.corr_gain_vcm / 100)
        drfg = self._delta_frequency_rfg * (self.corr_gain_rfg / 100)
        dcm = _np.r_[dhcm, dvcm]

        idcs = _np.isclose(dcm, 0, atol=_PowerSupply.TINY_STRENGTH)
        dcm[idcs] = _np.nan
        currs0 = self.famcms.get_strengths()
        freq = self.rfgen.frequency + drfg

        self.famcms.set_strengths(currs0 + dcm)
        tini = _time.time()
        boo = self.rfgen.set_frequency(freq, timeout=timeout)
        if not boo:
            return False
        timeout = max(0, timeout - (_time.time() - tini))
        currs0[~idcs] += dcm[~idcs]
        return self.famcms.wait_strengths(currs0, timeout=timeout)

    def _calc_inv_respmat(self, mat=None):
        sel_bpm = _np.r_[self.bpmx_enbl, self.bpmy_enbl]
        sel_cor = _np.r_[self.hcm_enbl, self.vcm_enbl, self.rfg_enbl]
        sel_mat = sel_bpm[:, None] * sel_cor[None, :]

        mat0 = self._respmat if mat is None else mat
        mat = mat0.copy()
        nr_svals = min(mat.shape)
        nr_bpms = self.nr_bpms
        nr_hcm = self.nr_hcms
        nr_hvcm = self.nr_hcms + self.nr_vcms
        if self._respmat_mode == self.RespMatMode.NoCoup:
            mat[:nr_bpms, nr_hcm:nr_hvcm] = 0
            mat[nr_bpms:, :nr_hcm] = 0
            mat[nr_bpms:, nr_hvcm:] = 0
        elif self._respmat_mode == self.RespMatMode.Mxx:
            mat[nr_bpms:] = 0
        elif self._respmat_mode == self.RespMatMode.Myy:
            mat[:nr_bpms] = 0

        mat = mat[sel_mat]
        mat = _np.reshape(mat, [_np.sum(sel_bpm), _np.sum(sel_cor)])

        uuu, sing, vvv = _np.linalg.svd(mat, full_matrices=False)
        if _np.isclose(sing[0], 0):
            raise ValueError("All singular values are zero.")
        idcs = sing / sing[0] > self.min_relative_singval
        singr = sing[idcs]
        nr_sv = _np.sum(idcs)

        # Apply Tikhonov regularization:
        regc = self.tikhonov_reg_const
        regc *= regc
        inv_s = _np.zeros(sing.size, dtype=float)
        inv_s[idcs] = singr / (singr * singr + regc)

        # calculate processed singular values
        singp = _np.zeros(sing.size, dtype=float)
        singp[idcs] = 1 / inv_s[idcs]
        inv_mat = _np.dot(vvv.T * inv_s, uuu.T)
        is_nan = _np.any(_np.isnan(inv_mat))
        is_inf = _np.any(_np.isinf(inv_mat))
        if is_nan or is_inf:
            raise ValueError("Inverse contains nan or inf.")

        sing_vals = _np.zeros(nr_svals, dtype=float)
        sing_vals[: sing.size] = sing
        self._sing_vals_raw = sing_vals

        sing_vals = _np.zeros(nr_svals, dtype=float)
        sing_vals[: singp.size] = singp
        self._sing_vals_proc = sing_vals
        self._sing_vals_nr = nr_sv

        self._inv_respmat = _np.zeros_like(mat0).T
        self._inv_respmat[sel_mat.T] = inv_mat.ravel()
        self._respmat_processed = _np.zeros_like(mat0)
        self._respmat_processed[sel_mat] = _np.dot(uuu * singp, vvv).ravel()
        return self._inv_respmat
