"""."""
import time as _time

import numpy as _np

from ..utils import get_namedtuple as _get_namedtuple
from .base import DeviceSet
from .fambpms import FamBPMs as _FamBPMs
from .famcms import FamCMs as _FamCMs


class SOFB(DeviceSet):
    """."""

    BPM_UPDATE_RATE = 10.4  # [Hz]
    RespMatMode = _get_namedtuple(
        'RespMatMode',
        ('Mxx', 'Myy', 'NoCoup', 'Full')
    )

    def __init__(self, accelerator=None):
        """."""
        self.fambpms = _FamBPMs(accelerator=accelerator)
        self.famcms = _FamCMs(accelerator=accelerator)
        super().__init__([self.fambpms, self.famcms])

        self.nr_bpms = len(self.fambpms.devices)
        self.nr_hcms = self.famcms.nhcms
        self.nr_vcms = self.famcms.nvcms
        self.nr_cors = self.nr_hcms + self.nr_vcms + 1

        self.bpmx_enbl = _np.ones(self.nr_bpms, dtype=float)
        self.bpmy_enbl = _np.ones(self.nr_bpms, dtype=float)
        self.hcm_enbl = _np.ones(self.nr_hcms, dtype=float)
        self.vcm_enbl = _np.ones(self.nr_vcms, dtype=float)
        self.rfg_enbl = True

        self.corr_gain_hcm = 100  # [%]
        self.corr_gain_vcm = 100  # [%]
        self.corr_gain_rfg = 100  # [%]

        self.ref_orbx = _np.zeros(self.nr_bpms, dtype=float)
        self.ref_orby = _np.zeros(self.nr_bpms, dtype=float)

        self._respmat = _np.zeros((self.nr_bpms*2, self.nr_cors), dtype=float)
        self._respmat_mode = SOFB.RespMatMode.Full

    @property
    def orbx(self):
        """."""
        _orb = self.fambpms.get_orbit()
        return _orb[: self.nbpms // 2]

    @property
    def orby(self):
        """."""
        _orb = self.fambpms.get_orbit()
        return _orb[self.nbpms // 2 :]

    @property
    def kicks_hcm(self):
        """."""
        _kicks = self.famcms.get_currents()
        return _kicks[: self.nhcms]

    @kicks_hcm.setter
    def kicks_hcm(self, values):
        for i, hcm in enumerate(self.famcms.hcms):
            hcm.current = values[i]

    @property
    def kicks_vcm(self):
        """."""
        _kicks = self.famcms.get_currents()
        return _kicks[self.nhcms : self.nhcms + self.nvcms]

    @kicks_vcm.setter
    def kicks_vcm(self, values):
        for i, vcm in enumerate(self.famcms.vcms):
            vcm.current = values[i]

    @property
    def respmat(self):
        return self._respmat

    @respmat.setter
    def respmat(self, value):
        value = _np.asarray(value)



    def get_orbit(self, nr_points=10, method='average'):
        """Return concatenated orbit with desired statistics.

        Args:
            nr_points (int, optional): Number of orbits to calculate
                statistics. Defaults to 10. Each orbit will be acquired with
                an interval defined by SOFB.BPM_UPDATE_RATE.
            method (str, optional): Desired statistics. Supported values are
                'average' and 'median'. Defaults to 'average'.

        Returns:
            _np.ndarray, (2*NR_BPMs, ): Concatenation of horizontal and
                vertical orbits.

        """
        orbs = []
        for _ in range(nr_points):
            orbs.append(_np.hstack([self.fambpms.orbx, self.fambpms.orby]))
            _time.sleep(1/SOFB.BPM_UPDATE_RATE)
        func = _np.mean if method.lower().startswith('average') else _np.median
        return func(orbs, axis=0)

    def correct_orbit(self, nr_iters=5, residue=5):


    def _calc_inv_respmat(self):
        sel_bpm = _np.r_[self.bpmx_enbl, self.bpmy_enbl]
        sel_cor = _np.r_[self.hcm_enbl, self.vcm_enbl, self.rfg_enbl]
        sel_mat = sel_bpm[:, None] * sel_cor[None, :]

        if self._respmat_mode != self._csorb.RespMatMode.Full:
            mat[:nr_bpms, nr_ch:nr_chcv] = 0
            mat[nr_bpms:, :nr_ch] = 0
            mat[nr_bpms:, nr_chcv:] = 0
        if self._respmat_mode == self._csorb.RespMatMode.Mxx:
            mat[nr_bpms:] = 0
        elif self._respmat_mode == self._csorb.RespMatMode.Myy:
            mat[:nr_bpms] = 0
