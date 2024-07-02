"""."""

import numpy as _np

from .. import _get_facility
from .base import DeviceSet
from .bpm import BPM as _BPM


class FamBPMs(DeviceSet):
    """."""

    def __init__(self, accelerator=None, bpmnames=None):
        """."""
        facil = _get_facility()
        self.accelerator = accelerator or facil.default_accelerator
        if bpmnames is None:
            bpmnames = self._get_default_bpmnames()

        bpmdevs = [_BPM(dev, auto_monitor_mon=False) for dev in bpmnames]
        super().__init__(bpmdevs)
        self._bpm_names = bpmnames

    @property
    def bpm_names(self):
        """."""
        return self._bpm_names

    @property
    def orbx(self):
        """."""
        return _np.array([bpm.posx for bpm in self.devices])

    @property
    def orby(self):
        """."""
        return _np.array([bpm.posy for bpm in self.devices])

    # ---------------- helper methods -----------------------
    def _get_default_bpmnames(self):
        facil = _get_facility()
        bpmnames = facil.find_aliases_from_accelerator(self.accelerator)
        bpmnames = facil.find_aliases_from_cs_devtype(
            {facil.CSDevType.BPM, facil.CSDevType.SOFB}, aliases=bpmnames,
        )
        return facil.sort_aliases_by_indices(bpmnames)
