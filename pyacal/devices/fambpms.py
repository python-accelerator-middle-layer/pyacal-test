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
            bpmnames = [
                alias
                for alias, amap in facil.alias_map.items()
                if amap["accelerator"] == self.accelerator
                and facil.CSDevTypes.BPM in amap["cs_devtype"]
            ]
            bpmnames.sort(
                key=lambda alias: facil.alias_map[alias]["sim_info"]["indices"]
            )

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
