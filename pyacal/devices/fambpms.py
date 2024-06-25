"""."""

import numpy as _np

from .. import FACILITY
from .base import DeviceSet
from .bpm import BPM as _BPM


class FamBPMs(DeviceSet):
    """."""

    def __init__(self, accelerator=None, bpmnames=None):
        """."""
        self.accelerator = accelerator or FACILITY.default_accelerator
        if bpmnames is None:
            bpmnames = [
                alias
                for alias, amap in FACILITY.alias_map.items()
                if amap["accelerator"] == self.accelerator
                and "BPM" in amap["cs_devtype"]
            ]
            bpmnames.sort(
                key=lambda alias: FACILITY.alias_map[alias]["sim_info"][
                    "indices"
                ]
            )

        bpmdevs = [_BPM(dev, auto_monitor_mon=False) for dev in bpmnames]
        super().__init__(bpmdevs)
        self._bpm_names = bpmnames

    @property
    def bpm_names(self):
        """."""
        return self._bpm_names

    def get_orbit(self):
        """."""
        orbx, orby = [], []
        for bpm in self.devices:
            orbx.append(bpm.posx)
            orby.append(bpm.posy)
        return _np.array(orbx), _np.array(orby)
