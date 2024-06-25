"""."""
from .. import FACILITY

from .base import DeviceSet
from . import get_device_class as _get_class


class FamBPMs(DeviceSet):
    """."""

    def __init__(self, accelerator=None, bpmnames=None):
        """."""
        bpmidcs = None
        accelerator = accelerator or FACILITY.default_accelerator
        if bpmnames is None:
            bpmidcs, bpmnames = zip(
                *sorted(
                    (amap["sim_info"]["indices"], alias)
                    for alias, amap in FACILITY.alias_map.items()
                    if amap["accelerator"] == accelerator
                    and "BPM" in amap["cs_devtype"]
                )
            )
            bpmidcs, bpmnames = list(bpmidcs), list(bpmnames)

        _bpm_class = _get_class('BPM')
        bpmdevs = [_bpm_class(dev, auto_monitor_mon=False) for dev in bpmnames]
        super().__init__(bpmdevs)
        self._bpm_names = bpmnames
        self._bpm_idcs = bpmidcs

    @property
    def bpm_names(self):
        """."""
        return self._bpm_names

    @property
    def bpm_indices(self):
        """."""
        return self._bpm_idcs

    def get_orbit(self):
        """."""
        orbx, orby = [], []
        for bpm in self.devices:
            orbx.append(bpm.posx)
            orby.append(bpm.posy)
        return orbx, orby
