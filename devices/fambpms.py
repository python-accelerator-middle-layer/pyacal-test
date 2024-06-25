from .. import ALIAS_MAP

from .base import DeviceSet
from .bpm import BPM


class FamBPMs(DeviceSet):
    """."""

    PROPERTIES_DEFAULT = BPM.PROPERTIES_DEFAULT

    def __init__(self, accelerator=DEFAULT_ACCELERATOR, bpmnames=None):
        """."""
        bpmidcs = None
        if bpmnames is None:
            bpmidcs, bpmnames = zip(
                *sorted(
                    (amap["sim_info"]["indices"], alias)
                    for alias, amap in ALIAS_MAP.items()
                    if amap["accelerator"] == accelerator
                    and amap["cs_devtype"] == "BPM"
                )
            )
            bpmidcs, bpmnames = list(bpmidcs), list(bpmnames)

        bpmdevs = [BPM(dev, auto_monitor_mon=False) for dev in bpmnames]
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
