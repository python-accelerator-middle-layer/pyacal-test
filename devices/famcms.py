from .. import ALIAS_MAP

from .base import DeviceSet
from .power_supply import PowerSupply


class FamCMs(DeviceSet):
    """."""

    PROPERTIES_DEFAULT = PowerSupply.PROPERTIES_DEFAULT

    def __init__(
        self, accelerator=DEFAULT_ACCELERATOR, cmnames=None, plane="HV"
    ):
        """."""
        cmidcs = []
        self.accelerator = accelerator
        if cmnames is None:
            cmnames = []
            plane = plane.upper()
            if "H" in plane:
                hcmidcs, hcmnames = self._get_cm_names(
                    devtype="Corrector Horizontal"
                )
                cmnames.extend(hcmnames)
                cmidcs.extend(hcmidcs)
                self.nhcms = len(hcmnames)
            if "V" in plane.upper():
                vcmidcs, vcmnames = self._get_cm_names(
                    devtype="Corrector Vertical"
                )
                cmnames.extend(vcmnames)
                cmidcs.extend(vcmidcs)
                self.nvcms = len(vcmnames)

        cmdevs = [PowerSupply(dev, auto_monitor_mon=False) for dev in cmnames]
        super().__init__(cmdevs)
        self._cm_names = cmnames
        self._cm_idcs = cmidcs

    def _get_cm_names(self, devtype):
        cmidcs, cmnames = zip(
            *sorted(
                (amap["sim_info"]["indices"], alias)
                for alias, amap in ALIAS_MAP.items()
                if amap["accelerator"] == self.accelerator
                and amap["cs_devtype"] == devtype
            )
        )
        return list(cmidcs), list(cmnames)

    @property
    def cm_names(self):
        """."""
        return self._cm_names

    @property
    def cm_indices(self):
        """."""
        return self._cm_idcs

    @property
    def hcms(self):
        """."""
        return self.devices[:self.nhcms]

    @property
    def vcms(self):
        """."""
        return self.devices[self.nhcms : self.nhcms + self.nvcms]

    def get_currents(self):
        """."""
        return [cm.current for cm in self.devices]

    def set_currents(self, currents):
        """."""
        for i, cm in enumerate(self.devices):
            cm.current = currents[i]
