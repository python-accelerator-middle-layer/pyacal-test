"""."""

import numpy as _np

from .. import FACILITY
from .base import DeviceSet
from .power_supply import PowerSupply as _PowerSupply


class FamCMs(DeviceSet):
    """."""

    def __init__(self, accelerator=None, cmnames=None, plane="HV"):
        """."""
        self.accelerator = accelerator or FACILITY.default_accelerator
        if cmnames is None:
            cmnames = []
            plane = plane.upper()
            if "H" in plane:
                hcmnames = self._get_cm_names(devtype="Corrector Horizontal")
                cmnames.extend(hcmnames)
                self.nhcms = len(hcmnames)
            if "V" in plane.upper():
                vcmnames = self._get_cm_names(devtype="Corrector Vertical")
                cmnames.extend(vcmnames)
                self.nvcms = len(vcmnames)

        cmdevs = [_PowerSupply(dev) for dev in cmnames]
        super().__init__(cmdevs)
        self._cm_names = cmnames

    def _get_cm_names(self, devtype):
        cmnames = [
            alias
            for alias, amap in FACILITY.alias_map.items()
            if amap["accelerator"] == self.accelerator
            and devtype in amap["cs_devtype"]
        ]
        return cmnames.sort(
            key=lambda alias: FACILITY.alias_map[alias]["sim_info"]["indices"]
        )

    @property
    def cm_names(self):
        """."""
        return self._cm_names

    @property
    def hcms(self):
        """."""
        return self.devices[: self.nhcms]

    @property
    def vcms(self):
        """."""
        return self.devices[self.nhcms : self.nhcms + self.nvcms]

    def get_currents(self):
        """."""
        return _np.array([cm.current for cm in self.devices])

    def set_currents(self, currents):
        """."""
        for i, cm in enumerate(self.devices):
            cm.current = currents[i]
