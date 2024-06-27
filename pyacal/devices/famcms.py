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
                self.nr_hcms = len(hcmnames)
            if "V" in plane.upper():
                vcmnames = self._get_cm_names(devtype="Corrector Vertical")
                cmnames.extend(vcmnames)
                self.nr_vcms = len(vcmnames)

        cmdevs = [_PowerSupply(dev) for dev in cmnames]

        super().__init__(cmdevs)
        self._cm_names = cmnames

    def _get_cm_names(self, devtype):
        names = [
            alias
            for alias, amap in FACILITY.alias_map.items()
            if amap["accelerator"] == self.accelerator
            and devtype in amap["cs_devtype"]
        ]
        names.sort(
            key=lambda alias: FACILITY.alias_map[alias]["sim_info"]["indices"]
        )
        return names

    @property
    def cm_names(self):
        """."""
        return self._cm_names

    @property
    def hcms(self):
        """."""
        return self.devices[: self.nr_hcms]

    @property
    def vcms(self):
        """."""
        return self.devices[self.nr_hcms :]

    @property
    def currents_hcm(self):
        """."""
        return _np.array([cm.current for cm in self.hcms])

    @property
    def currents_vcm(self):
        """."""
        return _np.array([cm.current for cm in self.vcms])

    def get_currents(self):
        """."""
        return _np.array([cm.current for cm in self.devices])

    def set_currents(self, currents):
        """."""
        for dev, curr in zip(self.devices, currents):
            if curr is None or _np.isnan(curr):
                continue
            dev.current = curr

    def wait_currents(self, currents, timeout=10):
        """."""
        return self.wait_devices_propty(
            self.devices,
            'current_rb',
            currents,
            comp='isclose',
            timeout=timeout,
            abs_tol=_PowerSupply.TINY_CURRENT,
            rel_tol=0,
        )
