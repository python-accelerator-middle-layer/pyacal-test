from .. import FACILITY

from .base import DeviceSet
from . import get_device_class as _get_class


class FamCMs(DeviceSet):
    """."""

    def __init__(
        self, accelerator=None, cmnames=None, plane="HV"
    ):
        """."""
        cmidcs = []
        self.accelerator = accelerator or FACILITY.default_accelerator
        if cmnames is None:
            cmnames = []
            plane = plane.upper()
            if "H" in plane:
                hcmidcs, hcmnames = self._get_cm_names(
                    devtype="Corrector Horizontal"
                )
                cmnames.extend(hcmnames)
                cmidcs.extend(hcmidcs)
            if "V" in plane.upper():
                vcmidcs, vcmnames = self._get_cm_names(
                    devtype="Corrector Vertical"
                )
                cmnames.extend(vcmnames)
                cmidcs.extend(vcmidcs)

        _ps_class = _get_class('PowerSupply')
        cmdevs = [_ps_class(dev, auto_monitor_mon=False) for dev in cmnames]
        super().__init__(cmdevs)
        self._cm_names = cmnames
        self._cm_idcs = cmidcs

    def _get_cm_names(self, devtype):
        cmidcs, cmnames = zip(
            *sorted(
                (amap["sim_info"]["indices"], alias)
                for alias, amap in FACILITY.alias_map.items()
                if amap["accelerator"] == self.accelerator
                and devtype in amap["cs_devtype"]
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

    def get_currents(self):
        """."""
        return [cm.current for cm in self.devices]

    def set_currents(self, currents):
        """."""
        for i, cm in enumerate(self.devices):
            cm.current = currents[i]
