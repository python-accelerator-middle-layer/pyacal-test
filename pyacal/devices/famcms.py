"""."""

import numpy as _np

from .. import _get_facility
from .base import DeviceSet
from .power_supply import PowerSupply as _PowerSupply


class FamCMs(DeviceSet):
    """."""

    def __init__(self, accelerator=None, cmnames=None, plane="HV"):
        """."""
        facil = _get_facility()
        self.accelerator = accelerator or facil.default_accelerator
        if cmnames is None:
            cmnames = self._get_default_cmnames(plane)
        cmdevs = [_PowerSupply(dev) for dev in cmnames]

        super().__init__(cmdevs)
        self._cm_names = cmnames

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

    # -------------------- helper methods ---------------------
    def _get_default_cmnames(self, plane):
        cmnames = []
        plane = plane.upper()
        if "H" in plane:
            hcmnames = self._get_cm_names('H')
            cmnames.extend(hcmnames)
            self.nr_hcms = len(hcmnames)
        if "V" in plane.upper():
            vcmnames = self._get_cm_names('V')
            cmnames.extend(vcmnames)
            self.nr_vcms = len(vcmnames)
        return cmnames

    def _get_cm_names(self, plane):
        facil = _get_facility()

        devtype = facil.CSDevTypes.CorrectorHorizontal if "H" in plane else \
            facil.CSDevTypes.CorrectorVertical

        cmnames = facil.find_aliases_from_accelerator(self.accelerator)
        cmnames = facil.find_aliases_from_cs_devtype(
            {facil.CSDevTypes.SOFB, devtype}, aliases=cmnames
        )
        return facil.sort_aliases_by_indices(cmnames)
