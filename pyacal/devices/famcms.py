"""."""

import numpy as _np

from .. import _get_facility, _get_devices
from .base import DeviceSet


class FamCMs(DeviceSet):
    """."""

    def __init__(self, accelerator=None, cmnames=None, plane="HV"):
        """."""
        facil = _get_facility()
        devices = _get_devices()
        self.accelerator = accelerator or facil.default_accelerator
        if cmnames is None:
            cmnames = self._get_default_cmnames(plane)
        cmdevs = [devices.PowerSupply(dev) for dev in cmnames]

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
    def strengths_hcm(self):
        """."""
        return _np.array([cm.strength for cm in self.hcms])

    @property
    def strengths_vcm(self):
        """."""
        return _np.array([cm.strength for cm in self.vcms])

    def get_strengths(self):
        """."""
        return _np.array([cm.strength for cm in self.devices])

    def set_strengths(self, strengths):
        """."""
        for dev, stren in zip(self.devices, strengths):
            if stren is None or _np.isnan(stren):
                continue
            dev.strength = stren

    def wait_strengths(self, strengths, timeout=10):
        """."""
        return self.wait_devices_propty(
            self.devices,
            'strength_rb',
            strengths,
            comp='isclose',
            timeout=timeout,
            abs_tol=_get_devices().PowerSupply.TINY_STRENGTH,
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
        return facil.sort_aliases_by_model_positions(cmnames)
