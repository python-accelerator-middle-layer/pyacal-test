"""."""

import numpy as _np

from ... import _get_facility, _get_devices
from ..base import DeviceSet, Device


class FamBPMs(DeviceSet):
    """."""

    def __init__(self, accelerator=None, bpmnames=None):
        """."""
        facil = _get_facility()
        devices = _get_devices()
        self.accelerator = accelerator or facil.default_accelerator
        devnames = facil.find_aliases_from_accelerator(self.accelerator)
        if bpmnames is None:
            bpmnames = self._get_default_bpmnames(facil, devnames)
        bpmdevs = [devices.BPM(dev, auto_monitor_mon=False) for dev in bpmnames]
        super().__init__(bpmdevs)
        self._bpm_names = bpmnames
        bpmfam = facil.find_aliases_from_cs_devtype(
            {facil.CSDevTypes.BPM, facil.CSDevTypes.Family}, aliases=devnames)
        self.famdev = Device(bpmfam[0], props2init=['orbx', 'orby'])
        self.indx = [facil.get_attribute_from_aliases(
            'cs_propties.posx.index', devname) for devname in bpmnames]
        self.indy = [facil.get_attribute_from_aliases(
            'cs_propties.posy.index', devname) for devname in bpmnames]

    @property
    def bpm_names(self):
        """."""
        return self._bpm_names

    @property
    def orbx(self):
        """."""
        return self.famdev['orbx'][self.indx]

    @property
    def orby(self):
        """."""
        return self.famdev['orby'][self.indy]

    # ---------------- helper methods -----------------------
    @staticmethod
    def _get_default_bpmnames(facil, devnames):
        bpmnames = facil.find_aliases_from_cs_devtype(
            {facil.CSDevTypes.BPM, facil.CSDevTypes.SOFB}, aliases=devnames,
        )
        return facil.sort_aliases_by_model_positions(bpmnames)
