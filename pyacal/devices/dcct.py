"""."""

from .. import _get_facility
from .base import Device

MIN_CURRENT = 0.01  # [mA]


class DCCT(Device):
    """."""

    PROPERTIES_DEFAULT = ("current",)

    def __init__(self, devname=None, accelerator=None):
        """."""
        fac = _get_facility()
        self.accelerator = accelerator or fac.default_accelerator
        if devname is None:
            devname = fac.find_aliases_from_cs_devtype(fac.CSDevTypes.DCCT)
            devname = fac.find_aliases_from_accelerator(
                self.accelerator, devname)[0]

        if not fac.is_alias_in_cs_devtype(devname, fac.CSDevTypes.DCCT):
            raise ValueError(f"Device name: {devname} not valid for a DCCT.")

        super().__init__(devname, props2init=DCCT.PROPERTIES_DEFAULT)

    @property
    def current(self):
        """."""
        return self["current"]

    @property
    def storedbeam(self):
        """."""
        return self.current > MIN_CURRENT

    @property
    def havebeam(self):
        """."""
        return self.connected and self.storedbeam
