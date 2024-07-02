"""."""

from .. import _get_facility
from .base import Device

MIN_CURRENT = 0.01  # [mA]


class DCCT(Device):
    """."""

    PROPERTIES_DEFAULT = ("current",)

    def __init__(self, devname):
        """."""
        super().__init__(devname, props2init=DCCT.PROPERTIES_DEFAULT)

        facil = _get_facility()
        if facil.check_alias_in_csdevtype(devname, facil.CSDevType.DCCT):
            raise ValueError(f"Device name: {devname} not valid for a DCCT.")

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
